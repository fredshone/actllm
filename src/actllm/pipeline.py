from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jsonlines
import pandas as pd
import yaml
from tqdm import tqdm

from .api.client import LLMClient, ModelConfig
from .api.retry import RetryState
from .parse.parser import ResponseParser
from .parse.validator import Validator
from .prompt.builder import PromptBuilder
from .prompt.few_shot import FewShotSelector
from .schedule.schema import Activity, ActivityType, GenerationRecord, Schedule

logger = logging.getLogger(__name__)
_prompt_logger = logging.getLogger("actllm.prompts")


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class ScheduleGenerator:
    def __init__(self, config_path: Path) -> None:
        cfg = _load_config(config_path)
        self._cfg = cfg
        self._model_config = ModelConfig.from_dict(cfg["model"])
        self._client = LLMClient(self._model_config)

        prompt_cfg = cfg.get("prompt", {})
        mode = prompt_cfg.get("mode", "zero_shot")
        n_examples = prompt_cfg.get("n_examples", 0)
        example_selection = prompt_cfg.get("example_selection", "stratified")

        few_shot_selector: FewShotSelector | None = None
        if mode in ("few_shot", "cot_few_shot") and n_examples > 0:
            pool_path = Path(
                prompt_cfg.get("few_shot_pool", "data/few_shot/pool.jsonl")
            )
            few_shot_selector = FewShotSelector(pool_path)

        self._builder = PromptBuilder(
            attribute_fields=cfg["attributes"]["fields"],
            mode=mode,
            few_shot_selector=few_shot_selector,
            n_examples=n_examples,
            example_selection=example_selection,
        )
        self._parser = ResponseParser()
        self._validator = Validator()

    def generate(self, attributes: dict[str, Any], record_id: str) -> GenerationRecord:
        system_prompt = self._builder.system_prompt
        user_prompt = self._builder.build(attributes)

        state = RetryState(
            schedule=None,
            raw_response="",
            attempts=0,
            valid=False,
            violations=[],
            fallback_level=3,
        )

        max_retries = self._model_config.max_retries
        prompt = user_prompt

        for attempt in range(1, max_retries + 1):
            state.attempts = attempt
            logger.debug("Attempt %d/%d for record %s", attempt, max_retries, record_id)

            _prompt_logger.debug(
                "=== SYSTEM [%s] ===\n%s\n=== USER [%s attempt %d] ===\n%s",
                record_id,
                system_prompt,
                record_id,
                attempt,
                prompt,
            )
            raw = self._client.generate(system_prompt, prompt)
            state.raw_response = raw
            _prompt_logger.debug("=== RESPONSE [%s attempt %d] ===\n%s", record_id, attempt, raw)

            schedule, fallback_level, parse_errors = self._parser.parse(raw)
            state.fallback_level = fallback_level

            if schedule is None:
                errors = ["Could not parse a valid JSON schedule from the response."] + parse_errors
                prompt = self._builder.build_retry(attributes, errors)
                continue

            result = self._validator.validate(schedule)
            state.schedule = schedule
            state.valid = result.valid
            state.violations = result.violations

            if result.valid:
                break

            # Build retry prompt with specific violations
            prompt = self._builder.build_retry(attributes, result.violations)

        schedule_data: list[dict[str, Any]] | None = None
        if state.schedule is not None:
            schedule_data = [
                {"activity": str(a.activity), "start": a.start}
                for a in state.schedule.activities
            ]

        return GenerationRecord(
            id=record_id,
            attributes=attributes,
            schedule=schedule_data,
            valid=state.valid,
            violations=state.violations,
            retries=state.attempts - 1,
            model=self._model_config.name,
            prompt_mode=self._cfg["prompt"]["mode"],
            raw_response=state.raw_response,
        )

    def export_csv(self, jsonl_path: Path, csv_path: Path) -> int:
        """Convert a generated JSONL file to CSV. Returns number of rows written."""
        all_rows: list[dict[str, Any]] = []
        with jsonlines.open(jsonl_path) as reader:
            for record in reader:
                if not record.get("valid") or record.get("schedule") is None:
                    continue
                schedule = Schedule(
                    activities=[
                        Activity(activity=ActivityType(a["activity"]), start=a["start"])
                        for a in record["schedule"]
                    ]
                )
                all_rows.extend(schedule.to_nts_rows(str(record["id"])))
        if all_rows:
            pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        else:
            pd.DataFrame(
                columns=["pid", "hid", "act", "start", "end", "duration"]
            ).to_csv(csv_path, index=False)
        logger.info("Exported %d rows to %s", len(all_rows), csv_path)
        return len(all_rows)

    def generate_batch(
        self,
        records: list[dict[str, Any]],
        output_path: Path,
        csv_path: Path | None = None,
    ) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        concurrency = self._cfg.get("concurrency", 1)
        n_valid = 0
        n_invalid = 0
        parse_failures = 0
        violation_counts: dict[str, int] = {}
        lock = threading.Lock()

        def _task(record: dict[str, Any], idx: int) -> GenerationRecord:
            record_id = str(record.get("pid", idx))
            attrs = {
                field: record.get(field) for field in self._cfg["attributes"]["fields"]
            }
            return self.generate(attrs, record_id)

        with jsonlines.open(output_path, mode="w") as writer:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(_task, record, i): i
                    for i, record in enumerate(records, 1)
                }
                with tqdm(total=len(records), unit="schedule") as bar:
                    for future in as_completed(futures):
                        gen = future.result()
                        with lock:
                            writer.write(json.loads(gen.model_dump_json()))
                            if gen.schedule is None:
                                parse_failures += 1
                                n_invalid += 1
                            elif gen.valid:
                                n_valid += 1
                            else:
                                n_invalid += 1
                                for v in gen.violations:
                                    key = v.split(":")[0]
                                    violation_counts[key] = (
                                        violation_counts.get(key, 0) + 1
                                    )
                            bar.update(1)

        if csv_path is not None:
            self.export_csv(output_path, csv_path)

        total = len(records)
        return {
            "total": total,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "parse_failures": parse_failures,
            "parse_success_rate": (total - parse_failures) / total if total else 0.0,
            "validity_rate": n_valid / total if total else 0.0,
            "violation_counts": violation_counts,
        }
