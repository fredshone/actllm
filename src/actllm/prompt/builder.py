from __future__ import annotations

import json
from typing import Any

from .few_shot import FewShotSelector
from .templates import (
    COT_FEW_SHOT_TEMPLATE,
    COT_SYSTEM_PROMPT,
    COT_ZERO_SHOT_TEMPLATE,
    FEW_SHOT_TEMPLATE,
    RETRY_TEMPLATE,
    SYSTEM_PROMPT,
    ZERO_SHOT_TEMPLATE,
)

_ATTR_LABELS: dict[str, str] = {
    "age": lambda x: f"Age: {x}",
    "gender": lambda x: f"Gender: {x}",
    "sex": lambda x: f"Sex: {x}",
    "car_access": lambda x: f"Car access: {x}",
    "work_status": lambda x: f"Employment status: {x}",
    "income": lambda x: f"Annual household income: {x} Euros",
    "area": lambda x: f"Area type: {x}",
    "hh_children": lambda x: f"Has children in household: {x}",
    "hh_size": lambda x: f"Household number of members: {x}",
    "education": lambda x: f"Education level: {x}",
    "license": lambda x: f"Driving licence: {x}",
    "dwelling": lambda x: f"Dwelling type: {x}",
    "ownership": lambda x: f"Home ownership: {x}",
    "vehicles": lambda x: f"Household number of vehicles: {x}",
    "disability": lambda x: f"Disability status: {x}",
    "can_wfh": lambda x: f"Can work from home: {x}",
    "occupation": lambda x: f"Occupation: {x}",
    "race": lambda x: f"Race/ethnicity: {x}",
    "has_licence": lambda x: f"Has driving licence: {x}",
    "relationship": lambda x: f"Relationship to head of household: {x}",
    "employment": lambda x: f"Employment status: {x}",
    "country": lambda x: f"Country: {x}",
    "source": lambda x: f"Data source: {x}",
    "year": lambda x: f"Year of survey: {x}",
    "month": lambda x: f"Month of survey: {x}",
    "day": lambda x: f"Day of survey: {x}",
    "hh_income": lambda x: f"Household income: {x} Euros per year",
    "hh_zone": lambda x: f"Household location type: {x}",
    "weight": lambda x: f"Survey weight: {x}",
    "access_egress_distance": lambda x: f"Distance to public transit: {x} km",
    "max_temp_c": lambda x: f"Max day temperature on day of survey: {x} °C",
    "rain": lambda x: f"Rain on day of survey: {x} mm",
    "avg_speed": lambda x: f"Average speed on day of survey: {x} km/hr",
}


def _normalise_entry(entry: dict[str, Any]) -> dict[str, str]:
    """Convert verbose {"activity": x, "start": t} to compact {x: t}."""
    if "activity" in entry:
        return {entry["activity"]: entry["start"]}
    return entry


_COT_MODES = {"cot_zero_shot", "cot_few_shot"}
_FEW_SHOT_MODES = {"few_shot", "cot_few_shot"}


class PromptBuilder:
    def __init__(
        self,
        attribute_fields: list[str],
        mode: str = "zero_shot",
        few_shot_selector: FewShotSelector | None = None,
        n_examples: int = 3,
        example_selection: str = "stratified",
    ) -> None:
        self._fields = attribute_fields
        self._mode = mode
        self._few_shot_selector = few_shot_selector
        self._n_examples = n_examples
        self._example_selection = example_selection

    @property
    def system_prompt(self) -> str:
        return COT_SYSTEM_PROMPT if self._mode in _COT_MODES else SYSTEM_PROMPT

    def build(self, attributes: dict) -> str:
        attrs_block = self._render_attributes(attributes)

        if self._mode == "zero_shot":
            return ZERO_SHOT_TEMPLATE.format(attributes_block=attrs_block)

        if self._mode == "few_shot":
            examples = self._get_examples(attributes)
            return FEW_SHOT_TEMPLATE.format(
                attributes_block=attrs_block,
                examples_block=self._render_examples(examples),
            )

        if self._mode == "cot_zero_shot":
            return COT_ZERO_SHOT_TEMPLATE.format(attributes_block=attrs_block)

        if self._mode == "cot_few_shot":
            examples = self._get_examples(attributes)
            return COT_FEW_SHOT_TEMPLATE.format(
                attributes_block=attrs_block,
                examples_block=self._render_examples(examples),
            )

        raise ValueError(f"Unknown prompt mode: {self._mode!r}")

    def build_retry(self, attributes: dict, errors: list[str]) -> str:
        attrs_block = self._render_attributes(attributes)
        errors_block = "\n".join(f"- {e}" for e in errors)
        return RETRY_TEMPLATE.format(
            attributes_block=attrs_block,
            errors_block=errors_block,
        )

    def _get_examples(self, attributes: dict) -> list[dict[str, Any]]:
        if self._few_shot_selector is None or self._n_examples == 0:
            return []
        return self._few_shot_selector.select(
            attributes, self._n_examples, self._example_selection
        )

    def _render_attributes(self, attributes: dict) -> str:
        lines = []
        for field in self._fields:
            value = attributes.get(field)
            if value is None or str(value).strip() == "":
                continue
            labeller = _ATTR_LABELS.get(
                field, lambda x: f"{field.replace("_", " ").title()}: {x}"
            )
            lines.append(f"- {labeller(value)}")
        return "\n".join(lines) if lines else "- No attributes provided"

    def _render_examples(self, examples: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for i, ex in enumerate(examples, 1):
            attrs_block = self._render_attributes(ex.get("attributes", {}))
            schedule = [_normalise_entry(e) for e in ex.get("schedule", [])]
            sched_json = json.dumps(schedule)
            blocks.append(
                f"### Example {i}\nPerson:\n{attrs_block}\n\nSchedule:\n{sched_json}"
            )
        return "\n\n".join(blocks)
