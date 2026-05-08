"""Build fine-tuning train/val splits from NTS activities and binned attributes."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Prepare fine-tuning data from NTS activities and attributes.")

MODEL = "google/gemma-4-E2B-it"
ATTRIBUTE_FIELDS = [
    "age", "sex", "vehicles", "employment",
    "hh_income", "hh_zone", "day", "access_egress_distance",
]

_DEFAULT_ACTIVITIES = Path.home() / "Data/foundata/out/nts/2023/activities.csv"
_DEFAULT_ATTRIBUTES = Path.home() / "Data/foundata/out/nts/2023/attributes_binned.csv"


def _minutes_to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _build_attr_dict(row: pd.Series, fields: list[str]) -> dict:
    return {f: v for f in fields if f in row.index and pd.notna(v := row[f])}


def _build_schedule(group: pd.DataFrame) -> list[dict]:
    return [{act: _minutes_to_hhmm(int(start))} for act, start in zip(group["act"], group["start"])]


def _stratified_split(examples: list[dict], val_fraction: float = 0.1, seed: int = 42) -> tuple[list, list]:
    strata: dict[tuple, list] = defaultdict(list)
    for ex in examples:
        strata[ex["_strata"]].append(ex)
    rng = random.Random(seed)
    train, val = [], []
    for group in strata.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


@app.command()
def main(
    activities: Path = typer.Option(_DEFAULT_ACTIVITIES, "--activities"),
    attributes: Path = typer.Option(_DEFAULT_ATTRIBUTES, "--attributes"),
    output_dir: Path = typer.Option(Path("data/finetune"), "--output-dir"),
    model_name: str = typer.Option(MODEL, "--model"),
    verify: bool = typer.Option(False, "--verify", help="Print token count statistics"),
    max_examples: int = typer.Option(0, "--max-examples", help="Cap total examples (0 = no cap, use for testing)"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from actllm.prompt.builder import PromptBuilder
    from actllm.prompt.templates import SYSTEM_PROMPT

    builder = PromptBuilder(attribute_fields=ATTRIBUTE_FIELDS, mode="zero_shot")

    typer.echo(f"Loading {activities} …")
    acts_df = pd.read_csv(activities).sort_values(["pid", "seq"])

    typer.echo(f"Loading {attributes} …")
    attrs_df = pd.read_csv(attributes).set_index("pid")

    pids = acts_df["pid"].unique()
    typer.echo(f"Building examples for {len(pids):,} persons …")

    examples = []
    for pid, group in acts_df.groupby("pid", sort=False):
        if pid not in attrs_df.index:
            continue
        row = attrs_df.loc[pid]
        attr_dict = _build_attr_dict(row, ATTRIBUTE_FIELDS)
        schedule = _build_schedule(group)
        strata_key = (
            str(row.get("age", "")),
            str(row.get("employment", "")),
            str(row.get("hh_zone", "")),
            str(row.get("day", "")),
        )
        examples.append({"attributes": attr_dict, "schedule": schedule, "_strata": strata_key})

    typer.echo(f"Built {len(examples):,} examples")

    if max_examples and len(examples) > max_examples:
        rng_cap = random.Random(seed)
        examples = rng_cap.sample(examples, max_examples)
        typer.echo(f"Capped to {max_examples:,} examples")

    train, val = _stratified_split(examples, seed=seed)
    rng = random.Random(seed)
    typer.echo(f"Split: {len(train):,} train / {len(val):,} val")

    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_text(ex: dict) -> str:
        user_prompt = builder.build(ex["attributes"])
        schedule_json = json.dumps(ex["schedule"], separators=(",", ":"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": schedule_json},
        ]
        return tokenizer.apply_chat_template(  # type: ignore[no-any-return]
            messages, tokenize=False, add_generation_prompt=False
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("val", val)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps({"text": to_text(ex)}) + "\n")
        typer.echo(f"Wrote {len(split_data):,} examples → {out_path}")

    if verify:
        typer.echo("\n--- Token count stats (sample of 2000) ---")
        sample = rng.sample(train + val, min(2000, len(train) + len(val)))
        counts = sorted(len(tokenizer.encode(to_text(ex))) for ex in sample)
        n = len(counts)
        typer.echo(f"Min:  {counts[0]}")
        typer.echo(f"Max:  {counts[-1]}")
        typer.echo(f"Mean: {sum(counts) / n:.0f}")
        typer.echo(f"P95:  {counts[int(n * 0.95)]}")
        over = sum(1 for c in counts if c > 512)
        if over:
            typer.echo(f"WARNING: {over}/{n} sampled examples exceed 512 tokens", err=True)
        else:
            typer.echo("All sampled examples fit within 512 tokens.")


if __name__ == "__main__":
    app()
