"""Curate a few-shot example pool from NTS data.

Samples ~50 constraint-satisfying schedules with diverse attribute strata
and writes them to data/few_shot/pool.jsonl.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Curate few-shot example pool from NTS data.")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ALLOWED_ACTS = {
    "home",
    "work",
    "education",
    "shop",
    "medical",
    "other",
    "escort",
    "visit",
}
ATTR_FIELDS = [
    "age",
    "sex",
    "vehicles",
    "has_licence",
    "employment",
    "hh_income",
    "hh_zone",
    "day",
    "access_egress_distance",
]


def _str(val, default: str = "unknown") -> str:
    if pd.isna(val):
        return default
    return str(val)


def _is_valid(group: pd.DataFrame) -> bool:
    rows = group.sort_values("start")
    if rows.iloc[0]["act"] != "home" or rows.iloc[0]["start"] != 0:
        return False
    acts = rows["act"].tolist()
    if any(a not in ALLOWED_ACTS for a in acts):
        return False
    starts = rows["start"].tolist()
    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            return False
    for i in range(1, len(acts)):
        if acts[i] == acts[i - 1]:
            return False
    if starts[-1] > 1425:
        return False
    return True


@app.command()
def main(
    attrs_path: Path = typer.Option(
        Path("data/attributes.csv"), "--attrs", "-a", help="NTS attributes CSV"
    ),
    sched_path: Path = typer.Option(
        Path("data/schedules.csv"),
        "--schedules",
        "-s",
        help="NTS schedules CSV",
    ),
    out_path: Path = typer.Option(
        Path("data/few_shot/pool.jsonl"), "--output", "-o", help="Output pool JSONL"
    ),
    target_per_stratum: int = typer.Option(
        10,
        "--per-stratum",
        help="Examples to sample per age-band × work-status stratum",
    ),
    seed: int = typer.Option(12345, "--seed", help="Random seed"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show strata analysis without writing output"
    ),
) -> None:
    logger.info("Loading data…")
    attrs = pd.read_csv(attrs_path, index_col="pid")
    schedules = pd.read_csv(sched_path)

    logger.info("Validating schedules…")
    valid_pids: list[str] = []
    for pid, grp in schedules.groupby("pid"):
        if _is_valid(grp):
            valid_pids.append(str(pid))

    logger.info("Valid schedules: %d / %d", len(valid_pids), schedules["pid"].nunique())

    strata: dict[tuple[str, str, str, str, str], list[str]] = {}
    for pid in valid_pids:
        if pid not in attrs.index:
            continue
        row = attrs.loc[pid]
        age = _str(row.get("age"))
        sex = _str(row.get("sex"))
        ws = _str(row.get("employment"))
        zone = _str(row.get("hh_zone"))
        day = _str(row.get("day"))

        strata.setdefault((age, sex, ws, zone, day), []).append(pid)

    logger.info("Strata found: %d", len(strata))
    total_selected = 0
    for k, v in sorted(strata.items()):
        n = min(target_per_stratum, len(v))
        total_selected += n
        logger.info(
            "  %s × %s × %s × %s × %s: %d available, %d selected",
            k[0],
            k[1],
            k[2],
            k[3],
            k[4],
            len(v),
            n,
        )

    logger.info("Total selected: %d", total_selected)

    if dry_run:
        typer.echo(f"\n[dry-run] Would write {total_selected} examples to {out_path}")
        return

    random.seed(seed)
    selected_pids: list[str] = []
    for key, pids in sorted(strata.items()):
        selected_pids.extend(random.sample(pids, min(target_per_stratum, len(pids))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sched_grouped = schedules.groupby("pid")
    written = 0
    with open(out_path, "w") as f:
        for pid in selected_pids:
            if pid not in attrs.index:
                continue
            row = attrs.loc[pid]
            grp = sched_grouped.get_group(pid).sort_values("start")

            attributes: dict = {}
            for field in ATTR_FIELDS:
                if field not in row.index:
                    continue
                val = row[field]
                if hasattr(val, "item"):
                    val = val.item()
                if pd.isna(val):
                    val = "unknown"
                attributes[field] = val

            schedule = [
                {r["act"]: f"{int(r['start']) // 60:02d}:{int(r['start']) % 60:02d}"}
                for _, r in grp.iterrows()
            ]
            age = _str(row.get("age"))
            example = {
                "attributes": attributes,
                "schedule": schedule,
                "stratum": {
                    "age_band": age,
                    "sex": _str(row.get("sex")),
                    "employment": _str(row.get("employment")),
                    "hh_zone": _str(row.get("hh_zone")),
                    "day": _str(row.get("day")),
                },
            }
            f.write(json.dumps(example) + "\n")
            written += 1

    logger.info("Wrote %d examples to %s", written, out_path)


if __name__ == "__main__":
    app()
