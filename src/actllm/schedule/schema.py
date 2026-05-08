from __future__ import annotations

from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator


def _hhmm_to_min(t: str) -> int:
    hh, mm = map(int, t.split(":"))
    return hh * 60 + mm


class ActivityType(StrEnum):
    home = "home"
    work = "work"
    education = "education"
    shop = "shop"
    medical = "medical"
    other = "other"
    escort = "escort"
    visit = "visit"


class Activity(BaseModel):
    activity: ActivityType
    start: str

    @field_validator("start")
    @classmethod
    def validate_start(cls, v: str) -> str:
        hh, mm = map(int, v.split(":"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(f"start {v} out of range [00:00, 23:59]")
        return v


class Schedule(BaseModel):
    activities: list[Activity]

    @model_validator(mode="after")
    def validate_schedule(self) -> Schedule:
        acts = self.activities
        if not acts:
            raise ValueError("Schedule must have at least one activity")
        if acts[0].start != "00:00":
            raise ValueError("First activity must be at start='00:00'")
        for i in range(1, len(acts)):
            if acts[i].start <= acts[i - 1].start:
                raise ValueError(
                    f"Times not strictly increasing at index {i}: "
                    f"{acts[i - 1].start} >= {acts[i].start}"
                )
        return self

    def durations(self) -> list[dict[str, Any]]:
        result = []
        for i, act in enumerate(self.activities):
            end_str = (
                self.activities[i + 1].start
                if i + 1 < len(self.activities)
                else "24:00"
            )
            start_min = _hhmm_to_min(act.start)
            end_min = _hhmm_to_min(end_str)
            result.append(
                {
                    "activity": act.activity,
                    "start": act.start,
                    "duration": end_min - start_min,
                }
            )
        return result

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.durations())

    def to_nts_rows(self, pid: str) -> list[dict[str, Any]]:
        rows = []
        for i, act in enumerate(self.activities):
            end_str = (
                self.activities[i + 1].start
                if i + 1 < len(self.activities)
                else "24:00"
            )
            start_min = _hhmm_to_min(act.start)
            end_min = _hhmm_to_min(end_str)
            rows.append(
                {
                    "pid": pid,
                    "hid": pid,
                    "act": str(act.activity),
                    "start": start_min,
                    "end": end_min,
                    "duration": end_min - start_min,
                }
            )
        return rows


class GenerationRecord(BaseModel):
    id: str
    attributes: dict[str, Any]
    schedule: list[dict[str, Any]] | None
    valid: bool
    violations: list[str]
    retries: int
    model: str
    prompt_mode: str
    raw_response: str
