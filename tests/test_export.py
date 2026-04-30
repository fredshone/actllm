import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from actllm.schedule.schema import Activity, ActivityType, Schedule


def make_schedule(*pairs: tuple[str, str]) -> Schedule:
    return Schedule(
        activities=[Activity(activity=ActivityType(act), start=t) for act, t in pairs]
    )


def test_to_nts_rows_basic():
    schedule = make_schedule(("home", "00:00"), ("work", "08:00"), ("home", "17:00"))
    rows = schedule.to_nts_rows("42")
    assert len(rows) == 3
    assert rows[0] == {"pid": "42", "hid": "42", "act": "home", "start": 0, "end": 480, "duration": 480}
    assert rows[1] == {"pid": "42", "hid": "42", "act": "work", "start": 480, "end": 1020, "duration": 540}
    assert rows[2] == {"pid": "42", "hid": "42", "act": "home", "start": 1020, "end": 1440, "duration": 420}


def test_to_nts_rows_final_end_is_1440():
    schedule = make_schedule(("home", "00:00"), ("shop", "15:00"))
    rows = schedule.to_nts_rows("x")
    assert rows[-1]["end"] == 1440


def test_to_nts_rows_duration_correct():
    schedule = make_schedule(("home", "00:00"), ("work", "08:00"), ("home", "17:00"))
    rows = schedule.to_nts_rows("1")
    for row in rows:
        assert row["duration"] == row["end"] - row["start"]


def test_to_nts_rows_single_activity():
    schedule = make_schedule(("home", "00:00"),)
    rows = schedule.to_nts_rows("solo")
    assert len(rows) == 1
    assert rows[0]["start"] == 0
    assert rows[0]["end"] == 1440
    assert rows[0]["duration"] == 1440


def test_to_nts_rows_pid_propagated():
    schedule = make_schedule(("home", "00:00"), ("work", "09:00"))
    rows = schedule.to_nts_rows("person_99")
    for row in rows:
        assert row["pid"] == "person_99"
        assert row["hid"] == "person_99"


def test_export_csv_writes_file():
    from actllm.pipeline import ScheduleGenerator

    record = {
        "id": "test_001",
        "attributes": {"age": 35},
        "schedule": [
            {"activity": "home", "start": "00:00"},
            {"activity": "work", "start": "08:00"},
            {"activity": "home", "start": "17:00"},
        ],
        "valid": True,
        "violations": [],
        "retries": 0,
        "model": "test",
        "prompt_mode": "zero_shot",
        "raw_response": "",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test.jsonl"
        csv_path = Path(tmpdir) / "test.csv"

        with open(jsonl_path, "w") as f:
            f.write(json.dumps(record) + "\n")

        # Instantiate generator without API call by using export_csv directly
        # We bypass __init__ since we only need export_csv which doesn't use the LLM
        gen = object.__new__(ScheduleGenerator)
        n_rows = gen.export_csv(jsonl_path, csv_path)

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["pid", "hid", "act", "start", "end", "duration"]
        assert len(df) == 3
        assert n_rows == 3
        assert df.iloc[0]["act"] == "home"
        assert df.iloc[1]["act"] == "work"
        assert df.iloc[2]["end"] == 1440


def test_export_csv_skips_invalid_records():
    from actllm.pipeline import ScheduleGenerator

    records = [
        {
            "id": "valid_1",
            "schedule": [{"activity": "home", "start": "00:00"}, {"activity": "work", "start": "08:00"}],
            "valid": True,
        },
        {
            "id": "invalid_1",
            "schedule": [{"activity": "home", "start": "00:00"}],
            "valid": False,
        },
        {
            "id": "null_sched",
            "schedule": None,
            "valid": False,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test.jsonl"
        csv_path = Path(tmpdir) / "test.csv"

        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        gen = object.__new__(ScheduleGenerator)
        n_rows = gen.export_csv(jsonl_path, csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 2  # only valid_1's 2 activities
        assert n_rows == 2
