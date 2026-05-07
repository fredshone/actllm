import pytest

from actllm.parse.validator import Validator
from actllm.schedule.schema import Activity, ActivityType, Schedule


def make_schedule(*pairs: tuple[str, str]) -> Schedule:
    return Schedule(
        activities=[Activity(activity=ActivityType(act), start=t) for act, t in pairs]
    )


def test_valid_schedule_passes():
    validator = Validator()
    schedule = make_schedule(("home", "00:00"), ("work", "08:00"), ("home", "17:00"))
    result = validator.validate(schedule)
    assert result.valid
    assert result.violations == []



def test_v3_non_monotonic():
    validator = Validator()
    schedule = Schedule.__new__(Schedule)
    object.__setattr__(schedule, "activities", [
        Activity(activity=ActivityType.home, start="00:00"),
        Activity(activity=ActivityType.work, start="08:00"),
        Activity(activity=ActivityType.home, start="05:00"),
    ])
    result = validator.validate(schedule)
    assert not result.valid
    assert any("V3" in v for v in result.violations)


def test_v4_invalid_hhmm_format():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Activity(activity=ActivityType.home, start="25:00")


def test_v1_invalid_activity_caught_by_enum():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Activity(activity="leisure", start="00:00")  # type: ignore[arg-type]


def test_multiple_violations_all_reported():
    validator = Validator()
    schedule = Schedule.__new__(Schedule)
    object.__setattr__(schedule, "activities", [
        Activity(activity=ActivityType.home, start="00:00"),
        Activity(activity=ActivityType.work, start="08:00"),
        Activity(activity=ActivityType.work, start="05:00"),
    ])
    result = validator.validate(schedule)
    assert not result.valid
    assert len(result.violations) >= 2
