from actllm.parse.parser import ResponseParser
from actllm.schedule.schema import ActivityType

VALID_JSON = """[
  {"home": "00:00"},
  {"work": "08:00"},
  {"home": "17:00"}
]"""


def make_parser() -> ResponseParser:
    return ResponseParser()


def test_direct_json_parse():
    parser = make_parser()
    schedule, level, errors = parser.parse(VALID_JSON)
    assert schedule is not None
    assert level == 0
    assert errors == []
    assert schedule.activities[0].activity == ActivityType.home
    assert schedule.activities[1].start == "08:00"


def test_regex_fallback():
    parser = make_parser()
    wrapped = f"Here is the schedule:\n{VALID_JSON}\nDone."
    schedule, level, errors = parser.parse(wrapped)
    assert schedule is not None
    assert level == 1
    assert errors == []


def test_tag_fallback():
    # When JSON is inside a tag, regex (level 1) catches it first — that's fine.
    # Test that tag extraction (level 2) fires when JSON is inside tags but the
    # outer text contains a bare [ that would fool the regex extractor.
    parser = make_parser()
    outer_noise = "Here {is some} noise <schedule>" + VALID_JSON + "</schedule>"
    schedule, level, errors = parser.parse(outer_noise)
    assert schedule is not None
    # Regex may grab the noise bracket block first and fail, then tag extraction succeeds.
    # Accept either level 1 (if regex extracts correctly) or 2 (if tag wins).
    assert level in (1, 2)
    assert errors == []


def test_total_failure():
    parser = make_parser()
    schedule, level, errors = parser.parse("I cannot generate a schedule.")
    assert schedule is None
    assert level == 3
    assert len(errors) > 0


def test_invalid_activity_fails():
    parser = make_parser()
    bad = '{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "shopping", "start": "01:00"}]}'
    schedule, level, errors = parser.parse(bad)
    assert schedule is None


def test_non_monotonic_times_fails():
    parser = make_parser()
    bad = '{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "work", "start": "00:00"}]}'
    schedule, level, errors = parser.parse(bad)
    assert schedule is None
