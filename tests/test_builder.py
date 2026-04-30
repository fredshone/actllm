import pytest

from actllm.prompt.builder import PromptBuilder
from actllm.prompt.templates import ZERO_SHOT_TEMPLATE


FIELDS = ["age", "gender", "car_access", "work_status", "income", "area"]


def make_builder() -> PromptBuilder:
    return PromptBuilder(FIELDS)


def test_constraint_block_always_present():
    builder = make_builder()
    prompt = builder.build({"age": 35, "gender": "M"})
    assert "Hard Constraints" in prompt
    assert "home, work, education, shop, medical, other, escort, visit" in prompt
    assert "00:00" in prompt


def test_missing_attrs_omitted():
    builder = make_builder()
    prompt = builder.build({"age": 40})
    assert "Age: 40" in prompt
    assert "Gender" not in prompt
    assert "Car access" not in prompt


def test_all_attrs_rendered():
    builder = make_builder()
    attrs = {
        "age": 30,
        "gender": "F",
        "car_access": "yes",
        "work_status": "employed",
        "income": "high",
        "area": "urban",
    }
    prompt = builder.build(attrs)
    assert "Age: 30" in prompt
    assert "Gender: F" in prompt
    assert "Car access: yes" in prompt
    assert "Employment status: employed" in prompt
    assert "Household income: high" in prompt
    assert "Area type: urban" in prompt


def test_empty_attrs_fallback():
    builder = make_builder()
    prompt = builder.build({})
    assert "No attributes provided" in prompt


def test_retry_prompt_contains_errors():
    builder = make_builder()
    errors = ["V2: consecutive duplicate activity 'home' at index 2"]
    prompt = builder.build_retry({"age": 45}, errors)
    assert "V2: consecutive duplicate" in prompt
    assert "Hard Constraints" in prompt


def test_system_prompt_not_empty():
    builder = make_builder()
    assert len(builder.system_prompt) > 10
