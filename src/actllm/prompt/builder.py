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
    "age": "Age",
    "gender": "Gender",
    "sex": "Sex",
    "car_access": "Car access",
    "work_status": "Employment status",
    "income": "Annual household income",
    "area": "Area type",
    "hh_children": "Has children in household",
    "hh_size": "Household number of members",
    "education": "Education level",
    "license": "Driving licence",
    "dwelling": "Dwelling type",
    "ownership": "Home ownership",
    "vehicles": "Household number of vehicles",
    "disability": "Disability status",
    "can_wfh": "Can work from home",
    "occupation": "Occupation",
    "race": "Race/ethnicity",
    "has_licence": "Has driving licence",
    "relationship": "Relationship to head of household",
    "employment": "Employment status",
    "country": "Country",
    "source": "Data source",
    "year": "Year of survey",
    "month": "Month of survey",
    "day": "Day of survey",
    "hh_income": "Annual household income (euros)",
    "hh_zone": "Household location type",
    "weight": "Survey weight",
    "access_egress_distance": "Access/egress distance (km) to public transit",
    "max_temp_c": "Max day temperature (°C) on day of survey",
    "rain": "Rain (mm) on day of survey",
    "avg_speed": "Average speed (km/h) on day of survey",
}

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
            label = _ATTR_LABELS.get(field, field.replace("_", " ").title())
            lines.append(f"- {label}: {value}")
        return "\n".join(lines) if lines else "- No attributes provided"

    def _render_examples(self, examples: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for i, ex in enumerate(examples, 1):
            attrs_block = self._render_attributes(ex.get("attributes", {}))
            sched_json = json.dumps({"schedule": ex.get("schedule", [])})
            blocks.append(
                f"### Example {i}\nPerson:\n{attrs_block}\n\nSchedule:\n{sched_json}"
            )
        return "\n\n".join(blocks)
