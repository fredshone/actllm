from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..schedule.schema import ActivityType, Schedule

logger = logging.getLogger(__name__)

ALLOWED_ACTIVITIES = {a.value for a in ActivityType}


@dataclass
class ValidationResult:
    valid: bool
    violations: list[str] = field(default_factory=list)


class Validator:
    def validate(self, schedule: Schedule) -> ValidationResult:
        violations: list[str] = []
        acts = schedule.activities

        # V1: all activities in allowed set (already enforced by ActivityType enum,
        # but check explicitly for clear violation messages)
        for i, act in enumerate(acts):
            if act.activity not in ALLOWED_ACTIVITIES:
                violations.append(
                    f"V1: activity '{act.activity}' at index {i} not in allowed set "
                    f"({', '.join(sorted(ALLOWED_ACTIVITIES))})"
                )

        # V2: no consecutive duplicate activities
        for i in range(1, len(acts)):
            if acts[i].activity == acts[i - 1].activity:
                violations.append(
                    f"V2: consecutive duplicate activity '{acts[i].activity}' at index {i}"
                )

        # V3: times strictly increasing (lexicographic comparison valid for zero-padded HH:MM)
        for i in range(1, len(acts)):
            if acts[i].start <= acts[i - 1].start:
                violations.append(
                    f"V3: times not strictly increasing at index {i}: "
                    f"{acts[i - 1].start} >= {acts[i].start}"
                )

        if violations:
            for v in violations:
                logger.warning("Constraint violation: %s", v)

        return ValidationResult(valid=len(violations) == 0, violations=violations)
