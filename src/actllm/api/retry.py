from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schedule.schema import Schedule

logger = logging.getLogger(__name__)


@dataclass
class RetryState:
    schedule: Schedule | None
    raw_response: str
    attempts: int
    valid: bool
    violations: list[str]
    fallback_level: int
