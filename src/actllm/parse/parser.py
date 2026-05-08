from __future__ import annotations

import json
import logging
import re

from ..schedule.schema import Activity, ActivityType, Schedule

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\[.*\]", re.DOTALL)
_SCHEDULE_TAG_RE = re.compile(r"<schedule>(.*?)</schedule>", re.DOTALL | re.IGNORECASE)


def _build_schedule(raw_list: list) -> Schedule:
    activities = [
        Activity(
            activity=ActivityType(next(iter(item))), start=next(iter(item.values()))
        )
        for item in raw_list
    ]
    return Schedule(activities=activities)


class ResponseParser:
    def parse(self, raw: str) -> tuple[Schedule | None, int, list[str]]:
        """Return (schedule, fallback_level, errors) where level 0=direct, 1=regex, 2=tag, 3=failed."""
        errors: list[str] = []

        # Level 0: direct JSON parse
        try:
            obj = json.loads(raw.strip())
            schedule = _build_schedule(obj)
            logger.debug("Parsed at fallback level 0 (direct JSON)")
            return schedule, 0, []
        except Exception as e:
            errors.append(f"Direct parse: {e}")

        # Level 1: extract JSON substring via regex
        match = _JSON_BLOCK_RE.search(raw)
        if match:
            try:
                obj = json.loads(match.group())
                schedule = _build_schedule(obj)
                logger.debug("Parsed at fallback level 1 (regex extraction)")
                return schedule, 1, []
            except Exception as e:
                errors.append(f"Regex extraction: {e}")
        else:
            errors.append("Regex extraction: no JSON block found in response")

        # Level 2: XML-style tag extraction
        tag_match = _SCHEDULE_TAG_RE.search(raw)
        if tag_match:
            try:
                obj = json.loads(tag_match.group(1).strip())
                schedule = _build_schedule(obj)
                logger.debug("Parsed at fallback level 2 (tag extraction)")
                return schedule, 2, []
            except Exception as e:
                errors.append(f"Tag extraction: {e}")
        else:
            errors.append("Tag extraction: no <schedule> tag found in response")

        snippet = raw[:200].replace("\n", " ")
        logger.debug(
            "All parse attempts failed. Errors: %s. Response: %.200s", errors, snippet
        )
        return None, 3, errors
