from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


def _vec_age(age: Any) -> float:
    if age == "≤16":
        return 0.0
    if age == "16-34":
        return 0.25
    if age == "34-49":
        return 0.5
    if age == "49-64":
        return 0.75
    if age == ">64":
        return 1.0
    return 0.5


def _vec_employment(ws: Any) -> float:
    if ws.lower() in ("employed", "ft-employed", "pt-employed"):
        return 1.0
    if ws.lower() in ("unemployed", "not working", "retired"):
        return 0.0
    if ws.lower() in ("student", "education"):
        return 0.5
    return 0.5


def _vec_zone(zone: Any) -> float:
    if zone.lower() == "urban":
        return 1.0
    if zone.lower() == "rural":
        return 0.0
    return 0.5


def _vec_day(day: Any) -> float:
    if day.lower() in ["saturday", "sunday"]:
        return 0.0
    if day.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        return 1.0
    return 0.5


def _vec_sex(sex: Any) -> float:
    if sex.lower() in ("m", "male"):
        return 1.0
    if sex.lower() in ("f", "female"):
        return 0.0
    return 0.5


def _vec_vehices(ca: Any) -> float:
    try:
        n = int(ca)
        if n == 0:
            return 0.0
        if n == 1:
            return 0.5
        return 1.0
    except Exception:
        return 0.5


def _vec_income(income: Any) -> float:
    if income == "≤20148":
        return 0.0
    if income == "20148-29499":
        return 0.25
    if income == "29499-40228":
        return 0.5
    if income == "40228-60798":
        return 0.75
    if income == ">60798":
        return 1.0
    return 0.5


def _encode(attributes: dict) -> list[float]:
    vec: list[float] = []

    age = _vec_age(attributes.get("age"))
    vec.append(age)

    ws = _vec_employment(attributes.get("employment"))
    vec.append(ws)

    zone = _vec_zone(attributes.get("hh_zone"))
    vec.append(zone)

    day = _vec_day(attributes.get("day"))
    vec.append(day)

    sex = _vec_sex(attributes.get("sex"))
    vec.append(sex)

    ca = _vec_vehices(attributes.get("vehicles"))
    vec.append(ca)

    income = _vec_income(attributes.get("hh_income"))
    vec.append(income)

    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class FewShotSelector:
    def __init__(self, pool_path: Path) -> None:
        self._pool: list[dict[str, Any]] = []
        with open(pool_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._pool.append(json.loads(line))

    def select(self, attributes: dict, n: int, strategy: str) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        n = min(n, len(self._pool))
        if strategy == "random":
            return random.sample(self._pool, n)
        if strategy == "stratified":
            return self._stratified(attributes, n)
        if strategy == "nearest_neighbour":
            return self._nearest_neighbour(attributes, n)
        raise ValueError(f"Unknown few-shot strategy: {strategy!r}")

    def _stratified(self, attributes: dict, n: int) -> list[dict[str, Any]]:
        target_age = str(attributes.get("age", "unknown"))
        target_ws = str(attributes.get("employment", "unknown"))
        target_zone = str(attributes.get("hh_zone", "unknown"))
        target_day = str(attributes.get("day", "unknown"))

        scored: list[tuple[int, dict[str, Any]]] = []
        for ex in self._pool:
            ex_attrs = ex.get("attributes", {})
            score = 0
            if str(ex_attrs.get("age")) == target_age:
                score += 1
            if str(ex_attrs.get("employment")) == target_ws:
                score += 1
            if str(ex_attrs.get("hh_zone", "unknown")) == target_zone:
                score += 1
            if str(ex_attrs.get("day", "unknown")) == target_day:
                score += 1
            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)

        result: list[dict[str, Any]] = []
        i = 0
        while len(result) < n and i < len(scored):
            tier_score = scored[i][0]
            tier = [ex for s, ex in scored if s == tier_score]
            random.shuffle(tier)
            for ex in tier:
                if len(result) < n:
                    result.append(ex)
            i += len(tier)

        return result[:n]

    def _nearest_neighbour(self, attributes: dict, n: int) -> list[dict[str, Any]]:
        query_vec = _encode(attributes)
        sims = [
            (_cosine(query_vec, _encode(ex.get("attributes", {}))), ex)
            for ex in self._pool
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in sims[:n]]
