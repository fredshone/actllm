import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_finetune_data import (
    _build_attr_dict,
    _build_schedule,
    _minutes_to_hhmm,
    _stratified_split,
)
from finetune import _CompletionOnlyCollator


# ---------------------------------------------------------------------------
# _minutes_to_hhmm
# ---------------------------------------------------------------------------


def test_minutes_midnight():
    assert _minutes_to_hhmm(0) == "00:00"


def test_minutes_noon():
    assert _minutes_to_hhmm(720) == "12:00"


def test_minutes_half_past():
    assert _minutes_to_hhmm(630) == "10:30"


def test_minutes_single_digit_padding():
    assert _minutes_to_hhmm(65) == "01:05"


def test_minutes_near_midnight():
    assert _minutes_to_hhmm(1435) == "23:55"


# ---------------------------------------------------------------------------
# _build_attr_dict
# ---------------------------------------------------------------------------


def test_attr_dict_drops_nan():
    row = pd.Series({"age": "30-40", "employment": float("nan"), "sex": "female"})
    result = _build_attr_dict(row, ["age", "employment", "sex"])
    assert "employment" not in result
    assert result == {"age": "30-40", "sex": "female"}


def test_attr_dict_drops_field_not_in_row():
    row = pd.Series({"age": "30-40"})
    result = _build_attr_dict(row, ["age", "hh_zone"])
    assert "hh_zone" not in result
    assert "age" in result


def test_attr_dict_keeps_zero():
    # 0 is falsy but not NaN — must be kept
    row = pd.Series({"vehicles": 0.0, "age": "30-40"})
    result = _build_attr_dict(row, ["vehicles", "age"])
    assert result["vehicles"] == 0.0


def test_attr_dict_all_nan_returns_empty():
    row = pd.Series({"age": float("nan"), "employment": float("nan")})
    result = _build_attr_dict(row, ["age", "employment"])
    assert result == {}


# ---------------------------------------------------------------------------
# _build_schedule
# ---------------------------------------------------------------------------


def test_build_schedule_basic():
    group = pd.DataFrame({"act": ["home", "work", "home"], "start": [0, 480, 1020]})
    assert _build_schedule(group) == [
        {"home": "00:00"},
        {"work": "08:00"},
        {"home": "17:00"},
    ]


def test_build_schedule_preserves_activity_order():
    group = pd.DataFrame({
        "act": ["home", "escort", "home", "shop", "home"],
        "start": [0, 540, 600, 900, 960],
    })
    result = _build_schedule(group)
    assert [list(d.keys())[0] for d in result] == ["home", "escort", "home", "shop", "home"]
    assert [list(d.values())[0] for d in result] == ["00:00", "09:00", "10:00", "15:00", "16:00"]


def test_build_schedule_float_start_times():
    # pandas reads CSV integers as int64 but may encounter floats
    group = pd.DataFrame({"act": ["home", "work"], "start": [0.0, 510.0]})
    result = _build_schedule(group)
    assert result == [{"home": "00:00"}, {"work": "08:30"}]


# ---------------------------------------------------------------------------
# _stratified_split
# ---------------------------------------------------------------------------


def test_split_total_preserved():
    examples = [{"_strata": ("a",), "n": i} for i in range(100)]
    train, val = _stratified_split(examples, seed=42)
    assert len(train) + len(val) == 100


def test_split_approximate_ratio():
    examples = [{"_strata": ("a",), "n": i} for i in range(100)]
    _, val = _stratified_split(examples, seed=42)
    assert 8 <= len(val) <= 12  # ~10% ± rounding


def test_split_all_strata_in_val():
    examples = (
        [{"_strata": ("young",), "n": i} for i in range(20)]
        + [{"_strata": ("old",), "n": i} for i in range(20)]
    )
    _, val = _stratified_split(examples, seed=42)
    val_strata = {e["_strata"][0] for e in val}
    assert {"young", "old"} == val_strata


def test_split_singleton_stratum_goes_to_val():
    # max(1, round(1 * 0.1)) == 1, so the single example goes to val
    examples = [{"_strata": ("only",), "n": 0}]
    train, val = _stratified_split(examples, seed=42)
    assert len(val) == 1
    assert len(train) == 0


def test_split_reproducible():
    examples = [{"_strata": (str(i % 5),), "n": i} for i in range(50)]
    t1, v1 = _stratified_split(examples, seed=42)
    t2, v2 = _stratified_split(examples, seed=42)
    assert [e["n"] for e in t1] == [e["n"] for e in t2]
    assert [e["n"] for e in v1] == [e["n"] for e in v2]


# ---------------------------------------------------------------------------
# _CompletionOnlyCollator
# ---------------------------------------------------------------------------


class _MockTokenizer:
    pad_token_id = 0

    def pad(self, features: list[dict], padding: bool = True, return_tensors: str | None = None) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask = [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [0] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


TEMPLATE = [10, 20, 30]


def _make_collator() -> _CompletionOnlyCollator:
    return _CompletionOnlyCollator(tokenizer=_MockTokenizer(), response_template_ids=TEMPLATE)


def test_collator_masks_prompt_keeps_completion():
    # template at position 2; tokens after it (5, 6, 7) are supervised
    ids = [1, 2, 10, 20, 30, 5, 6, 7]
    features = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = _make_collator()(features)
    labels = batch["labels"][0].tolist()
    assert labels[:5] == [-100] * 5
    assert labels[5:] == [5, 6, 7]


def test_collator_masks_padding():
    ids_a = [1, 2, 10, 20, 30, 5, 6]
    ids_b = [1, 10, 20, 30, 9]  # shorter — padded to length 7
    features = [
        {"input_ids": ids_a, "attention_mask": [1] * len(ids_a)},
        {"input_ids": ids_b, "attention_mask": [1] * len(ids_b)},
    ]
    batch = _make_collator()(features)
    # last token of ids_b row is padding (0) — must be -100 in labels
    assert batch["labels"][1, -1].item() == -100


def test_collator_template_not_found_masks_all():
    ids = [1, 2, 3, 4, 5]
    features = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = _make_collator()(features)
    assert all(l == -100 for l in batch["labels"][0].tolist())


def test_collator_template_at_start():
    # template is the very first tokens — nothing supervised before it,
    # everything after should be unmasked
    ids = [10, 20, 30, 7, 8, 9]
    features = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = _make_collator()(features)
    labels = batch["labels"][0].tolist()
    assert labels[:3] == [-100] * 3
    assert labels[3:] == [7, 8, 9]


def test_collator_batch_independence():
    # Two examples with templates at different positions should be masked independently
    ids_a = [1, 10, 20, 30, 5]   # template at pos 1
    ids_b = [1, 2, 3, 10, 20, 30, 9]  # template at pos 3
    features = [
        {"input_ids": ids_a, "attention_mask": [1] * len(ids_a)},
        {"input_ids": ids_b, "attention_mask": [1] * len(ids_b)},
    ]
    batch = _make_collator()(features)
    labels_a = batch["labels"][0].tolist()
    labels_b = batch["labels"][1].tolist()
    # ids_a padded to 7: [1, 10, 20, 30, 5, 0, 0]
    assert labels_a[4] == 5      # supervised
    assert labels_a[5] == -100   # padding
    # ids_b: template at 3, supervised = [9]
    assert labels_b[6] == 9
    assert labels_b[3] == -100   # template itself masked
