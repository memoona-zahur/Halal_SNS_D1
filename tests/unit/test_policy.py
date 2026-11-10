from __future__ import annotations

import math
from dataclasses import replace

import pytest

from moderation.domain import (
    CATEGORIES,
    ItemResult,
    ItemSource,
    Label,
    TextScore,
    TextTier,
)
from moderation.policy import D1_POLICY, Policy, Threshold, worst_item


@pytest.fixture()
def policy() -> Policy:
    return D1_POLICY


def test_default_policy_values(policy: Policy) -> None:
    assert policy.version == "d1.6.1"
    assert policy.ocr_min_confidence == 0.6
    assert policy.ocr_min_chars == 3


# --- surfaces -> line ------------------------------------------------------------


def test_post_is_accepted(policy: Policy) -> None:
    assert policy.accepts("post")


@pytest.mark.parametrize("surface", ["POST", "unknown", "", "post ", "comment", "dm", "reel"])
def test_surfaces_outside_d1_are_not_accepted(policy: Policy, surface: str) -> None:
    assert not policy.accepts(surface)


# --- text ------------------------------------------------------------------------


def test_text_unsafe_mapped_category_blocks(policy: Policy) -> None:
    item = policy.judge_text(TextScore("Unsafe", "Violent", 0.97))
    assert item == ItemResult(ItemSource.TEXT, None, Label.BLOCK, "G-01", 0.97)


def test_text_controversial_mapped_category_holds(policy: Policy) -> None:
    item = policy.judge_text(TextScore("Controversial", "Unethical Acts", 0.6))
    assert item.label is Label.HOLD and item.category == "G-04"


def test_text_safe_allows(policy: Policy) -> None:
    item = policy.judge_text(TextScore("Safe", None, 0.99))
    assert item.label is Label.ALLOW and item.category is None


@pytest.mark.parametrize("category", ["Politically Sensitive Topics", "Copyright Violation", "Jailbreak"])
def test_allow_and_log_lowers_a_hold_to_allow(policy: Policy, category: str) -> None:
    item = policy.judge_text(TextScore("Controversial", category, 0.6))
    assert item.label is Label.ALLOW and item.category is None


@pytest.mark.parametrize("category", ["Politically Sensitive Topics", "Copyright Violation", "Jailbreak"])
def test_unsafe_is_blocked_whatever_the_category(policy: Policy, category: str) -> None:
    assert policy.judge_text(TextScore("Unsafe", category, 0.95)).label is Label.BLOCK


@pytest.mark.parametrize("tier", ["Controversial", "Unsafe"])
def test_pii_is_allowed_at_any_tier(policy: Policy, tier: TextTier) -> None:
    item = policy.judge_text(TextScore(tier, "PII", 0.9))
    assert item.label is Label.ALLOW and item.category is None


def test_unmapped_unsafe_category_blocks_unclassified(policy: Policy) -> None:
    item = policy.judge_text(TextScore("Unsafe", "Something new", 0.9))
    assert item.label is Label.BLOCK and item.category is None


def test_text_nan_score_holds(policy: Policy) -> None:
    assert policy.judge_text(TextScore("Safe", None, math.nan)).label is Label.HOLD


# --- images ----------------------------------------------------------------------


def _low(value: float = 0.02) -> dict[str, float]:
    return {code: value for code in CATEGORIES}


def test_image_between_thresholds_holds(policy: Policy) -> None:
    item = policy.judge_image("m1", _low() | {"G-03": 0.85})
    assert item.label is Label.HOLD and item.category == "G-03" and item.media_id == "m1"


def test_image_above_block_threshold_blocks(policy: Policy) -> None:
    item = policy.judge_image("m1", _low() | {"G-02": 0.9})
    assert item.label is Label.BLOCK and item.category == "G-02"


def test_image_all_low_allows(policy: Policy) -> None:
    item = policy.judge_image("m1", _low(0.1))
    assert item.label is Label.ALLOW and item.category is None and item.score == 0.1


def test_image_override_applies_to_its_category_only() -> None:
    strict = replace(D1_POLICY, image_overrides={"G-03": Threshold(hold=0.2, block=0.4)})
    assert strict.judge_image("m1", _low() | {"G-03": 0.3}).label is Label.HOLD
    assert strict.judge_image("m1", _low() | {"G-02": 0.3}).label is Label.ALLOW


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_image_non_finite_score_holds(policy: Policy, bad: float) -> None:
    assert policy.judge_image("m1", _low() | {"G-05": bad}).label is Label.HOLD


def test_image_missing_code_holds(policy: Policy) -> None:
    scores = {code: 0.02 for code in CATEGORIES if code != "G-06"}
    assert policy.judge_image("m1", scores).label is Label.HOLD


# --- worst wins ------------------------------------------------------------------


def _item(label: Label, score: float | None, category: str | None = None) -> ItemResult:
    return ItemResult(ItemSource.IMAGE, "m1", label, category, score)


def test_block_beats_hold_beats_allow() -> None:
    items = [_item(Label.ALLOW, 0.1), _item(Label.HOLD, 0.6, "G-03"), _item(Label.BLOCK, 0.9, "G-01")]
    assert worst_item(items).label is Label.BLOCK


def test_hold_beats_allow() -> None:
    assert worst_item([_item(Label.ALLOW, 0.1), _item(Label.HOLD, 0.6, "G-03")]).label is Label.HOLD


def test_all_allow_is_allow() -> None:
    assert worst_item([_item(Label.ALLOW, 0.1), _item(Label.ALLOW, 0.2)]).label is Label.ALLOW


def test_ties_go_to_higher_score() -> None:
    assert worst_item([_item(Label.HOLD, 0.55, "G-04"), _item(Label.HOLD, 0.7, "G-05")]).category == "G-05"


def test_item_without_score_does_not_hide_a_real_block() -> None:
    assert worst_item([_item(Label.HOLD, None), _item(Label.BLOCK, 0.95, "G-02")]).label is Label.BLOCK


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        worst_item([])
def test_thresholds_differ_per_category(policy: Policy) -> None:
    """Chosen from the scan of real pictures: adult content needs a higher score than weapons to be held."""
    assert policy.judge_image("m1", _low() | {"G-03": 0.7}).label is Label.ALLOW
    assert policy.judge_image("m1", _low() | {"G-02": 0.7}).label is Label.HOLD
    assert policy.judge_image("m1", _low() | {"G-03": 0.9}).label is Label.HOLD
    assert policy.judge_image("m1", _low() | {"G-03": 0.96}).label is Label.BLOCK
    assert policy.judge_image("m1", _low() | {"G-05": 0.55}).label is Label.HOLD  # no override: default 0.50
