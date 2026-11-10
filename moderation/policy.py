"""Policy: the D1 rules, turned into item decisions, plus surface routing and worst-wins.

Thresholds are uncalibrated starting points. Calibration is D3.
Bump `D1_POLICY.version` on every edit: it is stored with each decision.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from moderation.domain import (
    CATEGORIES,
    ItemResult,
    ItemSource,
    Label,
    TextScore,
    TextTier,
)

log = logging.getLogger(__name__)

_SEVERITY: dict[Label, int] = {Label.ALLOW: 0, Label.HOLD: 1, Label.BLOCK: 2}


@dataclass(frozen=True)
class Threshold:
    hold: float
    block: float


@dataclass(frozen=True)
class Policy:
    version: str
    # Content types accepted. D2-D4 add their surfaces here.
    surfaces: frozenset[str]
    # Qwen3Guard tier -> label for the item.
    tier_labels: Mapping[TextTier, Label]
    # Qwen3Guard category -> G-code. A category in neither map keeps the tier label, unclassified.
    qwen_codes: Mapping[str, str]
    # Qwen3Guard categories whose HOLD is lowered to allow and logged. Never applied to a block (Unsafe).
    # NOT a client decision yet: plan §11 item 6 is open.
    allow_and_log: frozenset[str]
    # Image score thresholds per G-code: one default, with overrides.
    image_default: Threshold
    image_overrides: Mapping[str, Threshold] = field(default_factory=dict)
    # Qwen3Guard categories allowed and logged at any tier, Unsafe included.
    always_allow: frozenset[str] = frozenset()
    # Text found in an image: fewer than ocr_min_chars characters read is no text (a sun, a pattern):
    # ignored, whatever the box count. Otherwise below ocr_min_confidence it is unreadable, so hold.
    ocr_min_confidence: float = 0.6
    ocr_min_chars: int = 3

    def accepts(self, surface: str) -> bool:
        return surface in self.surfaces

    def judge_text(self, score: TextScore) -> ItemResult:
        if not math.isfinite(score.score):
            return ItemResult(ItemSource.TEXT, None, Label.HOLD, None, None)
        base = self.tier_labels[score.tier]
        if base is Label.ALLOW:
            return ItemResult(ItemSource.TEXT, None, Label.ALLOW, None, score.score)
        if score.category in self.always_allow or (base is Label.HOLD and score.category in self.allow_and_log):
            log.info("allow_and_log tier=%s category=%s", score.tier, score.category)
            return ItemResult(ItemSource.TEXT, None, Label.ALLOW, None, score.score)
        return ItemResult(ItemSource.TEXT, None, base, self.qwen_codes.get(score.category or ""), score.score)

    def judge_image(self, media_id: str, scores: Mapping[str, float]) -> ItemResult:
        """Highest-severity category wins. Ties go to the higher score. A broken score holds."""
        if any(code not in scores or not math.isfinite(scores[code]) for code in CATEGORIES):
            return ItemResult(ItemSource.IMAGE, media_id, Label.HOLD, None, None)
        best: ItemResult | None = None
        for code in CATEGORIES:
            value = scores[code]
            threshold = self.image_overrides.get(code, self.image_default)
            if value >= threshold.block:
                level = Label.BLOCK
            elif value >= threshold.hold:
                level = Label.HOLD
            else:
                continue
            if best is None or (_SEVERITY[level], value) > (_SEVERITY[best.label], best.score or 0.0):
                best = ItemResult(ItemSource.IMAGE, media_id, level, code, value)
        if best is not None:
            return best
        return ItemResult(ItemSource.IMAGE, media_id, Label.ALLOW, None, max(scores[c] for c in CATEGORIES))


def worst_item(items: Sequence[ItemResult]) -> ItemResult:
    """block > hold > allow. Ties go to the higher score; an item without a score ranks lowest."""

    def rank(item: ItemResult) -> tuple[int, float]:
        return _SEVERITY[item.label], -math.inf if item.score is None else item.score

    return max(items, key=rank)


D1_POLICY = Policy(
    version="d1.6.1",
    surfaces=frozenset({"post"}),
    tier_labels={"Safe": Label.ALLOW, "Controversial": Label.HOLD, "Unsafe": Label.BLOCK},
    qwen_codes={
        "Violent": "G-01",
        "Non-violent Illegal Acts": "G-07",
        "Sexual Content or Sexual Acts": "G-03",
        "Unethical Acts": "G-04",
        "Suicide & Self-Harm": "G-05",
    },
    allow_and_log=frozenset({"Politically Sensitive Topics", "Copyright Violation", "Jailbreak"}),
    # Personal details alone are allowed. A harmful clause in the same post is judged on its own and
    # can block it (moderate._judge_texts).
    always_allow=frozenset({"PII"}),
    image_default=Threshold(hold=0.5, block=0.85),
    # Chosen from the scan of 347 pictures (local calibration run, since removed): clean pictures flagged fell from 7% to 2.5%.
    # G-03 separates well; G-02 separates poorly; G-01 had only 2 pictures. G-04..G-07 had none and keep the default.
    image_overrides={
        "G-01": Threshold(hold=0.60, block=0.85),
        "G-02": Threshold(hold=0.60, block=0.85),
        "G-03": Threshold(hold=0.80, block=0.95),
    },
)
