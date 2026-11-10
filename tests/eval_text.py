"""Run the labelled captions in tests/data/text_cases.jsonl through the real text pipeline.

    python tests/eval_text.py              # show every case, failures, and changes since the last run

Pass rule: a clean post is never blocked, and a harmful post is never allowed. The last run is kept
in local/eval_text_last.json so a change to the prompt, the policy or the rules shows what moved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's package, even if the venv has another copy installed

from moderation.domain import Decision, Label, ModerationRequest
from moderation.moderate import Moderator
from moderation.policy import D1_POLICY, Policy

CASES_FILE = ROOT / "tests" / "data" / "text_cases.jsonl"
LAST_RUN = ROOT / "local" / "eval_text_last.json"


@dataclass(frozen=True)
class Case:
    expect: str  # "clean" or "harmful"
    lang: str
    note: str
    text: str
    known_gap: bool = False  # the model cannot do this yet; reported, not counted as a test failure


@dataclass(frozen=True)
class Outcome:
    case: Case
    label: Label
    category: str | None
    reason: str

    @property
    def passed(self) -> bool:
        bad = Label.BLOCK if self.case.expect == "clean" else Label.ALLOW
        return self.label is not bad


class _NoImages:
    """Text-only requests never reach the image model or OCR."""

    async def score(self, images: list[bytes]) -> list[Any]:
        return []

    async def read(self, images: list[bytes]) -> list[Any]:
        return []


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    return [Case(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run(text_model: Any, cases: list[Case], policy: Policy = D1_POLICY) -> list[Outcome]:
    moderator = Moderator(text=text_model, image=_NoImages(), ocr=_NoImages(), fetcher=None, policy=policy)
    outcomes = []
    for i, case in enumerate(cases):
        request = ModerationRequest(request_id=f"eval-{i}", surface="post", content_id=f"c{i}", content_version=1, text=case.text)
        decision: Decision = await moderator.moderate(request)
        outcomes.append(Outcome(case, decision.label, decision.category, decision.reason))
    return outcomes


def _report(outcomes: list[Outcome]) -> None:
    last = json.loads(LAST_RUN.read_text(encoding="utf-8")) if LAST_RUN.is_file() else {}
    for o in outcomes:
        before = last.get(o.case.text)
        moved = f"  (was {before})" if before and before != o.label.value else ""
        mark = "ok  " if o.passed else ("GAP " if o.case.known_gap else "FAIL")
        preview = o.case.text.replace("\n", " ")[:70]
        print(f"{mark} {o.case.expect:7} {o.case.lang:8} {o.label.value:5} {o.category or '-':5} {preview}{moved}")
    failed = sum(not o.passed and not o.case.known_gap for o in outcomes)
    gaps = sum(not o.passed and o.case.known_gap for o in outcomes)
    print(f"\n{len(outcomes) - failed - gaps}/{len(outcomes)} pass; {failed} fail; {gaps} known model gap")
    LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(json.dumps({o.case.text: o.label.value for o in outcomes}, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "qwen3guard-gen-0.6b"))
    args = parser.parse_args()

    import torch

    from moderation.classifiers.text_qwen3guard import Qwen3GuardClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    model = Qwen3GuardClassifier.load(Path(args.model_dir), device, dtype)
    try:
        _report(asyncio.run(run(model, load_cases())))
    finally:
        model.close()


if __name__ == "__main__":
    main()
