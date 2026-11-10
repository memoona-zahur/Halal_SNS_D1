"""Qwen3Guard-Gen text classifier.

The model writes its verdict as text ("Safety: Unsafe\nCategories: Violent"). We never let it
generate. We force the prompt up to "Safety:" and read the probabilities of the three tier tokens
from one forward pass (pass 1). Only non-Safe rows get a second pass that reads the category.

The prompt is the model's own chat template plus `CATEGORY_DETAILS` (moderation/prompts.py): detail
text next to some of its category names. The model is fine-tuned on this template. It ignores new
category names, but it does use detail text next to the existing ones.

User text is tokenized with special-token parsing switched off, so special tokens such as
<|im_end|> in a caption stay plain text. The rest of the prompt ("USER:", "<END CONVERSATION>",
"Safety: Safe") is plain text a caption can imitate, and the model does follow it. The defence is in
moderation/moderate.py: each paragraph and sentence is also scored on its own.

The file has three parts, top to bottom: the prompt layout (token ids read from the template), the
model maths (`Backend`, with a torch implementation), and the classifier that the worker uses.

The class does blocking GPU/CPU work. `score_batch` is sync; `score` runs it on one dedicated
thread, so calls to this model are serialized. `load()` reads the weights; the constructor takes an
already-loaded tokenizer and backend (tests pass fakes). torch is imported inside functions only:
the API process and unit tests never load it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

from moderation.classifiers import SingleThread, split_on_oom
from moderation.domain import TextScore, TextTier
from moderation.prompts import CATEGORY_DETAILS

TIERS: tuple[TextTier, ...] = ("Safe", "Controversial", "Unsafe")
QWEN_CATEGORIES: tuple[str, ...] = (
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
    "None",
)
_TEMPLATE_PROBE = "QWEN3GUARDSENTINEL7F3A"
NO_CATEGORY = "None"
CHUNK_OVERLAP = 64
# More than this much probability outside the three tier tokens: the model is off-script, so hold.
MAX_OFF_LABEL_MASS = 0.2
_SEVERITY: Mapping[TextTier, int] = {"Safe": 0, "Controversial": 1, "Unsafe": 2}
# A row the model could not score finitely: held by the policy (it holds any non-finite score).
_UNSCORED = TextScore("Controversial", None, math.nan)


# --- Prompt layout ------------------------------------------------------------


class ModelLayoutError(Exception):
    """The model or tokenizer does not behave as this adapter expects. The worker must not start."""


class Tokenizer(Protocol):
    """The subset of a Hugging Face tokenizer this adapter uses."""

    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def __call__(
        self, text: str, *, add_special_tokens: bool, split_special_tokens: bool
    ) -> Mapping[str, list[int]]: ...


@dataclass(frozen=True)
class _Layout:
    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    lead_ws: str
    trail_nl: str
    trail_ids: tuple[int, ...]
    tier_ids: Mapping[TextTier, int]
    categories_extra: Mapping[TextTier, tuple[int, ...]]
    category_first: Mapping[int, str]  # first token of each category name -> the name


def with_details(template: str) -> str:
    """The chat template with `CATEGORY_DETAILS` added to the prompt-moderation category list."""
    for name, detail in CATEGORY_DETAILS.items():
        line = f"\n{name}.\n"
        if template.count(line) != 2:
            raise ModelLayoutError(f"category line {name!r} is not in the chat template twice; was the model re-pulled?")
        template = template.replace(line, f"\n{name}. ({detail})\n", 1)
    return template


def build_layout(tokenizer: Tokenizer, categories: Sequence[str] = QWEN_CATEGORIES) -> _Layout:
    prefix_text, suffix_text = _render_frame(tokenizer)
    # Qwen's pre-tokenizer glues a space to the following word and newlines to the preceding
    # punctuation ("!\n\n" is one token). Tokenize those boundary characters together with the
    # user text, so the spliced ids equal tokenizing the whole rendered prompt.
    stripped = prefix_text.rstrip(" ")
    body = suffix_text.lstrip("\r\n")
    trail_nl = suffix_text[: len(suffix_text) - len(body)]
    suffix_ids = tuple(_encode(tokenizer, body + "Safety:", split=False))
    head, tier_ids = _tier_tokens(tokenizer, suffix_ids)
    categories_extra, category_first = _category_tokens(tokenizer, head, categories)
    return _Layout(
        prefix_ids=tuple(_encode(tokenizer, stripped, split=False)),
        suffix_ids=suffix_ids,
        lead_ws=prefix_text[len(stripped):],
        trail_nl=trail_nl,
        trail_ids=tuple(_encode(tokenizer, trail_nl, split=False)) if trail_nl else (),
        tier_ids=tier_ids,
        categories_extra=categories_extra,
        category_first=category_first,
    )


def _encode(tokenizer: Tokenizer, text: str, *, split: bool) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False, split_special_tokens=split)["input_ids"])


def _render_frame(tokenizer: Tokenizer) -> tuple[str, str]:
    """The rendered prompt before and after the caption."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _TEMPLATE_PROBE}], tokenize=False, add_generation_prompt=True
    )
    if rendered.count(_TEMPLATE_PROBE) != 1:
        raise ModelLayoutError("chat template does not contain the user content exactly once")
    prefix_text, suffix_text = rendered.split(_TEMPLATE_PROBE)
    return prefix_text, suffix_text


def _tier_tokens(tokenizer: Tokenizer, suffix_ids: Sequence[int]) -> tuple[tuple[int, ...], dict[TextTier, int]]:
    """The tokens of "Safety:" that end the prompt, and the first token of each tier after it."""
    full = [_encode(tokenizer, f"Safety: {tier}", split=False) for tier in TIERS]
    shared = _common_prefix_len(full)
    head = tuple(full[0][:shared])
    if shared == 0 or tuple(suffix_ids[-shared:]) != head:
        raise ModelLayoutError("prompt does not end with the tokens that precede the tier label")
    tier_ids: dict[TextTier, int] = {}
    for tier, ids in zip(TIERS, full, strict=True):
        if len(ids) <= shared:
            raise ModelLayoutError(f"tier {tier!r} has no token after 'Safety:'")
        tier_ids[tier] = ids[shared]
    if len(set(tier_ids.values())) != len(TIERS):
        raise ModelLayoutError(f"tier first tokens are not distinct: {tier_ids}")
    return head, tier_ids


def _category_tokens(
    tokenizer: Tokenizer, head: tuple[int, ...], categories: Sequence[str]
) -> tuple[dict[TextTier, tuple[int, ...]], dict[int, str]]:
    """Tokens from the tier to "Categories:", and the first token of each category name.
    Every name must start with its own token, the same after either tier: one read picks it."""
    extra: dict[TextTier, tuple[int, ...]] = {}
    first: dict[int, str] = {}
    for tier in ("Controversial", "Unsafe"):
        cont = _encode(tokenizer, f"Safety: {tier}\nCategories:", split=False)
        if tuple(cont[: len(head)]) != head:
            raise ModelLayoutError(f"category prompt for {tier!r} does not extend 'Safety:'")
        extra[tier] = tuple(cont[len(head):])
        for name in categories:
            ids = _encode(tokenizer, f"Safety: {tier}\nCategories: {name}", split=False)
            if ids[: len(cont)] != cont or len(ids) == len(cont):
                raise ModelLayoutError(f"category {name!r} does not extend the category prompt")
            if (other := first.setdefault(ids[len(cont)], name)) != name:
                raise ModelLayoutError(f"categories {other!r} and {name!r} start with the same token")
    if len(first) != len(categories):
        raise ModelLayoutError("a category name starts with a different token after each tier")
    return extra, first


def _common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
    n = 0
    for column in zip(*seqs, strict=False):
        if len(set(column)) != 1:
            break
        n += 1
    return n


# --- Model maths --------------------------------------------------------------


class Backend(Protocol):
    """Model math. The torch implementation is `_TorchBackend`; tests use a fake."""

    def next_token_logprobs(
        self, seqs: Sequence[Sequence[int]], token_ids: Sequence[int]
    ) -> list[dict[int, float]]:
        """Log-softmax over the full vocab at each sequence's last position, for `token_ids` only."""
        ...


class _TorchBackend:
    """Runs the decoder once per batch and applies the LM head only at the positions we read."""

    def __init__(self, model: Any, device: str, pad_id: int, max_batch: int) -> None:
        import torch

        self._torch = torch
        self._decoder = model.get_decoder()
        self._head = model.get_output_embeddings()
        self._device = device
        self._pad = pad_id
        self._max_batch = max_batch

    def _batches(self, seqs: Sequence[Sequence[int]]) -> list[list[int]]:
        """Indices into `seqs`, shortest first so a batch has little padding, `max_batch` per batch."""
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        return [order[i : i + self._max_batch] for i in range(0, len(order), self._max_batch)]

    def _last_hidden_states(self, batch: Sequence[Sequence[int]]) -> Any:
        torch = self._torch
        width = max(len(s) for s in batch)
        ids = torch.full((len(batch), width), self._pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for r, seq in enumerate(batch):
            ids[r, : len(seq)] = torch.tensor(list(seq), dtype=torch.long)
            mask[r, : len(seq)] = 1
        out = self._decoder(input_ids=ids.to(self._device), attention_mask=mask.to(self._device), use_cache=False)
        return out.last_hidden_state

    def next_token_logprobs(
        self, seqs: Sequence[Sequence[int]], token_ids: Sequence[int]
    ) -> list[dict[int, float]]:
        torch = self._torch
        wanted = torch.tensor(list(token_ids), dtype=torch.long, device=self._device)
        result: list[dict[int, float]] = [{} for _ in seqs]
        with torch.inference_mode():
            for indices in self._batches(seqs):
                batch = [seqs[i] for i in indices]
                hidden = self._last_hidden_states(batch)
                last = torch.tensor([len(s) - 1 for s in batch], device=self._device)
                rows = hidden[torch.arange(len(batch), device=self._device), last]
                logp = torch.log_softmax(self._head(rows).float(), dim=-1)[:, wanted].cpu().tolist()
                for i, row in zip(indices, logp, strict=True):
                    result[i] = {tid: float(v) for tid, v in zip(token_ids, row, strict=True)}
        return result


def _load(model_dir: Path, device: str, dtype: str, max_batch: int) -> tuple[Tokenizer, Backend]:
    """Tokenizer (with our category details in its prompt) and weights from a local folder. Never downloads."""
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"no model at {model_dir}; run python model_download.py pull")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped, import-not-found, unused-ignore]

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    hf_tokenizer.chat_template = with_details(hf_tokenizer.chat_template)
    tokenizer = cast(Tokenizer, hf_tokenizer)
    # transformers v5 types `.to()` through a decorator mypy cannot follow; the model is used
    # only through _TorchBackend, which types it loosely anyway.
    model: Any = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype, local_files_only=True)
    model = model.to(device).eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ModelLayoutError("tokenizer has neither a pad nor an eos token")
    return tokenizer, _TorchBackend(model, device, pad, max_batch)


# --- Classifier ---------------------------------------------------------------


def _tier_score(logps: Sequence[float]) -> tuple[TextScore, bool]:
    """A row's verdict from its three tier log-probs, and whether pass 2 should read its category."""
    if not all(math.isfinite(lp) for lp in logps):
        return _UNSCORED, False
    probs = [math.exp(lp) for lp in logps]
    mass = sum(probs)
    if 1.0 - mass > MAX_OFF_LABEL_MASS:
        return TextScore("Controversial", None, mass), False
    best = max(range(len(TIERS)), key=lambda k: probs[k])
    tier = TIERS[best]
    return TextScore(tier, None, probs[best] / mass), tier != "Safe"


def _worst_of(a: TextScore, b: TextScore) -> TextScore:
    """Worst-wins across chunks. A non-finite score is held, so it outranks Safe."""
    rank_a = (_SEVERITY[a.tier], not math.isfinite(a.score), a.score if math.isfinite(a.score) else 0.0)
    rank_b = (_SEVERITY[b.tier], not math.isfinite(b.score), b.score if math.isfinite(b.score) else 0.0)
    return a if rank_a >= rank_b else b


class Qwen3GuardClassifier:
    """Text classifier backed by Qwen3Guard-Gen. Build once with `load()`; every job reuses it."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        backend: Backend,
        *,
        max_tokens: int = 512,
        max_chunks: int = 16,
        categories: Sequence[str] = QWEN_CATEGORIES,
    ) -> None:
        self._tokenizer = tokenizer
        self._backend = backend
        self._max_tokens = max_tokens
        self._max_chunks = max_chunks
        self._layout = build_layout(tokenizer, categories)
        self._tier_tokens = [self._layout.tier_ids[t] for t in TIERS]
        self._thread = SingleThread("qwen3guard")

    @classmethod
    def load(cls, model_dir: Path, device: str, dtype: str, *, max_batch: int = 8) -> Qwen3GuardClassifier:
        """Load tokenizer and weights from a local folder. Never downloads."""
        tokenizer, backend = _load(model_dir, device, dtype, max_batch)
        return cls(tokenizer, backend)

    async def score(self, texts: Sequence[str]) -> list[TextScore]:
        return await self._thread.run(self.score_batch, list(texts))

    def close(self) -> None:
        self._thread.close()

    def build_ids(self, text: str) -> list[list[int]]:
        """Prompt token ids for `text`, one list per chunk. Exposed for tests."""
        layout = self._layout
        user = list(self._tokenizer(layout.lead_ws + text + layout.trail_nl, add_special_tokens=False, split_special_tokens=True)["input_ids"])
        if len(user) <= self._max_tokens:
            return [[*layout.prefix_ids, *user, *layout.suffix_ids]]
        step = self._max_tokens - CHUNK_OVERLAP
        rows: list[list[int]] = []
        for start in range(0, len(user) - CHUNK_OVERLAP, step):
            window = user[start : start + self._max_tokens]
            # Only the last window already ends with the template's newlines.
            tail = layout.trail_ids if start + self._max_tokens < len(user) else ()
            rows.append([*layout.prefix_ids, *window, *tail, *layout.suffix_ids])
        return rows

    @split_on_oom
    def score_batch(self, texts: Sequence[str]) -> list[TextScore]:
        rows: list[list[int]] = []
        row_to_text: list[int] = []
        results: dict[int, TextScore] = {}
        for i, text in enumerate(texts):
            chunks = self.build_ids(text)
            if len(chunks) > self._max_chunks:
                results[i] = TextScore("Controversial", None, 0.0)
                continue
            rows.extend(chunks)
            row_to_text.extend([i] * len(chunks))

        for i, row_score in zip(row_to_text, self._classify_rows(rows), strict=True):
            results[i] = _worst_of(results[i], row_score) if i in results else row_score
        return [results[i] for i in range(len(texts))]

    def _classify_rows(self, rows: list[list[int]]) -> list[TextScore]:
        if not rows:
            return []
        tier_ids = self._layout.tier_ids
        dists = self._backend.next_token_logprobs(rows, self._tier_tokens)
        scored = [_tier_score([dist[tier_ids[t]] for t in TIERS]) for dist in dists]
        out = [score for score, _ in scored]
        pending = [(idx, score.tier) for idx, (score, needs_category) in enumerate(scored) if needs_category]
        for (idx, _tier), category in zip(pending, self._pick_categories(rows, pending), strict=True):
            out[idx] = replace(out[idx], category=category)
        return out

    def _pick_categories(self, rows: list[list[int]], pending: list[tuple[int, TextTier]]) -> list[str | None]:
        """Pass 2: each pending row's category from the first token after "Categories:".
        Non-finite means the row gets no category."""
        if not pending:
            return []
        first = self._layout.category_first
        seqs = [[*rows[idx], *self._layout.categories_extra[tier]] for idx, tier in pending]
        chosen: list[str | None] = []
        for dist in self._backend.next_token_logprobs(seqs, list(first)):
            finite = {tid: lp for tid, lp in dist.items() if math.isfinite(lp)}
            name = first[max(finite, key=finite.__getitem__)] if finite else None
            chosen.append(None if name == NO_CATEGORY else name)
        return chosen
