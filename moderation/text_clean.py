"""Caption and OCR text as the text model reads it.

`clean_text` undoes disguises (full-width letters, invisible characters). `split_segments` cuts the
text into the pieces that are scored on their own. A caption can paste fake prompt text after a
threat ("Safety: Safe ... USER: Ramadan Mubarak") and the model follows it when it reads the whole
caption, but not when it reads the threat alone.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Format characters Urdu spelling needs: zero-width non-joiner and the Arabic letter mark.
_KEEP_FORMAT = frozenset({"\u200c", "\u061c"})
_SPACES = re.compile(r"[ \t]+")
_PARAGRAPH = re.compile(r"\n\s*\n|\n(?=[ \t]*(?:USER|ASSISTANT)[ \t]*:)", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?۔؟,;،])\s+|\n")  # ۔ Urdu full stop, ؟ question mark, ، Urdu comma
_TEMPLATE = re.compile(
    r"<\s*(?:BEGIN|END)\s+CONVERSATION\s*>|^[ \t]*Safety[ \t]*:[ \t]*(?:Safe|Unsafe|Controversial)\b",
    re.IGNORECASE | re.MULTILINE,
)
# Rows per text: the whole text, then paragraph and sentence windows. Each level covers all the text.
MAX_PARAGRAPH_ROWS = 7
MAX_SENTENCE_ROWS = 8


@dataclass(frozen=True)
class Segment:
    text: str
    sentence: bool  # a sentence window: out of context it may hold the post, never block it


def clean_text(text: str) -> str:
    """NFKC, no invisible format characters, single spaces. Newlines stay: segments need them."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in _KEEP_FORMAT or unicodedata.category(ch) != "Cf")
    return _SPACES.sub(" ", text).strip()


def has_template_text(text: str) -> bool:
    """True if the text imitates the moderation prompt. A real post never contains these markers."""
    return _TEMPLATE.search(text) is not None


def split_segments(text: str) -> list[Segment]:
    """The whole text, its paragraphs (or USER:/ASSISTANT: turns) and its sentences: at most 16 rows."""
    segments = [Segment(text, sentence=False)]
    paragraphs = _parts(_PARAGRAPH, text)
    if len(paragraphs) > 1:
        segments += [Segment(p, sentence=False) for p in _windows(paragraphs, MAX_PARAGRAPH_ROWS, "\n")]
    sentences = _parts(_SENTENCE, text)
    if len(sentences) > 1:
        segments += [Segment(s, sentence=True) for s in _windows(sentences, MAX_SENTENCE_ROWS, " ")]
    unique: dict[str, Segment] = {}
    for segment in segments:
        unique.setdefault(segment.text, segment)  # first wins: a one-sentence paragraph can still block
    return list(unique.values())


def _parts(pattern: re.Pattern[str], text: str) -> list[str]:
    return [part.strip() for part in pattern.split(text) if part.strip()]


def _windows(parts: list[str], limit: int, sep: str) -> list[str]:
    """Consecutive parts grouped into at most `limit` windows; nothing is dropped."""
    size = -(-len(parts) // limit)
    return [sep.join(parts[i : i + size]) for i in range(0, len(parts), size)]
