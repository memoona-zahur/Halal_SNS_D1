"""Check one request: caption and images through the models, then worst-wins across all pieces."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from moderation.domain import (
    CATEGORIES,
    UNCLASSIFIED,
    Decision,
    ImageRef,
    ItemResult,
    ItemSource,
    Label,
    ModerationRequest,
    OcrResult,
    RejectedImage,
    TextScore,
)
from moderation.errors import (
    MEDIA_FORBIDDEN,
    MEDIA_TOO_LARGE,
    MEDIA_UNDECODABLE,
    MEDIA_UNSUPPORTED,
    PermanentError,
)
from moderation.fetch import is_animated
from moderation.policy import Policy, worst_item
from moderation.text_clean import Segment, clean_text, has_template_text, split_segments

ANIMATED_REASON = "Animated image: not checked, held"
TEMPLATE_NOTE = "Text imitates the moderation prompt"

# What the backend is told for an image we could not check (error_code -> detail).
REJECT_DETAILS = {
    MEDIA_TOO_LARGE: "file is too large",
    MEDIA_UNSUPPORTED: "unsupported file type: only JPEG, PNG, WebP and GIF are accepted",
    MEDIA_UNDECODABLE: "file is unreadable or corrupt",
    MEDIA_FORBIDDEN: "image could not be accessed",
}


async def score_each[R](call: Callable[[list[bytes]], Awaitable[Sequence[R]]], blobs: list[bytes]) -> list[R | PermanentError]:
    """One batch call. If an image in it cannot be decoded, retry one by one so only that image is lost."""
    if not blobs:
        return []
    try:
        return list(await call(blobs))
    except PermanentError as exc:
        if exc.code != MEDIA_UNDECODABLE:
            raise
        if len(blobs) == 1:
            return [exc]
    return [(await score_each(call, [blob]))[0] for blob in blobs]


class Moderator:
    def __init__(
        self,
        *,
        text: Any,
        image: Any,
        ocr: Any,
        fetcher: Any,
        policy: Policy,
        model_versions: Mapping[str, str] | None = None,
    ) -> None:
        self._text = text
        self._image = image
        self._ocr = ocr
        self._fetcher = fetcher
        self._policy = policy
        self._model_versions = dict(model_versions or {})

    async def moderate(self, request: ModerationRequest) -> Decision:
        """Fetch errors propagate unchanged: PermanentError fails the job, RetryableError retries it."""
        # The text model runs in its own thread, so the caption is scored while the images are fetched and read.
        caption, (images, embeddings, rejected) = await asyncio.gather(
            self._judge_caption(request.text), self.moderate_images(request.images)
        )
        items = [*caption, *images]
        winner = worst_item(items)
        return Decision(
            label=winner.label,
            category=winner.category,
            reason=build_reason(winner),
            items=tuple(items),
            policy_version=self._policy.version,
            embeddings=embeddings,
            model_versions=self._model_versions,
            rejected_images=tuple(rejected),
        )

    async def _judge_caption(self, text: str | None) -> list[ItemResult]:
        if not text:
            return []
        cleaned = clean_text(text)
        if not cleaned:  # only spaces or invisible characters: nothing to check
            return [ItemResult(ItemSource.TEXT, None, Label.ALLOW, None, None)]
        [items] = await self._judge_texts([cleaned])
        return items

    async def moderate_images(
        self, images: Sequence[ImageRef]
    ) -> tuple[list[ItemResult], dict[str, tuple[float, ...]], list[RejectedImage]]:
        """An image we cannot check is held and reported in `rejected`; the other images are still checked."""
        fetched = await asyncio.gather(*(self._fetch(image) for image in images))
        embeddings: dict[str, tuple[float, ...]] = {}
        rejected: list[RejectedImage] = []
        by_position: dict[int, ItemResult] = {}
        usable: list[tuple[int, ImageRef, bytes]] = []
        for position, (image, blob) in enumerate(zip(images, fetched, strict=True)):
            if isinstance(blob, PermanentError):
                by_position[position] = self._reject(image.media_id, blob.code, rejected)
            elif is_animated(blob):
                by_position[position] = ItemResult(
                    ItemSource.IMAGE, image.media_id, Label.HOLD, None, None, ANIMATED_REASON
                )
            else:
                usable.append((position, image, blob))
        data = [blob for _, _, blob in usable]
        scores, ocr_results = await asyncio.gather(score_each(self._image.score, data), score_each(self._ocr.read, data))

        ids: list[str] = []
        good_ocr: list[OcrResult] = []
        for (position, image, _), score, ocr in zip(usable, scores, ocr_results, strict=True):
            if isinstance(score, PermanentError) or isinstance(ocr, PermanentError):
                by_position[position] = self._reject(image.media_id, MEDIA_UNDECODABLE, rejected)
                continue
            ids.append(image.media_id)
            good_ocr.append(ocr)
            embeddings[image.media_id] = score.embedding
            by_position[position] = self._policy.judge_image(image.media_id, score.categories)
        items = [by_position[position] for position in range(len(images))]
        items.extend(await self._check_image_text(ids, good_ocr))
        return items, embeddings, rejected

    async def _fetch(self, image: ImageRef) -> bytes | PermanentError:
        """A permanent fetch problem is about this image only. A retryable one still retries the whole job."""
        try:
            return await self._fetcher.fetch(image.url)
        except PermanentError as exc:
            return exc

    @staticmethod
    def _reject(media_id: str, code: str, rejected: list[RejectedImage]) -> ItemResult:
        detail = REJECT_DETAILS.get(code, "image could not be checked")
        rejected.append(RejectedImage(media_id, code, detail))
        return ItemResult(ItemSource.IMAGE, media_id, Label.HOLD, None, None, f"Image {media_id}: {detail}")

    async def _judge_texts(self, texts: list[str]) -> list[list[ItemResult]]:
        """Each cleaned text: its worst segment verdict, plus a hold for prompt-like text.
        All segments of all texts go to the model in one call."""
        segments = [split_segments(text) for text in texts]
        scores = iter(await self._text.score([s.text for segs in segments for s in segs]))
        out = []
        for text, segs in zip(texts, segments, strict=True):
            row_scores = [next(scores) for _ in segs]
            # The whole text is only personal details (allowed): its intent clause decides, so it may block.
            decides = row_scores[0].category in self._policy.always_allow
            items = [self._judge_segment(s, sc, decides) for s, sc in zip(segs, row_scores, strict=True)]
            out.append(self._text_items(text, items))
        return out

    def _judge_segment(self, segment: Segment, score: TextScore, decides: bool) -> ItemResult:
        item = self._policy.judge_text(score)
        if segment.sentence and item.label is Label.BLOCK and not decides:
            return replace(item, label=Label.HOLD)  # one sentence out of context holds, never blocks
        return item

    def _text_items(self, text: str, segment_items: list[ItemResult]) -> list[ItemResult]:
        items = [worst_item(segment_items)]
        if has_template_text(text):
            items.append(ItemResult(ItemSource.TEXT, None, Label.HOLD, None, None, TEMPLATE_NOTE))
        return items

    async def _check_image_text(self, media_ids: list[str], ocr_results: list[OcrResult]) -> list[ItemResult]:
        """No text: nothing to check. Text we cannot read: hold. Readable text: judged like a caption."""
        items: list[ItemResult] = []
        readable: list[tuple[str, str]] = []
        for media_id, result in zip(media_ids, ocr_results, strict=True):
            if len("".join(result.text.split())) < self._policy.ocr_min_chars:  # letters only, not spaces or line breaks
                continue  # nothing read, or a stray mark: a photo without writing is not held for it
            if result.mean_confidence is None or result.mean_confidence < self._policy.ocr_min_confidence:
                note = f"Image {media_id}: text could not be read"
                items.append(ItemResult(ItemSource.IMAGE_TEXT, media_id, Label.HOLD, None, None, note))
            elif text := clean_text(result.text.replace("\n", " ")):
                readable.append((media_id, text))  # OCR lines are joined: one OCR line is not a sentence
        if readable:
            judged = await self._judge_texts([text for _, text in readable])
            for (media_id, _), text_items in zip(readable, judged, strict=True):
                items.extend(replace(item, source=ItemSource.IMAGE_TEXT, media_id=media_id) for item in text_items)
        return items


def build_reason(item: ItemResult) -> str:
    if item.label is Label.ALLOW:
        return "Allowed"
    if item.note:
        return item.note
    name = CATEGORIES[item.category] if item.category is not None else UNCLASSIFIED
    if item.source is ItemSource.TEXT:
        where = "Caption"
    elif item.source is ItemSource.IMAGE_TEXT:
        where = f"Text in image {item.media_id}"
    else:
        where = f"Image {item.media_id}"
    return f"{where}: {name}"
