"""Structured-output OCR for math-heavy exam papers via Claude vision + tool use.

The original ``claude_ocr.py`` returned a markdown blob — fine for storage, but
unfit for programmatic validation. This module replaces that with a Claude call
that is **forced** to call the ``submit_exam_question`` tool, whose input JSON
schema is defined in ``schemas.py``. The tool input is then validated through
Pydantic before being returned. Result: every successful call yields an
``ExamQuestion`` with the same shape, every time.

Public surface
--------------
``StructuredOcrClient``
    Reusable, thread-safe client. Single-image and multi-image calls.
``StructuredOcrResult``
    Dataclass with the validated ``ExamQuestion`` plus usage metadata.
``StructuredOcrError`` (+ ``ImageDownloadError``, ``StructuredOcrApiError``,
``SchemaValidationError``)
    Exception hierarchy — every recoverable failure raises one of these.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import anthropic
import httpx
from pydantic import ValidationError

try:
    from .schemas import (
        EXAM_QUESTION_TOOL_NAME,
        EXAM_QUESTION_TOOL_SCHEMA,
        ExamQuestion,
    )
except ImportError:  # flat layout: files are siblings, not in a package
    from schemas import (  # type: ignore[no-redef]
        EXAM_QUESTION_TOOL_NAME,
        EXAM_QUESTION_TOOL_SCHEMA,
        ExamQuestion,
    )

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# .env loading (idempotent)
# --------------------------------------------------------------------------- #


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Look for `.env` in the enhanced/ folder, then its parent, then cwd."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _DOTENV_LOADED = True
        return
    here = Path(__file__).resolve().parent
    for candidate in (here / ".env", here.parent / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)
            break
    _DOTENV_LOADED = True


_load_dotenv_once()


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


_SYSTEM_PROMPT = """You are a precise OCR engine for math-heavy exam papers (Singapore/UK A-Level style).

You will be shown one or more images for a single exam question. Some images
may be the question itself; others may be the worked solution / mark scheme.
You MUST emit your transcription by calling the `submit_exam_question` tool —
do not produce any free-form text outside the tool call.

LaTeX & formatting rules (apply to every text field you populate):
- Use `$...$` for inline math and `$$...$$` for display math. NEVER use `\\(...\\)` or `\\[...\\]`.
- Every `\\begin{env}` must have a matching `\\end{env}`. Repair half-open environments.
- Keep column vectors VERTICAL: `\\begin{pmatrix} a \\\\ b \\\\ c \\end{pmatrix}`.
  Never flatten to `(a, b, c)`.
- Never nest one matrix environment inside another. Use `\\left( ... \\right)` to bracket.
- Preserve every step of working on its own line. If a multi-line derivation has been
  flattened into one sentence, split it back into one line per `=` or `\\implies` step.
- Question/part numbering: preserve EXACTLY (`1.`, `(a)`, `(i)`, `(ii)`, etc.).
- For diagrams that cannot be transcribed as text, populate the `diagrams[]` array
  with a factual description. NEVER invent detail you cannot see.
- Mark illegible text with `[illegible]` inline AND list the region in
  `illegible_regions[]`. Do not guess.
- Do not add solutions, hints, or content that is not visible in the source image(s).

Routing rules:
- If a solution / mark-scheme image is present, populate `solution`. Prefer the
  per-part breakdown (`solution.by_part`); fall back to `solution.raw_markdown`
  only when working can't be cleanly split.
- If no solution image is present, leave `solution` empty.
- `topics` is a best-effort tag list (e.g. "Integration", "Vectors") — keep it short.
- `confidence` is your honest self-rating of the transcription quality.

Call the tool exactly once."""


_USER_PROMPT_SINGLE = (
    "Transcribe this exam-paper image into structured fields by calling the "
    "submit_exam_question tool. Follow every rule in the system prompt."
)

_USER_PROMPT_MULTI = (
    "These images all belong to ONE exam question. Earlier images are typically "
    "the question; later images are typically the worked solution / mark scheme "
    "(filenames containing 'Ans', 'Solution', or 'MS' indicate a solution). "
    "Merge them into a single structured record by calling the "
    "submit_exam_question tool. Follow every rule in the system prompt."
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class StructuredOcrError(RuntimeError):
    """Base class for every error this module raises."""


class ImageDownloadError(StructuredOcrError):
    """The source image URL could not be fetched or decoded."""


class StructuredOcrApiError(StructuredOcrError):
    """The Anthropic API rejected the request or returned an unusable response."""


class SchemaValidationError(StructuredOcrError):
    """Claude called the tool but its input failed Pydantic validation."""

    def __init__(self, message: str, *, raw_input: dict) -> None:
        super().__init__(message)
        self.raw_input = raw_input


# --------------------------------------------------------------------------- #
# Image fetching
# --------------------------------------------------------------------------- #


_SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})


def _sniff_media_type(body: bytes) -> str:
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if body[:2] == b"\xff\xd8":
        return "image/jpeg"
    if body[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "image/webp"
    raise ImageDownloadError(
        "Image bytes did not match any supported format (PNG / JPEG / GIF / WebP)."
    )


def _normalize_media_type(content_type_header: str, body: bytes) -> str:
    ct = (content_type_header or "").split(";")[0].strip().lower()
    if ct in _SUPPORTED_MEDIA_TYPES:
        return ct
    if ct == "image/jpg":
        return "image/jpeg"
    return _sniff_media_type(body)


def download_image(url: str, *, timeout_s: float = 60.0) -> tuple[bytes, str]:
    """Fetch an image URL → ``(body_bytes, media_type)``."""
    headers = {"User-Agent": "LearningPlatform-StructuredOCR/1.0"}
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ImageDownloadError(
            f"HTTP {e.response.status_code} downloading image "
            f"(URL may be expired): {url[:120]}..."
        ) from e
    except httpx.RequestError as e:
        raise ImageDownloadError(f"Network error downloading image: {e}") from e

    body = r.content
    if not body:
        raise ImageDownloadError("Empty image body.")
    media_type = _normalize_media_type(r.headers.get("content-type", ""), body)
    return body, media_type


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StructuredOcrResult:
    """Successful structured-OCR outcome."""

    question: ExamQuestion
    raw_tool_input: dict
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    stop_reason: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


_TOOL_DEFINITION = {
    "name": EXAM_QUESTION_TOOL_NAME,
    "description": (
        "Submit the structured transcription of the exam question shown in the "
        "image(s). Call exactly once per request."
    ),
    "input_schema": EXAM_QUESTION_TOOL_SCHEMA,
}


class StructuredOcrClient:
    """Anthropic vision client that forces a structured tool call. Thread-safe."""

    DEFAULT_MODEL = "claude-opus-4-7"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8192,
        max_retries: int = 4,
        request_timeout_s: float = 180.0,
        download_timeout_s: float = 60.0,
    ) -> None:
        # Anthropic SDK reads ANTHROPIC_API_KEY itself, but pass explicitly when
        # provided so callers can override per-instance (Streamlit settings, etc.).
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise StructuredOcrError(
                "ANTHROPIC_API_KEY not set. Add it to your environment or pass "
                "api_key=... explicitly."
            )
        self._client = anthropic.Anthropic(
            api_key=resolved_key,
            max_retries=max_retries,
            timeout=request_timeout_s,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._download_timeout_s = download_timeout_s

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> str:
        return self._model

    def ocr_image_url(self, url: str) -> StructuredOcrResult:
        """Download a single URL, OCR it, return one structured record."""
        body, media_type = download_image(url, timeout_s=self._download_timeout_s)
        return self.ocr_image_bytes(body, media_type=media_type)

    def ocr_image_bytes(
        self, body: bytes, *, media_type: str
    ) -> StructuredOcrResult:
        """OCR a single image already in memory."""
        return self._call([(body, media_type)], _USER_PROMPT_SINGLE)

    def ocr_image_urls(
        self, urls: Sequence[str]
    ) -> StructuredOcrResult:
        """Download all URLs and OCR them together as one question + solution."""
        if not urls:
            raise StructuredOcrError("ocr_image_urls called with no URLs.")
        images = [
            download_image(u, timeout_s=self._download_timeout_s) for u in urls
        ]
        prompt = _USER_PROMPT_SINGLE if len(images) == 1 else _USER_PROMPT_MULTI
        return self._call(images, prompt)

    def ocr_images(
        self, images: Iterable[tuple[bytes, str]]
    ) -> StructuredOcrResult:
        """OCR a pre-fetched batch of ``(bytes, media_type)`` images together."""
        items = list(images)
        if not items:
            raise StructuredOcrError("ocr_images called with no images.")
        prompt = _USER_PROMPT_SINGLE if len(items) == 1 else _USER_PROMPT_MULTI
        return self._call(items, prompt)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _call(
        self,
        images: Sequence[tuple[bytes, str]],
        user_text: str,
    ) -> StructuredOcrResult:
        for _, media_type in images:
            if media_type not in _SUPPORTED_MEDIA_TYPES:
                raise StructuredOcrApiError(
                    f"Unsupported media type for Claude vision: {media_type!r}. "
                    f"Allowed: {sorted(_SUPPORTED_MEDIA_TYPES)}."
                )

        content: list[dict] = []
        for body, media_type in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(body).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": user_text})

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[_TOOL_DEFINITION],
                tool_choice={
                    "type": "tool",
                    "name": EXAM_QUESTION_TOOL_NAME,
                },
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIStatusError as e:
            raise StructuredOcrApiError(
                f"Anthropic API error ({e.status_code}): {e.message}"
            ) from e
        except anthropic.APIConnectionError as e:
            raise StructuredOcrApiError(
                f"Connection error reaching Anthropic API: {e}"
            ) from e

        if response.stop_reason == "refusal":
            raise StructuredOcrApiError(
                "Anthropic refused to transcribe the image."
            )

        tool_use = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )
        if tool_use is None:
            # Surface any free-text the model emitted instead, for debugging.
            stray = " ".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            raise StructuredOcrApiError(
                "Claude did not call the submit_exam_question tool "
                f"(stop_reason={response.stop_reason!r}). "
                f"Stray text: {stray[:200]!r}"
            )

        raw_input = dict(tool_use.input or {})
        try:
            question = ExamQuestion.model_validate(raw_input)
        except ValidationError as e:
            raise SchemaValidationError(
                f"Tool input failed Pydantic validation: {e}", raw_input=raw_input
            ) from e

        usage = response.usage
        return StructuredOcrResult(
            question=question,
            raw_tool_input=raw_input,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            stop_reason=response.stop_reason,
        )
