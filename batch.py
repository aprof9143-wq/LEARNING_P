"""Batch driver: Airtable CSV → structured ``ExamQuestion`` per row.

One CSV row = one logical exam question, even when it has multiple image
attachments (typical layout: question image + solution image, or question
spread across two pages). All images for a row are sent to Claude in a single
multi-image tool-use call, so the resulting ``ExamQuestion`` has its parts and
solution merged end-to-end.

Concurrency is at the row level — each worker thread handles one row's full
multi-image OCR call. The Anthropic SDK is thread-safe, so a shared client is
reused across workers.

Self-contained: re-implements the small bits of the parent ``csv_links``
parser so the ``enhanced/`` folder runs standalone without sys.path tricks.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from .enhanced_ocr import (
    SchemaValidationError,
    StructuredOcrClient,
    StructuredOcrError,
    StructuredOcrResult,
)
from .schemas import ExamQuestion, Source

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CSV parsing (self-contained — no dependency on the parent app)
# --------------------------------------------------------------------------- #


# "Label (https://...)" — label can contain commas / newlines.
_ATTACHMENT_RE = re.compile(
    r"(?P<label>[^\n(]+?)\s*\((?P<url>https?://[^)]+)\)",
    re.DOTALL,
)

# Airtable renamed this column; accept both.
_QUESTION_FILE_COLUMNS: tuple[str, ...] = (
    "Question File",
    "Question Title (main question and answer)",
)

# Filenames like "Ans1.PNG", "answer.jpg", "Solution2.png", "MS.pdf".
_ANSWER_FILENAME_RE = re.compile(
    r"^(ans(?:wer)?|sol(?:ution)?|ms)\b",
    re.IGNORECASE,
)

_SOURCE_KEYS: tuple[str, ...] = ("school", "year", "exam_type", "paper")


@dataclass(frozen=True)
class Attachment:
    """One image link from the Question File cell."""

    question_title: str
    label: str
    url: str
    source_row: int


def _clean_label(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    return s.lstrip(",").strip()


def _resolve_question_file_cell(row: dict) -> str:
    for key in _QUESTION_FILE_COLUMNS:
        value = row.get(key)
        if value:
            return value
    return ""


def extract_attachments_from_cell(
    question_title: str, cell: str, row_index: int
) -> list[Attachment]:
    if not cell or not cell.strip():
        return []
    out: list[Attachment] = []
    for m in _ATTACHMENT_RE.finditer(cell):
        label = _clean_label(m.group("label"))
        url = m.group("url").strip()
        if url:
            out.append(
                Attachment(
                    question_title=question_title.strip(),
                    label=label,
                    url=url,
                    source_row=row_index,
                )
            )
    return out


def iter_csv_rows(csv_bytes: bytes) -> Iterator[tuple[int, dict]]:
    """Yield ``(row_number, row_dict)`` for every data row in the CSV."""
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = csv_bytes.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    if not any(col in fieldnames for col in _QUESTION_FILE_COLUMNS):
        raise ValueError(
            f"CSV must have one of {_QUESTION_FILE_COLUMNS} as a column; "
            f"got: {fieldnames!r}"
        )
    for i, row in enumerate(reader, start=2):
        yield i, row


def is_answer_attachment(att: Attachment) -> bool:
    parts = [p.strip() for p in att.label.split(",")]
    filename = parts[-1] if parts else att.label
    return bool(_ANSWER_FILENAME_RE.match(filename))


def parse_source_from_label(label: str) -> dict[str, str]:
    parts = [p.strip() for p in label.split(",")]
    return {key: value for key, value in zip(_SOURCE_KEYS, parts) if value}


# --------------------------------------------------------------------------- #
# Row grouping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RowGroup:
    """All attachments belonging to one CSV row, ordered question-then-solution."""

    row_index: int
    question_title: str
    topic_hint: str
    learning_objective_hint: str
    question_attachments: list[Attachment] = field(default_factory=list)
    answer_attachments: list[Attachment] = field(default_factory=list)

    @property
    def ordered_attachments(self) -> list[Attachment]:
        return [*self.question_attachments, *self.answer_attachments]

    @property
    def has_solution(self) -> bool:
        return bool(self.answer_attachments)


def group_rows(csv_bytes: bytes, *, max_rows: Optional[int] = None) -> list[RowGroup]:
    """Parse the CSV into one RowGroup per row that has at least one attachment."""
    groups: list[RowGroup] = []
    for row_index, row in iter_csv_rows(csv_bytes):
        title = (row.get("Question Title") or "").strip()
        cell = _resolve_question_file_cell(row)
        if not cell.strip():
            continue
        atts = extract_attachments_from_cell(title, cell, row_index)
        if not atts:
            continue
        q_atts = [a for a in atts if not is_answer_attachment(a)]
        a_atts = [a for a in atts if is_answer_attachment(a)]
        groups.append(
            RowGroup(
                row_index=row_index,
                question_title=title,
                topic_hint=(row.get("Topics") or "").strip(),
                learning_objective_hint=(row.get("Learning Objectives") or "").strip(),
                question_attachments=q_atts,
                answer_attachments=a_atts,
            )
        )
        if max_rows is not None and len(groups) >= max_rows:
            break
    return groups


# --------------------------------------------------------------------------- #
# Per-row OCR + batch driver
# --------------------------------------------------------------------------- #


@dataclass
class RowResult:
    """Outcome of OCRing one row. Always present in the final summary."""

    row_index: int
    question_title: str
    attachment_urls: list[str]
    answer_urls: list[str]
    topic_hint: str = ""
    learning_objective_hint: str = ""
    status: str = "pending"  # ok | error | dry-run
    question: Optional[dict] = None  # serialised ExamQuestion
    raw_tool_input: Optional[dict] = None
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    duration_s: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class BatchSummary:
    results: list[RowResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    def to_json(self) -> list[dict]:
        return [r.to_json() for r in self.results]


ProgressCb = Callable[[RowResult], None]


def _merge_source_from_label(question: ExamQuestion, label: str) -> ExamQuestion:
    """Backfill ``Source`` from the Airtable label only when Claude left fields blank.

    The label format ``"ACJC,2025,Prelim,P2,Q1.PNG"`` is more reliable than
    inferring from the page itself, so prefer it for any field Claude missed,
    but never overwrite a value Claude already filled in.
    """
    parsed = parse_source_from_label(label)
    src = question.source
    merged_dict = src.model_dump()
    for key, value in parsed.items():
        if not merged_dict.get(key):
            if key == "year":
                try:
                    merged_dict[key] = int(value)
                except (TypeError, ValueError):
                    continue
            else:
                merged_dict[key] = value
    question = question.model_copy(update={"source": Source.model_validate(merged_dict)})
    return question


def _process_row(
    group: RowGroup,
    *,
    client: StructuredOcrClient,
) -> RowResult:
    result = RowResult(
        row_index=group.row_index,
        question_title=group.question_title,
        attachment_urls=[a.url for a in group.question_attachments],
        answer_urls=[a.url for a in group.answer_attachments],
        topic_hint=group.topic_hint,
        learning_objective_hint=group.learning_objective_hint,
    )

    urls = [a.url for a in group.ordered_attachments]
    if not urls:
        result.status = "error"
        result.error = "Row had no attachment URLs after grouping."
        return result

    start = time.monotonic()
    try:
        ocr: StructuredOcrResult = client.ocr_image_urls(urls)
    except SchemaValidationError as exc:
        result.status = "error"
        result.error = f"SchemaValidationError: {exc}"
        result.raw_tool_input = exc.raw_input
        result.duration_s = time.monotonic() - start
        logger.warning("Row %s schema validation failed: %s", group.row_index, exc)
        return result
    except StructuredOcrError as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        result.duration_s = time.monotonic() - start
        logger.warning("Row %s OCR failed: %s", group.row_index, exc)
        return result
    except Exception as exc:  # defensive: never let one row kill the batch
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        result.duration_s = time.monotonic() - start
        logger.exception("Row %s unexpected error", group.row_index)
        return result

    question = ocr.question
    # Backfill metadata from the first attachment's label (more reliable than the page).
    first_label = (
        group.question_attachments[0].label
        if group.question_attachments
        else group.ordered_attachments[0].label
    )
    question = _merge_source_from_label(question, first_label)

    result.status = "ok"
    result.question = question.model_dump()
    result.raw_tool_input = ocr.raw_tool_input
    result.input_tokens = ocr.input_tokens
    result.output_tokens = ocr.output_tokens
    result.cache_read_tokens = ocr.cache_read_tokens
    result.duration_s = time.monotonic() - start
    return result


def run_batch(
    csv_bytes: bytes,
    *,
    client: StructuredOcrClient,
    max_rows: Optional[int] = None,
    max_workers: int = 4,
    progress: Optional[ProgressCb] = None,
) -> BatchSummary:
    """Top-level entry point. Concurrency is per-row."""
    groups = group_rows(csv_bytes, max_rows=max_rows)
    summary = BatchSummary(results=[None] * len(groups))  # type: ignore[list-item]

    if not groups:
        return summary

    if max_workers <= 1:
        for i, group in enumerate(groups):
            r = _process_row(group, client=client)
            summary.results[i] = r
            if progress:
                progress(r)
        return summary

    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="enhanced-ocr"
    ) as ex:
        futures = {
            ex.submit(_process_row, group, client=client): i
            for i, group in enumerate(groups)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            summary.results[i] = r
            if progress:
                progress(r)

    return summary


# --------------------------------------------------------------------------- #
# Convenience CLI (for ad-hoc testing — Streamlit is the real UI)
# --------------------------------------------------------------------------- #


def _cli() -> int:
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        description="Run structured-OCR over an Airtable CSV (debug helper)."
    )
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Output JSON file.")
    p.add_argument("-n", "--max-rows", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--model", default=StructuredOcrClient.DEFAULT_MODEL)
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    client = StructuredOcrClient(model=args.model)

    def _on_progress(r: RowResult) -> None:
        marker = "[ok]" if r.status == "ok" else "[error]"
        print(f"{marker} row={r.row_index} ({r.duration_s:.1f}s) {r.error or ''}")

    summary = run_batch(
        args.csv.read_bytes(),
        client=client,
        max_rows=args.max_rows,
        max_workers=args.workers,
        progress=_on_progress,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"\n{summary.ok_count} ok / {summary.error_count} error — wrote {args.out}"
    )
    print(
        f"Tokens: {summary.total_input_tokens} in / {summary.total_output_tokens} out"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
