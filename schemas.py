"""Pydantic models + Anthropic tool JSON schema for structured exam-question OCR.

Two artefacts:

  * ``ExamQuestion`` (and friends) — Pydantic models used to validate
    Claude's tool output and to power the Streamlit editor.
  * ``EXAM_QUESTION_TOOL_SCHEMA`` — a hand-written JSON Schema attached to
    the Anthropic tool definition. Hand-written (rather than derived from
    Pydantic) so we can fine-tune the field descriptions Claude sees, which
    is the single biggest lever for output quality.

Keep the two in sync: every required property in the JSON schema must also
exist on the Pydantic model.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


class Source(BaseModel):
    """Provenance of the question. Parsed from the Airtable label or the image."""

    school: Optional[str] = None
    year: Optional[int] = None
    exam_type: Optional[str] = None  # Prelim, Final, Mock, etc.
    paper: Optional[str] = None      # P1, P2, P3


class SubPart(BaseModel):
    label: str  # e.g. "(i)", "(ii)"
    text: str   # LaTeX-ready markdown
    marks: Optional[int] = None


class Part(BaseModel):
    label: str  # e.g. "(a)", "(b)"
    text: str
    marks: Optional[int] = None
    sub_parts: list[SubPart] = Field(default_factory=list)


class Diagram(BaseModel):
    """Non-textual content (graph, geometric figure, force diagram, etc.)."""

    location: str       # which part it belongs to, e.g. "(b)" or "stem"
    description: str    # factual description only — never invented detail


class SolutionByPart(BaseModel):
    """Solution working broken down by question part."""

    part_label: Optional[str] = None  # which (a)/(b)/(i) this answers
    steps: list[str] = Field(default_factory=list)
    final_answer: Optional[str] = None


class Solution(BaseModel):
    by_part: list[SolutionByPart] = Field(default_factory=list)
    overall_final_answer: Optional[str] = None
    raw_markdown: Optional[str] = None  # fallback when working can't be split


class ExamQuestion(BaseModel):
    """One exam question with structured stem, parts, diagrams, and worked solution."""

    question_number: Optional[str] = None
    source: Source = Field(default_factory=Source)
    marks_total: Optional[int] = None
    stem: Optional[str] = None  # context text before part (a)
    parts: list[Part] = Field(default_factory=list)
    diagrams: list[Diagram] = Field(default_factory=list)
    solution: Solution = Field(default_factory=Solution)
    topics: list[str] = Field(default_factory=list)
    difficulty_hint: Optional[str] = None  # EASY | MEDIUM | HARD
    confidence: float = 0.0
    illegible_regions: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("difficulty_hint")
    @classmethod
    def _normalise_difficulty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        return v if v in {"EASY", "MEDIUM", "HARD"} else None

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# --------------------------------------------------------------------------- #
# Anthropic tool schema (hand-written so we control the descriptions)
# --------------------------------------------------------------------------- #


EXAM_QUESTION_TOOL_NAME = "submit_exam_question"

EXAM_QUESTION_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question_number": {
            "type": "string",
            "description": (
                "The question number as printed (e.g. '1', '12'). "
                "Omit if the image shows only a sub-part with no top-level number."
            ),
        },
        "source": {
            "type": "object",
            "description": (
                "Provenance metadata. Fill only what is explicitly visible on the "
                "page (header, watermark, footer). Never guess."
            ),
            "properties": {
                "school": {"type": "string"},
                "year": {"type": "integer", "minimum": 1980, "maximum": 2100},
                "exam_type": {
                    "type": "string",
                    "description": "e.g. Prelim, Final, Mock, Practice.",
                },
                "paper": {
                    "type": "string",
                    "description": "Paper identifier, e.g. P1, P2, P3.",
                },
            },
        },
        "marks_total": {
            "type": "integer",
            "minimum": 1,
            "description": "Total marks for the whole question, if printed.",
        },
        "stem": {
            "type": "string",
            "description": (
                "Context text shown BEFORE part (a). Use LaTeX delimited by $...$ "
                "for inline math and $$...$$ for display math. Empty if the "
                "question goes straight into (a)."
            ),
        },
        "parts": {
            "type": "array",
            "description": (
                "Each labelled part — typically (a), (b), (c). Preserve the original "
                "labels exactly, including parentheses. Order must match the page."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Original label, e.g. '(a)', '(b)'.",
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Body text of this part as LaTeX-ready markdown. Keep "
                            "multi-line derivations split on '=' or '\\implies'."
                        ),
                    },
                    "marks": {"type": "integer", "minimum": 1},
                    "sub_parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "e.g. '(i)', '(ii)'.",
                                },
                                "text": {"type": "string"},
                                "marks": {"type": "integer", "minimum": 1},
                            },
                            "required": ["label", "text"],
                        },
                    },
                },
                "required": ["label", "text"],
            },
        },
        "diagrams": {
            "type": "array",
            "description": (
                "Every non-textual figure on the page (graphs, geometric drawings, "
                "force diagrams, tree diagrams). Describe what is shown — do not "
                "invent detail."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "Which part the diagram belongs to, e.g. '(b)', 'stem', "
                            "or 'solution(a)'."
                        ),
                    },
                    "description": {"type": "string"},
                },
                "required": ["location", "description"],
            },
        },
        "solution": {
            "type": "object",
            "description": (
                "Worked solution / mark scheme. Populate ONLY if a solution image "
                "was provided. Prefer per-part breakdown; fall back to raw_markdown "
                "if the working cannot be cleanly split."
            ),
            "properties": {
                "by_part": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "part_label": {
                                "type": "string",
                                "description": "Which (a)/(b)/(i) this answers.",
                            },
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "One LaTeX-ready string per line of working.",
                            },
                            "final_answer": {"type": "string"},
                        },
                    },
                },
                "overall_final_answer": {"type": "string"},
                "raw_markdown": {
                    "type": "string",
                    "description": "Fallback only — use when working can't be split.",
                },
            },
        },
        "topics": {
            "type": "array",
            "description": (
                "Short topic tags inferred from the question content "
                "(e.g. 'Integration', 'Vectors', 'Complex numbers')."
            ),
            "items": {"type": "string"},
        },
        "difficulty_hint": {
            "type": "string",
            "enum": ["EASY", "MEDIUM", "HARD"],
            "description": (
                "Best-effort difficulty estimate based on the techniques required."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": (
                "Self-reported transcription confidence in [0, 1]. Lower it when "
                "regions are illegible, when LaTeX is ambiguous, or when the "
                "question text is partially cropped."
            ),
        },
        "illegible_regions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Short labels for regions you marked '[illegible]' in the text. "
                "e.g. 'part (b) third line'."
            ),
        },
        "notes": {
            "type": "string",
            "description": (
                "Anything the downstream reviewer should know: ambiguous symbols, "
                "missing diagram details, etc. Keep brief."
            ),
        },
    },
    "required": ["parts", "confidence"],
}
