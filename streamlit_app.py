"""Streamlit demo: validate structured-output OCR on an Airtable-export CSV.

Run from the ``LP/`` folder:

    streamlit run enhanced/streamlit_app.py

The app uploads a CSV, runs every row through ``StructuredOcrClient`` (forced
tool-use, validated by Pydantic), and shows each result as:

  * the original attachment images (thumbnails),
  * the structured fields rendered with KaTeX,
  * an editable form so a reviewer can correct mistakes,
  * the raw tool-input JSON,
  * approve / reject buttons.

Approved + edited results can be exported as one JSON file at the end.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Make the ``enhanced`` package importable when run via ``streamlit run``.
# Streamlit only adds the script's own directory to sys.path; we need the
# parent (LP/) so that ``import enhanced.x`` works for the relative imports
# inside enhanced_ocr.py and batch.py.
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent

# Put both the script's directory AND its parent on sys.path so the imports
# below work regardless of whether this file lives inside an ``enhanced/``
# package or at the repo root with its sibling source files.
for _p in (_HERE, _PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402  (path setup must run first)

try:
    # Layout A: ``enhanced/`` is a subfolder of the dir containing this file
    # (e.g. repo-root/streamlit_app.py + repo-root/enhanced/*.py).
    from enhanced.batch import (  # noqa: E402
        BatchSummary,
        RowResult,
        group_rows,
        run_batch,
    )
    from enhanced.enhanced_ocr import (  # noqa: E402
        StructuredOcrClient,
        StructuredOcrError,
    )
    from enhanced.schemas import ExamQuestion  # noqa: E402
except ModuleNotFoundError:
    # Layout B: every source file is a flat sibling of this script.
    from batch import (  # noqa: E402, F401  # type: ignore[no-redef]
        BatchSummary,
        RowResult,
        group_rows,
        run_batch,
    )
    from enhanced_ocr import (  # noqa: E402, F401  # type: ignore[no-redef]
        StructuredOcrClient,
        StructuredOcrError,
    )
    from schemas import ExamQuestion  # noqa: E402, F401  # type: ignore[no-redef]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (_HERE / ".env", _PARENT / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)
            return


_load_env()


def _validate_api_key(key: str) -> tuple[bool, str]:
    """Cheap probe to verify a runtime-entered key actually works.

    Always uses Haiku (smallest, cheapest model) regardless of the user's
    selected model, so a 'permission denied for opus' error can't mask a
    valid key.
    """
    import anthropic

    try:
        client = anthropic.Anthropic(api_key=key, timeout=15.0, max_retries=0)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True, "API key valid."
    except anthropic.AuthenticationError:
        return False, "Invalid API key (401)."
    except anthropic.APIStatusError as e:
        return False, f"API error ({e.status_code}): {str(e.message)[:120]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


st.set_page_config(
    page_title="Enhanced OCR Validator",
    page_icon="N",
    layout="wide",
)


def _init_state() -> None:
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("edited", {})
    st.session_state.setdefault("decisions", {})
    st.session_state.setdefault("preview_groups", None)
    st.session_state.setdefault("uploaded_csv_bytes", None)
    st.session_state.setdefault("uploaded_csv_name", None)
    st.session_state.setdefault("validated_key", None)
    st.session_state.setdefault("validation_msg", "")
    # Seed the API-key widget's session_state slot once. After this, the
    # widget owns the value; passing `value=` alongside `key=` would warn.
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = os.environ.get("ANTHROPIC_API_KEY", "")


_init_state()


with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        key="api_key_input",
        help=(
            "Paste your Anthropic API key. Stored only in this Streamlit "
            "session - never written to disk or logged. Auto-filled from "
            "ANTHROPIC_API_KEY if set in the environment."
        ),
    ).strip()

    # If the user edited the key, drop any prior validation result.
    if (
        st.session_state.get("validated_key") is not None
        and st.session_state["validated_key"] != api_key
    ):
        st.session_state.validated_key = None
        st.session_state.validation_msg = ""

    if api_key:
        if st.button("Validate key", use_container_width=True):
            with st.spinner("Testing key against Anthropic..."):
                ok, msg = _validate_api_key(api_key)
            st.session_state.validated_key = api_key if ok else None
            st.session_state.validation_msg = msg

        if st.session_state.get("validated_key") == api_key:
            st.success(st.session_state.get("validation_msg") or "Key validated.")
        elif st.session_state.get("validation_msg"):
            st.error(st.session_state["validation_msg"])
    else:
        st.info("Paste an Anthropic API key above to enable the run button.")

    model = st.selectbox(
        "Model",
        options=[
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        index=0,
        help="Opus 4.7 is the most accurate for math-heavy OCR; Sonnet is ~3x cheaper.",
    )

    max_rows = st.number_input(
        "Max rows to process",
        min_value=1,
        max_value=500,
        value=10,
        step=1,
        help="Each row becomes one structured ExamQuestion.",
    )

    max_workers = st.slider(
        "Parallel workers",
        min_value=1,
        max_value=8,
        value=3,
        help="Concurrent Claude calls. Lower if you hit rate limits.",
    )

    st.divider()
    st.caption(
        "Output schema is `ExamQuestion` from `enhanced/schemas.py`. "
        "Claude is forced to call the `submit_exam_question` tool, so every "
        "row produces the same JSON shape."
    )


st.title("Enhanced OCR Validator")
st.write(
    "Upload an Airtable-export CSV, structured-OCR each row, then review and "
    "correct the parsed exam questions before exporting."
)

uploaded = st.file_uploader(
    "Airtable export CSV",
    type=["csv"],
    help="Must have a 'Question File' or 'Question Title (main question and answer)' column.",
)

if uploaded is not None:
    csv_bytes = uploaded.read()
    if csv_bytes != st.session_state.uploaded_csv_bytes:
        st.session_state.uploaded_csv_bytes = csv_bytes
        st.session_state.uploaded_csv_name = uploaded.name
        st.session_state.summary = None
        st.session_state.edited = {}
        st.session_state.decisions = {}
        try:
            st.session_state.preview_groups = group_rows(
                csv_bytes, max_rows=int(max_rows)
            )
        except ValueError as e:
            st.error(f"Could not parse CSV: {e}")
            st.session_state.preview_groups = None


if st.session_state.preview_groups:
    groups = st.session_state.preview_groups
    total_imgs = sum(len(g.ordered_attachments) for g in groups)

    col_a, col_b, col_c = st.columns([2, 2, 3])
    col_a.metric("Rows to process", len(groups))
    col_b.metric("Total images", total_imgs)
    col_c.metric(
        "Rows with solution images",
        sum(1 for g in groups if g.has_solution),
    )

    with st.expander("Preview attachments (first 5 rows)", expanded=False):
        for g in groups[:5]:
            st.markdown(
                f"**Row {g.row_index}** - _{g.question_title or '(no title)'}_"
            )
            for a in g.question_attachments:
                st.markdown(f"&nbsp;&nbsp;Q: `{a.label[:80]}`")
            for a in g.answer_attachments:
                st.markdown(f"&nbsp;&nbsp;A: `{a.label[:80]}`")

    run_disabled = not api_key.strip()
    if run_disabled:
        st.warning("Enter an Anthropic API key in the sidebar to enable the run button.")

    if st.button(
        "Run structured OCR",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        progress_bar = st.progress(0.0, text="Starting...")
        status_box = st.empty()

        try:
            client = StructuredOcrClient(api_key=api_key.strip(), model=model)
        except StructuredOcrError as e:
            st.error(str(e))
            st.stop()

        total = len(groups)
        completed = {"n": 0}
        started_at = time.monotonic()

        def _on_progress(r: RowResult) -> None:
            completed["n"] += 1
            done = completed["n"]
            elapsed = time.monotonic() - started_at
            rate = done / max(elapsed, 1e-3)
            eta = (total - done) / max(rate, 1e-3)
            marker = "OK" if r.status == "ok" else "ERR"
            status_box.write(
                f"[{marker}] row {r.row_index} - {r.status} "
                f"({r.duration_s:.1f}s)"
                + (f" - {r.error}" if r.error else "")
            )
            progress_bar.progress(
                done / total,
                text=f"{done}/{total} rows - ~{eta:.0f}s remaining",
            )

        summary: BatchSummary = run_batch(
            st.session_state.uploaded_csv_bytes,
            client=client,
            max_rows=int(max_rows),
            max_workers=int(max_workers),
            progress=_on_progress,
        )
        progress_bar.empty()
        status_box.empty()

        st.session_state.summary = summary
        st.session_state.edited = {
            r.row_index: dict(r.question) if r.question else {}
            for r in summary.results
        }
        st.session_state.decisions = {
            r.row_index: "pending" for r in summary.results
        }
        st.success(
            f"Done - {summary.ok_count} ok, {summary.error_count} error. "
            f"Tokens: {summary.total_input_tokens} in / "
            f"{summary.total_output_tokens} out."
        )


def _render_question_preview(q: dict) -> None:
    src = q.get("source") or {}
    src_bits = [src.get(k) for k in ("school", "year", "exam_type", "paper") if src.get(k)]
    if src_bits:
        st.caption(" / ".join(str(b) for b in src_bits))

    header_bits = []
    if q.get("question_number"):
        header_bits.append(f"**Q{q['question_number']}**")
    if q.get("marks_total"):
        header_bits.append(f"[{q['marks_total']} marks]")
    if header_bits:
        st.markdown(" ".join(header_bits))

    if q.get("stem"):
        st.markdown(q["stem"])

    for part in q.get("parts") or []:
        marks = f" _[{part['marks']} marks]_" if part.get("marks") else ""
        st.markdown(f"**{part.get('label', '')}**{marks}")
        if part.get("text"):
            st.markdown(part["text"])
        for sp in part.get("sub_parts") or []:
            sp_marks = f" _[{sp['marks']} marks]_" if sp.get("marks") else ""
            st.markdown(f"&nbsp;&nbsp;**{sp.get('label', '')}**{sp_marks}")
            if sp.get("text"):
                st.markdown(f"&nbsp;&nbsp;{sp['text']}")

    diagrams = q.get("diagrams") or []
    if diagrams:
        st.markdown("**Diagrams**")
        for d in diagrams:
            st.markdown(f"- _{d.get('location', '?')}_: {d.get('description', '')}")

    sol = q.get("solution") or {}
    by_part = sol.get("by_part") or []
    if by_part or sol.get("overall_final_answer") or sol.get("raw_markdown"):
        st.markdown("**Solution**")
        for bp in by_part:
            label = bp.get("part_label") or ""
            st.markdown(f"_{label}_")
            for step in bp.get("steps") or []:
                st.markdown(step)
            if bp.get("final_answer"):
                st.markdown(f"**Answer:** {bp['final_answer']}")
        if sol.get("overall_final_answer"):
            st.markdown(f"**Final answer:** {sol['overall_final_answer']}")
        if sol.get("raw_markdown"):
            st.markdown(sol["raw_markdown"])

    chips = []
    if q.get("topics"):
        chips.append("Topics: " + ", ".join(q["topics"]))
    if q.get("difficulty_hint"):
        chips.append(f"Difficulty: {q['difficulty_hint']}")
    if q.get("confidence") is not None:
        chips.append(f"Confidence: {q['confidence']:.2f}")
    if chips:
        st.caption(" / ".join(chips))

    if q.get("illegible_regions"):
        st.warning("Illegible: " + "; ".join(q["illegible_regions"]))
    if q.get("notes"):
        st.info(f"Notes: {q['notes']}")


def _editable_form(row_index: int, q: dict) -> dict:
    edited = dict(q)
    edited["question_number"] = st.text_input(
        "Question number", value=q.get("question_number") or "", key=f"qn_{row_index}"
    )
    edited["marks_total"] = st.number_input(
        "Total marks",
        min_value=0,
        max_value=200,
        value=int(q.get("marks_total") or 0),
        key=f"mt_{row_index}",
    ) or None

    src = q.get("source") or {}
    c1, c2, c3, c4 = st.columns(4)
    new_src = {
        "school": c1.text_input("School", value=src.get("school") or "", key=f"sch_{row_index}"),
        "year": c2.number_input(
            "Year",
            min_value=1980,
            max_value=2100,
            value=int(src.get("year") or 2024),
            key=f"yr_{row_index}",
        ),
        "exam_type": c3.text_input(
            "Exam type", value=src.get("exam_type") or "", key=f"et_{row_index}"
        ),
        "paper": c4.text_input(
            "Paper", value=src.get("paper") or "", key=f"pp_{row_index}"
        ),
    }
    edited["source"] = {k: v for k, v in new_src.items() if v}

    edited["stem"] = st.text_area(
        "Stem (context before part (a))",
        value=q.get("stem") or "",
        key=f"stem_{row_index}",
        height=80,
    )

    new_parts = []
    parts = q.get("parts") or []
    if parts:
        st.markdown("**Parts**")
    for i, part in enumerate(parts):
        with st.container(border=True):
            pc1, pc2 = st.columns([1, 3])
            new_label = pc1.text_input(
                f"Label #{i}",
                value=part.get("label") or "",
                key=f"plabel_{row_index}_{i}",
            )
            new_marks = pc2.number_input(
                f"Marks #{i}",
                min_value=0,
                max_value=50,
                value=int(part.get("marks") or 0),
                key=f"pmarks_{row_index}_{i}",
            )
            new_text = st.text_area(
                f"Text #{i}",
                value=part.get("text") or "",
                key=f"ptext_{row_index}_{i}",
                height=100,
            )
            sub_parts = []
            for j, sp in enumerate(part.get("sub_parts") or []):
                sc1, sc2 = st.columns([1, 3])
                sp_label = sc1.text_input(
                    f"  Sub-label #{i}.{j}",
                    value=sp.get("label") or "",
                    key=f"splabel_{row_index}_{i}_{j}",
                )
                sp_marks = sc2.number_input(
                    f"  Sub-marks #{i}.{j}",
                    min_value=0,
                    max_value=50,
                    value=int(sp.get("marks") or 0),
                    key=f"spmarks_{row_index}_{i}_{j}",
                )
                sp_text = st.text_area(
                    f"  Sub-text #{i}.{j}",
                    value=sp.get("text") or "",
                    key=f"sptext_{row_index}_{i}_{j}",
                    height=70,
                )
                sub_part = {"label": sp_label, "text": sp_text}
                if sp_marks:
                    sub_part["marks"] = sp_marks
                sub_parts.append(sub_part)
            new_part = {"label": new_label, "text": new_text}
            if new_marks:
                new_part["marks"] = new_marks
            if sub_parts:
                new_part["sub_parts"] = sub_parts
            new_parts.append(new_part)
    edited["parts"] = new_parts

    edited["topics"] = [
        t.strip()
        for t in st.text_input(
            "Topics (comma-separated)",
            value=", ".join(q.get("topics") or []),
            key=f"topics_{row_index}",
        ).split(",")
        if t.strip()
    ]
    edited["difficulty_hint"] = st.selectbox(
        "Difficulty",
        options=["", "EASY", "MEDIUM", "HARD"],
        index=(
            ["", "EASY", "MEDIUM", "HARD"].index(q.get("difficulty_hint") or "")
        ),
        key=f"diff_{row_index}",
    ) or None
    edited["notes"] = st.text_area(
        "Reviewer notes",
        value=q.get("notes") or "",
        key=f"notes_{row_index}",
        height=70,
    )

    for k in ("diagrams", "solution", "confidence", "illegible_regions"):
        edited.setdefault(k, q.get(k))

    return edited


def _validate_edited(edited: dict) -> tuple[bool, Optional[str]]:
    try:
        ExamQuestion.model_validate(edited)
    except Exception as e:
        return False, str(e)
    return True, None


summary: Optional[BatchSummary] = st.session_state.summary
if summary is not None:
    st.divider()
    st.subheader("Results")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("OK", summary.ok_count)
    sc2.metric("Errors", summary.error_count)
    sc3.metric(
        "Approved",
        sum(1 for v in st.session_state.decisions.values() if v == "approved"),
    )
    sc4.metric(
        "Rejected",
        sum(1 for v in st.session_state.decisions.values() if v == "rejected"),
    )

    for r in summary.results:
        if r is None:
            continue
        decision = st.session_state.decisions.get(r.row_index, "pending")
        badge = {"approved": "[A]", "rejected": "[R]", "pending": "[?]"}.get(
            decision, "[?]"
        )
        title = f"{badge} Row {r.row_index} - {r.question_title or '(no title)'} - {r.status}"
        with st.expander(title, expanded=False):
            if r.status == "error":
                st.error(r.error or "Unknown error")
                if r.raw_tool_input:
                    with st.popover("Raw tool input that failed validation"):
                        st.json(r.raw_tool_input)
                continue

            tab_imgs, tab_view, tab_edit, tab_json = st.tabs(
                ["Images", "Rendered", "Edit", "Raw JSON"]
            )

            with tab_imgs:
                if r.attachment_urls:
                    st.markdown("**Question images**")
                    cols = st.columns(min(len(r.attachment_urls), 3))
                    for i, url in enumerate(r.attachment_urls):
                        with cols[i % len(cols)]:
                            try:
                                st.image(url, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Image preview failed: {e}")
                            st.caption(url[:80] + "...")
                if r.answer_urls:
                    st.markdown("**Solution images**")
                    cols = st.columns(min(len(r.answer_urls), 3))
                    for i, url in enumerate(r.answer_urls):
                        with cols[i % len(cols)]:
                            try:
                                st.image(url, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Image preview failed: {e}")
                            st.caption(url[:80] + "...")

            current = st.session_state.edited.get(r.row_index) or (r.question or {})

            with tab_view:
                _render_question_preview(current)
                st.caption(
                    f"Tokens: {r.input_tokens} in / {r.output_tokens} out / "
                    f"{r.duration_s:.1f}s"
                )

            with tab_edit:
                edited = _editable_form(r.row_index, current)
                cba, cbb, _ = st.columns([1, 1, 4])
                if cba.button("Save edits", key=f"save_{r.row_index}"):
                    ok, err = _validate_edited(edited)
                    if ok:
                        st.session_state.edited[r.row_index] = edited
                        st.success("Saved.")
                    else:
                        st.error(f"Validation failed: {err}")
                if cbb.button("Reset", key=f"reset_{r.row_index}"):
                    st.session_state.edited[r.row_index] = dict(r.question or {})
                    st.rerun()

            with tab_json:
                st.json(current)
                if r.raw_tool_input and r.raw_tool_input != current:
                    with st.popover("Show original tool input"):
                        st.json(r.raw_tool_input)

            d1, d2, _ = st.columns([1, 1, 4])
            if d1.button(
                "Approve",
                key=f"approve_{r.row_index}",
                type="primary" if decision != "approved" else "secondary",
            ):
                st.session_state.decisions[r.row_index] = "approved"
                st.rerun()
            if d2.button(
                "Reject",
                key=f"reject_{r.row_index}",
                type="primary" if decision != "rejected" else "secondary",
            ):
                st.session_state.decisions[r.row_index] = "rejected"
                st.rerun()

    st.divider()
    st.subheader("Export")

    export_mode = st.radio(
        "What to export",
        options=["Approved only", "All non-rejected", "Everything"],
        horizontal=True,
    )

    def _gather_export() -> list[dict]:
        out = []
        for r in summary.results:
            if r is None:
                continue
            decision = st.session_state.decisions.get(r.row_index, "pending")
            if export_mode == "Approved only" and decision != "approved":
                continue
            if export_mode == "All non-rejected" and decision == "rejected":
                continue
            payload = {
                "row_index": r.row_index,
                "question_title": r.question_title,
                "attachment_urls": r.attachment_urls,
                "answer_urls": r.answer_urls,
                "status": r.status,
                "decision": decision,
                "question": st.session_state.edited.get(r.row_index)
                or r.question,
                "error": r.error,
                "tokens": {
                    "input": r.input_tokens,
                    "output": r.output_tokens,
                    "cache_read": r.cache_read_tokens,
                },
            }
            out.append(payload)
        return out

    export_payload = _gather_export()
    st.write(f"{len(export_payload)} row(s) selected for export.")
    st.download_button(
        "Download JSON",
        data=json.dumps(export_payload, indent=2, ensure_ascii=False),
        file_name=(
            (st.session_state.uploaded_csv_name or "questions").rsplit(".", 1)[0]
            + ".validated.json"
        ),
        mime="application/json",
        use_container_width=True,
    )
