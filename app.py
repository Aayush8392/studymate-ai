"""StudyMate AI -- Streamlit app: upload notes, generate study material,
review flashcards, take quizzes, and watch the adaptive loop respond to
weak concepts. Export a portable offline HTML study pack anytime."""
import re
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from services import db, pipeline, adaptive_loop
from services.gemini_client import describe_error
from services.note_parser import parse_uploaded_file
from export.export_builder import build_html, build_pdf
from agents.style_rules import VALID_STYLES, VALID_DEPTHS, DEFAULT_STYLE, DEFAULT_DEPTH
from agents.vision_reader import extract_notes_from_images
from agents import pyq_solver

load_dotenv()
db.init_db()

STYLE_LABELS = {"simple": "Simple", "standard": "Standard", "technical": "Technical"}
DEPTH_LABELS = {"overview": "Overview", "standard": "Standard", "deep": "Deep dive"}


def _safe_filename(title: str) -> str:
    """The download filename should match the document title exactly -- just
    strip characters Windows/macOS/Linux all forbid in filenames, rather than
    forcing underscores or appending a generic suffix."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "", title).strip()
    return cleaned or "My Notes"


def style_depth_pickers(key_prefix: str):
    col1, col2 = st.columns(2)
    with col1:
        style = st.selectbox(
            "Explanation style", VALID_STYLES, index=VALID_STYLES.index(DEFAULT_STYLE),
            format_func=lambda s: STYLE_LABELS[s], key=f"{key_prefix}_style",
            help="Controls vocabulary and language complexity.",
        )
    with col2:
        depth = st.selectbox(
            "Depth", VALID_DEPTHS, index=VALID_DEPTHS.index(DEFAULT_DEPTH),
            format_func=lambda d: DEPTH_LABELS[d], key=f"{key_prefix}_depth",
            help="Controls how much content/nuance is covered, independent of style.",
        )
    return style, depth

def _make_progress_handler():
    """Renders a growing checklist of concepts instead of overwriting a single
    status line -- each concept appears pending (⏳) the moment its generation
    starts and flips to done (✅) in place once the pipeline reports it
    finished, so progress is visible as a running list, not a single line
    that gets replaced every time."""
    checklist_box = st.empty()
    status_box = st.empty()
    items: list[list] = []
    order: dict[str, int] = {}

    def handle(msg: str):
        if msg.startswith("Generating material for: "):
            name = msg[len("Generating material for: "):]
            order[name] = len(items)
            items.append([name, False])
        elif msg.startswith("Completed: "):
            name = msg[len("Completed: "):]
            if name in order:
                items[order[name]][1] = True
        else:
            status_box.info(msg)

        if items:
            checklist_box.markdown(
                "\n\n".join(f"{'✅' if done else '⏳'} {name}" for name, done in items)
            )

    return handle


def _grade_and_store_quiz(doc, concepts, questions_by_concept):
    """Grades every answered question in one pass (nothing is graded until the
    whole form is submitted) and records each attempt. Does NOT call the LLM --
    remediation is a separate, explicit step so the student sees their results
    first and decides when to trigger it."""
    wrong_counts = {}
    answered_counts = {}
    results_by_concept = {}

    for c in concepts:
        questions = questions_by_concept.get(c["id"])
        if not questions:
            continue
        rows = []
        for q in questions:
            choice = st.session_state.get(f"q_{q['id']}")
            if choice is None:
                rows.append({"question": q, "selected_index": None, "correct": None})
                continue
            selected_index = q["options"].index(choice)
            correct = selected_index == q["correct_index"]
            db.record_quiz_attempt(q["id"], c["id"], selected_index, correct)
            rows.append({"question": q, "selected_index": selected_index, "correct": correct})
            answered_counts[c["id"]] = answered_counts.get(c["id"], 0) + 1
            if not correct:
                wrong_counts[c["id"]] = wrong_counts.get(c["id"], 0) + 1
        results_by_concept[c["id"]] = rows

    for c in concepts:
        if answered_counts.get(c["id"], 0) > 0 and wrong_counts.get(c["id"], 0) == 0:
            adaptive_loop.clear_concept_if_correct(c["id"])

    st.session_state[f"quiz_results_{doc['id']}"] = {
        "results_by_concept": results_by_concept,
        "wrong_counts": wrong_counts,
        "answered_counts": answered_counts,
        "remediation_reports": {},
    }


def _render_quiz_results(doc, state, concepts):
    results_by_concept = state["results_by_concept"]
    wrong_counts = state["wrong_counts"]
    answered_counts = state["answered_counts"]
    remediation_reports = state["remediation_reports"]

    for c in concepts:
        rows = [r for r in results_by_concept.get(c["id"], []) if r["selected_index"] is not None]
        if not rows:
            continue
        st.markdown(f"### {c['name']}")
        for row in rows:
            q = row["question"]
            st.markdown(f"**{q['question']}**")
            if row["correct"]:
                st.success(f"Your answer: {q['options'][row['selected_index']]} — Correct")
            else:
                misconception = q["distractor_notes"].get(str(row["selected_index"]), "")
                st.error(
                    f"Your answer: {q['options'][row['selected_index']]} — Incorrect"
                    + (f" ({misconception})" if misconception else "")
                )
                st.info(f"Correct answer: {q['options'][q['correct_index']]}")

        report = remediation_reports.get(c["id"])
        if report:
            _render_remediation_report(c["name"], report)

    weak_concepts = [
        c for c in concepts
        if c["id"] not in remediation_reports
        and adaptive_loop.is_weak(wrong_counts.get(c["id"], 0), answered_counts.get(c["id"], 0))
    ]
    if weak_concepts:
        fixing_key = f"fixing_weak_{doc['id']}"
        fix_error_key = f"fix_error_{doc['id']}"
        if fixing_key not in st.session_state:
            st.session_state[fixing_key] = False
        if st.session_state.get(fix_error_key):
            st.error(st.session_state.pop(fix_error_key))

        st.divider()
        st.write(
            "Weak topic(s) detected: " + ", ".join(f"**{c['name']}**" for c in weak_concepts)
        )
        fix_clicked = st.button("Fix weak topics now", disabled=_BUSY)
        if fix_clicked and not st.session_state[fixing_key]:
            st.session_state[fixing_key] = True
            st.rerun()

        if st.session_state[fixing_key]:
            # Reset in `finally`, not after/else -- Streamlit's Stop control
            # raises a StopException (a BaseException), which a plain
            # `except Exception` doesn't catch; only `finally` is guaranteed
            # to still run and clear the flag.
            try:
                for c in weak_concepts:
                    wrong_misconceptions = [
                        row["question"]["distractor_notes"].get(str(row["selected_index"]), "")
                        for row in results_by_concept.get(c["id"], [])
                        if row["selected_index"] is not None and row["correct"] is False
                    ]
                    wrong_misconceptions = [m for m in wrong_misconceptions if m]
                    with st.spinner(f"Adapting material for '{c['name']}'..."):
                        remediation_reports[c["id"]] = adaptive_loop.run_remediation(
                            doc["id"], c["id"], doc["raw_text"], misconceptions=wrong_misconceptions
                        )
            except Exception as e:
                st.session_state[fix_error_key] = f"Remediation failed: {describe_error(e)}"
            finally:
                st.session_state[fixing_key] = False
            st.rerun()


def _render_remediation_report(concept_name, report):
    if report["action"] == "remediation":
        st.info(
            f"Weak spot detected on **{concept_name}** — re-explained it, "
            f"added {report['drill_cards_added']} drill flashcards, and "
            f"{report['followup_questions_added']} follow-up questions. "
            "Check the Summary and Flashcards tabs."
        )
    else:
        st.warning(report["message"])


import html as _html_lib

# Same hex values as export/html_template.html's --attempt-N vars, so the
# in-app colors and the exported HTML pack's colors actually match.
ATTEMPT_COLORS = {1: "#1c5f9e", 2: "#b5651d", 3: "#a4442e"}
ATTEMPT_DOTS = {1: "🔵", 2: "🟠", 3: "🔴"}
ATTEMPT_LEGEND = "🔵 Attempt 1&nbsp;&nbsp;&nbsp;🟠 Attempt 2&nbsp;&nbsp;&nbsp;🔴 Attempt 3"


def _remediation_box(text: str, attempt: int, header: str):
    """A colored callout using the SAME hex per attempt as the HTML export,
    instead of st.info/warning/error -- those are Streamlit's own theme
    colors (its "warning" renders olive/mustard in dark mode, not orange),
    which didn't match what the legend promised."""
    color = ATTEMPT_COLORS.get(attempt, ATTEMPT_COLORS[1])
    safe_text = _html_lib.escape(text).replace("\n\n", "<br><br>")
    st.markdown(
        f'<div style="border-left:4px solid {color}; background:{color}22; '
        f'border-radius:8px; padding:14px 18px; margin:10px 0; font-size:0.95em;">'
        f'<div style="font-weight:700; color:{color}; margin-bottom:6px;">{header}</div>'
        f"{safe_text}</div>",
        unsafe_allow_html=True,
    )


def _render_remediation_summaries(concept_id):
    remediations = db.list_concept_remediations(concept_id)
    for rem in remediations:
        attempt = rem["attempt"]
        summary = db.parse_summary(rem["summary"])
        text = summary.get("explained_further") or summary.get("from_notes") or ""
        _remediation_box(text, attempt, f"🔁 Remediation — Attempt {attempt}")


st.set_page_config(page_title="StudyMate AI", page_icon="📘", layout="wide")


def _is_busy() -> bool:
    """True while any LLM-backed call is in flight -- the three flags for
    Generate/Solve PYQ/Fix weak topics, checked directly, plus a scan for any
    per-document "fixing_weak_<doc_id>" flag since that one's key is dynamic."""
    if st.session_state.get("generating_notes") or st.session_state.get("solving_pyq"):
        return True
    return any(
        (k.startswith("fixing_weak_") or k.startswith("expanding_pyq_")) and v
        for k, v in st.session_state.items()
    )


_BUSY = _is_busy()

# Sidebar's drag-to-resize is disabled and its width fixed -- a resizable
# sidebar looked accidental/unfinished for this app's fixed two-pane layout.
# The header's own z-index is far above any app content, so it -- and the
# native running-indicator/Stop control inside it -- stays on top of and
# clickable through the busy overlay below regardless of the overlay's
# z-index; only the overlay's own value matters for covering the page body.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { width: 340px !important; }
    [data-testid="stSidebarResizeHandle"] { display: none !important; }
    [data-testid="stAppDeployButton"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if _BUSY:
    # A plain st.spinner dims the page visually but everything underneath
    # stays fully clickable -- clicks just queue up oddly instead of being
    # blocked. This overlay actually intercepts clicks across the whole app
    # while a call is running. Streamlit's own running-indicator/Stop control
    # lives in the fixed header above this in the stacking order, so it stays
    # usable -- it's the only way to interrupt a call, deliberately.
    st.markdown(
        """
        <style>
        .sm-busy-overlay {
            position: fixed; inset: 0; z-index: 999; cursor: not-allowed;
            background: rgba(0,0,0,0.55);
            display: flex; align-items: center; justify-content: center;
        }
        .sm-busy-card {
            background: #1c1f26; border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px; padding: 28px 36px; max-width: 380px;
            text-align: center; font-family: sans-serif;
            box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        }
        .sm-busy-spinner {
            width: 34px; height: 34px; margin: 0 auto 14px;
            border: 3px solid rgba(255,255,255,0.2); border-top-color: #7fc4b2;
            border-radius: 50%; animation: sm-busy-spin 0.8s linear infinite;
        }
        @keyframes sm-busy-spin { to { transform: rotate(360deg); } }
        .sm-busy-text { color: #fff; font-size: 1.05rem; font-weight: 600; margin-bottom: 6px; }
        .sm-busy-sub { color: rgba(255,255,255,0.7); font-size: 0.85rem; }
        </style>
        <div class="sm-busy-overlay">
          <div class="sm-busy-card">
            <div class="sm-busy-spinner"></div>
            <div class="sm-busy-text">Working on it&hellip;</div>
            <div class="sm-busy-sub">Click the &#9632; stop control in the top-right corner to stop the request.
            If it's stuck, it will give up automatically within 30 seconds.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

st.title("📘 StudyMate AI")
st.caption("Upload your notes. Get a verified summary, flashcards, and a quiz that adapts to what you actually get wrong.")

with st.sidebar:
    st.header("Your documents")
    docs = db.list_documents()
    for d in docs:
        col_name, col_del = st.columns([4, 1])
        with col_name:
            if st.button(d["title"], key=f"doc_{d['id']}", use_container_width=True):
                st.session_state.selected_doc = d["id"]
        with col_del:
            if st.button("🗑️", key=f"del_{d['id']}", help="Delete this document"):
                db.delete_document(d["id"])
                if st.session_state.selected_doc == d["id"]:
                    st.session_state.selected_doc = None
                st.rerun()

    st.divider()
    st.header("Upload new notes")
    uploaded_files = st.file_uploader(
        "File(s) or image(s) of notes (PDF, DOCX, PPTX, TXT, PNG, JPG)",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    pasted = st.text_area("...or paste notes directly", height=120)
    extra_concepts_raw = st.text_input("Extra concepts to include (comma-separated, optional)")
    title = st.text_input("Title for this document", value="My Notes")
    notes_style, notes_depth = style_depth_pickers("notes")

    if "generating_notes" not in st.session_state:
        st.session_state.generating_notes = False
    if st.session_state.get("gen_error"):
        st.error(st.session_state.pop("gen_error"))

    gen_clicked = st.button(
        "Generate study material", type="primary", use_container_width=True,
        disabled=_BUSY,
    )
    if gen_clicked and not st.session_state.generating_notes:
        st.session_state.generating_notes = True
        st.rerun()

    if st.session_state.generating_notes:
        IMAGE_EXTS = {"png", "jpg", "jpeg"}
        doc_files = [f for f in uploaded_files if f.name.rsplit(".", 1)[-1].lower() not in IMAGE_EXTS]
        image_files = [f for f in uploaded_files if f.name.rsplit(".", 1)[-1].lower() in IMAGE_EXTS]

        notes_parts = []
        error = None
        if doc_files:
            try:
                notes_parts.extend(parse_uploaded_file(f.name, f.read()) for f in doc_files)
            except ValueError as e:
                error = str(e)
        if image_files and error is None:
            try:
                with st.spinner("Reading text from image(s)..."):
                    image_text = extract_notes_from_images(
                        [(f.read(), f.type or "image/png") for f in image_files]
                    )
                if image_text:
                    notes_parts.append(image_text)
                else:
                    error = "Couldn't read any text from the image(s). Try a clearer photo."
            except Exception as e:
                error = f"Couldn't read the image(s): {e}"
        if pasted.strip():
            notes_parts.append(pasted.strip())

        notes_text = "\n\n".join(notes_parts) if notes_parts else None
        if error:
            st.session_state.gen_error = error
            st.session_state.generating_notes = False
            st.rerun()
        elif not notes_text:
            st.session_state.gen_error = "Upload file(s)/image(s) or paste some notes first."
            st.session_state.generating_notes = False
            st.rerun()
        else:
            extra = [c.strip() for c in extra_concepts_raw.split(",") if c.strip()] or None
            progress_cb = _make_progress_handler()
            doc_id = None
            # The flag reset lives in `finally`, not after the try block --
            # clicking Streamlit's native Stop control raises a StopException,
            # which is a BaseException (not Exception), so it skips `except`
            # entirely and would otherwise leave the button disabled forever
            # since nothing after a plain try/except runs either. `finally`
            # still executes even for that injected exception.
            try:
                with st.spinner("Running the agent pipeline..."):
                    doc_id = pipeline.generate_study_material(
                        title, notes_text, extra_concepts=extra,
                        progress_cb=progress_cb,
                        style=notes_style, depth=notes_depth,
                    )
            except Exception as e:
                st.session_state.gen_error = f"Generation failed: {describe_error(e)}"
            finally:
                st.session_state.generating_notes = False
            if doc_id:
                st.session_state.selected_doc = doc_id
                st.success("Study material generated.")
            st.rerun()

    st.divider()
    st.header("...or solve a question paper (PYQ)")
    st.caption("Upload a previous year question paper -- as a file (PDF, DOCX, PPTX, TXT) or photo(s). "
               "We'll read the questions, build study material around the topics they test, and write model answers.")
    pyq_files = st.file_uploader(
        "Question paper file(s) or image(s) (PDF, DOCX, PPTX, TXT, PNG, JPG)",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True, key="pyq_uploader",
    )
    pyq_title = st.text_input("Title for this question paper", value="My PYQ", key="pyq_title")
    pyq_style, pyq_depth = style_depth_pickers("pyq")

    if "solving_pyq" not in st.session_state:
        st.session_state.solving_pyq = False
    if st.session_state.get("pyq_error"):
        st.error(st.session_state.pop("pyq_error"))

    pyq_clicked = st.button(
        "Solve question paper", type="primary", use_container_width=True,
        disabled=_BUSY,
    )
    if pyq_clicked and not st.session_state.solving_pyq:
        st.session_state.solving_pyq = True
        st.rerun()

    if st.session_state.solving_pyq:
        if not pyq_files:
            st.session_state.pyq_error = "Upload at least one file or image of the question paper."
            st.session_state.solving_pyq = False
            st.rerun()
        else:
            PYQ_IMAGE_EXTS = {"png", "jpg", "jpeg"}
            pyq_doc_files = [f for f in pyq_files if f.name.rsplit(".", 1)[-1].lower() not in PYQ_IMAGE_EXTS]
            pyq_image_files = [f for f in pyq_files if f.name.rsplit(".", 1)[-1].lower() in PYQ_IMAGE_EXTS]

            pyq_text = None
            pyq_parse_error = None
            if pyq_doc_files:
                try:
                    pyq_text = "\n\n".join(parse_uploaded_file(f.name, f.read()) for f in pyq_doc_files)
                except ValueError as e:
                    pyq_parse_error = str(e)
            images = [(f.read(), f.type or "image/png") for f in pyq_image_files]

            if pyq_parse_error:
                st.session_state.pyq_error = pyq_parse_error
                st.session_state.solving_pyq = False
                st.rerun()

            progress_cb = _make_progress_handler()
            doc_id = None
            # Reset in `finally` -- Streamlit's Stop control raises a
            # StopException (a BaseException), which skips a plain `except`
            # entirely; only `finally` is guaranteed to still run.
            try:
                with st.spinner("Reading and solving the question paper..."):
                    doc_id = pipeline.generate_from_pyq(
                        pyq_title, text_notes=pyq_text, images=images, progress_cb=progress_cb,
                        style=pyq_style, depth=pyq_depth,
                    )
            except Exception as e:
                st.session_state.pyq_error = describe_error(e)
            finally:
                st.session_state.solving_pyq = False
            if doc_id:
                st.session_state.selected_doc = doc_id
                st.success("Question paper solved.")
            st.rerun()

# ---- Main panel ----

if not st.session_state.selected_doc:
    st.info("Upload notes in the sidebar to get started, or pick an existing document.")
    st.stop()

doc = db.get_document(st.session_state.selected_doc)
concepts = db.list_concepts(doc["id"])

st.subheader(doc["title"])
st.caption(
    f"{STYLE_LABELS.get(doc['explanation_style'], doc['explanation_style'])} style · "
    f"{DEPTH_LABELS.get(doc['depth_level'], doc['depth_level'])} depth"
)

col_export_html, col_export_pdf, _ = st.columns([1, 1, 4], gap="small")
with col_export_html:
    html_content = build_html(doc["id"])
    st.download_button(
        "⬇️ Download as HTML",
        data=html_content,
        file_name=f"{_safe_filename(doc['title'])}.html",
        mime="text/html",
    )
with col_export_pdf:
    try:
        pdf_content = build_pdf(doc["id"])
        st.download_button(
            "⬇️ Download as PDF",
            data=pdf_content,
            file_name=f"{_safe_filename(doc['title'])}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

pyq_solutions = db.list_pyq_solutions(doc["id"])
tab_names = ["Summary", "Flashcards", "Quiz", "Progress dashboard"]
if pyq_solutions:
    tab_names.append("PYQ Solutions")
tabs = st.tabs(tab_names)
tab_summary, tab_flashcards, tab_quiz, tab_dashboard = tabs[:4]
tab_pyq = tabs[4] if pyq_solutions else None

with tab_summary:
    if any(c["attempts"] > 0 for c in concepts):
        st.caption(f"Legend for remediation sections below: {ATTEMPT_LEGEND}", unsafe_allow_html=True)
    for c in concepts:
        with st.expander(c["name"], expanded=False):
            if c["attempts"] > 0:
                st.caption(f"Revisited — attempt {c['attempts']}")
            summary = db.parse_summary(c["summary"])

            if summary.get("from_notes"):
                st.markdown("**From your notes**")
                st.write(summary["from_notes"])

            if summary.get("explained_further"):
                st.markdown("**Explained further** *(beyond your notes)*")
                st.write(summary["explained_further"])

            if summary.get("comparisons_and_limits"):
                st.markdown("**Comparisons & limitations** *(Deep dive)*")
                st.write(summary["comparisons_and_limits"])

            if summary.get("analogy"):
                st.info(f"**Analogy:** {summary['analogy']}")

            if summary.get("key_terms"):
                st.markdown("**Key terms**")
                for kt in summary["key_terms"]:
                    if isinstance(kt, dict):
                        st.markdown(f"- **{kt.get('term', '')}** — {kt.get('definition', '')}")

            if summary.get("exam_answer_example"):
                st.markdown("**How to write this in an exam**")
                st.success(summary["exam_answer_example"])

            if summary.get("expand_hints"):
                st.markdown("**To go further, mention:**")
                for hint in summary["expand_hints"]:
                    st.markdown(f"- {hint}")

            if summary.get("why_it_matters"):
                st.caption(f"**Why it matters:** {summary['why_it_matters']}")

            if c["attempts"] > 0:
                _render_remediation_summaries(c["id"])

with tab_flashcards:
    if any(c["attempts"] > 0 for c in concepts):
        st.caption(f"Legend for remediation cards below: {ATTEMPT_LEGEND}", unsafe_allow_html=True)
    flashcards = db.list_flashcards(doc["id"])
    now = datetime.utcnow().isoformat()
    due_ids = {f["id"] for f in flashcards if f["next_review_at"] and f["next_review_at"] <= now}

    if due_ids:
        st.subheader(f"📌 Due for review ({len(due_ids)})")
        st.caption(
            "These cards use spaced repetition: getting a concept's quiz fully right moves its "
            "cards to a higher box (reviewed less often), getting it wrong drops them back to "
            "box 1 (reviewed immediately). Box 1 = due now, box 5 = due in 2 weeks."
        )
        due_by_concept = {}
        for f in flashcards:
            if f["id"] in due_ids:
                due_by_concept.setdefault(f["concept_id"], []).append(f)
        for c in concepts:
            cards = due_by_concept.get(c["id"], [])
            if not cards:
                continue
            st.markdown(f"**{c['name']}**")
            due_cols = st.columns(3)
            for i, card in enumerate(cards):
                with due_cols[i % 3]:
                    with st.expander(card["front"]):
                        st.write(card["back"])
                        st.caption(f"Box {card['box_level']}")
        st.divider()

    st.subheader("Not due yet" if due_ids else "Flashcards")
    not_due = [f for f in flashcards if f["id"] not in due_ids]
    if due_ids and not not_due:
        st.caption("Every topic needs review right now — nothing here yet. Check the section above.")
    by_concept = {}
    for f in not_due:
        by_concept.setdefault(f["concept_id"], []).append(f)
    for c in concepts:
        cards = by_concept.get(c["id"], [])
        if not cards:
            continue
        st.markdown(f"**{c['name']}**")
        original_cards = [card for card in cards if card["source"] != "drill"]
        drill_cards = [card for card in cards if card["source"] == "drill"]

        cols = st.columns(3)
        for i, card in enumerate(original_cards):
            with cols[i % 3]:
                with st.expander(card["front"]):
                    st.write(card["back"])
                    if card["next_review_at"]:
                        st.caption(f"Box {card['box_level']} · next review {card['next_review_at'][:10]}")

        for attempt in sorted({card["attempt"] for card in drill_cards}):
            dot = ATTEMPT_DOTS.get(attempt, ATTEMPT_DOTS[1])
            st.caption(f"Remediation flashcards — attempt {attempt}")
            attempt_cards = [card for card in drill_cards if card["attempt"] == attempt]
            drill_cols = st.columns(3)
            for i, card in enumerate(attempt_cards):
                with drill_cols[i % 3]:
                    with st.expander(f"{dot} {card['front']}"):
                        st.write(card["back"])
                        if card["next_review_at"]:
                            st.caption(f"Box {card['box_level']} · next review {card['next_review_at'][:10]}")

with tab_quiz:
    quiz_active_key = f"quiz_active_{doc['id']}"
    quiz_results_key = f"quiz_results_{doc['id']}"

    questions_by_concept = {
        c["id"]: db.list_quiz_questions(doc["id"], concept_id=c["id"]) for c in concepts
    }
    questions_by_concept = {cid: qs for cid, qs in questions_by_concept.items() if qs}

    if not questions_by_concept:
        st.info("No quiz questions available for this document yet.")
    elif quiz_results_key in st.session_state:
        _render_quiz_results(doc, st.session_state[quiz_results_key], concepts)
        if st.button("Take quiz again"):
            del st.session_state[quiz_results_key]
            st.session_state[quiz_active_key] = False
            for c in concepts:
                st.session_state.pop(f"skip_{doc['id']}_{c['id']}", None)
            st.rerun()
    elif not st.session_state.get(quiz_active_key):
        total_qs = sum(len(qs) for qs in questions_by_concept.values())
        st.write(
            f"This quiz has {total_qs} question(s) across {len(questions_by_concept)} concept(s). "
            "You'll see the correct answers and any weak-topic feedback after you submit."
        )
        if st.button("Start quiz", type="primary"):
            st.session_state[quiz_active_key] = True
            st.rerun()
    else:
        st.caption("Skip a topic to leave its questions unanswered -- skipped topics are excluded "
                   "from grading entirely, just like leaving individual questions blank.")
        skip_cols = st.columns(3)
        skipped_concepts = set()
        skippable = [c for c in concepts if c["id"] in questions_by_concept]
        for i, c in enumerate(skippable):
            with skip_cols[i % 3]:
                if st.checkbox(f"Skip: {c['name']}", key=f"skip_{doc['id']}_{c['id']}"):
                    skipped_concepts.add(c["id"])

        with st.form(key=f"quiz_form_{doc['id']}"):
            for c in concepts:
                questions = questions_by_concept.get(c["id"])
                if not questions or c["id"] in skipped_concepts:
                    continue
                st.markdown(f"### {c['name']}")
                for q in questions:
                    st.radio(q["question"], q["options"], key=f"q_{q['id']}", index=None)
            submitted = st.form_submit_button("Submit quiz", type="primary")

        if submitted:
            _grade_and_store_quiz(doc, concepts, questions_by_concept)
            st.session_state[quiz_active_key] = False
            st.rerun()

with tab_dashboard:
    st.write("Weak-concept tracking across this document.")
    for c in concepts:
        cols = st.columns([3, 1, 1])
        cols[0].write(c["name"])
        cols[1].write(f"Attempts: {c['attempts']}")
        status_emoji = {"unseen": "⚪", "learning": "🟡", "cleared": "🟢", "flagged": "🔴"}.get(c["status"], "⚪")
        cols[2].write(f"{status_emoji} {c['status']}")

PYQ_EXPANSION_WORD_CEILING = 700  # matches the tuned ~700-word target for genuine 10-mark depth


if tab_pyq is not None:
    with tab_pyq:
        st.write("Model answers for each question extracted from your uploaded paper.")
        for i, sol in enumerate(pyq_solutions, start=1):
            with st.expander(f"Q{i}. {sol['question']}", expanded=False):
                if sol.get("matched_concept"):
                    st.caption(f"Related concept: {sol['matched_concept']}")
                st.markdown("**Model answer**")
                st.success(sol["model_answer"])

                expand_key = f"expanding_pyq_{sol['id']}"
                expand_error_key = f"expand_error_{sol['id']}"
                if expand_key not in st.session_state:
                    st.session_state[expand_key] = False
                if st.session_state.get(expand_error_key):
                    st.error(st.session_state.pop(expand_error_key))

                word_count = len(sol["model_answer"].split())
                if sol.get("expand_hints") and word_count < PYQ_EXPANSION_WORD_CEILING:
                    st.markdown("**To go further, mention:**")
                    for hint in sol["expand_hints"]:
                        st.markdown(f"- {hint}")
                    expand_clicked = st.button(
                        "Expand answer using these hints", key=f"expand_btn_{sol['id']}",
                        disabled=_BUSY,
                    )
                    if expand_clicked and not st.session_state[expand_key]:
                        st.session_state[expand_key] = True
                        st.rerun()

                    if st.session_state[expand_key]:
                        try:
                            with st.spinner("Expanding this answer..."):
                                result = pyq_solver.expand_answer(
                                    sol["question"], sol.get("matched_concept", ""),
                                    sol["model_answer"], sol["expand_hints"],
                                    doc["raw_text"], style=pyq_style,
                                )
                            new_answer = sol["model_answer"] + "\n\n" + result["addition"]
                            new_word_count = len(new_answer.split())
                            # Cap enforced here in code, not left to the model --
                            # once the ceiling is reached, no new hints are kept
                            # regardless of what the model returned, so the
                            # button/hints disappear on the next render.
                            new_hints = result["new_hints"] if new_word_count < PYQ_EXPANSION_WORD_CEILING else []
                            db.update_pyq_solution(sol["id"], new_answer, new_hints)
                        except Exception as e:
                            st.session_state[expand_error_key] = f"Expansion failed: {describe_error(e)}"
                        finally:
                            st.session_state[expand_key] = False
                        st.rerun()
