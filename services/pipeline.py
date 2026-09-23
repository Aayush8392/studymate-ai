"""Ties the agents together: given raw notes, extracts concepts and generates
summary + flashcards + quiz for each, verified before being saved."""
import json
import re
from difflib import SequenceMatcher
from services import db
from services.json_utils import looks_like_json_blob
from services.logging_setup import get_logger
from agents import orchestrator, summarizer, flashcard_maker, quiz_maker, verifier, pyq_solver
from agents.style_rules import DEFAULT_STYLE, DEFAULT_DEPTH, flashcard_n, quiz_n

logger = get_logger(__name__)

_SIMILARITY_THRESHOLD = 0.75


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _is_duplicate(candidate: str, seen: list[str]) -> bool:
    norm = _normalize(candidate)
    for existing in seen:
        if SequenceMatcher(None, norm, existing).ratio() >= _SIMILARITY_THRESHOLD:
            return True
    return False


def generate_study_material(title: str, notes: str, extra_concepts: list[str] | None = None,
                             progress_cb=None, style: str = DEFAULT_STYLE,
                             depth: str = DEFAULT_DEPTH) -> int:
    """Runs the full pipeline and persists everything. Returns the document id.
    progress_cb(str) is called with status messages if provided, for UI feedback."""
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    doc_id = db.create_document(title, notes, explanation_style=style, depth_level=depth)

    report("Extracting concepts...")
    concepts = orchestrator.extract_concepts(notes, extra_concepts, depth=depth)

    # Track normalized flashcard fronts so overlapping concepts (common when an
    # outline covers related sub-topics) don't produce near-duplicate flashcards.
    # Key terms are NOT deduped here anymore -- a concept is allowed to list a
    # term another concept also uses (that's normal; real topics revisit shared
    # vocabulary from different angles). Deduping there was starving later
    # concepts of any genuinely distinct vocabulary, forcing weak filler terms
    # just to hit the count. Cross-document repetition is instead merged only
    # when the standalone Glossary view is assembled (see export_builder.py),
    # which is the only place a repeated term would actually look redundant.
    seen_flashcard_fronts: list[str] = []

    for concept_name in concepts:
        report(f"Generating material for: {concept_name}")
        source = "manual" if extra_concepts and concept_name in extra_concepts else "auto"
        concept_id = db.create_concept(doc_id, concept_name, source=source)

        summary = summarizer.summarize_concept(concept_name, notes, style=style, depth=depth)
        summary["key_terms"] = [
            kt for kt in summary.get("key_terms", []) if isinstance(kt, dict) and kt.get("term")
        ]
        db.update_concept_summary(concept_id, json.dumps(summary))

        # Ground flashcards/quiz in the notes PLUS the concept's own explanation,
        # not the raw notes alone. This matters most for a manually-added extra
        # concept the notes don't cover at all -- without this, flashcards get
        # silently verified into nothing (nothing to ground them in) while quiz
        # questions drift onto whatever unrelated topic the raw notes DO cover,
        # since the model still has to produce something notes-grounded. Adding
        # the concept's real explanation gives both a genuine, on-topic source.
        grounding_text = f"{notes}\n\n{summary.get('explained_further', '')}"

        requested_n = flashcard_n(depth)
        cards = flashcard_maker.make_flashcards(concept_name, grounding_text, n=requested_n, style=style, depth=depth)
        raw_count = len(cards)
        cards = [
            c for c in cards
            if isinstance(c, dict) and isinstance(c.get("front"), str) and c["front"].strip()
            and isinstance(c.get("back"), str) and c["back"].strip()
            and not looks_like_json_blob(c["front"]) and not looks_like_json_blob(c["back"])
        ]
        shape_count = len(cards)
        cards = verifier.verify_flashcards(cards, grounding_text, concept_name)
        verified_count = len(cards)
        cards = [c for c in cards if not _is_duplicate(c["front"], seen_flashcard_fronts)]
        final_count = len(cards)
        logger.info(
            "[%s] flashcards: requested=%d raw=%d after_shape_filter=%d "
            "after_verification=%d after_dedup=%d",
            concept_name, requested_n, raw_count, shape_count, verified_count, final_count,
        )
        for c in cards:
            db.add_flashcard(concept_id, c["front"], c["back"])
            seen_flashcard_fronts.append(_normalize(c["front"]))

        requested_quiz_n = quiz_n(depth)
        questions = quiz_maker.make_quiz(concept_name, grounding_text, n=requested_quiz_n, style=style, depth=depth)
        raw_quiz_count = len(questions)
        questions = [
            q for q in questions
            if isinstance(q, dict) and q.get("question") and isinstance(q.get("options"), list)
            and isinstance(q.get("correct_index"), int) and 0 <= q["correct_index"] < len(q["options"])
        ]
        shape_quiz_count = len(questions)
        questions = verifier.verify_quiz_questions(questions, grounding_text, concept_name)
        logger.info(
            "[%s] quiz: requested=%d raw=%d after_shape_filter=%d after_verification=%d",
            concept_name, requested_quiz_n, raw_quiz_count, shape_quiz_count, len(questions),
        )
        for q in questions:
            db.add_quiz_question(
                concept_id, q["question"], q["options"], q["correct_index"],
                q.get("distractor_notes", {})
            )

        db.update_concept_status(concept_id, "unseen")
        report(f"Completed: {concept_name}")

    report("Done.")
    return doc_id


def generate_from_pyq(title: str, text_notes: str | None = None,
                       images: list[tuple[bytes, str]] | None = None,
                       progress_cb=None, style: str = DEFAULT_STYLE,
                       depth: str = DEFAULT_DEPTH) -> int:
    """For an uploaded question paper -- as a text-based file (PDF/DOCX/PPTX/TXT,
    already extracted to plain text) and/or photographed/scanned image(s): reads
    the questions out, builds study material (concepts/summary/flashcards/quiz)
    around the topics those questions test, and produces a model answer +
    expand-marks hints for each individual question. Returns the document id."""
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    report("Reading the question paper...")
    questions = []
    if text_notes and text_notes.strip():
        questions.extend(pyq_solver.extract_questions_from_text(text_notes))
    if images:
        questions.extend(pyq_solver.extract_questions_from_images(images))
    if not questions:
        raise ValueError("Couldn't read any questions from the uploaded file(s). Try a clearer photo or a different file.")

    # The extracted questions become the "notes" for the rest of the pipeline --
    # concept extraction, summaries, flashcards, and quiz all run exactly as they
    # would for uploaded notes, just grounded in the question text instead.
    notes_text = "\n".join(f"- {q['text']}" for q in questions)

    doc_id = generate_study_material(title, notes_text, progress_cb=progress_cb, style=style, depth=depth)

    report("Solving the question paper...")
    concepts = db.list_concepts(doc_id)
    concept_names = [c["name"] for c in concepts]
    solutions = pyq_solver.solve_questions(
        questions, concept_names, notes_text, progress_cb=report, style=style, depth=depth
    )
    for s in solutions:
        if not isinstance(s, dict) or not s.get("question"):
            continue
        db.add_pyq_solution(
            doc_id,
            s.get("question", ""),
            s.get("matched_concept", ""),
            s.get("model_answer", ""),
            s.get("expand_hints", []) if isinstance(s.get("expand_hints"), list) else [],
        )

    report("Done.")
    return doc_id
