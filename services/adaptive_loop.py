"""The adaptive feedback loop: when a concept is flagged weak, re-explain it,
generate drill flashcards, reschedule review, and produce a follow-up quiz.
Capped at MAX_ATTEMPTS per concept -- after that, checks for a prerequisite
gap once, then honestly flags the concept for human follow-up instead of
looping forever.
"""
import json
from services import db
from services.pipeline import _is_duplicate, _normalize
from agents import summarizer, flashcard_maker, quiz_maker, verifier
from agents.style_rules import flashcard_n, quiz_n
from services.json_utils import safe_get, looks_like_json_blob
from services.logging_setup import get_logger

logger = get_logger(__name__)

MAX_ATTEMPTS = 3

# A concept is "weak" for one completed quiz attempt when BOTH hold: a flat
# wrong count alone doesn't scale (2/3 wrong is a real problem, 2/10 isn't),
# but a bare percentage breaks down at low question counts (1 wrong out of 1
# would be "100% wrong"), so we require a minimum wrong count too.
WEAK_PERCENTAGE_THRESHOLD = 0.5
WEAK_MIN_WRONG = 2


def is_weak(wrong: int, answered: int) -> bool:
    if answered == 0 or wrong < WEAK_MIN_WRONG:
        return False
    return (wrong / answered) >= WEAK_PERCENTAGE_THRESHOLD

PREREQUISITE_PROMPT = """The student has struggled to understand "{concept}" after
{attempts} attempts. Based on the notes below, is there a simpler, more foundational
concept in these notes that this idea depends on? If yes, name it briefly. If no
clear prerequisite is present in the notes, say so.

Return JSON: {{"has_prerequisite": true/false, "prerequisite": "name or empty string"}}

NOTES:
---
{notes}
---
"""


def run_remediation(document_id: int, concept_id: int, notes: str,
                     misconceptions: list[str] | None = None) -> dict:
    """Runs one round of the adaptive loop for a weak concept. Returns a report
    dict describing what was done, for display in the UI. misconceptions: the
    specific wrong-answer reasons from the quiz attempt that triggered this
    remediation, if available -- lets the re-explanation target the actual
    gap instead of guessing at a generic "different angle"."""
    concept = db.get_concept(concept_id)
    document = db.get_document(document_id)
    style = document["explanation_style"]
    depth = document["depth_level"]
    attempts = concept["attempts"]

    if attempts >= MAX_ATTEMPTS:
        return _handle_cap_reached(concept, notes)

    # 1. Re-explain -- stored as a SEPARATE remediation entry, never overwriting
    # the original summary, so both stay visible and distinctly labeled.
    prior_remediations = db.list_concept_remediations(concept_id)
    previous_summary = (
        json.loads(prior_remediations[-1]["summary"]) if prior_remediations
        else db.parse_summary(concept["summary"])
    )
    new_summary = summarizer.reexplain_concept(
        concept["name"], notes, previous_summary, attempts + 1, style=style, depth=depth,
        misconceptions=misconceptions,
    )
    db.add_concept_remediation(concept_id, attempts + 1, json.dumps(new_summary))

    # Same reasoning as pipeline.py: ground drill flashcards/follow-up quiz in
    # the notes PLUS this attempt's own explanation, not the raw notes alone --
    # matters most for a manually-added concept the notes never covered.
    grounding_text = f"{notes}\n\n{new_summary.get('explained_further', '')}"

    # 2. Drill flashcards
    all_existing = db.list_flashcards(document_id)
    existing = [c for c in all_existing if c["concept_id"] == concept_id]
    requested_n = flashcard_n(depth)
    drill_cards = flashcard_maker.make_drill_flashcards(
        concept["name"], grounding_text, existing, n=requested_n, style=style, depth=depth
    )
    raw_count = len(drill_cards)
    drill_cards = [
        c for c in drill_cards
        if isinstance(c, dict) and isinstance(c.get("front"), str) and c["front"].strip()
        and isinstance(c.get("back"), str) and c["back"].strip()
        and not looks_like_json_blob(c["front"]) and not looks_like_json_blob(c["back"])
    ]
    shape_count = len(drill_cards)
    drill_cards = verifier.verify_flashcards(drill_cards, grounding_text, concept["name"])
    verified_count = len(drill_cards)
    seen_fronts = [_normalize(c["front"]) for c in all_existing]
    drill_cards = [c for c in drill_cards if not _is_duplicate(c["front"], seen_fronts)]
    logger.info(
        "[%s] drill flashcards (attempt %d): requested=%d raw=%d after_shape_filter=%d "
        "after_verification=%d after_dedup=%d",
        concept["name"], attempts + 1, requested_n, raw_count, shape_count, verified_count, len(drill_cards),
    )
    for c in drill_cards:
        db.add_flashcard(concept_id, c["front"], c["back"], source="drill", attempt=attempts + 1)

    # 3. Reschedule (push this concept's cards back to box 1 / review soon)
    db.reschedule_flashcards_for_concept(concept_id, correct=False)

    # 4. Follow-up quiz
    followup_qs = quiz_maker.make_followup_quiz(
        concept["name"], grounding_text, n=quiz_n(depth), style=style, depth=depth
    )
    followup_qs = [
        q for q in followup_qs
        if isinstance(q, dict) and q.get("question") and isinstance(q.get("options"), list)
        and isinstance(q.get("correct_index"), int) and 0 <= q["correct_index"] < len(q["options"])
    ]
    followup_qs = verifier.verify_quiz_questions(followup_qs, grounding_text, concept["name"])
    for q in followup_qs:
        db.add_quiz_question(
            concept_id, q["question"], q["options"], q["correct_index"], q.get("distractor_notes", {})
        )

    db.update_concept_status(concept_id, "learning", increment_attempt=True)

    return {
        "action": "remediation",
        "attempt": attempts + 1,
        "new_summary": new_summary["explained_further"] or new_summary["from_notes"],
        "drill_cards_added": len(drill_cards),
        "followup_questions_added": len(followup_qs),
    }


def _handle_cap_reached(concept: dict, notes: str) -> dict:
    from services.gemini_client import generate_json

    prompt = PREREQUISITE_PROMPT.format(
        concept=concept["name"], attempts=concept["attempts"], notes=notes[:20000]
    )
    result = generate_json(prompt)

    if safe_get(result, "has_prerequisite", False) and not concept.get("_prereq_tried"):
        db.update_concept_status(concept["id"], "flagged")
        return {
            "action": "prerequisite_suggested",
            "prerequisite": safe_get(result, "prerequisite", ""),
            "message": (
                f"Still stuck on \"{concept['name']}\" after {concept['attempts']} attempts. "
                f"This may depend on understanding \"{safe_get(result, 'prerequisite', '')}\" first."
            ),
        }

    db.update_concept_status(concept["id"], "flagged")
    return {
        "action": "handoff",
        "message": (
            f"You've attempted \"{concept['name']}\" {concept['attempts']} times. "
            "Further AI-generated explanations may not help at this point -- "
            "consider a different resource (your instructor, textbook, or a study group)."
        ),
    }


def clear_concept_if_correct(concept_id: int):
    db.update_concept_status(concept_id, "cleared")
    db.reschedule_flashcards_for_concept(concept_id, correct=True)
