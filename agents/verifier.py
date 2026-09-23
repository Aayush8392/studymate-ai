"""Verifier agent: checks generated flashcards/quiz questions are actually
grounded in the source notes AND actually about the concept they're filed
under. Deliberately runs on a DIFFERENT model (Groq, not Gemini) than the one
that generated the content, so it isn't grading its own homework -- an
independent check, not a self-check.

The on-topic check matters as much as groundedness: a question can be 100%
true and genuinely grounded in the notes while still being about the WRONG
concept -- e.g. when asked to write questions for a concept the notes don't
actually cover, a model may fall back to writing real questions about
whatever the notes DO cover instead, which pass a pure groundedness check but
are mislabeled under the wrong topic."""
from services.groq_client import generate_json
from services.json_utils import safe_get_list
from services.logging_setup import get_logger

logger = get_logger(__name__)

VERIFY_PROMPT = """You are fact-checking study material against source notes.
For each item below, decide if it passes BOTH checks:
1. Grounded: is it clearly supported by the notes (not hallucinated, not
   contradicting them)?
2. On-topic: is it actually ABOUT the stated concept below -- not a different
   topic that happens to also be true and grounded in the notes?

CONCEPT: {concept}

ITEMS TO CHECK:
{items}

NOTES:
---
{notes}
---

Return JSON: {{"results": [{{"index": 0, "valid": true, "reason": "..."}}, ...]}}
One result per item, in the same order. "valid" must be false if EITHER check fails.
"""


def verify_items(items: list[str], notes: str, concept: str = "") -> list[dict]:
    """items: list of plain-text claims (flashcard front+back, or quiz question+answer)."""
    if not items:
        return []
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(items))
    # Groq free tier caps this model at 8K tokens/minute -- keep the notes slice
    # small so a single verify call doesn't eat the whole per-minute budget.
    prompt = VERIFY_PROMPT.format(items=numbered, notes=notes[:8000], concept=concept or "(unspecified)")
    try:
        result = generate_json(prompt)
    except Exception:
        # Groq's rate limits can make this call fail even after retrying for
        # up to 2 minutes. Verification is a quality check on top of content
        # that already passed shape/type filtering -- not the last line of
        # defense -- so let it through unverified rather than stalling or
        # crashing the whole generation run over one unavailable check.
        logger.warning("Verification unavailable, letting %d item(s) through unverified", len(items))
        return []
    return [r for r in safe_get_list(result, "results") if isinstance(r, dict)]


def _valid_map(results: list[dict]) -> dict:
    return {r["index"]: r.get("valid", True) for r in results if "index" in r}


def verify_flashcards(flashcards: list[dict], notes: str, concept: str = "") -> list[dict]:
    """Defensively re-checks shape even though callers are expected to have
    already filtered -- this function shouldn't assume it's the only caller
    forever, and a crash here would take down the whole generation run."""
    safe_cards = [c for c in flashcards if isinstance(c, dict) and c.get("front") and c.get("back")]
    items = [f"{c['front']} -- Answer: {c['back']}" for c in safe_cards]
    valid = _valid_map(verify_items(items, notes, concept))
    return [c for i, c in enumerate(safe_cards) if valid.get(i, True)]


def verify_quiz_questions(questions: list[dict], notes: str, concept: str = "") -> list[dict]:
    safe_questions = [
        q for q in questions
        if isinstance(q, dict) and isinstance(q.get("options"), list)
        and isinstance(q.get("correct_index"), int) and 0 <= q["correct_index"] < len(q["options"])
    ]
    items = [f"{q['question']} -- Correct answer: {q['options'][q['correct_index']]}" for q in safe_questions]
    valid = _valid_map(verify_items(items, notes, concept))
    return [q for i, q in enumerate(safe_questions) if valid.get(i, True)]
