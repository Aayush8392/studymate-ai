"""Flashcard-maker agent: generates flashcards per concept, and small targeted
drill sets for concepts flagged weak by the adaptive loop."""
from services.gemini_client import generate_json
from services.json_utils import safe_get_list
from agents.style_rules import style_only_block, DEFAULT_STYLE, DEFAULT_DEPTH

FLASHCARD_PROMPT = """Create {n} flashcards for the concept "{concept}", based only
on the notes below. Each flashcard should test one specific fact or idea.

{style_depth_rules}
(Apply this to how the question and answer are worded and how substantial the
tested fact is -- Overview depth means simpler recall facts, Deep dive means
flashcards can test nuance/comparisons too.)

Return JSON: {{"flashcards": [{{"front": "question/prompt", "back": "answer"}}, ...]}}

NOTES:
---
{notes}
---
"""

DRILL_PROMPT = """The student is struggling with the concept "{concept}". Create {n}
NEW flashcards drilling this concept from different angles than these existing cards
(don't repeat their exact wording), based only on the notes below.

{style_depth_rules}

Existing cards:
{existing}

Return JSON: {{"flashcards": [{{"front": "question/prompt", "back": "answer"}}, ...]}}

NOTES:
---
{notes}
---
"""


def make_flashcards(concept: str, notes: str, n: int = 5, style: str = DEFAULT_STYLE,
                     depth: str = DEFAULT_DEPTH) -> list[dict]:
    prompt = FLASHCARD_PROMPT.format(
        concept=concept, notes=notes[:20000], n=n,
        style_depth_rules=style_only_block(style),
    )
    result = generate_json(prompt)
    return safe_get_list(result, "flashcards")


def make_drill_flashcards(concept: str, notes: str, existing_cards: list[dict], n: int = 4,
                           style: str = DEFAULT_STYLE, depth: str = DEFAULT_DEPTH) -> list[dict]:
    existing_str = "\n".join(f"- {c['front']}" for c in existing_cards) or "(none yet)"
    prompt = DRILL_PROMPT.format(
        concept=concept, notes=notes[:20000], existing=existing_str, n=n,
        style_depth_rules=style_only_block(style),
    )
    result = generate_json(prompt)
    return safe_get_list(result, "flashcards")
