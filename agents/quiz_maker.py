"""Quiz-maker agent: generates MCQs per concept with DIAGNOSTIC distractors —
each wrong option is designed to represent a specific plausible misconception,
so which wrong answer a student picks reveals what they actually misunderstand."""
import random
from services.gemini_client import generate_json
from services.json_utils import safe_get_list, looks_like_json_blob
from agents.style_rules import style_only_block, DEFAULT_STYLE, DEFAULT_DEPTH

QUIZ_PROMPT = """Create {n} multiple-choice questions for the concept "{concept}",
based only on the notes below. Each question has exactly 4 options.

{style_depth_rules}
(Apply this to question wording and difficulty -- Overview depth means testing
the core fact directly, Deep dive means questions can probe nuance/comparisons.)

IMPORTANT: design the 3 wrong options (distractors) deliberately — each one should
represent a specific, plausible misconception a student might have (e.g. confusing
two similar terms, a common reasoning error), not just a random wrong fact. Note
what misconception each distractor represents.

Return JSON:
{{"questions": [
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_index": 0,
    "distractor_notes": {{"1": "misconception this option represents", "2": "...", "3": "..."}}
  }}
]}}

NOTES:
---
{notes}
---
"""

FOLLOWUP_PROMPT = """The student previously got questions wrong on the concept "{concept}".
Create {n} NEW multiple-choice questions on just this concept, with fresh wording and
fresh diagnostic distractors (different from before), based only on the notes below.

{style_depth_rules}

Return JSON in the same format as before:
{{"questions": [
  {{"question": "...", "options": ["A","B","C","D"], "correct_index": 0,
    "distractor_notes": {{"1": "...", "2": "...", "3": "..."}}}}
]}}

NOTES:
---
{notes}
---
"""


def _is_valid_question(question) -> bool:
    """Guards every assumption the rest of this module and downstream code
    makes about a question's shape, before anything touches it -- the model
    occasionally deviates (wrong types, out-of-range correct_index, duplicate
    option text that would make a question unanswerable)."""
    if not isinstance(question, dict):
        return False
    if not isinstance(question.get("question"), str) or not question["question"].strip():
        return False
    if looks_like_json_blob(question["question"]):
        return False
    options = question.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return False
    if not all(isinstance(o, str) and o.strip() and not looks_like_json_blob(o) for o in options):
        return False
    if len(set(options)) != len(options):  # duplicate option text -> unanswerable/ambiguous
        return False
    correct_index = question.get("correct_index")
    if not isinstance(correct_index, int) or not (0 <= correct_index < len(options)):
        return False
    return True


def _shuffle_options(question: dict) -> dict:
    """The model tends to put the correct answer in the same position (usually
    first) across most questions -- shuffle so answer position carries no signal,
    remapping correct_index and the distractor_notes keys to match. Caller must
    have already validated the question with _is_valid_question."""
    options = question["options"]
    correct_index = question["correct_index"]
    distractor_notes = question.get("distractor_notes")
    if not isinstance(distractor_notes, dict):
        distractor_notes = {}

    order = list(range(len(options)))
    random.shuffle(order)

    new_options = [options[i] for i in order]
    new_correct_index = order.index(correct_index)
    new_distractor_notes = {}
    for new_pos, old_pos in enumerate(order):
        if new_pos == new_correct_index:
            continue
        note = distractor_notes.get(str(old_pos))
        if isinstance(note, str) and note.strip():
            new_distractor_notes[str(new_pos)] = note

    return {
        "question": question["question"].strip(),
        "options": new_options,
        "correct_index": new_correct_index,
        "distractor_notes": new_distractor_notes,
    }


def make_quiz(concept: str, notes: str, n: int = 3, style: str = DEFAULT_STYLE,
              depth: str = DEFAULT_DEPTH) -> list[dict]:
    prompt = QUIZ_PROMPT.format(
        concept=concept, notes=notes[:20000], n=n,
        style_depth_rules=style_only_block(style),
    )
    result = generate_json(prompt)
    questions = [q for q in safe_get_list(result, "questions") if _is_valid_question(q)]
    return [_shuffle_options(q) for q in questions]


def make_followup_quiz(concept: str, notes: str, n: int = 3, style: str = DEFAULT_STYLE,
                        depth: str = DEFAULT_DEPTH) -> list[dict]:
    prompt = FOLLOWUP_PROMPT.format(
        concept=concept, notes=notes[:20000], n=n,
        style_depth_rules=style_only_block(style),
    )
    result = generate_json(prompt)
    questions = [q for q in safe_get_list(result, "questions") if _is_valid_question(q)]
    return [_shuffle_options(q) for q in questions]
