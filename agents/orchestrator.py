"""Orchestrator: extracts a shared concept vocabulary from the notes and merges
in any concepts the student explicitly asks to include, so all downstream
agents (Summarizer, Flashcard-maker, Quiz-maker) tag content consistently.
Concept granularity scales with the chosen depth -- Overview merges related
ideas into fewer, broader concepts; Deep dive splits them into more, granular
ones -- so the document's overall shape visibly reflects the depth setting."""
from services.gemini_client import generate_json
from services.json_utils import safe_get_list, safe_get, safe_get_str
from agents.style_rules import concept_granularity_text, DEFAULT_DEPTH

RELEVANCE_PROMPT = """You are checking whether the text below is genuinely academic
study material -- notes, textbook content, slides, or a question paper -- suitable
for generating summaries, flashcards, and a quiz from.

Return JSON: {{"is_study_material": true/false, "reason": "one short sentence explaining
why, e.g. what the text actually looks like instead if it's rejected"}}

Reject text that is not real study material -- for example a menu, a receipt, an
unrelated document, song lyrics, a personal letter, random or meaningless text, or
anything else that isn't coursework content. Accept it even if it's sparse, informal,
or only partially complete -- the bar is "is this genuinely study material," not
"is this well-written."

TEXT:
---
{text}
---
"""


def check_is_study_material(text: str) -> tuple[bool, str]:
    """A cheap, single extra AI call that runs right after text extraction and
    before the real pipeline starts -- catches uploads that have real, readable
    text but aren't actually study material (a menu, an unrelated document, a
    photo of something with no coursework content). This is a judgment call by
    a second model, not a hard rule, so it can occasionally be wrong -- same
    caveat as the Verifier agent (Section 6). Fails open: if the check itself
    errors out, the upload is allowed through rather than blocked, consistent
    with how Verifier failures are handled elsewhere in this app."""
    try:
        result = generate_json(RELEVANCE_PROMPT.format(text=text[:8000]))
    except Exception:
        return True, ""
    is_material = safe_get(result, "is_study_material", True)
    reason = safe_get_str(result, "reason")
    return bool(is_material), reason


EXTRACTION_PROMPT = """You are analyzing a student's study notes to identify the distinct
concepts covered, for building study material (summaries, flashcards, quiz questions).

Return JSON: {{"concepts": ["Concept name 1", "Concept name 2", ...]}}

Rules:
- {granularity}
- Each concept should be a specific, teachable idea (e.g. "Price elasticity of demand"),
  not a whole chapter title (e.g. "Chapter 3").
- Use short, clear names (3-6 words).
- Do not invent concepts not present in the notes.

NOTES:
---
{notes}
---
"""


def extract_concepts(notes: str, extra_concepts: list[str] | None = None,
                      depth: str = DEFAULT_DEPTH) -> list[str]:
    prompt = EXTRACTION_PROMPT.format(
        notes=notes[:20000], granularity=concept_granularity_text(depth)
    )
    result = generate_json(prompt)
    concepts = [c.strip() for c in safe_get_list(result, "concepts") if isinstance(c, str) and c.strip()]

    if extra_concepts:
        existing_lower = {c.lower() for c in concepts}
        for extra in extra_concepts:
            extra = extra.strip()
            if extra and extra.lower() not in existing_lower:
                concepts.append(extra)
                existing_lower.add(extra.lower())

    return concepts
