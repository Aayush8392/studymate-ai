"""Orchestrator: extracts a shared concept vocabulary from the notes and merges
in any concepts the student explicitly asks to include, so all downstream
agents (Summarizer, Flashcard-maker, Quiz-maker) tag content consistently.
Concept granularity scales with the chosen depth -- Overview merges related
ideas into fewer, broader concepts; Deep dive splits them into more, granular
ones -- so the document's overall shape visibly reflects the depth setting."""
from services.gemini_client import generate_json
from services.json_utils import safe_get_list
from agents.style_rules import concept_granularity_text, DEFAULT_DEPTH

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
