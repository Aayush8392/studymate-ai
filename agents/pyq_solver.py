"""PYQ (previous year question paper) solver: reads a question paper -- either
photographed/scanned via Gemini's vision input, or uploaded as a text-based
file (PDF/DOCX/PPTX/TXT, already extracted to plain text by note_parser) --
extracts the individual questions, and produces model answers in the same
exam-answer format used elsewhere: a concise model answer plus hints on what
to add for extra marks."""
from services.gemini_client import generate_json
from services.json_utils import safe_get_list, safe_get, safe_get_str
from agents.style_rules import (
    style_only_block, pyq_answer_length, pyq_needs_thinking, DEFAULT_STYLE, DEFAULT_DEPTH,
)

_EXTRACT_INSTRUCTIONS = """Extract each distinct question as its own entry. Ignore headers and
instructions -- only the actual questions. If the same question appears with
sub-parts (a), (b), (c), extract each sub-part as its own entry with the
sub-part text included. Also capture the marks allocation for each question if
one is given (e.g. "(10 marks)", "[5]", "10M") -- report it as a number and
strip it out of the question text itself; if no marks are stated for a
question, use null."""

# Used as-is (no .format() call, since vision prompts pass through generate_json
# unformatted) -- single braces in the JSON example.
EXTRACT_PROMPT = f"""You are reading photo(s) of a student's exam question paper
(previous year questions / PYQ). Read all the text in the image(s) carefully,
including handwriting if present.

Return JSON:
{{
  "questions": [
    {{"text": "question 1 text, exactly as written (marks notation removed)", "marks": 10}},
    {{"text": "question 2 text", "marks": null}}
  ]
}}

{_EXTRACT_INSTRUCTIONS}
"""

# Formatted with .format(text=...) before use -- so the JSON example needs
# doubled braces to survive that call and come out as single braces.
EXTRACT_TEXT_PROMPT = f"""You are reading the text of a student's exam question
paper (previous year questions / PYQ), extracted from an uploaded document.

Return JSON:
{{{{
  "questions": [
    {{{{"text": "question 1 text, exactly as written (marks notation removed)", "marks": 10}}}},
    {{{{"text": "question 2 text", "marks": null}}}}
  ]
}}}}

{_EXTRACT_INSTRUCTIONS}

DOCUMENT TEXT:
---
{{text}}
---
"""

SOLVE_PROMPT = """You are helping a student prepare model answers for the exam
questions below, based on their own study notes/concepts. For EACH question,
write a model answer the way a strong student should actually write it in the
exam -- correctly structured, using proper terminology, and long/thorough
enough to earn full marks on the specific mark weight given for it. Also
identify which of the student's known concepts (if any) this question relates
to, and give hints on what to add for extra marks.

{style_depth_rules}
Each question below is tagged with exactly how long/thorough its answer must
be -- follow that tag's rule EXACTLY for that question, don't default to the
same length for every question regardless of its tag. A short-answer tag
means stay concise; a long-answer tag means write a genuinely thorough,
multi-paragraph, exam-ready answer -- don't under-answer a question worth
many marks just because a shorter answer would be easier to write.

Return JSON:
{{
  "solutions": [
    {{
      "question": "the question text, exactly as given",
      "matched_concept": "name of the closest matching concept from the list below,
        or empty string if none match",
      "model_answer": "the model answer text -- follow that question's length/
        structure tag exactly",
      "expand_hints": ["2-4 short bullets on what to add for extra marks"]
    }}
  ]
}}

QUESTIONS (each tagged with its required answer length):
{questions}

KNOWN CONCEPTS FOR THIS DOCUMENT:
{concepts}

STUDY NOTES (for grounding the answers):
---
{notes}
---
"""

# Used ONLY for a single high-marks/thinking-enabled question (see
# _group_into_chunks -- those are always solved solo, never batched). Verified
# live that burying the length/depth spec inline inside a numbered question
# list (as SOLVE_PROMPT does, needed there to support several DIFFERENTLY
# short-tagged questions per call) visibly dilutes how much weight the model
# gives it -- the exact same spec as its own prominent up-front paragraph,
# immediately before the JSON schema, reliably produced 2x+ the length in
# side-by-side testing. Since a solo question has only one length rule to
# follow anyway, there's no batching reason not to give it this treatment.
SOLVE_PROMPT_SOLO = """You are helping a student prepare a model answer for the exam
question below, based on their own study notes/concepts. Write the model
answer the way a strong student should actually write it in the exam --
correctly structured, using proper terminology. Also identify which of the
student's known concepts (if any) this question relates to, and give hints on
what to add for extra marks.

{style_depth_rules}
The answer must be: {answer_length}

Return JSON:
{{
  "solutions": [
    {{
      "question": "the question text, exactly as given",
      "matched_concept": "name of the closest matching concept from the list below,
        or empty string if none match",
      "model_answer": "the model answer text -- follow the length/depth rule above exactly",
      "expand_hints": ["2-4 short bullets on what to add for extra marks"]
    }}
  ]
}}

QUESTION:
{question}

KNOWN CONCEPTS FOR THIS DOCUMENT:
{concepts}

STUDY NOTES (for grounding the answer):
---
{notes}
---
"""


def _parse_questions(raw_items: list) -> list[dict]:
    parsed = []
    for item in raw_items:
        if isinstance(item, str):
            text = item.strip()
            marks = None
        elif isinstance(item, dict):
            text = safe_get_str(item, "text")
            marks_raw = safe_get(item, "marks", None)
            try:
                marks = int(marks_raw) if marks_raw is not None else None
            except (TypeError, ValueError):
                marks = None
        else:
            continue
        if text:
            parsed.append({"text": text, "marks": marks})
    return parsed


def extract_questions_from_images(images: list[tuple[bytes, str]]) -> list[dict]:
    """images: list of (image_bytes, mime_type). Returns [{"text": ..., "marks": int|None}]."""
    result = generate_json(EXTRACT_PROMPT, images=images)
    return _parse_questions(safe_get_list(result, "questions"))


def extract_questions_from_text(text: str) -> list[dict]:
    """text: already-extracted plain text from a PDF/DOCX/PPTX/TXT question paper.
    Returns [{"text": ..., "marks": int|None}]."""
    prompt = EXTRACT_TEXT_PROMPT.format(text=text[:20000])
    result = generate_json(prompt)
    return _parse_questions(safe_get_list(result, "questions"))


_CHUNK_SIZE = 4  # keep each non-thinking call's output small enough to avoid truncated JSON
_THINKING_BUDGET = 8192


def _group_into_chunks(questions: list[dict], depth: str) -> list[list[dict]]:
    """Groups consecutive questions into batches of up to _CHUNK_SIZE, EXCEPT
    a question that needs thinking-enabled treatment always gets its own
    solo chunk -- a high-marks answer alone can run to 600+ words, so batching
    several of them into one call risks the combined response being cut off
    mid-JSON."""
    chunks = []
    current = []
    for q in questions:
        if pyq_needs_thinking(q["marks"], depth):
            if current:
                chunks.append(current)
                current = []
            chunks.append([q])
        else:
            current.append(q)
            if len(current) == _CHUNK_SIZE:
                chunks.append(current)
                current = []
    if current:
        chunks.append(current)
    return chunks


def solve_questions(questions: list[dict], concepts: list[str], notes: str,
                     progress_cb=None, style: str = DEFAULT_STYLE,
                     depth: str = DEFAULT_DEPTH) -> list[dict]:
    """questions: [{"text": ..., "marks": int|None}]. Solves in small chunks
    rather than one giant call -- a single call asked to write full model
    answers for a whole paper risks the response being cut off mid-JSON
    (breaking parsing) and produces rushed, lower-quality answers. Any
    question substantial enough to need thinking-enabled generation is solved
    on its own (see _group_into_chunks)."""
    if not questions:
        return []

    chunks = _group_into_chunks(questions, depth)

    all_solutions = []
    solved_so_far = 0
    for chunk in chunks:
        if progress_cb:
            progress_cb(f"Solving question(s) {solved_so_far + 1}-{solved_so_far + len(chunk)} of {len(questions)}...")
        concepts_text = ", ".join(concepts) if concepts else "(none extracted)"

        if len(chunk) == 1 and pyq_needs_thinking(chunk[0]["marks"], depth):
            prompt = SOLVE_PROMPT_SOLO.format(
                question=chunk[0]["text"],
                answer_length=pyq_answer_length(chunk[0]["marks"], depth),
                concepts=concepts_text,
                notes=notes[:20000],
                style_depth_rules=style_only_block(style),
            )
            result = generate_json(prompt, thinking_budget=_THINKING_BUDGET)
        else:
            lines = [
                f"{i+1}. [{q['marks']} marks -- required answer length: {pyq_answer_length(q['marks'], depth)}] {q['text']}"
                if q["marks"] is not None else
                f"{i+1}. [marks not stated -- required answer length: {pyq_answer_length(q['marks'], depth)}] {q['text']}"
                for i, q in enumerate(chunk)
            ]
            prompt = SOLVE_PROMPT.format(
                questions="\n".join(lines),
                concepts=concepts_text,
                notes=notes[:20000],
                style_depth_rules=style_only_block(style),
            )
            result = generate_json(prompt)

        chunk_solutions = safe_get_list(result, "solutions")
        # Match solutions to questions by POSITION, not by trusting the
        # model's echoed "question" field -- verified live that the model can
        # echo back the whole tagged line (numbering + our [marks -- required
        # answer length: ...] prefix included) instead of just the clean
        # question text, which would otherwise pollute what gets stored and
        # shown to the student. We already know each chunk's real question
        # text and its order, so overwrite rather than trust the echo.
        for i, q in enumerate(chunk):
            if i < len(chunk_solutions) and isinstance(chunk_solutions[i], dict):
                chunk_solutions[i]["question"] = q["text"]
        all_solutions.extend(chunk_solutions)
        solved_so_far += len(chunk)

    return all_solutions


# A PYQ answer can be expanded further on demand (student clicks "Expand
# answer" to get more material out of the current hints). The word-count
# ceiling this is checked against lives in app.py alongside the button --
# this function only ever produces ONE round's worth of addition; whether
# another round is offered afterward is a UI-level decision based on the
# resulting length, not something this function decides for itself.
EXPAND_PROMPT = """A student is expanding their exam model answer to get more study material
out of it (this is a study aid, not a literal exam submission -- more real
content is always welcome here, there is no such thing as "too long" for this
purpose). Do not repeat what the CURRENT ANSWER already covers.

{style_depth_rules}
For EACH of the hints below, write a genuinely developed addition covering:
what the hint is pointing at (a real explanation, not just naming it), how it
specifically applies to this question, and a concrete example or elaboration.
Do not compress a hint into one or two sentences -- give it the same level of
depth as the rest of the answer.

Write the WHOLE addition as one connected piece of writing, not one
self-contained mini-paragraph per hint stapled after another. Read the
CURRENT ANSWER below first and continue in the same voice, as if you were
still mid-answer -- do not restart with a generic opener like "Next," or
"Another point is." When you move from covering one hint to the next, connect
them with real reasoning (how the new point relates to, builds on, or
contrasts with what was just said), not a list-style transition.

Also produce a FRESH set of 2-4 "to go further" hints for what's still
missing after this addition -- do not repeat any of the hints already used.

Return JSON:
{{
  "addition": "the new text only -- this will be appended after the current answer, do not repeat the current answer",
  "new_hints": ["2-4 short bullets on what to add next"]
}}

QUESTION:
{question}

RELATED CONCEPT: {matched_concept}

CURRENT ANSWER:
---
{current_answer}
---

HINTS TO EXPAND ON:
{hints}

STUDY NOTES (for grounding):
---
{notes}
---
"""


def expand_answer(question: str, matched_concept: str, current_answer: str,
                   hints: list[str], notes: str, style: str = DEFAULT_STYLE) -> dict:
    """Returns {"addition": str, "new_hints": [str, ...]}."""
    prompt = EXPAND_PROMPT.format(
        question=question,
        matched_concept=matched_concept or "(none matched)",
        current_answer=current_answer,
        hints="\n".join(f"- {h}" for h in hints),
        notes=notes[:20000],
        style_depth_rules=style_only_block(style),
    )
    result = generate_json(prompt, thinking_budget=_THINKING_BUDGET)
    addition = safe_get_str(result, "addition")
    new_hints = [h.strip() for h in safe_get_list(result, "new_hints") if isinstance(h, str) and h.strip()]
    return {"addition": addition, "new_hints": new_hints}
