"""Summarizer agent: produces a per-concept explanation, split into what's
faithfully drawn from the student's own notes versus supplementary general
knowledge that fills gaps -- so the two are never presented as the same thing.
Also produces an exam-answer model and pointers for going further, plus a
comparisons/limitations field that only appears at Deep dive depth. Can
re-explain a weak concept differently (new analogy / different angle) for
the adaptive loop, while staying at the same style/depth level.

Two things that used to be "ask nicely in the prompt and hope" are now
deterministic, checked, and corrected if they fail:
  1. Explanation depth (paragraph count) is a schema requirement -- separate
     named fields per depth (explanation_core/_detail/_example) -- not a
     word-count request inside one blob. If a required field comes back
     empty/too short, one corrective follow-up call fills in just that field.
  2. Key-terms count is checked against the (style, depth) minimum after
     generation. If short, one follow-up call extracts more terms from the
     model's OWN explanation text (which is never as sparse as the notes),
     rather than re-asking the same prompt and hoping for a different result.
"""
from services.gemini_client import generate_json
from services.json_utils import safe_get_str, safe_get_list, looks_like_json_blob
from agents.style_rules import (
    style_depth_block, DEFAULT_STYLE, DEFAULT_DEPTH,
    explanation_fields, explanation_fields_schema, key_terms_min, MIN_FIELD_CHARS,
    escalated_explanation_fields, escalated_explanation_fields_schema,
)

GROUNDING_SPLIT_RULES = """SOURCE-HONESTY RULES -- this is important, follow strictly:
- "from_notes": restate and simplify ONLY what the notes actually contain for this
  concept. If the notes are detailed here, extract the real content faithfully and
  explain it in simpler/shorter words -- don't invent extra facts. If the notes are
  sparse (just a term or short phrase), this field should also be short and honest
  -- do NOT pad it with invented explanation. It's fine for this to be brief.
- If the notes do NOT mention this concept AT ALL (this happens when a student
  manually adds an extra topic to study that isn't actually in their uploaded
  material), do not invent a plausible-sounding "from_notes" line pretending it
  came from there. Instead set "from_notes" to exactly: "Not covered in your
  uploaded notes -- this topic was added separately." The explanation_* fields
  below should still teach the real concept properly using general knowledge.
- The explanation_* fields below are where you may bring in your own general
  knowledge to properly teach the concept beyond what the bare notes say -- fill
  gaps, add context, explain the "why" and "how" the notes don't spell out. Do
  not pretend this came from the notes.
- Never blend from_notes with the explanation_* fields -- a reader should be able
  to tell which parts are strictly from their own material and which parts are
  you teaching beyond it.
"""

SUMMARY_PROMPT = """You are writing an in-depth study explanation for the concept
"{concept}", based on the student's own notes below.

{style_depth_rules}

{grounding_split_rules}

Return JSON with these fields:
{{
  "from_notes": "A faithful, simplified restatement of exactly what the notes say
    about this concept. Short if the notes are short, longer if the notes are
    detailed. Don't add facts the notes don't contain.",
  {explanation_fields_schema},
  "comparisons_and_limits": "See the DEPTH rules above for whether to fill this in
    or leave it empty -- when filled in, real nuance: comparisons to related
    concepts, edge cases, or limitations.",
  "analogy": "See the STYLE rules above for what kind of analogy/comparison to use
    here -- this field is ALWAYS filled in, never left empty.",
  "exam_answer_example": "A model answer written the way a student should actually
    write it in an exam paper for this concept -- proper structure, correct
    terminology. Follow the exact length rule given above.",
  "expand_hints": ["Bullet points on what a student could mention further to
    strengthen or extend this answer for extra marks. Follow the exact count rule
    given above."],
  "why_it_matters": "2-3 sentences on why this concept matters / where it's applied.",
  "key_terms": [{{"term": "a SHORT label, 1-4 words, e.g. 'Proactiveness' -- never a
    full sentence", "definition": "a separate, complete sentence explaining that term"}}]
}}
Every explanation_* field listed above is REQUIRED and must be substantial (not a
one-liner) -- do not leave any of them blank. Follow the exact count rule given
above for how many key_terms to include. Every key_terms entry needs BOTH a short
term and a non-empty definition -- never leave definition blank, and never put a
full sentence in the term field.

NOTES:
---
{notes}
---
"""

REEXPLAIN_PROMPT = """The student has struggled to understand the concept "{concept}"
after a previous explanation. Re-explain it in a genuinely DIFFERENT way this time --
a new analogy/comparison or a different angle of approach entirely (don't just reword
the same explanation). Base "from_notes" only on the notes below. This is attempt
number {attempt} for this concept. Keep following the SAME style rules as before
(below) -- the goal is a different angle, NOT a simpler or harder vocabulary level
than before.

{style_depth_rules}

IMPORTANT -- this attempt requires MORE explanation fields than the original attempt
did (listed below). This isn't optional depth-creep: the student already didn't
grasp the shorter version, so simply rewording the same single point again won't
help. Use the extra field(s) to genuinely teach more -- a concrete worked example,
an additional angle, or a nuance the shorter version had no room for -- not padding.

{misconception_block}
{grounding_split_rules}

Previous attempt (do not just repeat this):
---
{previous}
---

Return JSON with these fields:
{{
  "from_notes": "faithful, simplified restatement of what the notes say",
  {explanation_fields_schema},
  "comparisons_and_limits": "per the DEPTH rules above",
  "analogy": "a NEW analogy/comparison, different from any used before, per the STYLE rules above",
  "exam_answer_example": "a model exam answer",
  "expand_hints": ["per the DEPTH rules above"],
  "why_it_matters": "2-3 sentences",
  "key_terms": [{{"term": "a SHORT label, 1-4 words, never a full sentence", "definition": "a separate sentence"}}]
}}
Every explanation_* field is REQUIRED and must be substantial, using a genuinely
different approach than the previous attempt. Every key_terms entry needs BOTH a
short term and a non-empty definition.

NOTES:
---
{notes}
---
"""

FILL_MISSING_FIELDS_PROMPT = """You previously wrote a partial explanation for the
concept "{concept}" but left some required fields empty or too short. Fill in ONLY
the missing field(s) listed below -- do not repeat or restate content already
written in the other fields shown for context.

{style_depth_rules}

Already written (for context, so you don't repeat it):
---
{existing_text}
---

Missing field(s) to fill in now: {missing_fields}

Return JSON with ONLY those field(s), e.g. {{"explanation_detail": "..."}}. Each
must be substantial (not a one-liner) and genuinely additive to what's already written.

NOTES:
---
{notes}
---
"""

EXTRA_KEY_TERMS_PROMPT = """For the concept "{concept}", you need {needed} MORE
distinct key terms with definitions than you already have. Do not repeat any of
these existing terms: {existing_terms}

Draw the additional terms from the explanation text below (it's fine if a term
wasn't in the original source notes, as long as it's genuinely part of explaining
this concept).

EXPLANATION TEXT:
---
{source_text}
---

Return JSON: {{"key_terms": [{{"term": "a SHORT label, 1-4 words, never a full
sentence", "definition": "a separate, complete sentence"}}]}}
"""


def summarize_concept(concept: str, notes: str, style: str = DEFAULT_STYLE,
                       depth: str = DEFAULT_DEPTH) -> dict:
    prompt = SUMMARY_PROMPT.format(
        concept=concept, notes=notes[:20000],
        style_depth_rules=style_depth_block(style, depth),
        grounding_split_rules=GROUNDING_SPLIT_RULES,
        explanation_fields_schema=explanation_fields_schema(depth),
    )
    result = generate_json(prompt)
    return _normalize(result, concept, notes, style, depth, explanation_fields(depth))


EXPAND_PROMPT = """You are helping a student go deeper on the concept "{concept}",
building on an explanation they've already read.

{style_depth_rules}
For EACH of the hints below, write a genuinely developed addition covering:
what the hint is pointing at (a real explanation, not just naming it), how it
specifically applies to this concept, and a concrete example or elaboration.
Do not compress a hint into one or two sentences -- give it the same level of
depth as the rest of the explanation. The goal is to help the student truly
understand this concept better, not just to add more words.

Write the WHOLE addition as one connected piece of writing, not one
self-contained mini-paragraph per hint stapled after another. Read the
CURRENT EXPLANATION below first and continue in the same voice, as if you were
still mid-explanation -- do not restart with a generic opener like "Next," or
"Another point is." When you move from covering one hint to the next, connect
them with real reasoning (how the new point relates to, builds on, or
contrasts with what was just said), not a list-style transition.

Also produce a FRESH set of 2-4 "to go further" hints for what's still missing
after this addition -- do not repeat any of the hints already used, and do not
suggest a hint that's really just a reworded version of one already covered.
If there is genuinely nothing substantively new left to add about this concept,
return an empty list for new_hints instead of inventing a weak or repetitive one.

Return JSON:
{{
  "addition": "the new text only -- this will be appended after the current explanation, do not repeat it",
  "new_hints": ["2-4 short bullets on what to add next, or an empty list if nothing genuinely new is left"]
}}

CURRENT EXPLANATION:
---
{current_explanation}
---

HINTS TO EXPAND ON:
{hints}

STUDY NOTES (for grounding):
---
{notes}
---
"""


def expand_explanation(concept: str, notes: str, current_explanation: str,
                        hints: list[str], style: str = DEFAULT_STYLE,
                        depth: str = DEFAULT_DEPTH) -> dict:
    """Returns {"addition": str, "new_hints": [str, ...]}. Mirrors pyq_solver's
    expand_answer, but for a concept's own explanation rather than a PYQ model
    answer -- same reasoning, no equivalent "marks" ceiling exists for a
    concept, so the stopping decision is made by the caller (app.py), by
    comparing new_hints against every hint already shown so far."""
    prompt = EXPAND_PROMPT.format(
        concept=concept,
        style_depth_rules=style_depth_block(style, depth),
        current_explanation=current_explanation,
        hints="\n".join(f"- {h}" for h in hints),
        notes=notes[:20000],
    )
    result = generate_json(prompt)
    addition = _safe_text(result, "addition")
    new_hints = [
        h.strip() for h in safe_get_list(result, "new_hints")
        if isinstance(h, str) and h.strip() and not looks_like_json_blob(h)
    ]
    return {"addition": addition, "new_hints": new_hints}


def reexplain_concept(concept: str, notes: str, previous_summary: dict, attempt: int,
                       style: str = DEFAULT_STYLE, depth: str = DEFAULT_DEPTH,
                       misconceptions: list[str] | None = None) -> dict:
    """Escalates the explanation schema with each attempt (see
    escalated_explanation_fields) instead of reusing the original depth's
    schema that already failed to get the concept across. misconceptions,
    when given, are the specific wrong-answer reasons from the quiz attempt
    that triggered this remediation -- letting the model target the actual
    gap instead of guessing at a generic "different angle"."""
    if isinstance(previous_summary, dict):
        previous_text = previous_summary.get("explained_further") or previous_summary.get("explanation", "")
    else:
        previous_text = str(previous_summary)

    misconception_block = ""
    real_misconceptions = [m for m in (misconceptions or []) if m and m.strip()]
    if real_misconceptions:
        bullets = "\n".join(f"- {m}" for m in real_misconceptions)
        misconception_block = (
            "SPECIFIC GAPS TO TARGET -- the student's wrong quiz answers suggest these "
            f"exact misconceptions; address them directly rather than re-explaining "
            f"generically:\n{bullets}\n"
        )

    prompt = REEXPLAIN_PROMPT.format(
        concept=concept, notes=notes[:20000], previous=previous_text, attempt=attempt,
        style_depth_rules=style_depth_block(style, depth),
        grounding_split_rules=GROUNDING_SPLIT_RULES,
        explanation_fields_schema=escalated_explanation_fields_schema(depth, attempt),
        misconception_block=misconception_block,
    )
    result = generate_json(prompt)
    return _normalize(result, concept, notes, style, depth, escalated_explanation_fields(depth, attempt))


_MAX_TERM_WORDS = 6  # a real "term" is a short label; longer than this is almost
                      # certainly the model accidentally putting a definition here


def _clean_key_term(kt) -> dict | None:
    if not isinstance(kt, dict):
        return None
    term = kt.get("term")
    definition = kt.get("definition")
    if not isinstance(term, str) or not term.strip():
        return None
    term = term.strip()
    if len(term.split()) > _MAX_TERM_WORDS:
        return None  # looks like a sentence, not a label -- drop rather than mislabel
    if not isinstance(definition, str) or not definition.strip():
        return None  # a term with no real definition isn't useful, drop it
    return {"term": term, "definition": definition.strip()}


def _safe_text(result, key: str) -> str:
    """safe_get_str, plus rejecting a field that's actually a stringified JSON
    blob (the model occasionally confuses this schema with another and dumps
    a nested object into what should be a plain-text field)."""
    value = safe_get_str(result, key)
    return "" if looks_like_json_blob(value) else value


def _fill_missing_explanation_fields(field_values: dict, missing: list[str], concept: str,
                                      notes: str, style: str, depth: str) -> dict:
    existing_text = "\n\n".join(f"[{k}] {v}" for k, v in field_values.items() if v) or "(nothing written yet)"
    prompt = FILL_MISSING_FIELDS_PROMPT.format(
        concept=concept, notes=notes[:20000],
        style_depth_rules=style_depth_block(style, depth),
        existing_text=existing_text,
        missing_fields=", ".join(missing),
    )
    try:
        result = generate_json(prompt)
    except Exception:
        return field_values  # best-effort -- keep what we had rather than fail the whole concept
    for field in missing:
        value = _safe_text(result, field)
        if len(value) >= MIN_FIELD_CHARS:
            field_values[field] = value
    return field_values


def _build_explanation(result, concept: str, notes: str, style: str, depth: str, fields: list[str]) -> str:
    field_values = {f: _safe_text(result, f) for f in fields}

    # Legacy fallback: older documents/responses used one "explained_further"
    # field -- if the new fields are all empty but that one has content, use it.
    if not any(field_values.values()):
        legacy = _safe_text(result, "explained_further")
        if legacy:
            field_values[fields[0]] = legacy

    missing = [f for f, v in field_values.items() if len(v) < MIN_FIELD_CHARS]
    if missing:
        field_values = _fill_missing_explanation_fields(field_values, missing, concept, notes, style, depth)

    return "\n\n".join(v for v in field_values.values() if v)


def ensure_key_terms_count(key_terms: list[dict], concept: str, style: str, depth: str,
                            source_text: str, extra_excluded_terms: list[str] | None = None) -> list[dict]:
    """Tops up key_terms to the (style, depth) minimum by extracting more from
    source_text. extra_excluded_terms lets a caller (e.g. the pipeline, after
    its own cross-concept dedup has stripped some terms) tell this call about
    terms used ELSEWHERE in the document too, so the top-up generates genuinely
    new ones instead of terms that would just get deduped away again."""
    needed = key_terms_min(style, depth) - len(key_terms)
    if needed <= 0 or not source_text.strip():
        return key_terms
    excluded_lower = {kt["term"].lower() for kt in key_terms}
    excluded_lower.update(t.lower() for t in (extra_excluded_terms or []))
    existing_names = ", ".join(sorted(excluded_lower)) or "(none yet)"
    prompt = EXTRA_KEY_TERMS_PROMPT.format(
        concept=concept, needed=needed, existing_terms=existing_names, source_text=source_text[:8000],
    )
    try:
        result = generate_json(prompt)
    except Exception:
        return key_terms  # best-effort -- ship what we have rather than fail the whole concept

    for raw in safe_get_list(result, "key_terms"):
        cleaned = _clean_key_term(raw)
        if cleaned and cleaned["term"].lower() not in excluded_lower:
            key_terms.append(cleaned)
            excluded_lower.add(cleaned["term"].lower())
    return key_terms


def _normalize(result, concept: str, notes: str, style: str, depth: str, fields: list[str]) -> dict:
    explained_further = _build_explanation(result, concept, notes, style, depth, fields)

    key_terms = [
        kt for kt in (_clean_key_term(raw) for raw in safe_get_list(result, "key_terms"))
        if kt is not None
    ]
    comparisons_and_limits = _safe_text(result, "comparisons_and_limits")
    key_terms = ensure_key_terms_count(
        key_terms, concept, style, depth,
        source_text=f"{explained_further}\n\n{comparisons_and_limits}",
    )

    expand_hints = [
        h for h in safe_get_list(result, "expand_hints")
        if isinstance(h, str) and h.strip() and not looks_like_json_blob(h)
    ]

    return {
        "from_notes": _safe_text(result, "from_notes"),
        "explained_further": explained_further,
        "comparisons_and_limits": comparisons_and_limits,
        "analogy": _safe_text(result, "analogy"),
        "exam_answer_example": _safe_text(result, "exam_answer_example"),
        "expand_hints": expand_hints,
        "why_it_matters": _safe_text(result, "why_it_matters"),
        "key_terms": key_terms,
    }
