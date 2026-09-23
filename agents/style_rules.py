"""Shared vocabulary/depth instruction blocks and numeric parameters, used by
every content-generating agent so a document's explanation style and depth
stay consistent -- and VISIBLY different -- across concept count, explanation
length, exam answers, flashcards, and quiz difficulty. Numeric parameters here
are the single source of truth: prompt text is built FROM these numbers so the
instructions and the actual generation counts never drift apart.

Depth controls WHICH sections exist and how much content they hold (structural
differences: expand_hints is entirely absent at Overview since "go further"
contradicts a bare-essentials depth; comparisons_and_limits only exists at
Deep dive). Style controls HOW the always-present sections are written
(vocabulary) and one structural lever of its own: how many key terms get
defined, since a beginner needs more scaffolding and an expert needs less.
"""

DEPTH_PARAMS = {
    "overview": {
        "exam_answer_length": "2-3 sentences only -- just the core definition, nothing more",
        "expand_hints_count": 0,
        "flashcard_n": 3,
        "quiz_n": 2,
        "include_comparisons": False,
        "concept_granularity": (
            "Keep concepts BROAD. Merge closely related sub-topics into a single "
            "concept rather than splitting them apart. Aim for the SMALLEST number "
            "of concepts that still covers the material -- roughly 3 to 6 concepts "
            "for typical notes, even if that means grouping several related ideas "
            "under one concept name."
        ),
    },
    "standard": {
        "exam_answer_length": "4-6 sentences",
        "expand_hints_count": 3,
        "flashcard_n": 5,
        "quiz_n": 3,
        "include_comparisons": False,
        "concept_granularity": (
            "Use a MODERATE number of concepts. Split into distinct sub-topics "
            "where they are genuinely separate ideas, but don't over-fragment. "
            "Aim for roughly 5 to 9 concepts for typical notes."
        ),
    },
    "deep": {
        "exam_answer_length": (
            "8+ sentences, or structured with clear sub-points if the question "
            "calls for a list -- a thorough, exam-ready answer, not a summary"
        ),
        "expand_hints_count": 6,
        "flashcard_n": 7,
        "quiz_n": 5,
        "include_comparisons": True,
        "concept_granularity": (
            "Use a GRANULAR breakdown. Split closely related ideas into their own "
            "separate concepts wherever they are distinct enough to teach "
            "separately -- don't merge things that deserve their own treatment. "
            "Aim for roughly 8 to 14 concepts for typical notes."
        ),
    },
}

# Explicit (style, depth) -> key-terms count range. Depth sets a base range,
# style applies a fixed modifier (+1 simple / +0 standard / -1 technical,
# floored at 1) -- written out explicitly here rather than computed, so the
# exact numbers are visible and can't silently drift.
KEY_TERMS_COUNT = {
    ("simple", "overview"): "2-3",
    ("simple", "standard"): "3-5",
    ("simple", "deep"): "5-7",
    ("standard", "overview"): "1-2",
    ("standard", "standard"): "2-4",
    ("standard", "deep"): "4-6",
    ("technical", "overview"): "exactly 1",
    ("technical", "standard"): "1-3",
    ("technical", "deep"): "3-5",
}

# The minimum acceptable count for each range above -- parsed out explicitly
# so code can check "did we actually get enough key terms" as a real,
# deterministic gate, not just trust the model followed the prompt.
KEY_TERMS_MIN = {
    ("simple", "overview"): 2, ("simple", "standard"): 3, ("simple", "deep"): 5,
    ("standard", "overview"): 1, ("standard", "standard"): 2, ("standard", "deep"): 4,
    ("technical", "overview"): 1, ("technical", "standard"): 1, ("technical", "deep"): 3,
}

# Depth controls how many distinct explanation fields the model must fill in
# -- turning "write N paragraphs" from a soft word-count request into a
# schema requirement that's actually checkable (is this named field present
# and non-trivial?), not something the model can silently under-deliver on.
EXPLANATION_FIELDS = {
    "overview": ["explanation_core"],
    "standard": ["explanation_core", "explanation_detail"],
    "deep": ["explanation_core", "explanation_detail", "explanation_example"],
}

EXPLANATION_FIELD_DESCRIPTIONS = {
    "explanation_core": "What the concept is and how it works, in your own teaching voice.",
    "explanation_detail": "A specific mechanism, nuance, or elaboration that ISN'T already "
        "covered in explanation_core -- genuinely additive, not a rephrasing.",
    "explanation_example": "A concrete, applied real-world scenario that illustrates the "
        "concept in action.",
}

MIN_FIELD_CHARS = 15  # below this, treat a field as effectively empty/dodged

STYLE_RULES = {
    "simple": """EXPLANATION STYLE: Simple.
- Write like you're explaining to a total beginner who has never heard of this
  topic before. Assume ZERO background knowledge, not even related terms.
- HARD RULE: keep sentences to roughly 15 words or fewer. One idea per sentence.
- HARD RULE: every single technical term must be immediately explained in plain
  words the moment it's used, every time, even terms that seem "basic" in this
  field -- never assume the reader already knows it.
- Prefer everyday words over academic ones (e.g. "figures out" not "ascertains").
- For the "analogy" field: use a playful, concrete, everyday-life comparison
  (a kitchen, a sports game, a pet, a familiar chore) -- something a total
  beginner immediately pictures. Avoid technical or domain comparisons here.
- No dense academic tone, no jargon-stacking, no corporate buzzwords.""",
    "standard": """EXPLANATION STYLE: Standard.
- Write at a normal undergraduate course level.
- Use standard academic vocabulary and technical terms, but briefly define each
  one the first time it's used so it's not assumed knowledge.
- Clear, well-structured sentences of moderate length -- not oversimplified,
  not dense.
- For the "analogy" field: a clear real-world or domain-adjacent comparison is
  fine, doesn't need to be as playful/simplistic as a beginner analogy.""",
    "technical": """EXPLANATION STYLE: Technical.
- Write for a reader who is comfortable with the field's terminology -- assume
  familiarity with standard concepts and vocabulary in this subject.
- HARD RULE: do NOT stop to define standard/basic terms in this field -- use
  them freely, the way a textbook or professional would.
- No sentence-length cap -- dense, precise, compound sentences are fine and
  expected.
- For the "analogy" field: the analogy must NEVER be left empty. Prefer a
  TECHNICAL comparison -- how this concept relates to, differs from, or is
  analogous to another concept/framework/model within the same field. If no
  clean cross-concept comparison exists, fall back to a structural or
  mechanism-level comparison instead (e.g. relate it to a simpler sub-case, a
  mathematical relationship, or an analogous process) -- but always provide
  something, never an empty string, and never a childish everyday analogy.
- Prioritize accuracy, precision, and nuance over accessibility.""",
}

VALID_STYLES = list(STYLE_RULES)
VALID_DEPTHS = list(DEPTH_PARAMS)

DEFAULT_STYLE = "simple"
DEFAULT_DEPTH = "standard"


def _depth_rules_text(style: str, depth: str) -> str:
    p = DEPTH_PARAMS.get(depth, DEPTH_PARAMS[DEFAULT_DEPTH])
    key_terms_count = KEY_TERMS_COUNT.get((style, depth), KEY_TERMS_COUNT[(DEFAULT_STYLE, DEFAULT_DEPTH)])

    comparisons_line = (
        'Also fill in "comparisons_and_limits" with 2-4 sentences on how this '
        "concept compares to related/competing concepts, edge cases, or its "
        "limitations -- real nuance, not filler."
        if p["include_comparisons"] else
        'Leave "comparisons_and_limits" as an empty string "" -- do not fill it in '
        "at this depth."
    )
    expand_hints_count = p["expand_hints_count"]
    expand_hints_line = (
        'Leave "expand_hints" as an empty list [] -- at this depth the point is the '
        "bare essentials, so there is deliberately nothing to \"go further\" with."
        if expand_hints_count == 0 else
        f'Include exactly {expand_hints_count} expand-hints, no more, no fewer.'
    )

    return f"""DEPTH: {depth}.
- The "exam_answer_example" must be {p['exam_answer_length']}.
- Include {key_terms_count} key terms, no more, no fewer than that. If the notes are
  too sparse to supply that many terms on their own, draw additional terms from
  your OWN explanation_core/explanation_detail/explanation_example content below --
  don't limit yourself to only what's literally in the notes for this count.
- {expand_hints_line}
- {comparisons_line}"""


def style_depth_block(style: str, depth: str) -> str:
    """Full style + depth instructions -- for the summarizer ONLY. The depth
    portion (key terms, expand-hints, comparisons_and_limits, exam answer
    length) refers to fields that exist ONLY in the summarizer's JSON schema.
    Flashcard/quiz agents must use style_only_block() instead -- giving them
    these schema-specific instructions with no matching field to put the
    content in caused the model to dump "Key terms:"/"Expand-hints:" text
    directly into flashcard answers instead."""
    style_text = STYLE_RULES.get(style, STYLE_RULES[DEFAULT_STYLE])
    depth_text = _depth_rules_text(style, depth)
    return f"{style_text}\n\n{depth_text}"


def style_only_block(style: str) -> str:
    """Just the vocabulary/tone rules, safe for any agent's prompt regardless
    of its own JSON schema -- use this for flashcards/quiz, not the full
    style_depth_block (see its docstring)."""
    return STYLE_RULES.get(style, STYLE_RULES[DEFAULT_STYLE])


def concept_granularity_text(depth: str) -> str:
    p = DEPTH_PARAMS.get(depth, DEPTH_PARAMS[DEFAULT_DEPTH])
    return p["concept_granularity"]


def exam_answer_length(depth: str) -> str:
    return DEPTH_PARAMS.get(depth, DEPTH_PARAMS[DEFAULT_DEPTH])["exam_answer_length"]


# A real exam question's own marks weight is a much stronger signal for how
# long/thorough its answer needs to be than the document-wide Depth setting --
# a 10-mark question needs a real multi-paragraph exam answer regardless of
# Depth, and a 2-mark question stays short even at Deep dive. Used for PYQ
# solving only; Depth still governs vocabulary/style and is the fallback
# length source when a question's marks can't be read off the paper.
#
# The 7+ marks tier is deliberately NOT phrased as a word/sentence/paragraph
# count -- three earlier attempts at count-based instructions (a raw length
# target, a paragraph-count requirement, a sentence-per-paragraph minimum) all
# failed to reliably produce real depth, because a small/fast model can
# satisfy a count with a one-line paragraph or a padded restatement without
# actually saying more. What worked, verified live: naming specific CONTENT
# the model must include for each distinct part of the question (a real
# definition, scenario-specific detail, a contrast, a risk, a closing
# significance line) -- content the model can't produce without actually
# reasoning about it -- combined with a `thinking_budget` (see
# gemini_client.generate_json) that lets it reason before writing instead of
# writing linearly and stopping early.
#
# The minimum number of developed points is ALSO deliberately decoupled from
# how many sub-parts the question happens to name -- verified live that two
# real 10-mark questions produced wildly different lengths (~185 words vs
# ~600 words) purely because one named 2 things and the other named 4, even
# though both were worth identical marks. Marks weight is the only thing that
# should set the floor; a question naming fewer parts than that floor must
# have the model supply its own additional angles to reach it, not stay short
# because the question phrasing under-specified how many parts to cover.
def _high_marks_answer_spec(min_points: int) -> str:
    return (
        f"a full exam-ready answer covering AT LEAST {min_points} distinct "
        "developed points, aiming for roughly 7-8 sentences EACH -- this "
        "minimum is set by the marks weight alone, regardless of how many "
        f"parts the question itself explicitly names. If the question names "
        f"{min_points} or more distinct sub-parts (domains, factors, "
        "functions, causes, etc.), give EACH one its own point. If it names "
        "FEWER than that (even just one, or a single indivisible concept), "
        "you must still reach the minimum by deriving additional genuinely "
        "relevant angles of your own -- e.g. an additional applicable "
        "model/factor not explicitly asked for, a cause, an underlying "
        "mechanism, a comparison to a related idea, a specific consequence, "
        "or a risk/limitation -- so the total answer reflects the full "
        "weight of the marks no matter how the question happens to be "
        "phrased. EVERY point (named or derived) must cover all of: (1) a "
        "precise definition of the concept, including what it's typically "
        "evaluated/measured against, (2) how it specifically applies here, "
        "referencing at least one concrete named detail from the "
        "question/scenario -- not a generic restatement, (3) a second, "
        "hypothetical or illustrative example beyond the scenario itself, "
        "to show general understanding of the concept, (4) a comparison or "
        "contrast with a related/adjacent idea, (5) a specific risk, "
        "limitation, or open question relevant to that point, and (6) a "
        "closing line on why that point matters for the overall outcome. Do "
        "not compress any point into one or two sentences -- some "
        "restatement/emphasis across points is fine, a short or padded "
        "answer is not."
    )


def pyq_answer_length(marks: int | None, depth: str) -> str:
    """Length/structure instruction for one PYQ solution, keyed primarily off
    that question's own marks weight when known."""
    if marks is None:
        return _high_marks_answer_spec(3) if depth == "deep" else exam_answer_length(depth)
    if marks <= 2:
        return "2-3 sentences -- just the core definition/fact, nothing more"
    if marks <= 4:
        return "4-6 sentences, a focused single-paragraph answer"
    if marks <= 6:
        return "a full paragraph of 6-10 sentences, including one concrete example"
    if marks <= 9:
        return _high_marks_answer_spec(3)
    if marks <= 14:
        return _high_marks_answer_spec(4)
    return _high_marks_answer_spec(5)


def pyq_needs_thinking(marks: int | None, depth: str) -> bool:
    """Whether this question's answer is substantial enough to warrant paying
    for a thinking-enabled call -- true for the same cases that get
    _HIGH_MARKS_ANSWER_SPEC. Kept as a separate check (not inferred from the
    length string) so the two stay obviously in sync."""
    if marks is None:
        return depth == "deep"
    return marks >= 7


def flashcard_n(depth: str) -> int:
    return DEPTH_PARAMS.get(depth, DEPTH_PARAMS[DEFAULT_DEPTH])["flashcard_n"]


def quiz_n(depth: str) -> int:
    return DEPTH_PARAMS.get(depth, DEPTH_PARAMS[DEFAULT_DEPTH])["quiz_n"]


def explanation_fields(depth: str) -> list[str]:
    return EXPLANATION_FIELDS.get(depth, EXPLANATION_FIELDS[DEFAULT_DEPTH])


def explanation_fields_schema(depth: str) -> str:
    """The JSON field descriptions for this depth's required explanation
    fields, to splice directly into a prompt's JSON schema block."""
    fields = explanation_fields(depth)
    lines = [f'"{f}": "{EXPLANATION_FIELD_DESCRIPTIONS[f]}"' for f in fields]
    return ",\n  ".join(lines)


def key_terms_min(style: str, depth: str) -> int:
    return KEY_TERMS_MIN.get((style, depth), KEY_TERMS_MIN[(DEFAULT_STYLE, DEFAULT_DEPTH)])


_DEPTH_TIERS = ["overview", "standard", "deep"]


def escalated_explanation_fields(depth: str, attempt: int) -> list[str]:
    """Once a concept needs remediation, the ORIGINAL depth's explanation
    schema already failed this student -- reusing that same schema (e.g.
    Overview's single paragraph) for every remediation attempt just produces
    more thin paragraphs reworded differently, not deeper teaching. Escalate
    toward the richer schema instead, independent of the document's original
    depth setting. Deep dive has no higher tier to escalate to -- repeated
    failure there is a stronger signal, handled by the existing
    prerequisite-check/handoff path (MAX_ATTEMPTS) rather than more prose."""
    start = _DEPTH_TIERS.index(depth) if depth in _DEPTH_TIERS else 0
    escalated_depth = _DEPTH_TIERS[min(start + attempt, len(_DEPTH_TIERS) - 1)]
    return EXPLANATION_FIELDS[escalated_depth]


def escalated_explanation_fields_schema(depth: str, attempt: int) -> str:
    fields = escalated_explanation_fields(depth, attempt)
    lines = [f'"{f}": "{EXPLANATION_FIELD_DESCRIPTIONS[f]}"' for f in fields]
    return ",\n  ".join(lines)
