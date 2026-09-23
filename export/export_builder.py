"""Builds the downloadable study pack for a document, as either the
interactive self-contained HTML or a print-friendly static PDF."""
import json
import re
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from jinja2 import Template
from xhtml2pdf import pisa
from services import db

TEMPLATE_PATH = Path(__file__).resolve().parent / "html_template.html"
PDF_TEMPLATE_PATH = Path(__file__).resolve().parent / "pdf_template.html"

_SIMILARITY_THRESHOLD = 0.75


def _normalize_term(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _merge_glossary_terms(concepts: list[dict]) -> list[dict]:
    """Each concept keeps its own natural key terms (even if another concept
    also uses the same term -- that's normal, real topics revisit shared
    vocabulary). The standalone Glossary view is the one place a repeated term
    would actually look redundant, so merging happens only here, at assembly
    time, not by stripping terms from individual concepts."""
    merged = []
    seen_norms = []
    for c in concepts:
        for kt in c["summary_parsed"].get("key_terms", []):
            norm = _normalize_term(kt.get("term", ""))
            if not norm:
                continue
            if any(SequenceMatcher(None, norm, s).ratio() >= _SIMILARITY_THRESHOLD for s in seen_norms):
                continue
            merged.append(kt)
            seen_norms.append(norm)
    return merged


def _build_context(document_id: int) -> dict:
    doc = db.get_document(document_id)
    concepts = db.list_concepts(document_id)
    has_remediation = False
    for c in concepts:
        c["summary_parsed"] = db.parse_summary(c["summary"])
        c["remediations_parsed"] = [
            {"attempt": rem["attempt"], "summary": db.parse_summary(rem["summary"])}
            for rem in db.list_concept_remediations(c["id"])
        ]
        if c["remediations_parsed"]:
            has_remediation = True
    flashcards = db.list_flashcards(document_id)
    if any(f["source"] == "drill" for f in flashcards):
        has_remediation = True
    quiz_questions = db.list_quiz_questions(document_id)
    pyq_solutions = db.list_pyq_solutions(document_id)
    glossary_terms = _merge_glossary_terms(concepts)

    return {
        "title": doc["title"],
        "concepts": concepts,
        "flashcards": flashcards,
        "quiz_questions": quiz_questions,
        "pyq_solutions": pyq_solutions,
        "glossary_terms": glossary_terms,
        "has_remediation": has_remediation,
    }


def build_html(document_id: int) -> str:
    ctx = _build_context(document_id)
    concept_names = {c["id"]: c["name"] for c in ctx["concepts"]}
    quiz_json = json.dumps([
        {
            "question": q["question"],
            "options": q["options"],
            "correct_index": q["correct_index"],
            "concept": concept_names.get(q["concept_id"], ""),
        }
        for q in ctx["quiz_questions"]
    ])

    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.render(
        title=ctx["title"],
        concepts=ctx["concepts"],
        flashcards=ctx["flashcards"],
        quiz_json=quiz_json,
        pyq_solutions=ctx["pyq_solutions"],
        glossary_terms=ctx["glossary_terms"],
        has_remediation=ctx["has_remediation"],
    )


def build_pdf(document_id: int) -> bytes:
    """A static, print-friendly PDF -- flashcard/quiz answers are revealed
    directly (no flip/click interaction is possible on paper)."""
    ctx = _build_context(document_id)
    template = Template(PDF_TEMPLATE_PATH.read_text(encoding="utf-8"))
    html = template.render(**ctx)

    buffer = BytesIO()
    result = pisa.CreatePDF(html, dest=buffer)
    if result.err:
        raise RuntimeError(f"PDF generation failed with {result.err} error(s).")
    return buffer.getvalue()
