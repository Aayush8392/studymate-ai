"""Reads plain study notes out of photographed/scanned pages via Gemini's vision
input -- separate from pyq_solver's question-paper reading, since this is meant
to extract general note content (headings, bullets, paragraphs), not questions."""
from services.gemini_client import generate_json
from services.json_utils import safe_get_str

EXTRACT_PROMPT = """You are reading photo(s) of a student's handwritten or printed
study notes, or textbook/slide pages. Read all the text carefully, including
handwriting if present, and preserve the structure as best you can -- headings,
bullet points, and paragraph breaks.

Return JSON: {"text": "the full extracted text, as plain text with line breaks preserved"}
"""


def extract_notes_from_images(images: list[tuple[bytes, str]]) -> str:
    """images: list of (image_bytes, mime_type)."""
    result = generate_json(EXTRACT_PROMPT, images=images)
    return safe_get_str(result, "text")
