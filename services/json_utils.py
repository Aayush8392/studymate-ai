"""Defensive helpers for reading fields out of LLM JSON output. Despite
requesting JSON mode with a specific object shape, Gemini/Groq occasionally
return a bare list (e.g. [...] instead of {"flashcards": [...]}) or otherwise
deviate from the requested schema. Every agent reads model output through
these helpers instead of raw .get() calls, so a malformed top-level shape
degrades to an empty result instead of crashing the whole pipeline."""
import json


def safe_get_list(result, key: str) -> list:
    """Reads a list field from a JSON result that should be a dict like
    {key: [...]}, but tolerates the model returning the bare list directly."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        value = result.get(key, [])
        return value if isinstance(value, list) else []
    return []


def safe_get(result, key: str, default=""):
    """Reads a scalar/dict field from a JSON result, tolerating a result that
    isn't a dict at all (falls back to default)."""
    if isinstance(result, dict):
        return result.get(key, default)
    return default


def safe_get_str(result, key: str, default: str = "") -> str:
    """Like safe_get, but also guards against the field itself being the wrong
    type (e.g. the model returns a list/number where a string was expected) --
    common enough with free-tier models to be worth checking explicitly rather
    than crashing on a later .strip() call."""
    value = safe_get(result, key, default)
    return value.strip() if isinstance(value, str) else default


def looks_like_json_blob(text: str) -> bool:
    """Detects a specific model glitch: instead of plain text, the field's
    value is itself a JSON-encoded object/array (e.g. a flashcard's "back"
    containing '{"explanation": ..., "key_terms": [...]}' instead of a real
    answer). A value that parses as JSON and yields a dict/list is essentially
    never a legitimate plain-text answer, so this is a safe, cheap check."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, (dict, list))
