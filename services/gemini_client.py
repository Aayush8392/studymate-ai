"""Thin wrapper around the Gemini API for structured JSON generation."""
import os
import re
import json
import time
import httpx
from google import genai
from google.genai import errors as genai_errors, types as genai_types

_client = None
_MAX_RETRIES = 5
_BASE_DELAY = 2  # seconds

# Free tier is rate-limited to a handful of requests/minute (not the same as
# the 1M-token context window, which is a per-request size limit, not a rate
# limit). Space calls out so we don't keep tripping 429s.
_MIN_SECONDS_BETWEEN_CALLS = 13  # ~4.6 req/min, under the 5/min free-tier cap
_last_call_at = 0.0

# Same reasoning as groq_client.py: cap total time spent waiting on retries so
# a run can't silently stall for many minutes on one call. Kept short (not the
# original 120s) because Streamlit's Stop control can't interrupt a call
# that's actually blocked in network I/O -- this cap is the only thing that
# unsticks the app if Stop is clicked mid-call, so it's tuned for a bearable
# worst-case wait, not for maximum retry patience.
_MAX_TOTAL_WAIT_SECONDS = 30


def _throttle():
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    if elapsed < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_at = time.monotonic()


def describe_error(e: Exception) -> str:
    """User-facing message for a failed call -- distinguishes the daily
    free-tier quota exhaustion (a known, actionable case worth naming
    specifically) from other failures, instead of surfacing the raw JSON
    error blob Gemini returns."""
    try:
        details = e.details.get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("QuotaFailure"):
                for v in d.get("violations", []):
                    if "PerDay" in v.get("quotaId", ""):
                        return (
                            "Daily free-tier limit reached for the Gemini API "
                            "(500 requests/day). This resets around midnight Pacific "
                            "time -- try again later, or use a different API key."
                        )
    except Exception:
        pass
    return str(e)


def _retry_delay_from_error(e) -> float | None:
    """Pull the server-suggested retry delay (seconds) out of a 429 error, if present."""
    try:
        details = e.details.get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                delay_str = d.get("retryDelay", "")  # e.g. "46s"
                if delay_str.endswith("s"):
                    return float(delay_str[:-1])
    except Exception:
        pass
    return None


def _call_with_retry(fn):
    """Retries on transient server/rate-limit errors AND on malformed JSON --
    a garbled response is usually a one-off generation glitch, and simply
    asking the model again is more reliable than trying to hand-repair
    arbitrary broken JSON."""
    last_err = None
    total_waited = 0.0
    for attempt in range(_MAX_RETRIES):
        _throttle()
        try:
            return fn()
        except genai_errors.ServerError as e:
            last_err = e
            delay = _BASE_DELAY * (2 ** attempt)
        except httpx.TransportError as e:
            # Connection-level failures (dropped connection, "server
            # disconnected without sending a response", timeouts, etc.) --
            # these come straight from httpx, not the genai SDK's own error
            # types, so they were previously falling through uncaught and
            # killing the whole generation run on a single transient blip.
            last_err = e
            delay = _BASE_DELAY * (2 ** attempt)
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) != 429:
                raise
            last_err = e
            delay = (_retry_delay_from_error(e) or (_BASE_DELAY * (2 ** attempt))) + 1
        except json.JSONDecodeError as e:
            last_err = e
            delay = _BASE_DELAY * (2 ** attempt)
        else:
            continue
        if total_waited + delay > _MAX_TOTAL_WAIT_SECONDS:
            break  # would blow the time budget -- give up now instead of waiting
        time.sleep(delay)
        total_waited += delay
    raise last_err


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")
        # attempts=1: the SDK otherwise retries 429s internally up to 5 times
        # with up to 60s delay each, silently and with no logging, stacking
        # underneath our own _call_with_retry. Same fix as groq_client.py --
        # all retrying happens in ONE place so the 2-minute cap actually holds.
        _client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(retry_options=genai_types.HttpRetryOptions(attempts=1)),
        )
    return _client


def _parse_json_text(text: str) -> dict:
    text = text.strip()
    # Some models occasionally wrap JSON in markdown fences even with JSON mode.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Cheap, common-case repair: trailing commas before a closing bracket
        # (e.g. `"a": 1,}`) are invalid JSON but a frequent model slip-up.
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(repaired)  # let this raise if still broken -- caller retries


def generate_json(prompt: str, model: str = "gemini-3.5-flash-lite",
                   images: list[tuple[bytes, str]] | None = None,
                   thinking_budget: int | None = None) -> dict:
    """Call Gemini and parse the response as JSON. Retries on transient errors
    and on malformed JSON (regenerating rather than trying to hand-repair
    arbitrary broken output). images: optional list of (image_bytes, mime_type)
    to include alongside the prompt, for vision tasks. thinking_budget: lets
    the model reason before writing its final answer instead of writing
    linearly -- same model, same one call, but produces markedly more
    complete/well-reasoned output for content that actually needs depth (e.g.
    PYQ exam answers). Left unset (no thinking) for high-volume calls like
    concept summaries/flashcards/quiz, where the added latency isn't worth it."""
    client = _get_client()
    contents = _build_contents(prompt, images)
    config = {"response_mime_type": "application/json"}
    if thinking_budget is not None:
        config["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=thinking_budget)

    def call_and_parse():
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _parse_json_text(response.text)

    return _call_with_retry(call_and_parse)


def _build_contents(prompt: str, images: list[tuple[bytes, str]] | None):
    if not images:
        return prompt
    parts = [genai_types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images]
    parts.append(genai_types.Part.from_text(text=prompt))
    return parts


def generate_text(prompt: str, model: str = "gemini-3.5-flash-lite") -> str:
    client = _get_client()

    def call():
        return client.models.generate_content(model=model, contents=prompt)

    response = _call_with_retry(call)
    return response.text.strip()
