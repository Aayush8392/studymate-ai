"""Thin wrapper around the Groq API — used only for the independent Verifier agent,
so groundedness checks aren't done by the same model that generated the content."""
import os
import re
import json
import time
from groq import Groq, RateLimitError, InternalServerError, APIConnectionError, APITimeoutError

_client = None
_MAX_RETRIES = 5
_BASE_DELAY = 2  # seconds

# Groq's free-tier 429s can carry a Retry-After of several minutes. Waiting
# that out made a single notes file take 10+ minutes end to end, which
# defeats the point of "instant" study material. Cap the TOTAL time spent
# waiting on retries so a stuck verifier call fails fast instead of stalling
# the whole run -- the caller (verifier.py) treats a failure here as
# "couldn't verify this batch" and lets the content through unverified
# rather than crashing the pipeline. Also kept short (not the original 120s)
# since Stop can't interrupt a blocked network call -- see gemini_client.py's
# note on the same constant.
_MAX_TOTAL_WAIT_SECONDS = 30

# qwen/qwen3-32b on the free tier has a fairly low tokens-per-minute cap, so
# space calls out a little even before hitting a 429, to reduce how often we
# have to wait out a real rate-limit error.
_MIN_SECONDS_BETWEEN_CALLS = 15
_last_call_at = 0.0


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")
        # max_retries=0: the SDK's own built-in retry-on-429 logic sleeps
        # silently (no logging) and stacks on top of our _call_with_retry,
        # which is how a single call ended up hanging for ~19 minutes with
        # nothing in the log. All retrying now happens in ONE place --
        # _call_with_retry -- which enforces the 2-minute total cap.
        _client = Groq(api_key=api_key, max_retries=0)
    return _client


def _throttle():
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    if elapsed < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_at = time.monotonic()


def _retry_after_seconds(e: RateLimitError) -> float | None:
    try:
        header = e.response.headers.get("retry-after")
        if header is not None:
            return float(header)
    except Exception:
        pass
    return None


def _call_with_retry(fn):
    """Retries on transient server/rate-limit errors AND on malformed JSON --
    a garbled response is usually a one-off generation glitch, and simply
    asking the model again is more reliable than hand-repairing broken JSON."""
    last_err = None
    total_waited = 0.0
    for attempt in range(_MAX_RETRIES):
        _throttle()
        try:
            return fn()
        except RateLimitError as e:
            last_err = e
            delay = _retry_after_seconds(e) or (_BASE_DELAY * (2 ** attempt))
            delay += 1  # small buffer past the server's suggestion
        except (InternalServerError, APIConnectionError, APITimeoutError) as e:
            last_err = e
            delay = _BASE_DELAY * (2 ** attempt)
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


def _parse_json_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Cheap, common-case repair: trailing commas before a closing bracket.
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(repaired)  # let this raise if still broken -- caller retries


def generate_json(prompt: str, model: str = "openai/gpt-oss-120b") -> dict:
    client = _get_client()

    def call_and_parse():
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return _parse_json_text(completion.choices[0].message.content)

    return _call_with_retry(call_and_parse)
