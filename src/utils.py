"""
src/utils.py - Shared utilities for the Spotify Support Agent.

Provides common functions used across the agent, evaluation harness,
and other pipeline scripts.
"""
import re
import time
from google.genai.errors import ClientError, ServerError


def call_with_retry(fn, max_retries=5, base_delay=15):
    """
    Execute `fn` with exponential backoff on transient API errors.

    Parses Google GenAI error messages for recommended retry delay.
    Falls back to exponential backoff (base_delay * 2^attempt) if
    no server-suggested delay is found.

    Args:
        fn: Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        The return value of `fn` on success.

    Raises:
        ClientError or ServerError if all retries are exhausted
        or the error is non-retriable.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except (ClientError, ServerError) as e:
            code = getattr(e, "code", None)
            if code in (429, 503) and attempt < max_retries - 1:
                err_msg = str(e)
                m = re.search(r"retry in ([\d\.]+)s", err_msg, re.IGNORECASE)
                if m:
                    delay = float(m.group(1)) + 2.0
                else:
                    delay = base_delay * (2 ** attempt)
                print(f"      [Retry {attempt+1}/{max_retries}] API {code} error, waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
