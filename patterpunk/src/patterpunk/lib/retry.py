"""
Retry utilities for handling rate limits and transient errors.

This module provides exponential backoff with jitter, designed specifically
for per-minute rate limits common with OpenAI and Azure OpenAI APIs, plus
schedule-driven retry execution for models configured with a RetryConfig.
"""

import asyncio
import random
import ssl
import time
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Optional

try:
    import httpx
except ImportError:
    httpx = None

from patterpunk.llm.retry_config import RetryConfig
from patterpunk.logger import logger

_TRANSPORT_ERROR_TYPES = (ConnectionError, TimeoutError, ssl.SSLError) + (
    (httpx.TransportError,) if httpx is not None else ()
)


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 60.0,
    max_delay: float = 300.0,
    min_delay: float = 45.0,
    jitter_factor: float = 0.5,
    retry_after: Optional[float] = None,
) -> float:
    """
    Calculate delay before next retry attempt using exponential backoff with jitter.

    The algorithm:
    1. Calculate base exponential delay: base_delay * (2 ** attempt)
    2. Apply random jitter: delay * (1 ± jitter_factor)
    3. Clamp to [min_delay, max_delay]
    4. Honor retry_after header if provided (use max of calculated vs retry_after)

    The min_delay ensures we never retry too early for per-minute rate limits.
    The jitter prevents thundering herd when multiple clients hit limits simultaneously.

    Args:
        attempt: Zero-indexed retry attempt number (0 = first retry)
        base_delay: Base delay in seconds for exponential calculation
        max_delay: Maximum delay cap in seconds
        min_delay: Minimum delay floor in seconds (hard minimum)
        jitter_factor: Jitter range (0.5 = ±50% randomization)
        retry_after: Optional Retry-After header value from the error response

    Returns:
        Delay in seconds before the next retry attempt

    Example delays with defaults (base=60, max=300, min=45, jitter=0.5):
        attempt=0: 60s base → 30-90s after jitter → 45-90s after min clamp
        attempt=1: 120s base → 60-180s after jitter → 60-180s
        attempt=2: 240s base → 120-360s after jitter → 120-300s after max clamp
    """
    # Exponential backoff
    delay = base_delay * (2**attempt)

    # Apply jitter: multiply by random factor in range [1-jitter, 1+jitter]
    jitter_multiplier = 1.0 + random.uniform(-jitter_factor, jitter_factor)
    delay *= jitter_multiplier

    # Clamp to [min_delay, max_delay]
    delay = max(min_delay, min(delay, max_delay))

    # Honor Retry-After header if provided
    if retry_after is not None and retry_after > 0:
        delay = max(delay, retry_after)

    return delay


def extract_retry_after(error: Exception) -> Optional[float]:
    """
    Extract Retry-After value from an API error response.

    OpenAI/Azure errors may include Retry-After in headers or response body.

    Args:
        error: The exception raised by the API

    Returns:
        Retry-After value in seconds, or None if not found
    """
    # Check for response attribute (httpx/requests style errors)
    response = getattr(error, "response", None)
    if response is not None:
        # Try headers first
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

    # Check for body attribute with retry_after field (OpenAI SDK errors)
    body = getattr(error, "body", None)
    if body is not None and isinstance(body, dict):
        retry_after = body.get("retry_after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

    return None


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.

    Retryable errors include:
    - 408 Request Timeout
    - 429 Too Many Requests (rate limit)
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    - Transport-level failures (connection reset, TLS errors, local timeouts)
    - AWS Bedrock throttling/service unavailable errors

    Args:
        error: The exception to check

    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_status_codes = (408, 429, 500, 502, 503, 504)

    # Check for status_code attribute (OpenAI, Anthropic, Ollama)
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return status_code in retryable_status_codes

    # Check for code attribute (Google genai)
    code = getattr(error, "code", None)
    if code is not None:
        return code in retryable_status_codes

    if isinstance(error, _TRANSPORT_ERROR_TYPES):
        return True

    # Check for AWS Bedrock ClientError format
    response = getattr(error, "response", None)
    if response is not None and isinstance(response, dict):
        error_info = response.get("Error", {})
        error_code = error_info.get("Code", "")
        if error_code in ("ThrottlingException", "ServiceUnavailableException"):
            return True

    # Check error message for rate limit indicators
    error_str = str(error).lower()
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
        "throttling",
    ]
    if any(indicator in error_str for indicator in rate_limit_indicators):
        return True

    # Check for connection/timeout errors
    connection_indicators = [
        "connection",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
    ]
    if any(indicator in error_str for indicator in connection_indicators):
        return True

    return False


def run_with_retry_config(
    retry_config: RetryConfig,
    attempt: Callable[[], Any],
    provider_name: str,
) -> Any:
    """
    Execute attempt() under a RetryConfig schedule.

    Makes len(delays_s) + 1 attempts total, sleeping delays_s[n] * uniform(*jitter)
    before each re-attempt. Non-retryable errors (per is_retryable_error) raise
    immediately; on exhaustion the last native error is re-raised unwrapped.
    """
    total_attempts = len(retry_config.delays_s) + 1
    for attempt_number, delay_s in enumerate(retry_config.delays_s, start=1):
        try:
            return attempt()
        except Exception as error:
            if not is_retryable_error(error):
                raise
            wait_time = delay_s * random.uniform(*retry_config.jitter)
            logger.warning(
                f"{provider_name}: retryable error on attempt "
                f"{attempt_number}/{total_attempts}: {error!r}. "
                f"Waiting {wait_time:.1f}s before retry."
            )
            time.sleep(wait_time)
    return attempt()


async def stream_with_retry_config(
    retry_config: RetryConfig,
    acquire_stream: Callable[[], Awaitable[AsyncIterable]],
    provider_name: str,
) -> AsyncIterator:
    """
    Async-generator counterpart of run_with_retry_config.

    Retries the whole stream: a mid-stream failure after partial yield
    re-acquires the stream and re-yields from the start, matching the
    behavior of the providers' legacy streaming retry loops.
    """
    total_attempts = len(retry_config.delays_s) + 1
    for attempt_number, delay_s in enumerate(retry_config.delays_s, start=1):
        try:
            stream = await acquire_stream()
            async for item in stream:
                yield item
            return
        except Exception as error:
            if not is_retryable_error(error):
                raise
            wait_time = delay_s * random.uniform(*retry_config.jitter)
            logger.warning(
                f"{provider_name}: retryable streaming error on attempt "
                f"{attempt_number}/{total_attempts}: {error!r}. "
                f"Waiting {wait_time:.1f}s before retry."
            )
            await asyncio.sleep(wait_time)
    stream = await acquire_stream()
    async for item in stream:
        yield item
