import os
from typing import Optional

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    pass  # openai package not installed, will fail at runtime if OpenAI methods are called

from ..defaults import MAX_RETRIES, SDK_MAX_RETRIES, resolve_timeout_default

OPENAI_API_KEY = os.getenv("PP_OPENAI_API_KEY", None)
OPENAI_MAX_RETRIES = os.getenv("PP_OPENAI_MAX_RETRIES") or MAX_RETRIES
OPENAI_DEFAULT_TIMEOUT = resolve_timeout_default("PP_OPENAI_DEFAULT_TIMEOUT")


def get_openai_client(timeout: int = OPENAI_DEFAULT_TIMEOUT):
    if not OPENAI_API_KEY:
        return None
    # Disable SDK internal retries - patterpunk handles retries with proper backoff
    return OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=SDK_MAX_RETRIES,
        timeout=timeout,
    )


def get_openai_async_client(timeout: int = OPENAI_DEFAULT_TIMEOUT):
    if not OPENAI_API_KEY:
        return None
    # Disable SDK internal retries - patterpunk handles retries with proper backoff
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=SDK_MAX_RETRIES,
        timeout=timeout,
    )


def is_openai_available() -> bool:
    return OPENAI_API_KEY is not None
