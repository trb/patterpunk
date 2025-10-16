import os
from typing import Optional

ANTHROPIC_API_KEY = os.getenv("PP_ANTHROPIC_API_KEY", None)
ANTHROPIC_DEFAULT_TEMPERATURE = os.getenv("PP_ANTHROPIC_DEFAULT_TEMPERATURE", 0.7)
ANTHROPIC_DEFAULT_TOP_P = os.getenv("PP_ANTHROPIC_DEFAULT_TOP_P", 1.0)
ANTHROPIC_DEFAULT_TOP_K = os.getenv("PP_ANTHROPIC_DEFAULT_TOP_K", 200)
ANTHROPIC_DEFAULT_MAX_TOKENS = os.getenv("PP_ANTHROPIC_DEFAULT_MAX_TOKENS", 8192)
ANTHROPIC_DEFAULT_TIMEOUT = 600

_anthropic_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None and ANTHROPIC_API_KEY:
        from anthropic import Anthropic

        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def is_anthropic_available() -> bool:
    return ANTHROPIC_API_KEY is not None


anthropic = get_anthropic_client()
