import os
from typing import Optional

from ..defaults import MAX_RETRIES

OPENAI_API_KEY = os.getenv("PP_OPENAI_API_KEY", None)
OPENAI_MAX_RETRIES = os.getenv("PP_OPENAI_MAX_RETRIES") or MAX_RETRIES

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def is_openai_available() -> bool:
    return OPENAI_API_KEY is not None


openai = get_openai_client()
