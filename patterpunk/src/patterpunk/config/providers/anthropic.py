import os
from typing import Optional

try:
    from anthropic import Anthropic, AsyncAnthropic
except ImportError:
    Anthropic = None
    AsyncAnthropic = None

from patterpunk.lib.async_client_per_event_loop import AsyncClientPerEventLoop
from ..defaults import resolve_timeout_default

ANTHROPIC_API_KEY = os.getenv("PP_ANTHROPIC_API_KEY", None)
ANTHROPIC_DEFAULT_TEMPERATURE = os.getenv("PP_ANTHROPIC_DEFAULT_TEMPERATURE", 0.7)
ANTHROPIC_DEFAULT_TOP_P = os.getenv("PP_ANTHROPIC_DEFAULT_TOP_P", 1.0)
ANTHROPIC_DEFAULT_TOP_K = os.getenv("PP_ANTHROPIC_DEFAULT_TOP_K", 200)
ANTHROPIC_DEFAULT_MAX_TOKENS = os.getenv("PP_ANTHROPIC_DEFAULT_MAX_TOKENS", 8192)
ANTHROPIC_DEFAULT_TIMEOUT = resolve_timeout_default("PP_ANTHROPIC_DEFAULT_TIMEOUT")

_anthropic_client = None
_anthropic_async_clients = AsyncClientPerEventLoop(
    lambda: AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
)


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None and ANTHROPIC_API_KEY:
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def get_anthropic_async_client():
    if not ANTHROPIC_API_KEY:
        return None
    return _anthropic_async_clients.for_running_loop()


def is_anthropic_available() -> bool:
    return ANTHROPIC_API_KEY is not None


anthropic = get_anthropic_client()
