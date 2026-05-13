import os
from typing import Optional

try:
    from ollama import Client
except ImportError:
    pass  # ollama package not installed, will fail at runtime if Ollama methods are called

from ..defaults import resolve_timeout_default

OLLAMA_API_ENDPOINT = os.getenv("PP_OLLAMA_API_ENDPOINT", None)

# Retry configuration for transient errors (503 server overloaded, 429 rate limit)
# Delay settings now use global defaults from config.defaults for consistency across providers
OLLAMA_MAX_RETRIES = int(os.getenv("PP_OLLAMA_MAX_RETRIES", "3"))
OLLAMA_DEFAULT_TIMEOUT = resolve_timeout_default("PP_OLLAMA_DEFAULT_TIMEOUT")


def get_ollama_client(timeout: int = OLLAMA_DEFAULT_TIMEOUT):
    if not OLLAMA_API_ENDPOINT:
        return None
    return Client(host=OLLAMA_API_ENDPOINT, timeout=timeout)


def is_ollama_available() -> bool:
    return OLLAMA_API_ENDPOINT is not None
