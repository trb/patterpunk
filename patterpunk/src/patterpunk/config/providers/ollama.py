import os
from typing import Optional

OLLAMA_API_ENDPOINT = os.getenv("PP_OLLAMA_API_ENDPOINT", None)

_ollama_client = None


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None and OLLAMA_API_ENDPOINT:
        from ollama import Client

        _ollama_client = Client(host=OLLAMA_API_ENDPOINT)
    return _ollama_client


def is_ollama_available() -> bool:
    return OLLAMA_API_ENDPOINT is not None


ollama = get_ollama_client()
