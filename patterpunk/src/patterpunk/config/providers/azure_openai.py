import os
from typing import Optional

from ..defaults import MAX_RETRIES

# Azure OpenAI Responses API uses v1 endpoint with standard OpenAI client
# Format: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
AZURE_OPENAI_ENDPOINT = os.getenv("PP_AZURE_OPENAI_ENDPOINT", None)
AZURE_OPENAI_API_KEY = os.getenv("PP_AZURE_OPENAI_API_KEY", None)
AZURE_OPENAI_MAX_RETRIES = os.getenv("PP_AZURE_OPENAI_MAX_RETRIES") or MAX_RETRIES

# Azure OpenAI reasoning endpoint (for gpt-5.2, o1, o3-mini, etc.)
# Uses the native AzureOpenAI client with api_version
AZURE_OPENAI_REASONING_ENDPOINT = os.getenv("PP_AZURE_OPENAI_REASONING_ENDPOINT", None)
AZURE_OPENAI_REASONING_API_KEY = os.getenv("PP_AZURE_OPENAI_REASONING_API_KEY", None)
AZURE_OPENAI_REASONING_API_VERSION = "2025-03-01-preview"

_azure_openai_client = None
_azure_openai_async_client = None
_azure_openai_reasoning_client = None
_azure_openai_reasoning_async_client = None


def _get_azure_base_url() -> str:
    """Build the Azure OpenAI base URL with correct formatting."""
    base_url = AZURE_OPENAI_ENDPOINT
    if not base_url.endswith("/openai/v1/"):
        if not base_url.endswith("/"):
            base_url += "/"
        base_url += "openai/v1/"
    return base_url


def get_azure_openai_client():
    global _azure_openai_client
    if _azure_openai_client is None and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        from openai import OpenAI

        _azure_openai_client = OpenAI(
            base_url=_get_azure_base_url(),
            api_key=AZURE_OPENAI_API_KEY,
        )
    return _azure_openai_client


def get_azure_openai_async_client():
    global _azure_openai_async_client
    if (
        _azure_openai_async_client is None
        and AZURE_OPENAI_ENDPOINT
        and AZURE_OPENAI_API_KEY
    ):
        from openai import AsyncOpenAI

        _azure_openai_async_client = AsyncOpenAI(
            base_url=_get_azure_base_url(),
            api_key=AZURE_OPENAI_API_KEY,
        )
    return _azure_openai_async_client


def get_azure_openai_reasoning_client():
    """Get sync client for Azure OpenAI reasoning models (gpt-5.2, o1, etc.)."""
    global _azure_openai_reasoning_client
    if (
        _azure_openai_reasoning_client is None
        and AZURE_OPENAI_REASONING_ENDPOINT
        and AZURE_OPENAI_REASONING_API_KEY
    ):
        from openai import AzureOpenAI

        _azure_openai_reasoning_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_REASONING_ENDPOINT,
            api_key=AZURE_OPENAI_REASONING_API_KEY,
            api_version=AZURE_OPENAI_REASONING_API_VERSION,
        )
    return _azure_openai_reasoning_client


def get_azure_openai_reasoning_async_client():
    """Get async client for Azure OpenAI reasoning models (gpt-5.2, o1, etc.)."""
    global _azure_openai_reasoning_async_client
    if (
        _azure_openai_reasoning_async_client is None
        and AZURE_OPENAI_REASONING_ENDPOINT
        and AZURE_OPENAI_REASONING_API_KEY
    ):
        from openai import AsyncAzureOpenAI

        _azure_openai_reasoning_async_client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_OPENAI_REASONING_ENDPOINT,
            api_key=AZURE_OPENAI_REASONING_API_KEY,
            api_version=AZURE_OPENAI_REASONING_API_VERSION,
        )
    return _azure_openai_reasoning_async_client


def is_azure_openai_available() -> bool:
    return AZURE_OPENAI_ENDPOINT is not None and AZURE_OPENAI_API_KEY is not None


def is_azure_openai_reasoning_available() -> bool:
    return (
        AZURE_OPENAI_REASONING_ENDPOINT is not None
        and AZURE_OPENAI_REASONING_API_KEY is not None
    )


azure_openai = get_azure_openai_client()
azure_openai_async = get_azure_openai_async_client()
azure_openai_reasoning = get_azure_openai_reasoning_client()
azure_openai_reasoning_async = get_azure_openai_reasoning_async_client()
