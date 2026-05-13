import os
from typing import Optional

try:
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
except ImportError:
    pass  # openai package not installed, will fail at runtime if Azure OpenAI methods are called

from ..defaults import MAX_RETRIES, SDK_MAX_RETRIES, resolve_timeout_default

# Azure OpenAI Responses API uses v1 endpoint with standard OpenAI client
# Format: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
AZURE_OPENAI_ENDPOINT = os.getenv("PP_AZURE_OPENAI_ENDPOINT", None)
AZURE_OPENAI_API_KEY = os.getenv("PP_AZURE_OPENAI_API_KEY", None)
AZURE_OPENAI_MAX_RETRIES = os.getenv("PP_AZURE_OPENAI_MAX_RETRIES") or MAX_RETRIES
AZURE_OPENAI_DEFAULT_TIMEOUT = resolve_timeout_default(
    "PP_AZURE_OPENAI_DEFAULT_TIMEOUT"
)

# Azure OpenAI reasoning endpoint (for gpt-5.2, o1, o3-mini, etc.)
# Uses the native AzureOpenAI client with api_version
AZURE_OPENAI_REASONING_ENDPOINT = os.getenv("PP_AZURE_OPENAI_REASONING_ENDPOINT", None)
AZURE_OPENAI_REASONING_API_KEY = os.getenv("PP_AZURE_OPENAI_REASONING_API_KEY", None)
AZURE_OPENAI_REASONING_API_VERSION = "2025-03-01-preview"


def _get_azure_base_url() -> str:
    """Build the Azure OpenAI base URL with correct formatting."""
    base_url = AZURE_OPENAI_ENDPOINT
    if not base_url.endswith("/openai/v1/"):
        if not base_url.endswith("/"):
            base_url += "/"
        base_url += "openai/v1/"
    return base_url


def get_azure_openai_client(timeout: int = AZURE_OPENAI_DEFAULT_TIMEOUT):
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY):
        return None
    # Disable SDK internal retries - patterpunk handles retries with proper backoff
    return OpenAI(
        base_url=_get_azure_base_url(),
        api_key=AZURE_OPENAI_API_KEY,
        max_retries=SDK_MAX_RETRIES,
        timeout=timeout,
    )


def get_azure_openai_async_client(timeout: int = AZURE_OPENAI_DEFAULT_TIMEOUT):
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY):
        return None
    # Disable SDK internal retries - patterpunk handles retries with proper backoff
    return AsyncOpenAI(
        base_url=_get_azure_base_url(),
        api_key=AZURE_OPENAI_API_KEY,
        max_retries=SDK_MAX_RETRIES,
        timeout=timeout,
    )


def get_azure_openai_reasoning_client(timeout: int = AZURE_OPENAI_DEFAULT_TIMEOUT):
    """Get sync client for Azure OpenAI reasoning models (gpt-5.2, o1, etc.)."""
    if not (AZURE_OPENAI_REASONING_ENDPOINT and AZURE_OPENAI_REASONING_API_KEY):
        return None
    # Disable SDK internal retries - patterpunk handles retries with proper backoff
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_REASONING_ENDPOINT,
        api_key=AZURE_OPENAI_REASONING_API_KEY,
        api_version=AZURE_OPENAI_REASONING_API_VERSION,
        max_retries=SDK_MAX_RETRIES,
        timeout=timeout,
    )


def get_azure_openai_reasoning_async_client(
    timeout: int = AZURE_OPENAI_DEFAULT_TIMEOUT,
):
    """Get async client for Azure OpenAI reasoning models (gpt-5.2, o1, etc.)."""
    if not (AZURE_OPENAI_REASONING_ENDPOINT and AZURE_OPENAI_REASONING_API_KEY):
        return None
    # Disable SDK internal retries - patterpunk handles retries with proper backoff
    return AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_REASONING_ENDPOINT,
        api_key=AZURE_OPENAI_REASONING_API_KEY,
        api_version=AZURE_OPENAI_REASONING_API_VERSION,
        max_retries=SDK_MAX_RETRIES,
        timeout=timeout,
    )


def is_azure_openai_available() -> bool:
    return AZURE_OPENAI_ENDPOINT is not None and AZURE_OPENAI_API_KEY is not None


def is_azure_openai_reasoning_available() -> bool:
    return (
        AZURE_OPENAI_REASONING_ENDPOINT is not None
        and AZURE_OPENAI_REASONING_API_KEY is not None
    )
