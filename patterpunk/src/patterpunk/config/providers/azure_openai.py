import os
from typing import Optional

from ..defaults import MAX_RETRIES

# Azure OpenAI Responses API uses v1 endpoint with standard OpenAI client
# Format: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
AZURE_OPENAI_ENDPOINT = os.getenv("PP_AZURE_OPENAI_ENDPOINT", None)
AZURE_OPENAI_API_KEY = os.getenv("PP_AZURE_OPENAI_API_KEY", None)
AZURE_OPENAI_MAX_RETRIES = os.getenv("PP_AZURE_OPENAI_MAX_RETRIES") or MAX_RETRIES

_azure_openai_client = None
_azure_openai_async_client = None


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


def is_azure_openai_available() -> bool:
    return AZURE_OPENAI_ENDPOINT is not None and AZURE_OPENAI_API_KEY is not None


azure_openai = get_azure_openai_client()
azure_openai_async = get_azure_openai_async_client()
