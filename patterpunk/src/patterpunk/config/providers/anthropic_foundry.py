import os

try:
    from anthropic import AnthropicFoundry, AsyncAnthropicFoundry
except ImportError:
    AnthropicFoundry = None
    AsyncAnthropicFoundry = None

ANTHROPIC_FOUNDRY_API_KEY = os.getenv("PP_ANTHROPIC_FOUNDRY_API_KEY", None)
ANTHROPIC_FOUNDRY_RESOURCE = os.getenv("PP_ANTHROPIC_FOUNDRY_RESOURCE", None)
ANTHROPIC_FOUNDRY_BASE_URL = os.getenv("PP_ANTHROPIC_FOUNDRY_BASE_URL", None)

_foundry_clients = {}
_foundry_async_clients = {}


def _client_kwargs(azure_ad_token_provider):
    # The SDK expands resource="x" to https://x.services.ai.azure.com/anthropic/.
    # An explicit base_url wins so nonstandard endpoints stay reachable.
    kwargs = {}
    if ANTHROPIC_FOUNDRY_BASE_URL:
        kwargs["base_url"] = ANTHROPIC_FOUNDRY_BASE_URL
    elif ANTHROPIC_FOUNDRY_RESOURCE:
        kwargs["resource"] = ANTHROPIC_FOUNDRY_RESOURCE
    if azure_ad_token_provider is not None:
        kwargs["azure_ad_token_provider"] = azure_ad_token_provider
    elif ANTHROPIC_FOUNDRY_API_KEY:
        kwargs["api_key"] = ANTHROPIC_FOUNDRY_API_KEY
    return kwargs


def get_anthropic_foundry_client(azure_ad_token_provider=None):
    if azure_ad_token_provider not in _foundry_clients:
        _foundry_clients[azure_ad_token_provider] = AnthropicFoundry(
            **_client_kwargs(azure_ad_token_provider)
        )
    return _foundry_clients[azure_ad_token_provider]


def get_anthropic_foundry_async_client(azure_ad_token_provider=None):
    if azure_ad_token_provider not in _foundry_async_clients:
        _foundry_async_clients[azure_ad_token_provider] = AsyncAnthropicFoundry(
            **_client_kwargs(azure_ad_token_provider)
        )
    return _foundry_async_clients[azure_ad_token_provider]


def has_foundry_endpoint() -> bool:
    return bool(ANTHROPIC_FOUNDRY_RESOURCE or ANTHROPIC_FOUNDRY_BASE_URL)


def is_anthropic_foundry_available() -> bool:
    return (
        AnthropicFoundry is not None
        and has_foundry_endpoint()
        and ANTHROPIC_FOUNDRY_API_KEY is not None
    )
