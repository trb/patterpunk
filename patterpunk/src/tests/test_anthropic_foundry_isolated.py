"""
These tests set or clear PP_ANTHROPIC_FOUNDRY_* environment variables and
call importlib.reload on the Foundry config and model modules. Reloading
rebinds the module globals other tests may hold references to. Run this
file alone:

    pytest tests/test_anthropic_foundry_isolated.py
"""

import copy
import importlib

import pytest

from patterpunk.config.providers import anthropic_foundry as foundry_config
from patterpunk.llm.models import anthropic_foundry as foundry_model
from patterpunk.llm.thinking import ThinkingConfig

_FOUNDRY_ENV_VARS = (
    "PP_ANTHROPIC_FOUNDRY_API_KEY",
    "PP_ANTHROPIC_FOUNDRY_RESOURCE",
    "PP_ANTHROPIC_FOUNDRY_BASE_URL",
)


@pytest.fixture
def foundry_env(monkeypatch):
    def apply(**env):
        for name in _FOUNDRY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        importlib.reload(foundry_config)
        importlib.reload(foundry_model)

    yield apply
    # monkeypatch's own restore runs after this finalizer, too late for the
    # reloads to see the original environment. Undo it here first.
    monkeypatch.undo()
    importlib.reload(foundry_config)
    importlib.reload(foundry_model)


def configured(apply, **extra):
    apply(
        PP_ANTHROPIC_FOUNDRY_RESOURCE="myresource",
        PP_ANTHROPIC_FOUNDRY_API_KEY="test-key",
        **extra,
    )


def test_missing_endpoint_raises(foundry_env):
    foundry_env(PP_ANTHROPIC_FOUNDRY_API_KEY="test-key")
    with pytest.raises(foundry_model.AnthropicFoundryMissingConfigurationError):
        foundry_model.AnthropicFoundryModel(deployment_name="claude-opus-5")


def test_missing_credentials_raises(foundry_env):
    foundry_env(PP_ANTHROPIC_FOUNDRY_RESOURCE="myresource")
    with pytest.raises(foundry_model.AnthropicFoundryMissingConfigurationError):
        foundry_model.AnthropicFoundryModel(deployment_name="claude-opus-5")


def test_token_provider_substitutes_for_api_key(foundry_env):
    foundry_env(PP_ANTHROPIC_FOUNDRY_RESOURCE="myresource")
    model = foundry_model.AnthropicFoundryModel(
        deployment_name="claude-opus-5",
        azure_ad_token_provider=lambda: "entra-token",
    )
    assert model.model == "claude-opus-5"


def test_resource_expands_to_azure_base_url(foundry_env):
    configured(foundry_env)
    client = foundry_config.get_anthropic_foundry_client()
    assert str(client.base_url) == "https://myresource.services.ai.azure.com/anthropic/"


def test_explicit_base_url_wins_over_resource(foundry_env):
    configured(foundry_env, PP_ANTHROPIC_FOUNDRY_BASE_URL="https://example.test/v1")
    kwargs = foundry_config._client_kwargs(None)
    assert kwargs["base_url"] == "https://example.test/v1"
    assert "resource" not in kwargs


def test_clients_cached_per_token_provider(foundry_env):
    configured(foundry_env)
    provider = lambda: "entra-token"
    default_first = foundry_config.get_anthropic_foundry_client()
    default_second = foundry_config.get_anthropic_foundry_client()
    with_provider = foundry_config.get_anthropic_foundry_client(provider)
    assert default_first is default_second
    assert with_provider is not default_first


def test_deployment_name_and_model_id_split(foundry_env):
    configured(foundry_env)
    model = foundry_model.AnthropicFoundryModel(
        deployment_name="my-claude-prod",
        model_id="claude-opus-4-8",
    )
    assert model._parse_model_version() == (4, 8)
    assert model._uses_adaptive_thinking_api() is True
    api_params = model._build_base_api_parameters([], None)
    assert api_params["model"] == "my-claude-prod"
    assert "temperature" not in api_params
    assert "top_p" not in api_params


def test_model_id_defaults_to_deployment_name(foundry_env):
    configured(foundry_env)
    model = foundry_model.AnthropicFoundryModel(deployment_name="claude-sonnet-4-5")
    assert model.model_id == "claude-sonnet-4-5"
    assert model._parse_model_version() == (4, 5)
    api_params = model._build_base_api_parameters([], None)
    assert api_params["model"] == "claude-sonnet-4-5"
    assert api_params["temperature"] is not None


def test_unparseable_deployment_name_gets_newest_family_rules(foundry_env, caplog):
    configured(foundry_env)
    model = foundry_model.AnthropicFoundryModel(deployment_name="prod-sonnet")
    with caplog.at_level("WARNING", logger="patterpunk"):
        assert model._parse_model_version() == (5, 0)
        api_params = model._build_base_api_parameters([], None)
    assert "temperature" not in api_params
    assert any("Cannot detect a Claude version" in r.message for r in caplog.records)


def test_deepcopy_shares_token_provider(foundry_env):
    configured(foundry_env)
    provider = lambda: "entra-token"
    model = foundry_model.AnthropicFoundryModel(
        deployment_name="my-claude-prod",
        model_id="claude-opus-4-8",
        azure_ad_token_provider=provider,
        thinking_config=ThinkingConfig(effort="high"),
    )
    clone = copy.deepcopy(model)
    assert clone is not model
    assert clone.model == "my-claude-prod"
    assert clone.model_id == "claude-opus-4-8"
    assert clone.azure_ad_token_provider is provider
