import pytest
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import SystemMessage, UserMessage


def test_thinking_config_validation():
    with pytest.raises(ValueError, match="Must specify exactly one"):
        ThinkingConfig()

    with pytest.raises(ValueError, match="Must specify exactly one"):
        ThinkingConfig(effort="medium", token_budget=4000)

    with pytest.raises(ValueError, match="token_budget must be non-negative"):
        ThinkingConfig(token_budget=-1)


def test_thinking_config_effort():
    config = ThinkingConfig(effort="medium")
    assert config.effort == "medium"
    assert config.token_budget is None
    assert config.include_thoughts is False


def test_thinking_config_token_budget():
    config = ThinkingConfig(token_budget=8000, include_thoughts=True)
    assert config.effort is None
    assert config.token_budget == 8000
    assert config.include_thoughts is True


def test_openai_model_thinking_config_integration():
    thinking_config = ThinkingConfig(effort="high")
    model = OpenAiModel(model="o3-mini", thinking_config=thinking_config)
    assert model.thinking_config == thinking_config
    assert model.reasoning_effort.name == "HIGH"


def test_anthropic_model_thinking_config_integration():
    thinking_config = ThinkingConfig(token_budget=10000)
    model = AnthropicModel(model="claude-3.7-sonnet", thinking_config=thinking_config)
    assert model.thinking_config == thinking_config
    assert model.thinking.budget_tokens == 10000


def test_google_model_thinking_config_integration():
    thinking_config = ThinkingConfig(token_budget=2000, include_thoughts=True)
    model = GoogleModel(
        model="gemini-2.5-flash",
        location="northamerica-northeast1",
        thinking_config=thinking_config,
    )
    assert model.thinking_config == thinking_config
    assert model.thinking_budget == 2000
    assert model.include_thoughts is True
