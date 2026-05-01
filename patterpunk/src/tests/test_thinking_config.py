import pytest
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage


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
        location="us-central1",
        thinking_config=thinking_config,
    )
    assert model.thinking_config == thinking_config
    assert model.thinking_budget == 2000
    assert model.include_thoughts is True


# =============================================================================
# Extended effort vocabulary (Opus 4.7's xhigh and max)
# =============================================================================


def test_thinking_config_accepts_xhigh():
    config = ThinkingConfig(effort="xhigh")
    assert config.effort == "xhigh"


def test_thinking_config_accepts_max():
    config = ThinkingConfig(effort="max")
    assert config.effort == "max"


def test_anthropic_opus_4_7_accepts_xhigh():
    """Opus 4.7+ uses the new effort vocabulary natively."""
    config = ThinkingConfig(effort="xhigh")
    model = AnthropicModel(model="claude-opus-4-7", thinking_config=config)
    api_params = model._apply_thinking_configuration({})
    assert api_params["output_config"] == {"effort": "xhigh"}


def test_anthropic_opus_4_7_accepts_max():
    config = ThinkingConfig(effort="max")
    model = AnthropicModel(model="claude-opus-4-7", thinking_config=config)
    api_params = model._apply_thinking_configuration({})
    assert api_params["output_config"] == {"effort": "max"}


def test_openai_clamps_xhigh_to_high_with_warning(caplog):
    """OpenAI doesn't support xhigh — clamp to high with a WARN, don't raise."""
    config = ThinkingConfig(effort="xhigh")
    with caplog.at_level("WARNING", logger="patterpunk"):
        model = OpenAiModel(model="o3-mini", thinking_config=config)
    assert model.reasoning_effort.name == "HIGH"
    assert any(
        "Anthropic-only" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_openai_clamps_max_to_high_with_warning(caplog):
    config = ThinkingConfig(effort="max")
    with caplog.at_level("WARNING", logger="patterpunk"):
        model = OpenAiModel(model="o3-mini", thinking_config=config)
    assert model.reasoning_effort.name == "HIGH"
    assert any(
        "Anthropic-only" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_google_clamps_xhigh_to_high_with_warning(caplog):
    """Google/Gemini doesn't support xhigh — clamp to high with a WARN, don't raise."""
    config = ThinkingConfig(effort="xhigh")
    with caplog.at_level("WARNING", logger="patterpunk"):
        model = GoogleModel(
            model="gemini-2.5-flash",
            location="us-central1",
            thinking_config=config,
        )
    # 'high' maps to 12000 in Google's effort_to_tokens
    assert model.thinking_budget == 12000
    assert any(
        "Anthropic-only" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_legacy_anthropic_clamps_xhigh_to_high_with_warning(caplog):
    """Pre-Opus-4.7 Anthropic models clamp xhigh/max to high in __init__."""
    config = ThinkingConfig(effort="xhigh")
    with caplog.at_level("WARNING", logger="patterpunk"):
        model = AnthropicModel(model="claude-sonnet-4-20250514", thinking_config=config)
    # 'high' maps to 24000 in Anthropic's legacy effort_to_tokens
    assert model.thinking.budget_tokens == 24000
    assert any(
        "only supported on Claude Opus 4.7+" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )
