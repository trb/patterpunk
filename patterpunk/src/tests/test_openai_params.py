import pytest

from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.thinking import ThinkingConfig


def make_model(**kwargs):
    return OpenAiModel(_INTERNAL__skip_client_validation=True, **kwargs)


def setup_params(model):
    return model._setup_model_parameters(
        model.model,
        model.temperature,
        model.top_p,
        model.frequency_penalty,
        model.presence_penalty,
        model.logit_bias,
        model.reasoning_effort,
    )


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gpt-5.2", True),
        ("o1", True),
        ("o3-mini", True),
        ("gpt-5.2-chat-latest", False),
        ("gpt-5-chat-latest", False),
        ("gpt-4.1", False),
        ("gpt-4o", False),
    ],
)
def test_reasoning_model_classification(model_id, expected):
    model = make_model(model=model_id)
    assert model._is_reasoning_model(model_id) is expected


def test_reasoning_model_drops_custom_sampling_with_warning(caplog):
    model = make_model(model="gpt-5.2", temperature=0.3, top_p=0.9)
    with caplog.at_level("WARNING", logger="patterpunk"):
        params = setup_params(model)
    assert "temperature" not in params
    assert "top_p" not in params
    assert "reasoning" in params
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "[OPENAI]" in m and "temperature=0.3" in m and "top_p=0.9" in m
        for m in warning_messages
    )


def test_reasoning_model_silent_at_defaults(caplog):
    model = make_model(model="o3-mini")
    with caplog.at_level("WARNING", logger="patterpunk"):
        params = setup_params(model)
    assert "reasoning" in params
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_reasoning_model_warns_for_penalties_and_logit_bias(caplog):
    model = make_model(
        model="gpt-5.2",
        frequency_penalty=0.5,
        presence_penalty=-0.5,
        logit_bias={"50256": -100},
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        setup_params(model)
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "frequency_penalty=0.5" in m
        and "presence_penalty=-0.5" in m
        and "logit_bias" in m
        for m in warning_messages
    )


def test_chat_variant_gets_sampling_and_no_reasoning_key(caplog):
    model = make_model(model="gpt-5.2-chat-latest", temperature=0.3, top_p=0.9)
    with caplog.at_level("WARNING", logger="patterpunk"):
        params = setup_params(model)
    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.9
    assert "reasoning" not in params
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_chat_variant_with_thinking_config_warns_ignored(caplog):
    model = make_model(
        model="gpt-5.2-chat-latest",
        thinking_config=ThinkingConfig(effort="high"),
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        params = setup_params(model)
    assert "reasoning" not in params
    assert any("ignoring thinking_config" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "model_id,requested,expected,should_warn",
    [
        ("gpt-5.6-terra", "xhigh", "xhigh", False),
        ("gpt-5.6", "max", "max", False),
        ("gpt-5.2", "xhigh", "high", True),
        ("gpt-5", "max", "high", True),
        ("o3-mini", "xhigh", "high", True),
    ],
)
def test_reasoning_effort_clamped_per_model(
    model_id, requested, expected, should_warn, caplog
):
    model = make_model(model=model_id, thinking_config=ThinkingConfig(effort=requested))
    with caplog.at_level("WARNING", logger="patterpunk"):
        params = setup_params(model)
    assert params["reasoning"]["effort"] == expected
    clamp_warnings = [r for r in caplog.records if "Clamping to 'high'" in r.message]
    assert bool(clamp_warnings) is should_warn


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gpt-5.6-luna", "none"),
        ("gpt-5.1", "none"),
        ("gpt-5", "low"),
        ("o3-mini", "low"),
    ],
)
def test_token_budget_zero_maps_to_none_where_supported(model_id, expected):
    model = make_model(model=model_id, thinking_config=ThinkingConfig(token_budget=0))
    params = setup_params(model)
    assert params["reasoning"]["effort"] == expected
