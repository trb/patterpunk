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
