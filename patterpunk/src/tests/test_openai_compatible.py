import copy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.openai_compatible import OpenAiCompatibleModel
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.types import ToolCall


class Verdict(BaseModel):
    answer: str
    confidence: float


def make_model(**kwargs):
    kwargs.setdefault("model", "llama-3.3-70b")
    kwargs.setdefault("base_url", "http://localhost:8000/v1")
    return OpenAiCompatibleModel(**kwargs)


def test_message_conversion_covers_all_roles():
    model = make_model()
    tool_call_message = ToolCallMessage(
        [ToolCall(id="call_1", name="get_weather", arguments='{"city": "Berlin"}')]
    )
    tool_result_message = ToolResultMessage(
        content="sunny", call_id="call_1", function_name="get_weather"
    )
    converted = model._convert_messages(
        [
            SystemMessage("be brief"),
            UserMessage("weather in Berlin?"),
            tool_call_message,
            tool_result_message,
        ]
    )
    assert converted == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather in Berlin?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Berlin"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]


def test_unset_sampling_params_are_omitted():
    model = make_model()
    params = model._build_request_params([UserMessage("hi")], None, None)
    assert "temperature" not in params
    assert "top_p" not in params
    assert "max_tokens" not in params
    assert "response_format" not in params
    assert "stream" not in params


def test_set_sampling_params_pass_through_for_third_party_ids():
    model = make_model(temperature=0.3, top_p=0.9, max_tokens=512)
    params = model._build_request_params([UserMessage("hi")], None, None)
    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.9
    assert params["max_tokens"] == 512


def test_openai_reasoning_family_strips_sampling_with_warning(caplog):
    model = make_model(
        model="gpt-5.2",
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
        thinking_config=ThinkingConfig(effort="high"),
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        params = model._build_request_params([UserMessage("hi")], None, None)
    assert "temperature" not in params
    assert "top_p" not in params
    assert params["max_completion_tokens"] == 512
    assert params["reasoning_effort"] == "high"
    assert any(
        "temperature=0.3" in r.message and "top_p=0.9" in r.message
        for r in caplog.records
    )


def test_third_party_id_with_thinking_config_warns_ignored(caplog):
    model = make_model(
        model="deepseek-r1", thinking_config=ThinkingConfig(effort="high")
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        params = model._build_request_params([UserMessage("hi")], None, None)
    assert "reasoning_effort" not in params
    assert any("ignoring thinking_config" in r.message for r in caplog.records)


def test_tools_pass_through_unchanged():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    model = make_model()
    params = model._build_request_params([UserMessage("hi")], tools, None)
    assert params["tools"] is tools


def test_structured_output_uses_response_format():
    model = make_model()
    params = model._build_request_params([UserMessage("hi")], None, Verdict)
    response_format = params["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "structured_response"
    assert response_format["json_schema"]["strict"] is True
    assert "properties" in response_format["json_schema"]["schema"]


def test_response_format_400_falls_back_to_prompted_schema(caplog):
    model = make_model()
    params = model._build_request_params([UserMessage("hi")], None, Verdict)
    error = Exception("400: response_format is not supported by this model")
    with caplog.at_level("WARNING", logger="patterpunk"):
        fallback = model._maybe_response_format_fallback(params, error)
    assert fallback is not None
    assert "response_format" not in fallback
    last_message = fallback["messages"][-1]
    assert last_message["role"] == "user"
    assert last_message["content"].startswith("hi\n")
    assert "properties" in last_message["content"]
    assert any("prompt-based structured output" in r.message for r in caplog.records)


def test_unrelated_400_does_not_fall_back():
    model = make_model()
    params = model._build_request_params([UserMessage("hi")], None, Verdict)
    error = Exception("400: model not found")
    assert model._maybe_response_format_fallback(params, error) is None


def test_api_key_provider_yields_fresh_token_per_request():
    tokens = iter(["token-1", "token-2"])
    model = make_model(api_key_provider=lambda: next(tokens))
    assert model._request_headers() == {"Authorization": "Bearer token-1"}
    assert model._request_headers() == {"Authorization": "Bearer token-2"}


def test_static_api_key_sends_no_extra_headers():
    model = make_model(api_key="sk-static")
    assert model._request_headers() is None


def test_default_headers_reach_the_client():
    headers = {"api-key": "azure-key", "extra-parameters": "drop"}
    model = make_model(default_headers=headers)
    assert model._client.default_headers["api-key"] == "azure-key"
    assert model._client.default_headers["extra-parameters"] == "drop"


def fake_response(message, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.MAX_TOKENS),
        ("content_filter", FinishReason.SAFETY),
        ("weird_custom_reason", FinishReason.OTHER),
    ],
)
def test_finish_reason_mapping(raw, expected):
    model = make_model()
    message = SimpleNamespace(content="hello", tool_calls=None)
    result = model._process_response(fake_response(message, raw), None)
    assert result.finish_reason == expected
    assert result._provider.raw_finish_reason == raw


def test_tool_call_response_becomes_tool_call_message():
    model = make_model()
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_9",
                function=SimpleNamespace(
                    name="get_weather", arguments='{"city": "Berlin"}'
                ),
            ),
            SimpleNamespace(
                id=None,
                function=SimpleNamespace(name="get_time", arguments="{}"),
            ),
        ],
    )
    result = model._process_response(fake_response(message, "tool_calls"), None)
    assert isinstance(result, ToolCallMessage)
    assert result.tool_calls[0].id == "call_9"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[1].id.startswith("call_get_time_")


def test_structured_output_parses_from_content():
    model = make_model()
    message = SimpleNamespace(
        content='{"answer": "42", "confidence": 0.9}', tool_calls=None
    )
    result = model._process_response(fake_response(message), Verdict)
    assert result.parsed_output.answer == "42"
    assert result.parsed_output.confidence == 0.9


def test_reasoning_content_becomes_thinking_block():
    model = make_model()
    message = SimpleNamespace(
        content="the answer", tool_calls=None, reasoning_content="let me think"
    )
    result = model._process_response(fake_response(message), None)
    assert result.thinking_blocks == [
        {"type": "thinking", "thinking": "let me think"}
    ]


def test_deepcopy_round_trip():
    provider = lambda: "token"
    model = make_model(
        api_key_provider=provider,
        temperature=0.3,
        thinking_config=ThinkingConfig(effort="low"),
    )
    clone = copy.deepcopy(model)
    assert clone is not model
    assert clone._client is not model._client
    assert clone.base_url == model.base_url
    assert clone.temperature == 0.3
    assert clone.api_key_provider is provider
