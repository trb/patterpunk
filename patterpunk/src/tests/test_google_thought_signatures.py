import base64
import json

import pytest
from google.genai import types

from patterpunk.llm.messages import deserialize_message
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.streaming import ChatStream
from patterpunk.llm.tool_types import ToolCall

SIGNATURE = b"\x00\x01opaque-signature\xff"
ENCODED_SIGNATURE = base64.b64encode(SIGNATURE).decode("ascii")
PLACEHOLDER_SIGNATURE = b"skip_thought_signature_validator"


def make_stub_model(model="gemini-4-hydrogen"):
    return GoogleModel(client=object(), model=model)


def response_with(parts):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


def function_call_part(city, thought_signature=None):
    return types.Part(
        function_call=types.FunctionCall(name="get_temperature", args={"city": city}),
        thought_signature=thought_signature,
    )


def function_call_parts_of_request(model, tool_calls):
    messages = [
        UserMessage("Temperature in Paris and Rome?"),
        ToolCallMessage(tool_calls),
        ToolResultMessage('{"celsius": 37}', function_name="get_temperature"),
    ]
    contents, _ = model._convert_messages_for_google_with_cache(messages)
    return contents[1].parts


def test_response_parsing_keeps_the_signature_of_each_function_call_part():
    response = response_with(
        [function_call_part("Paris", SIGNATURE), function_call_part("Rome")]
    )

    message = make_stub_model()._process_generation_response(response, None)

    assert [call.thought_signature for call in message.tool_calls] == [
        ENCODED_SIGNATURE,
        None,
    ]


def test_request_returns_the_signature_on_the_function_call_part_it_came_from():
    tool_calls = [
        ToolCall(
            id="call_1",
            name="get_temperature",
            arguments='{"city": "Paris"}',
            thought_signature=ENCODED_SIGNATURE,
        ),
        ToolCall(id="call_2", name="get_temperature", arguments='{"city": "Rome"}'),
    ]

    parts = function_call_parts_of_request(make_stub_model(), tool_calls)

    assert [part.thought_signature for part in parts] == [SIGNATURE, None]
    assert [dict(part.function_call.args) for part in parts] == [
        {"city": "Paris"},
        {"city": "Rome"},
    ]


@pytest.mark.parametrize(
    "model_name", ["gemini-3.5-flash", "gemini-4-hydrogen", "gemini-next"]
)
def test_unsigned_first_call_gets_the_placeholder_on_models_that_validate(model_name):
    tool_calls = [
        ToolCall(id="call_1", name="get_temperature", arguments='{"city": "Paris"}'),
        ToolCall(id="call_2", name="get_temperature", arguments='{"city": "Rome"}'),
    ]

    parts = function_call_parts_of_request(make_stub_model(model_name), tool_calls)

    assert [part.thought_signature for part in parts] == [PLACEHOLDER_SIGNATURE, None]


def test_unsigned_call_stays_unsigned_on_gemini_25():
    tool_calls = [
        ToolCall(id="call_1", name="get_temperature", arguments='{"city": "Paris"}')
    ]

    parts = function_call_parts_of_request(
        make_stub_model("gemini-2.5-flash"), tool_calls
    )

    assert parts[0].thought_signature is None


def test_stream_events_carry_the_signature_into_the_accumulated_tool_call():
    chunk = response_with([function_call_part("Paris", SIGNATURE)])
    stream = ChatStream(raw_stream=None, chat_factory=None)

    for event in make_stub_model()._convert_stream_chunk_to_events(chunk):
        stream._handle_tool_event(event)

    tool_call = stream._accumulator.tool_calls[0]
    assert tool_call.name == "get_temperature"
    assert json.loads(tool_call.arguments) == {"city": "Paris"}
    assert tool_call.thought_signature == ENCODED_SIGNATURE


def test_tool_call_message_keeps_the_signature_through_json_storage():
    message = ToolCallMessage(
        [
            ToolCall(
                id="call_1",
                name="get_temperature",
                arguments='{"city": "Paris"}',
                thought_signature=ENCODED_SIGNATURE,
            ),
            ToolCall(id="call_2", name="get_temperature", arguments='{"city": "Rome"}'),
        ]
    )

    stored = json.dumps(message.serialize())
    restored = deserialize_message(json.loads(stored))

    assert restored.tool_calls == message.tool_calls
    assert "thought_signature" not in json.loads(stored)["tool_calls"][1]


def test_tool_call_stored_without_a_signature_field_deserializes():
    stored = {
        "type": "tool_call",
        "id": "message-1",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_temperature", "arguments": "{}"},
            }
        ],
    }

    restored = deserialize_message(stored)

    assert restored.tool_calls[0].thought_signature is None


def test_openai_format_omits_the_signature():
    tool_call = ToolCall(
        id="call_1",
        name="get_temperature",
        arguments="{}",
        thought_signature=ENCODED_SIGNATURE,
    )

    assert "thought_signature" not in tool_call.to_openai_format()


def test_function_call_without_arguments_parses_to_an_empty_object():
    response = response_with(
        [types.Part(function_call=types.FunctionCall(name="get_time"))]
    )

    message = make_stub_model()._process_generation_response(response, None)

    assert message.tool_calls[0].arguments == "{}"
