import json

import pytest

from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages import deserialize_message
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.tool_types import ToolCall

pytestmark = pytest.mark.integration

MODELS_THAT_VALIDATE_THOUGHT_SIGNATURES = [
    ("gemini-3.5-flash", "northamerica-northeast1"),
    ("gemini-4-hydrogen", "global"),
]

TEMPERATURES_CELSIUS = {"Paris": 37, "Rome": 41}

PARIS_PROMPT = (
    "What is the temperature in Paris right now? Call the get_temperature tool, "
    "then answer with the number it returns."
)
PARIS_AND_ROME_PROMPT = (
    "What are the temperatures in Paris and in Rome right now? Call the "
    "get_temperature tool once per city, then answer with both numbers."
)


@pytest.fixture(
    params=MODELS_THAT_VALIDATE_THOUGHT_SIGNATURES,
    ids=["gemini-3.5-flash-montreal", "gemini-4-hydrogen-global"],
)
def model(request):
    model_name, location = request.param
    return GoogleModel(model=model_name, location=location)


def make_temperature_tool():
    requested_cities = []

    def get_temperature(city: str) -> str:
        """Get the current temperature for a city.

        Args:
            city: Name of the city, without the country
        """
        requested_cities.append(city)
        return json.dumps(
            {"city": city, "temperature_celsius": TEMPERATURES_CELSIUS.get(city)}
        )

    return get_temperature, requested_cities


def tool_call_messages(chat):
    return [
        message for message in chat.messages if isinstance(message, ToolCallMessage)
    ]


def assert_first_call_of_every_step_is_signed(chat):
    steps = tool_call_messages(chat)
    assert steps
    for step in steps:
        assert step.tool_calls[0].thought_signature


def test_complete_finishes_the_tool_loop(model):
    get_temperature, requested_cities = make_temperature_tool()

    final_chat = (
        Chat(model=model)
        .with_tools([get_temperature])
        .add_message(UserMessage(PARIS_PROMPT))
        .complete()
    )

    assert requested_cities == ["Paris"]
    assert "37" in final_chat.latest_message.content
    assert_first_call_of_every_step_is_signed(final_chat)


async def test_complete_stream_finishes_the_tool_loop(model):
    get_temperature, requested_cities = make_temperature_tool()
    chat = (
        Chat(model=model)
        .with_tools([get_temperature])
        .add_message(UserMessage(PARIS_PROMPT))
    )

    async with chat.complete_stream() as stream:
        async for _ in stream.content:
            pass
    final_chat = await stream.chat

    assert requested_cities == ["Paris"]
    assert "37" in final_chat.latest_message.content
    assert_first_call_of_every_step_is_signed(final_chat)


def test_tool_calls_for_two_cities_finish_with_every_step_signed(model):
    get_temperature, requested_cities = make_temperature_tool()

    final_chat = (
        Chat(model=model)
        .with_tools([get_temperature])
        .add_message(UserMessage(PARIS_AND_ROME_PROMPT))
        .complete()
    )

    assert sorted(requested_cities) == ["Paris", "Rome"]
    assert "37" in final_chat.latest_message.content
    assert "41" in final_chat.latest_message.content
    assert_first_call_of_every_step_is_signed(final_chat)


def test_conversation_stored_mid_turn_resumes_after_deserialization(model):
    get_temperature, _ = make_temperature_tool()
    paused_chat = (
        Chat(model=model)
        .with_tools([get_temperature])
        .add_message(UserMessage(PARIS_PROMPT))
        .complete(execute_tools=False)
    )
    signed_calls = paused_chat.latest_message.tool_calls
    for tool_call in signed_calls:
        paused_chat = paused_chat.add_message(
            ToolResultMessage(
                get_temperature(**json.loads(tool_call.arguments)),
                call_id=tool_call.id,
                function_name=tool_call.name,
            )
        )
    stored = json.dumps([message.serialize() for message in paused_chat.messages])

    restored_messages = [deserialize_message(item) for item in json.loads(stored)]
    resumed_chat = (
        Chat(messages=restored_messages, model=model)
        .with_tools([get_temperature])
        .complete()
    )

    assert signed_calls[0].thought_signature
    assert restored_messages[1].tool_calls == signed_calls
    assert "37" in resumed_chat.latest_message.content


def test_unsigned_tool_call_in_the_current_turn_is_accepted(model):
    get_temperature, _ = make_temperature_tool()
    call_from_another_provider = ToolCall(
        id="toolu_01", name="get_temperature", arguments='{"city": "Paris"}'
    )

    final_chat = (
        Chat(model=model)
        .with_tools([get_temperature])
        .add_message(UserMessage(PARIS_PROMPT))
        .add_message(ToolCallMessage([call_from_another_provider]))
        .add_message(
            ToolResultMessage(
                get_temperature(city="Paris"),
                call_id="toolu_01",
                function_name="get_temperature",
            )
        )
        .complete()
    )

    assert "37" in final_chat.latest_message.content
