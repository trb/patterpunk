"""
Chat-level structured output guarantees that hold for every provider.
"""

import pytest
from pydantic import BaseModel

from patterpunk.llm.chat.core import Chat
from patterpunk.llm.chat.exceptions import StructuredOutputParsingError
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.types import ToolCall


class Answer(BaseModel):
    value: int


def _chat():
    return Chat(model=AnthropicModel(model="claude-haiku-4-5-20251001"))


def test_parsed_output_is_none_when_nothing_was_requested():
    chat = _chat().add_message(UserMessage("hi")).add_message(AssistantMessage("hello"))
    assert chat.parsed_output is None


def test_parsed_output_raises_when_requested_but_latest_message_has_none():
    """A structured request answered by a real tool call must fail loudly instead
    of returning None and crashing the caller later."""
    chat = (
        _chat()
        .add_message(UserMessage("count", structured_output=Answer))
        .add_message(
            ToolCallMessage([ToolCall(id="c1", name="lookup", arguments="{}")])
        )
    )
    with pytest.raises(StructuredOutputParsingError, match="requested"):
        chat.parsed_output


def test_parsed_output_parses_json_text_reply():
    chat = (
        _chat()
        .add_message(UserMessage("count", structured_output=Answer))
        .add_message(AssistantMessage('{"value": 3}', structured_output=Answer))
    )
    assert chat.parsed_output == Answer(value=3)
