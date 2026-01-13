import pytest
from typing import Dict, List, Optional

from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.base import Model


class MockModel(Model):
    def generate_assistant_message(self, messages, **kwargs):
        return AssistantMessage("Mock response")

    def count_tokens(self, content) -> int:
        return 0  # Mock implementation

    @staticmethod
    def get_name() -> str:
        return "mock-model"

    @staticmethod
    def get_available_models() -> List[str]:
        return ["mock-model"]


def sample_function(param: str, optional: int = 42) -> str:
    """Sample function for testing."""
    return f"Result: {param}, {optional}"


def another_function(x: float, y: Optional[str] = None) -> Dict:
    return {"x": x, "y": y}


class TestChatToolsIntegration:
    def test_with_tools_accepts_function_list(self):
        chat = Chat(model=MockModel())

        chat_with_tools = chat.with_tools([sample_function, another_function])

        assert len(chat_with_tools.tools) == 2

        tool1 = chat_with_tools.tools[0]
        assert tool1["type"] == "function"
        assert tool1["function"]["name"] == "sample_function"
        assert "Sample function for testing" in tool1["function"]["description"]

        params1 = tool1["function"]["parameters"]
        assert params1["type"] == "object"
        assert "param" in params1["properties"]
        assert "optional" in params1["properties"]
        assert params1["required"] == ["param"]

        tool2 = chat_with_tools.tools[1]
        assert tool2["function"]["name"] == "another_function"
        assert params1["required"] == ["param"]

    def test_with_tools_accepts_legacy_definitions(self):
        chat = Chat(model=MockModel())

        legacy_tools = [
            {
                "type": "function",
                "function": {
                    "name": "legacy_tool",
                    "description": "A legacy tool definition",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param": {"type": "string", "description": "A parameter"}
                        },
                        "required": ["param"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

        chat_with_tools = chat.with_tools(legacy_tools)

        assert len(chat_with_tools.tools) == 1
        assert chat_with_tools.tools[0] == legacy_tools[0]

    def test_with_tools_mixed_input_not_supported(self):

        chat = Chat(model=MockModel())

        legacy_tool = {
            "type": "function",
            "function": {
                "name": "legacy_tool",
                "description": "A legacy tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        chat.with_tools([sample_function])

        chat.with_tools([legacy_tool])

    def test_with_tools_empty_list(self):

        chat = Chat(model=MockModel())

        chat_with_tools = chat.with_tools([])

        assert chat_with_tools.tools == []

    def test_with_tools_preserves_other_attributes(self):

        chat = Chat(model=MockModel())
        chat = chat.add_message(UserMessage("Hello"))

        original_messages = chat.messages
        original_model = chat.model

        chat_with_tools = chat.with_tools([sample_function])

        assert len(chat_with_tools.messages) == len(original_messages)
        assert chat_with_tools.messages[0].content == original_messages[0].content
        assert type(chat_with_tools.model) == type(original_model)

        assert len(chat_with_tools.tools) == 1

    def test_with_tools_immutability(self):

        chat = Chat(model=MockModel())

        chat_with_tools = chat.with_tools([sample_function])

        assert chat.tools is None

        assert len(chat_with_tools.tools) == 1

        assert chat is not chat_with_tools

    def test_function_without_docstring(self):

        def no_docstring_func(x: int) -> int:
            return x * 2

        chat = Chat(model=MockModel())
        chat_with_tools = chat.with_tools([no_docstring_func])

        tool = chat_with_tools.tools[0]

        assert tool["function"]["description"] == "no_docstring_func"

    def test_function_without_annotations(self):

        def no_annotations_func(x, y=None):
            return str(x) + str(y or "")

        chat = Chat(model=MockModel())
        chat_with_tools = chat.with_tools([no_annotations_func])

        tool = chat_with_tools.tools[0]
        params = tool["function"]["parameters"]

        assert params["properties"]["x"]["type"] == "string"
        assert params["properties"]["y"]["type"] == "string"
        assert params["required"] == ["x"]
