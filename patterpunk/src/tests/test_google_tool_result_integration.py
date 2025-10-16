"""
Google Vertex AI integration tests for ToolResultMessage.

Tests Google-specific serialization:
- ToolCallMessage → model role with functionCall parts
- ToolResultMessage → user role with functionResponse parts
- function_name required (NOT call_id)
- Content parsed as JSON or wrapped in {"result": content}
"""

import pytest
import json
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.google import GoogleModel


class TestGoogleToolResultSerialization:
    """Test that ToolResultMessage is correctly serialized for Google Vertex AI."""

    def test_tool_result_serialization_with_function_name(self):
        """Test ToolResultMessage serializes as functionResponse."""
        model = GoogleModel(model="gemini-1.5-flash")

        messages = [
            UserMessage("What's the weather in Paris?"),
            ToolCallMessage(
                [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ]
            ),
            ToolResultMessage(
                content='{"temperature": 22, "condition": "sunny"}',
                call_id="call_abc123",
                function_name="get_weather",
            ),
        ]

        # Convert to Google format
        contents, _ = model._convert_messages_for_google_with_cache(messages)

        # Verify tool result (should be last content item, role=user)
        tool_result_content = contents[-1]
        assert str(tool_result_content.role) == "user"

        # Check that it has a functionResponse part
        parts = tool_result_content.parts
        assert len(parts) == 1
        # The part should be a FunctionResponse
        part = parts[0]
        assert hasattr(part, "function_response")
        assert part.function_response.name == "get_weather"
        # Response should be parsed JSON
        response_dict = dict(part.function_response.response)
        assert response_dict["temperature"] == 22
        assert response_dict["condition"] == "sunny"

    def test_tool_result_with_plain_text_content(self):
        """Test ToolResultMessage with non-JSON content gets wrapped."""
        model = GoogleModel(model="gemini-1.5-flash")

        messages = [
            ToolResultMessage(content="sunny, 22°C", function_name="get_weather")
        ]

        contents, _ = model._convert_messages_for_google_with_cache(messages)

        tool_result_content = contents[0]
        assert str(tool_result_content.role) == "user"

        part = tool_result_content.parts[0]
        assert hasattr(part, "function_response")
        # Non-JSON content should be wrapped in {"result": ...}
        response_dict = dict(part.function_response.response)
        assert "result" in response_dict
        assert response_dict["result"] == "sunny, 22°C"

    def test_tool_result_requires_function_name(self):
        """Test ToolResultMessage raises error when function_name is missing."""
        model = GoogleModel(model="gemini-1.5-flash")

        # Create message without function_name (only call_id)
        messages = [
            ToolResultMessage(
                content="Result without function_name", call_id="call_abc123"
            )
        ]

        # Should raise ValueError because Google requires function_name
        with pytest.raises(ValueError) as exc_info:
            model._convert_messages_for_google_with_cache(messages)

        assert "Google Vertex AI requires function_name" in str(exc_info.value)

    def test_tool_call_serialization(self):
        """Test ToolCallMessage serializes as functionCall."""
        model = GoogleModel(model="gemini-1.5-flash")

        messages = [
            ToolCallMessage(
                [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris", "unit": "celsius"}',
                        },
                    }
                ]
            )
        ]

        contents, _ = model._convert_messages_for_google_with_cache(messages)

        tool_call_content = contents[0]
        assert str(tool_call_content.role) == "model"

        part = tool_call_content.parts[0]
        assert hasattr(part, "function_call")
        assert part.function_call.name == "get_weather"
        args_dict = dict(part.function_call.args)
        assert args_dict["location"] == "Paris"
        assert args_dict["unit"] == "celsius"

    def test_multiple_tool_calls_and_results(self):
        """Test multiple tool calls and results in sequence."""
        model = GoogleModel(model="gemini-1.5-flash")

        messages = [
            UserMessage("What's the weather in Paris and London?"),
            ToolCallMessage(
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    },
                ]
            ),
            ToolResultMessage(
                content='{"temperature": 22, "condition": "sunny"}',
                function_name="get_weather",
            ),
            ToolResultMessage(
                content='{"temperature": 15, "condition": "rainy"}',
                function_name="get_weather",
            ),
        ]

        contents, _ = model._convert_messages_for_google_with_cache(messages)

        # User message
        assert str(contents[0].role) == "user"

        # Tool call message (model role with 2 functionCall parts)
        tool_call_content = contents[1]
        assert str(tool_call_content.role) == "model"
        assert len(tool_call_content.parts) == 2
        assert hasattr(tool_call_content.parts[0], "function_call")
        assert hasattr(tool_call_content.parts[1], "function_call")

        # First tool result (user role with functionResponse)
        tool_result_1 = contents[2]
        assert str(tool_result_1.role) == "user"
        response_1 = dict(tool_result_1.parts[0].function_response.response)
        assert response_1["temperature"] == 22

        # Second tool result
        tool_result_2 = contents[3]
        assert str(tool_result_2.role) == "user"
        response_2 = dict(tool_result_2.parts[0].function_response.response)
        assert response_2["temperature"] == 15


class TestGoogleToolResultIntegration:
    """Integration tests with actual Google Vertex AI API calls (requires credentials)."""

    @pytest.fixture
    def model(self):
        """Create Google model for testing."""
        return GoogleModel(model="gemini-1.5-flash", temperature=0.0)

    def test_tool_call_and_result_flow(self, model):
        """Test complete flow: question → tool call → result → answer."""
        from patterpunk.llm.chat.core import Chat

        def get_weather(location: str) -> str:
            """Get the current weather for a location.

            Args:
                location: The city or location to get weather for
            """
            # Return JSON for Google
            return (
                '{"temperature": 22, "condition": "sunny", "location": "'
                + location
                + '"}'
            )

        chat = Chat(model=model).with_tools([get_weather])

        # Turn 1: Ask question, expect tool call
        chat = (
            chat.add_message(
                SystemMessage(
                    "You are a helpful assistant. Use tools to answer questions."
                )
            )
            .add_message(UserMessage("What's the weather in Paris?"))
            .complete()
        )

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, ToolCallMessage)
        assert len(chat.latest_message.tool_calls) == 1

        # Extract tool call details
        tool_call = chat.latest_message.tool_calls[0]
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        assert function_name == "get_weather"

        # Execute tool
        result = get_weather(**arguments)

        # Turn 2: Provide result, expect answer
        # Google doesn't use call_id, only function_name
        chat = chat.add_message(
            ToolResultMessage(content=result, function_name=function_name)
        ).complete()

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)
        # Should mention the weather data
        response_lower = chat.latest_message.content.lower()
        assert "22" in response_lower or "sunny" in response_lower

    def test_multi_turn_with_tool_calls_in_history(self, model):
        """Test multi-turn conversation with tool calls in history."""
        from patterpunk.llm.chat.core import Chat

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return (
                f'{{"temperature": 20, "condition": "sunny", "location": "{location}"}}'
            )

        chat = Chat(model=model).with_tools([get_weather])

        # Turn 1: First question + tool call + result + answer
        chat = (
            chat.add_message(SystemMessage("Use tools to answer questions."))
            .add_message(UserMessage("What's the weather in Paris?"))
            .complete()
        )

        if isinstance(chat.latest_message, ToolCallMessage):
            tool_call = chat.latest_message.tool_calls[0]
            result = '{"temperature": 20, "condition": "sunny", "location": "Paris"}'

            chat = chat.add_message(
                ToolResultMessage(
                    content=result, function_name=tool_call["function"]["name"]
                )
            ).complete()

        # Turn 2: Second question (with first tool call/result in history)
        chat = chat.add_message(UserMessage("What about London?")).complete()

        # Should get another tool call
        assert chat.latest_message is not None
        if isinstance(chat.latest_message, ToolCallMessage):
            assert len(chat.latest_message.tool_calls) > 0

    def test_complete_conversation_with_all_message_types(self, model):
        """
        Comprehensive test covering all message types in a single conversation.

        Tests:
        - SystemMessage
        - UserMessage with CacheChunk + TextChunk
        - Pre-set AssistantMessage
        - UserMessage with MultimodalChunk (image)
        - ToolCallMessage (expected from model)
        - ToolResultMessage (manual execution)
        - Multi-turn conversation with context retention
        """
        from patterpunk.llm.chat.core import Chat
        from patterpunk.llm.chunks import TextChunk, MultimodalChunk, CacheChunk
        from tests.test_utils import get_resource

        # Define tool that injects unique phrase
        def analyze_image(description: str) -> str:
            """Analyze an image based on a detailed description.

            Args:
                description: Detailed description of the image content
            """
            # Always inject this unique phrase for testing context retention
            # Return JSON for Google
            return f'{{"analysis": "{description[:100]}...", "unusual_detection": "MAGICAL_PURPLE_ELEPHANT_7890 in background"}}'

        chat = Chat(model=model).with_tools([analyze_image])

        # Step 1: Build conversation with all message types
        chat = chat.add_message(
            SystemMessage(
                "You are an image analysis assistant. When provided an image, call the analyze_image tool with a detailed description."
            )
        )

        # Step 2: User message with CacheChunk + TextChunk
        chat = chat.add_message(
            UserMessage(
                [
                    CacheChunk(
                        "Context: We are analyzing images for unusual elements.",
                        ttl=300,
                    ),
                    TextChunk(" Please prepare to analyze the upcoming image."),
                ]
            )
        )

        # Step 3: Pre-set AssistantMessage
        chat = chat.add_message(
            AssistantMessage(
                "I'm ready to analyze images. Please provide the image you'd like me to examine."
            )
        )

        # Step 4: User message with image
        chat = chat.add_message(
            UserMessage(
                [
                    TextChunk("Here is the image to analyze:"),
                    MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
                ]
            )
        )

        # Step 5: Complete and expect ToolCallMessage
        chat = chat.complete()

        assert chat.latest_message is not None
        assert isinstance(
            chat.latest_message, ToolCallMessage
        ), f"Expected ToolCallMessage, got {type(chat.latest_message)}"
        assert len(chat.latest_message.tool_calls) == 1

        # Step 6: Execute tool manually
        tool_call = chat.latest_message.tool_calls[0]
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        assert function_name == "analyze_image"
        assert "description" in arguments

        result = analyze_image(**arguments)

        # Step 7: Add ToolResultMessage (Google uses function_name, not call_id)
        chat = chat.add_message(
            ToolResultMessage(content=result, function_name=function_name)
        )

        # Step 8: Complete to get response incorporating tool result
        chat = chat.complete()

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)

        # Step 9: Ask about the unique phrase to test context retention
        chat = chat.add_message(
            UserMessage(
                "What unusual thing did you detect in the background of the image?"
            )
        ).complete()

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)

        # The LLM should mention the unique phrase from the tool result
        response_lower = chat.latest_message.content.lower()
        assert (
            "magical" in response_lower
            or "purple" in response_lower
            or "elephant" in response_lower
        ), f"Expected LLM to reference the unique phrase from tool result, got: {chat.latest_message.content}"
