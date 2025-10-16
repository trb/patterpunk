"""
AWS Bedrock integration tests for ToolResultMessage.

Tests Bedrock-specific serialization (Claude models):
- ToolCallMessage → assistant message with toolUse content blocks
- ToolResultMessage → user message with toolResult content blocks
- call_id required (as toolUseId)
- status field (success/error) instead of is_error boolean
"""

import pytest
import json
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.bedrock import BedrockModel


class TestBedrockToolResultSerialization:
    """Test that ToolResultMessage is correctly serialized for AWS Bedrock."""

    def test_tool_result_serialization_with_call_id(self):
        """Test ToolResultMessage serializes as user message with toolResult block."""
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

        messages = [
            UserMessage("What's the weather in Paris?"),
            ToolCallMessage(
                [
                    {
                        "id": "tooluse_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}'
                        }
                    }
                ]
            ),
            ToolResultMessage(
                content="sunny, 22°C",
                call_id="tooluse_abc123",
                function_name="get_weather"
            )
        ]

        # Convert messages to Bedrock format
        bedrock_messages = model._convert_messages_for_bedrock(messages)

        # Verify tool result serialization
        tool_result_msg = bedrock_messages[2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) == 1
        assert "toolResult" in tool_result_msg["content"][0]

        tool_result = tool_result_msg["content"][0]["toolResult"]
        assert tool_result["toolUseId"] == "tooluse_abc123"
        assert tool_result["content"][0]["text"] == "sunny, 22°C"
        assert tool_result["status"] == "success"

    def test_tool_result_with_error_flag(self):
        """Test ToolResultMessage with is_error=True sets status to 'error'."""
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

        messages = [
            ToolResultMessage(
                content="Tool execution failed: Invalid location",
                call_id="tooluse_abc123",
                function_name="get_weather",
                is_error=True
            )
        ]

        bedrock_messages = model._convert_messages_for_bedrock(messages)

        tool_result_msg = bedrock_messages[0]
        assert tool_result_msg["role"] == "user"
        tool_result = tool_result_msg["content"][0]["toolResult"]
        assert tool_result["status"] == "error"
        assert "failed" in tool_result["content"][0]["text"]

    def test_tool_result_requires_call_id(self):
        """Test ToolResultMessage raises error when call_id is missing."""
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

        # Create message without call_id
        messages = [
            ToolResultMessage(
                content="Result without call_id",
                function_name="get_weather"
            )
        ]

        # Should raise ValueError because Bedrock requires call_id
        with pytest.raises(ValueError) as exc_info:
            model._convert_messages_for_bedrock(messages)

        assert "AWS Bedrock requires call_id" in str(exc_info.value)
        assert "toolUseId" in str(exc_info.value)

    def test_tool_call_serialization(self):
        """Test ToolCallMessage serializes as assistant message with toolUse blocks."""
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

        messages = [
            ToolCallMessage(
                [
                    {
                        "id": "tooluse_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris", "unit": "celsius"}'
                        }
                    }
                ]
            )
        ]

        bedrock_messages = model._convert_messages_for_bedrock(messages)

        tool_call_msg = bedrock_messages[0]
        assert tool_call_msg["role"] == "assistant"
        assert len(tool_call_msg["content"]) == 1
        assert "toolUse" in tool_call_msg["content"][0]

        tool_use = tool_call_msg["content"][0]["toolUse"]
        assert tool_use["toolUseId"] == "tooluse_abc123"
        assert tool_use["name"] == "get_weather"
        assert tool_use["input"]["location"] == "Paris"
        assert tool_use["input"]["unit"] == "celsius"

    def test_multiple_tool_calls_and_results(self):
        """Test multiple tool calls and results in sequence."""
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

        messages = [
            UserMessage("What's the weather in Paris and London?"),
            ToolCallMessage(
                [
                    {
                        "id": "tooluse_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}'
                        }
                    },
                    {
                        "id": "tooluse_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}'
                        }
                    }
                ]
            ),
            ToolResultMessage(
                content="sunny, 22°C",
                call_id="tooluse_1",
                function_name="get_weather"
            ),
            ToolResultMessage(
                content="rainy, 15°C",
                call_id="tooluse_2",
                function_name="get_weather"
            )
        ]

        bedrock_messages = model._convert_messages_for_bedrock(messages)

        # User message
        assert bedrock_messages[0]["role"] == "user"

        # Tool call message with 2 toolUse blocks
        tool_call_msg = bedrock_messages[1]
        assert tool_call_msg["role"] == "assistant"
        assert len(tool_call_msg["content"]) == 2
        assert tool_call_msg["content"][0]["toolUse"]["toolUseId"] == "tooluse_1"
        assert tool_call_msg["content"][1]["toolUse"]["toolUseId"] == "tooluse_2"

        # First tool result
        tool_result_1 = bedrock_messages[2]
        assert tool_result_1["role"] == "user"
        assert tool_result_1["content"][0]["toolResult"]["toolUseId"] == "tooluse_1"
        assert tool_result_1["content"][0]["toolResult"]["content"][0]["text"] == "sunny, 22°C"

        # Second tool result
        tool_result_2 = bedrock_messages[3]
        assert tool_result_2["role"] == "user"
        assert tool_result_2["content"][0]["toolResult"]["toolUseId"] == "tooluse_2"
        assert tool_result_2["content"][0]["toolResult"]["content"][0]["text"] == "rainy, 15°C"


class TestBedrockToolResultIntegration:
    """Integration tests with actual AWS Bedrock API calls (requires AWS credentials)."""

    @pytest.fixture
    def model(self):
        """Create Bedrock model for testing."""
        return BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.0)

    def test_tool_call_and_result_flow(self, model):
        """Test complete flow: question → tool call → result → answer."""
        from patterpunk.llm.chat.core import Chat
        from patterpunk.llm.messages.assistant import AssistantMessage

        def get_weather(location: str) -> str:
            """Get the current weather for a location.

            Args:
                location: The city or location to get weather for
            """
            return f"The weather in {location} is sunny and 22°C"

        chat = Chat(model=model).with_tools([get_weather])

        # Turn 1: Ask question, expect tool call
        chat = (
            chat.add_message(SystemMessage("You are a helpful assistant. Use tools to answer questions."))
            .add_message(UserMessage("What's the weather in Paris?"))
            .complete()
        )

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, ToolCallMessage)
        assert len(chat.latest_message.tool_calls) == 1

        # Extract tool call details
        tool_call = chat.latest_message.tool_calls[0]
        call_id = tool_call["id"]
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        assert function_name == "get_weather"
        assert "location" in arguments

        # Execute tool
        result = get_weather(**arguments)

        # Turn 2: Provide result, expect answer
        chat = chat.add_message(
            ToolResultMessage(
                content=result,
                call_id=call_id,
                function_name=function_name
            )
        ).complete()

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)
        assert "22" in chat.latest_message.content
        assert "sunny" in chat.latest_message.content.lower()

    def test_tool_call_with_error_result(self, model):
        """Test handling of tool execution errors."""
        from patterpunk.llm.chat.core import Chat
        from patterpunk.llm.messages.assistant import AssistantMessage

        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Weather data for {location}"

        chat = Chat(model=model).with_tools([get_weather])

        # Get a tool call
        chat = (
            chat.add_message(SystemMessage("Use tools to answer questions."))
            .add_message(UserMessage("What's the weather?"))
            .complete()
        )

        if isinstance(chat.latest_message, ToolCallMessage):
            tool_call = chat.latest_message.tool_calls[0]

            # Provide error result
            chat = chat.add_message(
                ToolResultMessage(
                    content="Error: Invalid location provided",
                    call_id=tool_call["id"],
                    function_name=tool_call["function"]["name"],
                    is_error=True
                )
            ).complete()

            # Model should handle the error gracefully
            assert chat.latest_message is not None
            assert isinstance(chat.latest_message, AssistantMessage)

    def test_multi_turn_with_tool_calls_in_history(self, model):
        """Test multi-turn conversation with tool calls in history."""
        from patterpunk.llm.chat.core import Chat

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: sunny, 20°C"

        chat = Chat(model=model).with_tools([get_weather])

        # Turn 1: First question + tool call + result + answer
        chat = (
            chat.add_message(SystemMessage("Use tools to answer questions."))
            .add_message(UserMessage("What's the weather in Paris?"))
            .complete()
        )

        if isinstance(chat.latest_message, ToolCallMessage):
            tool_call = chat.latest_message.tool_calls[0]
            result = "Weather in Paris: sunny, 20°C"

            chat = chat.add_message(
                ToolResultMessage(
                    content=result,
                    call_id=tool_call["id"],
                    function_name=tool_call["function"]["name"]
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
        from patterpunk.llm.messages.assistant import AssistantMessage
        from tests.test_utils import get_resource

        # Define tool that injects unique phrase
        def analyze_image(description: str) -> str:
            """Analyze an image based on a detailed description.

            Args:
                description: Detailed description of the image content
            """
            # Always inject this unique phrase for testing context retention
            return f"Image analysis: {description[:100]}... MAGICAL_PURPLE_ELEPHANT_7890 detected in background."

        chat = Chat(model=model).with_tools([analyze_image])

        # Step 1: Build conversation with all message types
        chat = chat.add_message(
            SystemMessage("You are an image analysis assistant. When provided an image, call the analyze_image tool with a detailed description.")
        )

        # Step 2: User message with CacheChunk + TextChunk
        chat = chat.add_message(
            UserMessage([
                CacheChunk("Context: We are analyzing images for unusual elements.", ttl=300),
                TextChunk(" Please prepare to analyze the upcoming image.")
            ])
        )

        # Step 3: Pre-set AssistantMessage
        chat = chat.add_message(
            AssistantMessage("I'm ready to analyze images. Please provide the image you'd like me to examine.")
        )

        # Step 4: User message with image
        chat = chat.add_message(
            UserMessage([
                TextChunk("Here is the image to analyze:"),
                MultimodalChunk.from_file(get_resource("ducks_pond.jpg"))
            ])
        )

        # Step 5: Complete and expect ToolCallMessage
        chat = chat.complete()

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, ToolCallMessage), \
            f"Expected ToolCallMessage, got {type(chat.latest_message)}"
        assert len(chat.latest_message.tool_calls) == 1

        # Step 6: Execute tool manually
        tool_call = chat.latest_message.tool_calls[0]
        call_id = tool_call["id"]
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        assert function_name == "analyze_image"
        assert "description" in arguments

        result = analyze_image(**arguments)

        # Step 7: Add ToolResultMessage
        chat = chat.add_message(
            ToolResultMessage(
                content=result,
                call_id=call_id,
                function_name=function_name
            )
        )

        # Step 8: Complete to get response incorporating tool result
        chat = chat.complete()

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)

        # Step 9: Ask about the unique phrase to test context retention
        chat = chat.add_message(
            UserMessage("What unusual thing did you detect in the background of the image?")
        ).complete()

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)

        # The LLM should mention the unique phrase from the tool result
        response_lower = chat.latest_message.content.lower()
        assert "magical" in response_lower or "purple" in response_lower or "elephant" in response_lower, \
            f"Expected LLM to reference the unique phrase from tool result, got: {chat.latest_message.content}"
