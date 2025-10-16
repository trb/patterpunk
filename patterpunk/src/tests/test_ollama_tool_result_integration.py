"""
Ollama integration tests for ToolResultMessage.

Tests Ollama-specific serialization (OpenAI-compatible):
- ToolCallMessage → assistant message with tool_calls
- ToolResultMessage → tool message with tool_call_id
- call_id required
- OpenAI-compatible format

Note: These tests require a running Ollama instance and cannot be run in CI.
"""

import pytest
import json
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.chunks import TextChunk, MultimodalChunk, CacheChunk
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.ollama import OllamaModel
from tests.test_utils import get_resource


@pytest.mark.skip(reason="Requires running Ollama instance - run manually")
class TestOllamaToolResultIntegration:
    """Integration tests with actual Ollama API calls (requires Ollama running)."""

    @pytest.fixture
    def model(self):
        """Create Ollama model for testing."""
        return OllamaModel(model="llama3.2-vision", temperature=0.0)

    def test_tool_call_and_result_flow(self, model):
        """Test complete flow: question → tool call → result → answer."""
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
