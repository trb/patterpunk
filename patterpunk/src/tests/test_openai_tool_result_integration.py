"""
OpenAI/Azure OpenAI integration tests for ToolResultMessage and message history serialization.

Tests cover:
1. ToolResultMessage serialization as function_call_output in the Responses API
2. ToolCallMessage and ToolResultMessage in conversation history
3. Assistant message output_text vs input_text content type serialization

Note: This file is OpenAI-specific. See test_<provider>_tool_result_integration.py for other providers.
"""

import json
import pytest
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.chunks import TextChunk, MultimodalChunk, CacheChunk
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.models.azure_openai import AzureOpenAiModel
from tests.test_utils import get_resource


class TestMessageHistorySerialization:
    """Test that message history is correctly serialized in the Responses API."""

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4o-mini"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_assistant_message_in_history_with_text(self, model_class, model_name):
        """
        Test that assistant messages in conversation history use output_text type.

        This is the core bug fix test: assistant messages should use "output_text"
        not "input_text" when included in the input array.
        """
        # Create model - if Azure isn't configured, the error will be clear
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.1)
        else:
            model = model_class(model=model_name, temperature=0.1)

        chat = Chat(model=model)

        # First turn: Get a response from the assistant
        chat = (
            chat.add_message(SystemMessage("You are a helpful math tutor."))
            .add_message(UserMessage("What is 2 + 2?"))
            .complete()
        )

        # Verify we got a response
        assert chat.latest_message is not None
        first_response = chat.latest_message.content
        assert "4" in first_response

        # Second turn: Include the assistant's response in history and continue
        # This is where the bug would manifest - the assistant message from the
        # previous turn needs to be serialized correctly
        chat = chat.add_message(UserMessage("What about 3 + 3?")).complete()

        # Verify we got a response to the second question
        assert chat.latest_message is not None
        second_response = chat.latest_message.content
        assert "6" in second_response

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4o-mini"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_multi_turn_conversation(self, model_class, model_name):
        """Test multiple turns of conversation to ensure history is maintained correctly."""
        # Create model - if Azure isn't configured, the error will be clear
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.1)
        else:
            model = model_class(model=model_name, temperature=0.1)

        chat = Chat(model=model)

        # Setup: System message
        chat = chat.add_message(
            SystemMessage("You are a helpful assistant. Keep responses brief.")
        )

        # Turn 1
        chat = chat.add_message(UserMessage("Hello! My name is Alice.")).complete()
        assert chat.latest_message is not None

        # Turn 2 - Should remember the context
        chat = chat.add_message(
            UserMessage("What name did I just tell you?")
        ).complete()
        assert chat.latest_message is not None
        assert "alice" in chat.latest_message.content.lower()

        # Turn 3 - Continue building context
        chat = chat.add_message(UserMessage("I like pizza.")).complete()
        assert chat.latest_message is not None

        # Turn 4 - Test full context retention
        chat = chat.add_message(
            UserMessage("What's my name and what do I like?")
        ).complete()
        assert chat.latest_message is not None
        response = chat.latest_message.content.lower()
        assert "alice" in response
        assert "pizza" in response

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4o-mini"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_assistant_message_with_chunks(self, model_class, model_name):
        """Test assistant messages created with TextChunk are serialized correctly."""
        # Create model - if Azure isn't configured, the error will be clear
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.1)
        else:
            model = model_class(model=model_name, temperature=0.1)

        chat = Chat(model=model)

        # Create an assistant message with explicit chunks
        assistant_msg = AssistantMessage([TextChunk("The answer is 42.")])

        # Add to conversation and continue
        chat = (
            chat.add_message(SystemMessage("You are a helpful assistant."))
            .add_message(UserMessage("What was the answer you gave?"))
            .add_message(assistant_msg)
            .add_message(UserMessage("Can you repeat that number?"))
            .complete()
        )

        assert chat.latest_message is not None
        assert "42" in chat.latest_message.content

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4o-mini"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_assistant_message_with_cache_chunks(self, model_class, model_name):
        """Test assistant messages with CacheChunk are serialized correctly."""
        # Create model - if Azure isn't configured, the error will be clear
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.1)
        else:
            model = model_class(model=model_name, temperature=0.1)

        chat = Chat(model=model)

        # Create an assistant message with cache chunks
        assistant_msg = AssistantMessage(
            [
                TextChunk("Here is some information: "),
                CacheChunk("The capital of France is Paris.", ttl=300),
            ]
        )

        # Add to conversation and continue
        chat = (
            chat.add_message(SystemMessage("You are a helpful geography tutor."))
            .add_message(UserMessage("Tell me about France."))
            .add_message(assistant_msg)
            .add_message(UserMessage("What capital did you just mention?"))
            .complete()
        )

        assert chat.latest_message is not None
        assert "paris" in chat.latest_message.content.lower()

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4o-mini"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_tool_call_in_message_history(self, model_class, model_name):
        """
        Test that conversations with tool calls in history are serialized correctly.

        This tests the critical scenario where a tool call message appears in conversation
        history and must be properly serialized when continuing the conversation.
        """
        # Create model - if Azure isn't configured, the error will be clear
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.0)
        else:
            model = model_class(model=model_name, temperature=0.0)

        def get_weather(location: str) -> str:
            """Get the current weather for a location.

            Args:
                location: The city or location to get weather for
            """
            return f"The weather in {location} is sunny and 22°C"

        chat = Chat(model=model).with_tools([get_weather])

        # First turn: Ask about weather, model should make a tool call
        chat = (
            chat.add_message(
                SystemMessage(
                    "You are a helpful assistant that uses tools to answer questions."
                )
            )
            .add_message(UserMessage("What's the weather in Paris?"))
            .complete(
                execute_tools=False
            )  # Don't auto-execute to verify ToolCallMessage
        )

        # Verify we got a tool call
        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, ToolCallMessage)

        # Second turn: Continue conversation with tool call in history
        # This is the critical test - the ToolCallMessage should be properly serialized
        chat = chat.add_message(UserMessage("And what about London?")).complete(
            execute_tools=False
        )

        # Should get another tool call for London
        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, ToolCallMessage)

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4.1-nano"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_multimodal_in_message_history(self, model_class, model_name):
        """
        Test that conversations with multimodal content in history are serialized correctly.

        This tests image/file content appearing in conversation history and being
        properly serialized with input_image/input_file types.
        """
        # Create model - if Azure isn't configured, the error will be clear
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.1)
        else:
            model = model_class(model=model_name, temperature=0.1)

        chat = Chat(model=model)

        # First turn: Send an image
        chat = (
            chat.add_message(
                SystemMessage(
                    "You are a helpful assistant. Analyze images and answer questions."
                )
            )
            .add_message(
                UserMessage(
                    [
                        TextChunk("What animals are in this image?"),
                        MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
                    ]
                )
            )
            .complete()
        )

        # Verify we got a response
        assert chat.latest_message is not None
        first_response = chat.latest_message.content.lower()
        assert "duck" in first_response

        # Second turn: Continue conversation with multimodal message in history
        # The UserMessage with MultimodalChunk should be properly serialized
        chat = chat.add_message(
            UserMessage("How many of those animals do you see?")
        ).complete()

        # Should get a response referring to the image
        assert chat.latest_message is not None
        assert len(chat.latest_message.content) > 0

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4o-mini"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_tool_call_and_multimodal_in_message_history(self, model_class, model_name):
        """
        Test conversations with both tool calls AND multimodal content in history.

        This is the most complex real-world scenario: a conversation that includes
        both images/files and tool calls, all of which must be properly serialized.
        """
        # Create model - if Azure isn't configured, the error will be clear
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.1)
        else:
            model = model_class(model=model_name, temperature=0.1)

        def identify_animal(animal_name: str) -> str:
            """Get information about an animal.

            Args:
                animal_name: The name of the animal
            """
            animals = {
                "duck": "Ducks are waterfowl in the family Anatidae.",
                "swan": "Swans are large waterfowl with long necks.",
                "goose": "Geese are waterfowl larger than ducks.",
            }
            return animals.get(animal_name.lower(), f"Unknown animal: {animal_name}")

        chat = Chat(model=model).with_tools([identify_animal])

        # Turn 1: Send an image asking what's in it
        chat = (
            chat.add_message(
                SystemMessage(
                    "You are a wildlife expert. Analyze images and use tools to provide "
                    "detailed information about animals you see."
                )
            )
            .add_message(
                UserMessage(
                    [
                        TextChunk("What animals are in this image?"),
                        MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
                    ]
                )
            )
            .complete()
        )

        # Should respond about ducks
        assert chat.latest_message is not None

        # Turn 2: Ask for more info - should trigger tool call
        # This turn has the multimodal message in history
        chat = chat.add_message(
            UserMessage("Tell me more about those animals using your tools.")
        ).complete()

        # Should get either a tool call or a response with tool info
        assert chat.latest_message is not None

        # Turn 3: Continue conversation - now has BOTH multimodal AND tool call in history
        chat = chat.add_message(UserMessage("Are they common in parks?")).complete()

        assert chat.latest_message is not None
        # Could be either an AssistantMessage with text or another ToolCallMessage
        if isinstance(chat.latest_message, ToolCallMessage):
            assert len(chat.latest_message.tool_calls) > 0
        else:
            assert len(chat.latest_message.content) > 0


class TestComprehensiveMessageFlow:
    """Comprehensive integration test covering all message types and tool calling."""

    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (OpenAiModel, "gpt-4o-mini"),
            (AzureOpenAiModel, "gpt-4"),
        ],
    )
    def test_complete_conversation_with_all_message_types(
        self, model_class, model_name
    ):
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
        # Create model
        if model_class == AzureOpenAiModel:
            model = model_class(deployment_name=model_name, temperature=0.0)
        else:
            model = model_class(model=model_name, temperature=0.0)

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
        chat = chat.complete(
            execute_tools=False
        )  # Don't auto-execute to verify ToolCallMessage

        assert chat.latest_message is not None
        assert isinstance(
            chat.latest_message, ToolCallMessage
        ), f"Expected ToolCallMessage, got {type(chat.latest_message)}"
        assert len(chat.latest_message.tool_calls) == 1

        # Step 6: Execute tool manually
        tool_call = chat.latest_message.tool_calls[0]
        call_id = tool_call.id
        function_name = tool_call.name
        arguments = json.loads(tool_call.arguments)

        assert function_name == "analyze_image"
        assert "description" in arguments

        result = analyze_image(**arguments)

        # Step 7: Add ToolResultMessage
        from patterpunk.llm.messages.tool_result import ToolResultMessage

        chat = chat.add_message(
            ToolResultMessage(
                content=result, call_id=call_id, function_name=function_name
            )
        )

        # Step 8: Complete to get response incorporating tool result
        chat = chat.complete(execute_tools=False)

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)

        # Step 9: Ask about the unique phrase to test context retention
        chat = chat.add_message(
            UserMessage(
                "What unusual thing did you detect in the background of the image?"
            )
        ).complete(execute_tools=False)

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)

        # The LLM should mention the unique phrase from the tool result
        response_lower = chat.latest_message.content.lower()
        assert (
            "magical" in response_lower
            or "purple" in response_lower
            or "elephant" in response_lower
        ), f"Expected LLM to reference the unique phrase from tool result, got: {chat.latest_message.content}"


class TestMessageContentTypeSerialization:
    """Test the internal serialization of different message types."""

    def test_user_message_uses_input_text(self):
        """Verify user messages use input_text type in serialization."""
        model = OpenAiModel(model="gpt-4o-mini", temperature=0.1)
        user_msg = UserMessage("Hello")

        # Call the internal conversion method
        result = model._convert_messages_to_responses_input([user_msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_assistant_message_uses_output_text(self):
        """Verify assistant messages use output_text type in serialization."""
        model = OpenAiModel(model="gpt-4o-mini", temperature=0.1)
        assistant_msg = AssistantMessage("The answer is 42")

        # Call the internal conversion method
        result = model._convert_messages_to_responses_input([assistant_msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "output_text"
        assert result[0]["content"][0]["text"] == "The answer is 42"

    def test_system_message_uses_input_text(self):
        """Verify system messages (developer role) use input_text type."""
        model = OpenAiModel(model="gpt-4o-mini", temperature=0.1)
        system_msg = SystemMessage("You are a helpful assistant")

        # Call the internal conversion method
        result = model._convert_messages_to_responses_input([system_msg])

        assert len(result) == 1
        assert result[0]["role"] == "developer"
        assert result[0]["content"][0]["type"] == "input_text"

    def test_assistant_message_with_text_chunks(self):
        """Verify assistant messages with TextChunk use output_text."""
        model = OpenAiModel(model="gpt-4o-mini", temperature=0.1)
        assistant_msg = AssistantMessage([TextChunk("Part 1. "), TextChunk("Part 2.")])

        result = model._convert_messages_to_responses_input([assistant_msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # TextChunks get combined into a single content string by AssistantMessage
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "output_text"
        assert "Part 1." in result[0]["content"][0]["text"]
        assert "Part 2." in result[0]["content"][0]["text"]

    def test_user_message_with_image(self):
        """Verify user messages with images use input_image type."""
        model = OpenAiModel(model="gpt-4o-mini", temperature=0.1)
        user_msg = UserMessage(
            [
                TextChunk("Look at this image:"),
                MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
            ]
        )

        result = model._convert_messages_to_responses_input([user_msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][1]["type"] == "input_image"

    def test_assistant_message_with_multimodal_content_is_skipped(self):
        """
        Verify that multimodal content in assistant messages is skipped with warning.

        Assistant-generated images come through image_generation_call tool outputs,
        not as direct content items. If someone tries to create an AssistantMessage
        with multimodal content, it should be skipped.
        """
        model = OpenAiModel(model="gpt-4o-mini", temperature=0.1)
        assistant_msg = AssistantMessage(
            [
                TextChunk("Here is an image:"),
                MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
            ]
        )

        result = model._convert_messages_to_responses_input([assistant_msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # Should only have the text chunk, image should be skipped
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "output_text"
        assert result[0]["content"][0]["text"] == "Here is an image:"

    def test_mixed_message_history(self):
        """Test a complete conversation with mixed message types."""
        model = OpenAiModel(model="gpt-4o-mini", temperature=0.1)

        messages = [
            SystemMessage("You are helpful"),
            UserMessage("Hello"),
            AssistantMessage("Hi there!"),
            UserMessage("What's 2+2?"),
            AssistantMessage("The answer is 4."),
        ]

        result = model._convert_messages_to_responses_input(messages)

        assert len(result) == 5
        assert result[0]["role"] == "developer"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["type"] == "input_text"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"][0]["type"] == "output_text"
        assert result[3]["role"] == "user"
        assert result[3]["content"][0]["type"] == "input_text"
        assert result[4]["role"] == "assistant"
        assert result[4]["content"][0]["type"] == "output_text"
