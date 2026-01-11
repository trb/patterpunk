"""
Anthropic Claude integration tests for ToolResultMessage.

Tests Anthropic-specific serialization:
- ToolCallMessage → assistant message with tool_use content blocks
- ToolResultMessage → user message with tool_result content blocks
- call_id required (as tool_use_id)
- is_error flag handling
"""

import pytest
import json
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.tool_types import ToolCall


class TestAnthropicToolResultSerialization:
    """Test that ToolResultMessage is correctly serialized for Anthropic API."""

    def test_tool_result_serialization_with_call_id(self):
        """Test ToolResultMessage serializes as user message with tool_result block."""
        model = AnthropicModel(model="claude-3-5-sonnet-20241022")

        messages = [
            SystemMessage("You are helpful"),
            UserMessage("What's the weather in Paris?"),
            ToolCallMessage(
                [
                    ToolCall(
                        id="toolu_abc123",
                        name="get_weather",
                        arguments='{"location": "Paris"}',
                    )
                ]
            ),
            ToolResultMessage(
                content="sunny, 22°C",
                call_id="toolu_abc123",
                function_name="get_weather",
            ),
        ]

        # Convert messages to Anthropic format
        anthropic_messages = model._convert_messages_for_anthropic(messages[1:])

        # Verify tool result serialization
        tool_result_msg = anthropic_messages[2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) == 1
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_abc123"
        assert tool_result_msg["content"][0]["content"] == "sunny, 22°C"
        assert tool_result_msg["content"][0]["is_error"] is False

    def test_tool_result_with_error_flag(self):
        """Test ToolResultMessage with is_error=True sets error flag."""
        model = AnthropicModel(model="claude-3-5-sonnet-20241022")

        messages = [
            ToolResultMessage(
                content="Tool execution failed: Invalid location",
                call_id="toolu_abc123",
                function_name="get_weather",
                is_error=True,
            )
        ]

        anthropic_messages = model._convert_messages_for_anthropic(messages)

        tool_result_msg = anthropic_messages[0]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["is_error"] is True
        assert "failed" in tool_result_msg["content"][0]["content"]

    def test_tool_result_requires_call_id(self):
        """Test ToolResultMessage raises error when call_id is missing."""
        model = AnthropicModel(model="claude-3-5-sonnet-20241022")

        # Create message without call_id
        messages = [
            ToolResultMessage(
                content="Result without call_id", function_name="get_weather"
            )
        ]

        # Should raise ValueError because Anthropic requires call_id
        with pytest.raises(ValueError) as exc_info:
            model._convert_messages_for_anthropic(messages)

        assert "Anthropic requires call_id" in str(exc_info.value)
        assert "tool_use_id" in str(exc_info.value)

    def test_tool_call_serialization(self):
        """Test ToolCallMessage serializes as assistant message with tool_use blocks."""
        model = AnthropicModel(model="claude-3-5-sonnet-20241022")

        messages = [
            ToolCallMessage(
                [
                    ToolCall(
                        id="toolu_abc123",
                        name="get_weather",
                        arguments='{"location": "Paris", "unit": "celsius"}',
                    )
                ]
            )
        ]

        anthropic_messages = model._convert_messages_for_anthropic(messages)

        tool_call_msg = anthropic_messages[0]
        assert tool_call_msg["role"] == "assistant"
        assert len(tool_call_msg["content"]) == 1
        assert tool_call_msg["content"][0]["type"] == "tool_use"
        assert tool_call_msg["content"][0]["id"] == "toolu_abc123"
        assert tool_call_msg["content"][0]["name"] == "get_weather"
        assert tool_call_msg["content"][0]["input"]["location"] == "Paris"
        assert tool_call_msg["content"][0]["input"]["unit"] == "celsius"

    def test_multiple_tool_calls_and_results(self):
        """Test multiple tool calls and results in sequence."""
        model = AnthropicModel(model="claude-3-5-sonnet-20241022")

        messages = [
            UserMessage("What's the weather in Paris and London?"),
            ToolCallMessage(
                [
                    ToolCall(
                        id="toolu_1",
                        name="get_weather",
                        arguments='{"location": "Paris"}',
                    ),
                    ToolCall(
                        id="toolu_2",
                        name="get_weather",
                        arguments='{"location": "London"}',
                    ),
                ]
            ),
            ToolResultMessage(
                content="sunny, 22°C", call_id="toolu_1", function_name="get_weather"
            ),
            ToolResultMessage(
                content="rainy, 15°C", call_id="toolu_2", function_name="get_weather"
            ),
        ]

        anthropic_messages = model._convert_messages_for_anthropic(messages)

        # User message
        assert anthropic_messages[0]["role"] == "user"

        # Tool call message with 2 tool_use blocks
        tool_call_msg = anthropic_messages[1]
        assert tool_call_msg["role"] == "assistant"
        assert len(tool_call_msg["content"]) == 2
        assert tool_call_msg["content"][0]["id"] == "toolu_1"
        assert tool_call_msg["content"][1]["id"] == "toolu_2"

        # First tool result
        tool_result_1 = anthropic_messages[2]
        assert tool_result_1["role"] == "user"
        assert tool_result_1["content"][0]["tool_use_id"] == "toolu_1"
        assert tool_result_1["content"][0]["content"] == "sunny, 22°C"

        # Second tool result
        tool_result_2 = anthropic_messages[3]
        assert tool_result_2["role"] == "user"
        assert tool_result_2["content"][0]["tool_use_id"] == "toolu_2"
        assert tool_result_2["content"][0]["content"] == "rainy, 15°C"


class TestAnthropicToolResultIntegration:
    """Integration tests with actual Anthropic API calls (requires API key)."""

    @pytest.fixture
    def model(self):
        """Create Anthropic model for testing - using Claude 4.5 Sonnet (regular mode)."""
        return AnthropicModel(model="claude-sonnet-4-5-20250929", temperature=0.0)

    @pytest.fixture(
        params=[
            {"use_thinking": False, "temperature": 0.0},
            {"use_thinking": True, "budget_tokens": 1500},
        ],
        ids=["regular_mode", "thinking_mode"],
    )
    def model_with_modes(self, request):
        """Create Anthropic model for testing in both regular and thinking modes."""
        from patterpunk.llm.thinking import ThinkingConfig

        config = request.param
        if config["use_thinking"]:
            return AnthropicModel(
                model="claude-sonnet-4-5-20250929",
                thinking_config=ThinkingConfig(token_budget=config["budget_tokens"]),
                max_tokens=4000,
                temperature=1.0,
            )
        else:
            return AnthropicModel(
                model="claude-sonnet-4-5-20250929",
                temperature=config["temperature"],
                max_tokens=4000,
            )

    def test_tool_call_and_result_flow(self, model):
        """Test complete flow: question → tool call → result → answer."""
        from patterpunk.llm.chat.core import Chat

        def get_weather(location: str) -> str:
            """Get the current weather for a location.

            Args:
                location: The city or location to get weather for
            """
            return f"The weather in {location} is sunny and 22°C"

        chat = Chat(model=model).with_tools([get_weather])

        # Turn 1: Ask question, expect tool call
        chat = (
            chat.add_message(
                SystemMessage(
                    "You are a helpful assistant. Use tools to answer questions."
                )
            )
            .add_message(UserMessage("What's the weather in Paris?"))
            .complete(execute_tools=False)  # Don't auto-execute to verify ToolCallMessage
        )

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, ToolCallMessage)
        assert len(chat.latest_message.tool_calls) == 1

        # Extract tool call details
        tool_call = chat.latest_message.tool_calls[0]
        call_id = tool_call.id
        function_name = tool_call.name
        arguments = json.loads(tool_call.arguments)

        assert function_name == "get_weather"
        assert "location" in arguments

        # Execute tool
        result = get_weather(**arguments)

        # Turn 2: Provide result, expect answer
        chat = chat.add_message(
            ToolResultMessage(
                content=result, call_id=call_id, function_name=function_name
            )
        ).complete(execute_tools=False)

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)
        assert "22" in chat.latest_message.content
        assert "sunny" in chat.latest_message.content.lower()

    def test_tool_call_with_error_result(self, model):
        """Test handling of tool execution errors."""
        from patterpunk.llm.chat.core import Chat

        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Weather data for {location}"

        chat = Chat(model=model).with_tools([get_weather])

        # Get a tool call
        chat = (
            chat.add_message(SystemMessage("Use tools to answer questions."))
            .add_message(UserMessage("What's the weather?"))
            .complete(execute_tools=False)  # Don't auto-execute to verify ToolCallMessage
        )

        if isinstance(chat.latest_message, ToolCallMessage):
            tool_call = chat.latest_message.tool_calls[0]

            # Provide error result
            chat = chat.add_message(
                ToolResultMessage(
                    content="Error: Invalid location provided",
                    call_id=tool_call.id,
                    function_name=tool_call.name,
                    is_error=True,
                )
            ).complete(execute_tools=False)

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
            .complete(execute_tools=False)  # Don't auto-execute to verify ToolCallMessage
        )

        if isinstance(chat.latest_message, ToolCallMessage):
            tool_call = chat.latest_message.tool_calls[0]
            result = "Weather in Paris: sunny, 20°C"

            chat = chat.add_message(
                ToolResultMessage(
                    content=result,
                    call_id=tool_call.id,
                    function_name=tool_call.name,
                )
            ).complete(execute_tools=False)

        # Turn 2: Second question (with first tool call/result in history)
        chat = chat.add_message(UserMessage("What about London?")).complete(execute_tools=False)

        # Should get another tool call
        assert chat.latest_message is not None
        if isinstance(chat.latest_message, ToolCallMessage):
            assert len(chat.latest_message.tool_calls) > 0

    def test_complete_conversation_with_all_message_types(self, model_with_modes):
        """
        Comprehensive test covering all message types in a single conversation.

        This test runs in both regular mode and thinking mode to verify that
        thinking blocks are properly preserved in multi-turn conversations.

        Tests:
        - SystemMessage
        - UserMessage with CacheChunk + TextChunk
        - Pre-set AssistantMessage
        - UserMessage with MultimodalChunk (image)
        - ToolCallMessage (expected from model)
        - ToolResultMessage (manual execution)
        - Multi-turn conversation with context retention
        - Thinking block preservation (when in thinking mode)
        """
        model = model_with_modes
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
            return f"Image analysis: {description[:100]}... MAGICAL_PURPLE_ELEPHANT_7890 detected in background."

        chat = Chat(model=model).with_tools([analyze_image])

        # Step 1: Build conversation with all message types
        chat = chat.add_message(
            SystemMessage(
                "You are an image analysis assistant. You have access to the analyze_image tool to help analyze images. "
                "When answering questions about images you've already analyzed, use the tool results from your previous analysis."
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
        chat = chat.complete(execute_tools=False)  # Don't auto-execute to verify ToolCallMessage

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
        # This tests that the model can read and reference previous tool results
        chat = chat.add_message(
            UserMessage(
                "Based on your analysis, what unusual thing did you find in the background?"
            )
        ).complete(execute_tools=False)

        assert chat.latest_message is not None
        assert isinstance(
            chat.latest_message, AssistantMessage
        ), f"Expected AssistantMessage using previous tool result, got {type(chat.latest_message).__name__}. The model should answer from previous tool results, not call tools again."

        # The LLM should mention the unique phrase from the tool result
        # This verifies the model can read and reference previous tool results
        response_lower = chat.latest_message.content.lower()
        assert (
            "magical" in response_lower
            or "purple" in response_lower
            or "elephant" in response_lower
        ), f"Expected LLM to reference the unique phrase (MAGICAL_PURPLE_ELEPHANT_7890) from the previous tool result. This indicates the model is NOT reading previous tool results. Got: {chat.latest_message.content}"


class TestClaude45Models:
    """Test Claude 4.5 models (Haiku and Sonnet) in both reasoning and regular modes."""

    @pytest.mark.parametrize(
        "model_name,use_thinking,expected_temp",
        [
            # Claude 4.5 Haiku - Regular mode with temperature
            ("claude-haiku-4-5-20251001", False, 0.7),
            # Claude 4.5 Haiku - Reasoning mode
            ("claude-haiku-4-5-20251001", True, 1.0),
            # Claude 4.5 Sonnet - Regular mode with temperature
            ("claude-sonnet-4-5-20250929", False, 0.7),
            # Claude 4.5 Sonnet - Reasoning mode
            ("claude-sonnet-4-5-20250929", True, 1.0),
        ],
    )
    def test_claude_45_basic_completion(
        self, model_name: str, use_thinking: bool, expected_temp: float
    ):
        """Test basic completion for Claude 4.5 models in reasoning and regular modes."""
        from patterpunk.llm.chat.core import Chat
        from patterpunk.llm.thinking import ThinkingConfig

        # Create model with appropriate configuration
        if use_thinking:
            model = AnthropicModel(
                model=model_name,
                thinking_config=ThinkingConfig(token_budget=2000),
                max_tokens=4000,
            )
            # Verify reasoning mode is detected
            assert model._is_reasoning_model() is True
            assert model.thinking.budget_tokens == 2000
        else:
            model = AnthropicModel(
                model=model_name, temperature=expected_temp, max_tokens=4000
            )
            # Verify reasoning mode is detected even without thinking config
            assert model._is_reasoning_model() is True

        chat = Chat(model=model)

        response = (
            chat.add_message(
                SystemMessage("You are a helpful assistant. Answer concisely.")
            )
            .add_message(UserMessage("What is 7 times 8? Just give the number."))
            .complete()
        )

        assert response.latest_message is not None
        assert isinstance(response.latest_message, AssistantMessage)
        content = response.latest_message.content
        assert "56" in content, f"Expected '56' in response, got: {content}"

    @pytest.mark.parametrize(
        "model_name,use_thinking",
        [
            ("claude-haiku-4-5-20251001", False),
            ("claude-sonnet-4-5-20250929", False),
        ],
    )
    def test_claude_45_tool_calling(self, model_name: str, use_thinking: bool):
        """Test tool calling with Claude 4.5 models in regular mode.

        Note: Tool calling with thinking mode has special message structure requirements
        in the Anthropic API and is tested separately in the main test_anthropic.py file.
        """
        from patterpunk.llm.chat.core import Chat

        def calculate_sum(a: int, b: int) -> str:
            """Calculate the sum of two numbers.

            Args:
                a: First number
                b: Second number
            """
            return f"The sum of {a} and {b} is {a + b}"

        model = AnthropicModel(model=model_name, temperature=0.7, max_tokens=4000)
        chat = Chat(model=model).with_tools([calculate_sum])

        # Turn 1: Ask question, expect tool call
        chat = (
            chat.add_message(
                SystemMessage(
                    "You are a helpful assistant. Use the calculate_sum tool to answer math questions."
                )
            )
            .add_message(UserMessage("What is 15 plus 27?"))
            .complete(execute_tools=False)  # Don't auto-execute to verify ToolCallMessage
        )

        assert chat.latest_message is not None
        assert isinstance(
            chat.latest_message, ToolCallMessage
        ), f"Expected ToolCallMessage, got {type(chat.latest_message).__name__}"

        # Extract and execute tool call
        tool_call = chat.latest_message.tool_calls[0]
        call_id = tool_call.id
        function_name = tool_call.name
        arguments = json.loads(tool_call.arguments)

        assert function_name == "calculate_sum"
        result = calculate_sum(**arguments)

        # Turn 2: Provide result, expect answer
        chat = chat.add_message(
            ToolResultMessage(
                content=result, call_id=call_id, function_name=function_name
            )
        ).complete(execute_tools=False)

        assert chat.latest_message is not None
        assert isinstance(chat.latest_message, AssistantMessage)
        assert "42" in chat.latest_message.content

    @pytest.mark.parametrize(
        "model_name,use_thinking",
        [
            ("claude-haiku-4-5-20251001", False),
            ("claude-haiku-4-5-20251001", True),
            ("claude-sonnet-4-5-20250929", False),
            ("claude-sonnet-4-5-20250929", True),
        ],
    )
    def test_claude_45_structured_output(self, model_name: str, use_thinking: bool):
        """Test structured output with Claude 4.5 models in both modes."""
        from patterpunk.llm.chat.core import Chat
        from patterpunk.llm.thinking import ThinkingConfig
        from pydantic import BaseModel, Field

        class MathProblem(BaseModel):
            problem: str = Field(description="The math problem being solved")
            answer: int = Field(description="The numerical answer")
            explanation: str = Field(description="Brief explanation of the solution")

        # Create model with appropriate configuration
        if use_thinking:
            model = AnthropicModel(
                model=model_name,
                thinking_config=ThinkingConfig(token_budget=2000),
                max_tokens=4000,
            )
        else:
            model = AnthropicModel(model=model_name, temperature=0.5, max_tokens=4000)

        chat = Chat(model=model)

        response = (
            chat.add_message(SystemMessage("You are a math tutor."))
            .add_message(
                UserMessage(
                    "Solve: What is 12 multiplied by 5?",
                    structured_output=MathProblem,
                )
            )
            .complete()
        )

        parsed = response.parsed_output
        assert parsed is not None
        assert isinstance(parsed, MathProblem)
        assert parsed.answer == 60
        assert parsed.problem
        assert parsed.explanation

    @pytest.mark.parametrize(
        "model_name,use_thinking",
        [
            ("claude-haiku-4-5-20251001", False),
            ("claude-haiku-4-5-20251001", True),
            ("claude-sonnet-4-5-20250929", False),
            ("claude-sonnet-4-5-20250929", True),
        ],
    )
    def test_claude_45_temperature_compatibility(
        self, model_name: str, use_thinking: bool
    ):
        """Test that temperature parameters are handled correctly for Claude 4.5."""
        from patterpunk.llm.thinking import ThinkingConfig

        if use_thinking:
            # With thinking mode: temperature should be forced to 1.0, top_p/top_k removed
            model = AnthropicModel(
                model=model_name,
                thinking_config=ThinkingConfig(token_budget=2000),
                temperature=0.5,
                top_p=0.9,
                top_k=40,
            )

            api_params = {
                "model": model_name,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1000,
            }

            filtered = model._get_compatible_params(api_params)
            assert filtered["temperature"] == 1.0
            assert "top_p" not in filtered
            assert "top_k" not in filtered
        else:
            # Without thinking mode: Claude 4+ can't use both temperature and top_p
            # Should keep temperature, remove top_p if both specified
            model = AnthropicModel(
                model=model_name, temperature=0.7, top_p=1.0, top_k=40
            )

            api_params = {
                "model": model_name,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 40,
                "max_tokens": 1000,
            }

            filtered = model._get_compatible_params(api_params)
            # When top_p is 1.0 (default), it should be removed
            assert "top_p" not in filtered
            assert "top_k" not in filtered
            assert filtered["temperature"] == 0.7
