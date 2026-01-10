"""
Tests for async streaming support with Google Vertex AI.

These tests validate the complete_stream() interface and streaming behavior
with the Google Vertex AI provider. They test content streaming and tool calling
during streaming.
"""

from typing import Optional

import pytest
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.models.google import GoogleModel, ToolResultBatch
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.streaming import StreamIncompleteError, ToolExecutionAbortError
from patterpunk.llm.messages.tool_call import ToolCallMessage


# Use a flash model for faster tests
GOOGLE_TEST_MODEL = "gemini-2.5-flash"
GOOGLE_TEST_LOCATION = (
    "us-central1"  # Iowa - Canada regions no longer support Gemini models
)


@pytest.fixture(autouse=True)
def reset_google_client():
    """Reset the shared GoogleModel client before each test.

    The GoogleModel uses a class-level shared client. When running multiple async
    tests, the event loop closes between tests but the httpx client inside the
    Google genai client becomes stale. Resetting forces a fresh client per test.
    """
    GoogleModel.client = None
    yield
    GoogleModel.client = None


def get_test_model(
    max_tokens: int = 256,
    temperature: float = 0.0,
    thinking_config: Optional[ThinkingConfig] = None,
) -> GoogleModel:
    """Create a GoogleModel for testing."""
    return GoogleModel(
        model=GOOGLE_TEST_MODEL,
        location=GOOGLE_TEST_LOCATION,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_config=thinking_config,
    )


# Tool functions for testing
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        A string describing the current weather
    """
    return f"The weather in {location} is sunny and 72F."


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


def failing_tool(message: str) -> str:
    """A tool that always fails.

    Args:
        message: A message to include in the error

    Returns:
        Never returns, always raises
    """
    raise ValueError(f"Tool failed: {message}")


def abort_tool(reason: str) -> str:
    """A tool that aborts the stream.

    Args:
        reason: The reason for aborting

    Returns:
        Never returns, always aborts
    """
    raise ToolExecutionAbortError(f"Aborting: {reason}")


@pytest.mark.asyncio
async def test_stream_content_basic():
    """
    Test basic content streaming without thinking blocks.

    Validates that:
    1. complete_stream() returns an async context manager
    2. stream.content yields accumulated content strings
    3. stream.chat is available after context exit with complete response

    Note: Google may return content in a single chunk for short responses,
    so we only require at least 1 iteration (not multiple).
    """
    chat = Chat(model=get_test_model(max_tokens=256))

    chat = chat.add_message(
        SystemMessage("You are a helpful assistant. Be brief.")
    ).add_message(UserMessage("Say hello in exactly 3 words."))

    accumulated_content = ""
    iteration_count = 0
    async with chat.complete_stream() as stream:
        async for content in stream.content:
            iteration_count += 1
            # Each yield should be the accumulated content so far
            assert len(content) >= len(accumulated_content)
            accumulated_content = content

    # Verify at least some content was received
    assert (
        iteration_count >= 1
    ), f"Expected at least one streaming iteration, got {iteration_count}"

    # After context exit, await stream.chat to get the final result
    final_chat = await stream.chat

    # Final chat should have the response
    assert final_chat is not None
    assert final_chat.latest_message is not None
    assert final_chat.latest_message.content is not None
    assert len(final_chat.latest_message.content) > 0

    # The accumulated content should match the final message
    assert accumulated_content == final_chat.latest_message.content


@pytest.mark.asyncio
async def test_stream_chat_cancelled_raises():
    """
    Test that awaiting stream.chat after cancellation raises StreamIncompleteError.

    When the user exits the stream early (e.g., break from loop), the stream is
    cancelled and there's no complete chat to return.

    Note: Google may return all content in a single chunk for short responses,
    so this test may not be able to cancel the stream if the response completes
    before we can break. We track which path was taken to ensure at least one
    code path is always verified.
    """
    chat = Chat(model=get_test_model(max_tokens=1024))

    chat = chat.add_message(SystemMessage("Be very verbose and detailed.")).add_message(
        UserMessage(
            "Write a very detailed story about a journey from 1 to 100. "
            "Describe each number in multiple sentences with creative details. "
            "Make it at least 500 words long."
        )
    )

    was_cancelled = False
    iteration_count = 0
    async with chat.complete_stream() as stream:
        # Start iterating but try to exit early on first iteration
        # Breaking on count >= 1 increases chance of actually cancelling
        async for _ in stream.content:
            iteration_count += 1
            if iteration_count >= 1:
                was_cancelled = True
                break  # Exit early - this should cancel the stream

    # At least one of these paths must execute and pass
    if was_cancelled:
        # If we broke early, await stream.chat should raise
        with pytest.raises(StreamIncompleteError) as exc_info:
            _ = await stream.chat
        assert "cancelled" in str(exc_info.value).lower()
    else:
        # If the stream completed before we could cancel (iteration_count == 0),
        # that's still valid - verify the stream completed successfully
        final_chat = await stream.chat
        assert final_chat is not None
        # Log for debugging: this path means Google returned 0 chunks before completion
        # which is unusual but technically possible


@pytest.mark.asyncio
async def test_stream_chat_waits_for_completion():
    """
    Test that await stream.chat waits for completion if called during iteration.

    The await is non-blocking if already complete, or waits if still in progress.
    """
    chat = Chat(model=get_test_model(max_tokens=100))

    chat = chat.add_message(SystemMessage("Be brief.")).add_message(
        UserMessage("Say hi")
    )

    async with chat.complete_stream() as stream:
        # Drain the stream normally
        async for _ in stream.content:
            pass

    # After complete iteration, await should return immediately
    final_chat = await stream.chat
    assert final_chat is not None
    assert final_chat.latest_message is not None


@pytest.mark.asyncio
async def test_stream_delta_iterators():
    """
    Test the delta iterators (content_delta vs content).

    Delta iterators yield only new text, not accumulated text.
    Useful for manual accumulation or logging.

    Note: Google may return content in a single chunk for short responses.
    """
    chat = Chat(model=get_test_model(max_tokens=256))

    chat = chat.add_message(SystemMessage("You are a helpful assistant.")).add_message(
        UserMessage("Count from 1 to 5, one number per line.")
    )

    deltas = []
    async with chat.complete_stream() as stream:
        async for delta in stream.content_delta:
            deltas.append(delta)

    # We should have received at least one delta
    assert len(deltas) >= 1

    # Joining deltas should give us the full content
    full_content = "".join(deltas)
    final_chat = await stream.chat
    assert full_content == final_chat.latest_message.content


@pytest.mark.asyncio
async def test_stream_with_function_tool_call():
    """
    Test that function tools are automatically called during streaming.

    The model should:
    1. Call the get_weather tool
    2. Receive the result
    3. Continue generating a response with the weather info
    """
    chat = Chat(model=get_test_model(max_tokens=256)).with_tools([get_weather])

    chat = chat.add_message(
        SystemMessage(
            "You are a helpful assistant. Use the get_weather tool when asked about weather."
        )
    ).add_message(UserMessage("What's the weather like in Paris?"))

    accumulated_content = ""
    content_iterations = 0

    async with chat.complete_stream() as stream:
        async for content in stream.content:
            content_iterations += 1
            accumulated_content = content

    # Verify streaming occurred - tool calling should produce multiple iterations
    # (tool call chunk + response chunk at minimum)
    assert (
        content_iterations > 1
    ), f"Expected multiple iterations for tool calling, got {content_iterations}"

    # The final response should mention the weather from our tool
    # Tool returns "sunny and 72F", so both should be present
    final_chat = await stream.chat
    assert "sunny" in accumulated_content.lower() and "72" in accumulated_content
    assert final_chat.latest_message.content == accumulated_content


@pytest.mark.asyncio
async def test_stream_tool_error_continues():
    """
    Test that normal tool exceptions are sent to the model as errors.

    When a tool raises a regular exception:
    1. A ToolResultMessage with is_error=True is created
    2. The model receives the error and can respond appropriately
    3. Streaming continues
    """
    chat = Chat(model=get_test_model(max_tokens=256)).with_tools([failing_tool])

    chat = chat.add_message(
        SystemMessage(
            "You are a helpful assistant. Use tools when asked. If a tool fails, apologize and explain."
        )
    ).add_message(UserMessage("Please use the failing_tool with message 'test error'"))

    accumulated_content = ""

    async with chat.complete_stream() as stream:
        async for content in stream.content:
            accumulated_content = content

    # The model should have received the error and responded
    final_chat = await stream.chat
    # Model should acknowledge the failure somehow (apologize, mention error, etc.)
    response_lower = accumulated_content.lower()
    assert any(
        word in response_lower
        for word in ["error", "fail", "sorry", "unable", "couldn't", "cannot"]
    )


@pytest.mark.asyncio
async def test_stream_tool_abort_stops():
    """
    Test that ToolExecutionAbortError stops the stream immediately.

    When a tool raises ToolExecutionAbortError:
    1. The stream stops immediately
    2. The exception propagates to the caller
    3. No further content is yielded
    """
    chat = Chat(model=get_test_model(max_tokens=256)).with_tools([abort_tool])

    chat = chat.add_message(
        SystemMessage("You are a helpful assistant. Use the abort_tool when asked.")
    ).add_message(
        UserMessage("Please use the abort_tool with reason 'user requested abort'")
    )

    with pytest.raises(ToolExecutionAbortError) as exc_info:
        async with chat.complete_stream() as stream:
            async for content in stream.content:
                pass

    assert "user requested abort" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_multiple_tool_rounds():
    """
    Test that multiple rounds of tool calling work seamlessly.

    The model should be able to:
    1. Call one tool, get result
    2. Call another tool (or same tool again), get result
    3. Finally generate a response using all tool results
    """
    chat = Chat(model=get_test_model(max_tokens=512)).with_tools(
        [get_weather, calculate_sum]
    )

    chat = chat.add_message(
        SystemMessage(
            "You are a helpful assistant. When asked about weather and math together, "
            "use both the get_weather tool and calculate_sum tool to answer."
        )
    ).add_message(
        UserMessage(
            "What's the weather in Tokyo, and also what is 15 + 27? "
            "Please tell me both answers in your response."
        )
    )

    accumulated_content = ""

    async with chat.complete_stream() as stream:
        async for content in stream.content:
            accumulated_content = content

    # Response should contain results from both tools
    # Weather tool returns "sunny and 72F" for any location - both should be present
    # Calculate sum returns 42 for 15 + 27
    assert "sunny" in accumulated_content.lower() and "72" in accumulated_content
    assert "42" in accumulated_content


@pytest.mark.asyncio
async def test_stream_thinking_and_content():
    """
    Test streaming with thinking blocks enabled (reasoning model).

    Validates that:
    1. Thinking content streams first in the thinking phase
    2. Content streams second in the content phase
    3. Both thinking and content are captured in the final message

    Note: Google's thought summaries require includeThoughts=True and a prompt
    complex enough to trigger extended reasoning.
    """
    chat = Chat(
        model=get_test_model(
            max_tokens=8000,
            temperature=1.0,
            thinking_config=ThinkingConfig(token_budget=8000, include_thoughts=True),
        )
    )

    # Use a brain teaser that requires extended reasoning to trigger thought summaries
    chat = chat.add_message(
        SystemMessage("You are a helpful puzzle solver. Show your complete reasoning.")
    ).add_message(
        UserMessage(
            "A farmer needs to cross a river with a wolf, a goat, and a cabbage. "
            "The boat can only carry the farmer and one item at a time. "
            "If left alone together, the wolf will eat the goat, and the goat will eat the cabbage. "
            "How can the farmer get everything across safely?"
        )
    )

    accumulated_thinking = ""
    accumulated_content = ""
    thinking_iterations = 0
    content_iterations = 0

    async with chat.complete_stream() as stream:
        # First iterate through thinking
        async for thinking in stream.thinking:
            thinking_iterations += 1
            # Each yield should be accumulated
            assert len(thinking) >= len(accumulated_thinking)
            accumulated_thinking = thinking

        # Then iterate through content
        async for content in stream.content:
            content_iterations += 1
            assert len(content) >= len(accumulated_content)
            accumulated_content = content

    # Verify streaming occurred (Google may batch more aggressively)
    assert (
        thinking_iterations >= 1
    ), f"Expected at least one thinking iteration, got {thinking_iterations}"
    assert (
        content_iterations >= 1
    ), f"Expected at least one content iteration, got {content_iterations}"

    # Verify final chat
    final_chat = await stream.chat
    assert final_chat is not None
    assert final_chat.latest_message is not None

    # Content should contain part of the solution (goat is taken first)
    assert "goat" in accumulated_content.lower()

    # Thinking should have been generated and captured
    assert accumulated_thinking, "Expected thinking content to be generated"
    assert final_chat.latest_message.has_thinking
    assert len(final_chat.latest_message.thinking_blocks) > 0


@pytest.mark.asyncio
async def test_stream_content_only_auto_drains_thinking():
    """
    Test that iterating content without thinking auto-drains thinking.

    When a consumer only cares about content, they shouldn't have to
    explicitly iterate thinking first. The stream should auto-drain it.
    """
    chat = Chat(
        model=get_test_model(
            max_tokens=4000,
            temperature=1.0,
            thinking_config=ThinkingConfig(token_budget=2000, include_thoughts=True),
        )
    )

    chat = chat.add_message(SystemMessage("You are a helpful assistant.")).add_message(
        UserMessage("What is 5 + 7?")
    )

    accumulated_content = ""
    content_iterations = 0

    async with chat.complete_stream() as stream:
        # Skip thinking, go straight to content
        async for content in stream.content:
            content_iterations += 1
            accumulated_content = content

    # Verify streaming occurred
    assert (
        content_iterations >= 1
    ), f"Expected at least one content iteration, got {content_iterations}"

    # Should still work and have the answer
    final_chat = await stream.chat
    assert final_chat is not None
    assert "12" in accumulated_content


@pytest.mark.asyncio
async def test_stream_with_thinking_and_tool_call():
    """
    Test that tool calling works correctly when thinking is enabled.

    Verifies:
    1. Tool is called correctly during streaming
    2. Tool result is incorporated into the response
    3. Message history shows correct sequence
    4. Thinking is captured if returned (Google's thought summaries are best-effort)
    """
    event_log = []

    def tracked_calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            The sum of a and b
        """
        event_log.append(("tool_called", {"a": a, "b": b, "result": a + b}))
        return a + b

    chat = Chat(
        model=get_test_model(
            max_tokens=8000,
            temperature=1.0,
            thinking_config=ThinkingConfig(token_budget=8000, include_thoughts=True),
        )
    ).with_tools([tracked_calculate_sum])

    chat = chat.add_message(
        SystemMessage(
            "You are a math assistant. You MUST use the tracked_calculate_sum tool "
            "for ALL calculations. After getting the result, explain whether the sum "
            "is odd or even and why."
        )
    ).add_message(UserMessage("What is 8473 + 9156? Is the result odd or even?"))

    accumulated_content = ""
    thinking_iterations = 0
    content_iterations = 0

    async with chat.complete_stream() as stream:
        # Consume thinking (may or may not have content - Google's thought summaries are best-effort)
        async for _ in stream.thinking:
            thinking_iterations += 1

        async for content in stream.content:
            content_iterations += 1
            accumulated_content = content

    # Verify content streaming occurred
    assert (
        content_iterations >= 1
    ), f"Expected content iterations, got {content_iterations}"

    # Response should contain the correct answer (17629, possibly formatted as 17,629)
    assert "17629" in accumulated_content or "17,629" in accumulated_content

    final_chat = await stream.chat

    # Verify tool was actually invoked via event log
    tool_events = [e for e in event_log if e[0] == "tool_called"]
    assert len(tool_events) >= 1, f"Tool was never invoked. Event log: {event_log}"
    assert (
        tool_events[0][1]["a"] == 8473
    ), f"Expected a=8473, got {tool_events[0][1]['a']}"
    assert (
        tool_events[0][1]["b"] == 9156
    ), f"Expected b=9156, got {tool_events[0][1]['b']}"
    assert tool_events[0][1]["result"] == 17629

    # Verify message history shows the correct sequence
    message_types = [type(m).__name__ for m in final_chat.messages]
    assert any(
        isinstance(m, ToolCallMessage) for m in final_chat.messages
    ), f"No ToolCallMessage found in message history. Types: {message_types}"
    assert any(
        isinstance(m, ToolResultMessage) for m in final_chat.messages
    ), f"No ToolResultMessage found in message history. Types: {message_types}"

    # Verify sequence: ToolCallMessage → ToolResultMessage → final AssistantMessage
    call_idx = next(
        i for i, m in enumerate(final_chat.messages) if isinstance(m, ToolCallMessage)
    )
    tool_result_idx = next(
        i for i, m in enumerate(final_chat.messages) if isinstance(m, ToolResultMessage)
    )
    assert call_idx < tool_result_idx, (
        f"ToolCallMessage (idx {call_idx}) should come before "
        f"ToolResultMessage (idx {tool_result_idx})"
    )


# =============================================================================
# Unit tests for _batch_consecutive_tool_results and _parse_tool_response_data
# =============================================================================


class TestBatchConsecutiveToolResults:
    """Unit tests for the _batch_consecutive_tool_results method."""

    def test_batch_consecutive_tool_results_groups_consecutive(self):
        """Test that consecutive tool_result messages are grouped into a single batch."""
        model = get_test_model()

        # Create messages: user -> tool_result -> tool_result -> assistant
        messages = [
            UserMessage("What is 1+1 and 2+2?"),
            ToolResultMessage(content="2", function_name="add", call_id="call_1"),
            ToolResultMessage(content="4", function_name="add", call_id="call_2"),
            AssistantMessage("The answers are 2 and 4."),
        ]

        result = model._batch_consecutive_tool_results(messages)

        # Should be: UserMessage, ToolResultBatch, AssistantMessage
        assert len(result) == 3
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], ToolResultBatch)
        assert isinstance(result[2], AssistantMessage)

        # The batch should contain both tool results
        batch = result[1]
        assert len(batch.results) == 2
        assert batch.results[0]["name"] == "add"
        # "2" is valid JSON that parses to integer 2, then wrapped in {"result": ...}
        assert batch.results[0]["response"] == {"result": 2}
        assert batch.results[1]["name"] == "add"
        assert batch.results[1]["response"] == {"result": 4}

    def test_batch_tool_results_at_end(self):
        """Test that tool_result messages at the end are properly batched."""
        model = get_test_model()

        messages = [
            UserMessage("Calculate something"),
            ToolResultMessage(content="42", function_name="calc", call_id="call_1"),
        ]

        result = model._batch_consecutive_tool_results(messages)

        # Should be: UserMessage, ToolResultBatch
        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], ToolResultBatch)
        assert len(result[1].results) == 1

    def test_batch_non_consecutive_tool_results(self):
        """Test that non-consecutive tool_results are in separate batches."""
        model = get_test_model()

        messages = [
            UserMessage("First question"),
            ToolResultMessage(content="1", function_name="func1", call_id="call_1"),
            AssistantMessage("First answer"),
            UserMessage("Second question"),
            ToolResultMessage(content="2", function_name="func2", call_id="call_2"),
        ]

        result = model._batch_consecutive_tool_results(messages)

        # Should be: UserMessage, ToolResultBatch, AssistantMessage, UserMessage, ToolResultBatch
        assert len(result) == 5
        assert isinstance(result[1], ToolResultBatch)
        assert isinstance(result[4], ToolResultBatch)
        # Each batch should have exactly one result
        assert len(result[1].results) == 1
        assert len(result[4].results) == 1

    def test_batch_raises_on_missing_function_name(self):
        """Test that ValueError is raised when function_name is missing."""
        model = get_test_model()

        messages = [
            UserMessage("Test"),
            ToolResultMessage(
                content="result",
                function_name=None,  # Missing function_name
                call_id="call_1",
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            model._batch_consecutive_tool_results(messages)

        assert "function_name" in str(exc_info.value).lower()


class TestParseToolResponseData:
    """Unit tests for the _parse_tool_response_data method."""

    def test_parse_valid_json_dict(self):
        """Test that valid JSON dict is returned as-is."""
        model = get_test_model()
        result = model._parse_tool_response_data('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_parse_valid_json_non_dict_wrapped(self):
        """Test that valid JSON non-dict is wrapped in result dict."""
        model = get_test_model()

        # String
        result = model._parse_tool_response_data('"hello"')
        assert result == {"result": "hello"}

        # Number
        result = model._parse_tool_response_data("42")
        assert result == {"result": 42}

        # Array
        result = model._parse_tool_response_data("[1, 2, 3]")
        assert result == {"result": [1, 2, 3]}

        # Boolean
        result = model._parse_tool_response_data("true")
        assert result == {"result": True}

    def test_parse_invalid_json_wrapped(self):
        """Test that invalid JSON is wrapped as string in result dict."""
        model = get_test_model()

        result = model._parse_tool_response_data("This is not JSON")
        assert result == {"result": "This is not JSON"}

        result = model._parse_tool_response_data("{invalid json}")
        assert result == {"result": "{invalid json}"}

    def test_parse_empty_string(self):
        """Test that empty string is wrapped in result dict."""
        model = get_test_model()
        result = model._parse_tool_response_data("")
        assert result == {"result": ""}
