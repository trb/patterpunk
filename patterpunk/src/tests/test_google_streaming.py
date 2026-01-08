"""
Tests for async streaming support with Google Vertex AI.

These tests validate the complete_stream() interface and streaming behavior
with the Google Vertex AI provider. They test content streaming and tool calling
during streaming.
"""

import pytest
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.streaming import StreamIncompleteError, ToolExecutionAbortError
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage


# Use a flash model for faster tests
GOOGLE_TEST_MODEL = "gemini-2.5-flash"
GOOGLE_TEST_LOCATION = "northamerica-northeast1"


def get_test_model(max_tokens: int = 256, temperature: float = 0.0) -> GoogleModel:
    """Create a GoogleModel for testing."""
    return GoogleModel(
        model=GOOGLE_TEST_MODEL,
        location=GOOGLE_TEST_LOCATION,
        temperature=temperature,
        max_tokens=max_tokens,
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
    before we can break. In that case, we verify the stream completed successfully.
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
    async with chat.complete_stream() as stream:
        # Start iterating but try to exit early
        count = 0
        async for _ in stream.content:
            count += 1
            if count >= 2:
                was_cancelled = True
                break  # Exit early - this should cancel the stream

    if was_cancelled:
        # If we broke early, await stream.chat should raise
        with pytest.raises(StreamIncompleteError) as exc_info:
            _ = await stream.chat
        assert "cancelled" in str(exc_info.value).lower()
    else:
        # If the stream completed before we could cancel, verify it worked
        final_chat = await stream.chat
        assert final_chat is not None


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

    # Verify streaming occurred
    assert (
        content_iterations > 1
    ), f"Expected multiple iterations, got {content_iterations}"

    # The final response should mention the weather from our tool
    final_chat = await stream.chat
    assert "sunny" in accumulated_content.lower() or "72" in accumulated_content
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
    # Weather tool returns "sunny and 72F" for any location
    # Calculate sum returns 42 for 15 + 27
    assert "sunny" in accumulated_content.lower() or "72" in accumulated_content
    assert "42" in accumulated_content
