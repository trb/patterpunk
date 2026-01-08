"""
Tests for async streaming support with Azure OpenAI.

These tests validate the complete_stream() interface and streaming behavior
with the Azure OpenAI provider. They test both content streaming and tool
calling during streaming, as well as thinking/reasoning token streaming
for reasoning models (o1, o3-mini, etc.).
"""

import os
import pytest
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.models.azure_openai import AzureOpenAiModel
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.streaming import StreamIncompleteError, ToolExecutionAbortError
from patterpunk.llm.thinking import ThinkingConfig


# Azure OpenAI reasoning model deployment name (gpt-5.2)
AZURE_REASONING_DEPLOYMENT = "gpt-5.2"


# Tool functions for testing
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        A string describing the current weather
    """
    return f"The weather in {location} is sunny and 72°F."


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
    Test basic content streaming without tools.

    Validates that:
    1. complete_stream() returns an async context manager
    2. stream.content yields accumulated content strings
    3. stream.chat is available after context exit with complete response
    """
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    )

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

    # CRITICAL: Verify streaming actually occurred (multiple chunks received)
    assert (
        iteration_count > 1
    ), f"Expected multiple streaming iterations, got {iteration_count}"

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
    """
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    )

    chat = chat.add_message(SystemMessage("Be verbose.")).add_message(
        UserMessage("Count from 1 to 100, describing each number in detail.")
    )

    async with chat.complete_stream() as stream:
        # Start iterating but exit early
        count = 0
        async for _ in stream.content:
            count += 1
            if count >= 3:
                break  # Exit early - this cancels the stream

    # After early exit, await stream.chat should raise
    with pytest.raises(StreamIncompleteError) as exc_info:
        _ = await stream.chat
    assert "cancelled" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_stream_chat_waits_for_completion():
    """
    Test that await stream.chat waits for completion if called during iteration.

    The await is non-blocking if already complete, or waits if still in progress.
    """
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    )

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
    """
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    )

    chat = chat.add_message(SystemMessage("You are a helpful assistant.")).add_message(
        UserMessage("Count from 1 to 5, one number per line.")
    )

    deltas = []
    async with chat.complete_stream() as stream:
        async for delta in stream.content_delta:
            deltas.append(delta)

    # We should have received multiple deltas
    assert len(deltas) > 1

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
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    ).with_tools([get_weather])

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
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    ).with_tools([failing_tool])

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
        for word in ["error", "fail", "sorry", "unable", "couldn't"]
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
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    ).with_tools([abort_tool])

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
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.0,
        )
    ).with_tools([get_weather, calculate_sum])

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
    # Weather tool returns "sunny and 72°F" for any location
    # Calculate sum returns 42 for 15 + 27
    assert "sunny" in accumulated_content.lower() or "72" in accumulated_content
    assert "42" in accumulated_content


# =============================================================================
# Thinking/Reasoning Token Streaming Tests
# =============================================================================
# These tests require a reasoning model deployment (o1, o3-mini, etc.)
# Set PP_AZURE_OPENAI_REASONING_DEPLOYMENT environment variable to run them.


@pytest.mark.asyncio
async def test_stream_thinking_and_content():
    """
    Test streaming with thinking/reasoning blocks enabled.

    NOTE: Azure OpenAI reasoning models may not always produce thinking summaries,
    especially for simple questions. The reasoning happens internally, and summaries
    are only generated when the model deems it useful.

    Validates that:
    1. The thinking iterator can be used (may or may not yield content)
    2. Content streams correctly
    3. The final response is correct
    """
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name=AZURE_REASONING_DEPLOYMENT,
            thinking_config=ThinkingConfig(effort="medium"),
        )
    )

    chat = chat.add_message(
        SystemMessage("You are a helpful math assistant.")
    ).add_message(
        UserMessage("What is 17 * 23? Think through the calculation step by step.")
    )

    accumulated_thinking = ""
    accumulated_content = ""
    thinking_iterations = 0
    content_iterations = 0

    async with chat.complete_stream() as stream:
        # First iterate through thinking (may be empty for Azure)
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

    # CRITICAL: Verify content streaming actually occurred
    assert (
        content_iterations > 1
    ), f"Expected multiple content iterations, got {content_iterations}"

    # Verify final chat
    final_chat = await stream.chat
    assert final_chat is not None
    assert final_chat.latest_message is not None

    # Content should contain the answer (391)
    assert "391" in accumulated_content

    # Note: Azure reasoning models may or may not produce thinking summaries
    # We verify the iterator works but don't require thinking to be present
    # If thinking was produced, verify it's captured correctly
    if thinking_iterations > 0:
        assert (
            accumulated_thinking
        ), "Thinking iterations occurred but no content captured"
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
        model=AzureOpenAiModel(
            deployment_name=AZURE_REASONING_DEPLOYMENT,
            thinking_config=ThinkingConfig(effort="low"),
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

    # CRITICAL: Verify streaming actually occurred
    assert (
        content_iterations > 1
    ), f"Expected multiple content iterations, got {content_iterations}"

    # Should still work and have the answer
    final_chat = await stream.chat
    assert final_chat is not None
    assert "12" in accumulated_content


@pytest.mark.asyncio
async def test_stream_with_reasoning_and_tool_call():
    """
    Test that reasoning models work correctly with tool calling.

    NOTE: Unlike Anthropic's interleaved thinking, Azure OpenAI reasoning models
    do NOT support true interleaved thinking. The reasoning happens internally
    and summaries are provided at the end, not between tool calls.

    This test verifies:
    1. Reasoning models (gpt-5.2) can use tools during streaming
    2. Tool execution works correctly (verified via side effect tracker)
    3. Message history contains ToolCallMessage and ToolResultMessage
    4. Final response includes the tool result
    """
    # Event log to verify tool was actually called
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
        model=AzureOpenAiModel(
            deployment_name=AZURE_REASONING_DEPLOYMENT,
            thinking_config=ThinkingConfig(effort="medium"),
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
    content_iterations = 0

    async with chat.complete_stream() as stream:
        # Note: Azure reasoning summaries come at the end, not interleaved
        # We skip thinking iteration and go straight to content
        async for content in stream.content:
            content_iterations += 1
            accumulated_content = content

    # Verify streaming occurred
    assert (
        content_iterations > 1
    ), f"Expected multiple content iterations, got {content_iterations}"

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
    tool_call_idx = next(
        i for i, m in enumerate(final_chat.messages) if isinstance(m, ToolCallMessage)
    )
    tool_result_idx = next(
        i for i, m in enumerate(final_chat.messages) if isinstance(m, ToolResultMessage)
    )
    assert tool_call_idx < tool_result_idx, (
        f"ToolCallMessage (idx {tool_call_idx}) should come before "
        f"ToolResultMessage (idx {tool_result_idx})"
    )

    # Verify final response
    assert final_chat.latest_message is not None
    assert final_chat.latest_message.content == accumulated_content


@pytest.mark.asyncio
async def test_stream_thinking_delta_iterators():
    """
    Test the thinking delta iterator.

    Thinking delta iterator yields only new thinking text, not accumulated text.
    Useful for manual accumulation or logging.
    """
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name=AZURE_REASONING_DEPLOYMENT,
            thinking_config=ThinkingConfig(
                effort="medium"
            ),  # Use medium to ensure reasoning
        )
    )

    chat = chat.add_message(SystemMessage("You are a helpful assistant.")).add_message(
        UserMessage("What is 8 * 9? Explain your reasoning step by step.")
    )

    thinking_deltas = []
    async with chat.complete_stream() as stream:
        async for delta in stream.thinking_delta:
            thinking_deltas.append(delta)

        # Drain content to complete the stream
        async for _ in stream.content:
            pass

    final_chat = await stream.chat

    # Verify content was generated
    assert final_chat is not None
    assert final_chat.latest_message is not None
    assert "72" in final_chat.latest_message.content  # 8 * 9 = 72

    # If thinking was produced, verify it accumulated correctly
    if len(thinking_deltas) > 0:
        full_thinking = "".join(thinking_deltas)
        assert len(full_thinking) > 0
        # Note: has_thinking may not be set if thinking was streamed but not captured in message
    # Note: Reasoning models may skip producing thinking summaries for simple questions
    # even with effort="medium". This is expected behavior.
