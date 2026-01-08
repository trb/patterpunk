"""
Tests for automatic tool calling during async streaming.

These tests validate that:
1. Python function tools are automatically executed during streaming
2. Tool results flow back to the model for continued response
3. ToolExecutionAbortError stops the stream immediately
4. Normal exceptions are sent to the model as errors, allowing recovery
5. Multiple rounds of tool calling work seamlessly
"""

import pytest
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.streaming import ToolExecutionAbortError


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
async def test_stream_with_function_tool_call():
    """
    Test that function tools are automatically called during streaming.

    The model should:
    1. Call the get_weather tool
    2. Receive the result
    3. Continue generating a response with the weather info
    """
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=256,
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
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=256,
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
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=256,
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
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=512,
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


@pytest.mark.asyncio
async def test_stream_with_thinking_and_tool_call():
    """
    Test that thinking blocks work correctly with tool calling (interleaved thinking).

    With interleaved thinking enabled (Claude 4.5+), verifies the full sequence:
    1. Model thinks about the request (pre-tool thinking)
    2. Model decides to call a tool
    3. Tool executes (verified via side effect tracker)
    4. Model thinks about the tool result (post-tool thinking)
    5. Model generates final response

    Verification uses:
    - Event log: Tracks the sequence thinking → tool call → more thinking
    - Side effect tracker: Confirms tool was actually invoked with correct args
    - Message history: Confirms ToolCallMessage and ToolResultMessage exist
    - Final message thinking: Proves thinking occurred after tool execution
    """
    # Event log to track the sequence: thinking → tool → thinking → content
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
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
        )
    ).with_tools([tracked_calculate_sum])

    # Use harder numbers and a prompt that requires reasoning about the result
    chat = chat.add_message(
        SystemMessage(
            "You are a math assistant. You MUST use the tracked_calculate_sum tool "
            "for ALL calculations. After getting the result, explain whether the sum "
            "is odd or even and why. Think through your reasoning carefully."
        )
    ).add_message(UserMessage("What is 8473 + 9156? Is the result odd or even?"))

    accumulated_content = ""
    thinking_iterations = 0
    content_iterations = 0
    pre_tool_thinking_iterations = 0
    post_tool_thinking_iterations = 0

    async with chat.complete_stream() as stream:
        async for thinking in stream.thinking:
            thinking_iterations += 1
            # Check if tool has been called yet
            tool_called = any(e[0] == "tool_called" for e in event_log)
            if tool_called:
                post_tool_thinking_iterations += 1
            else:
                pre_tool_thinking_iterations += 1

        async for content in stream.content:
            content_iterations += 1
            accumulated_content = content

    # Verify streaming occurred
    assert (
        thinking_iterations > 0
    ), f"Expected thinking iterations, got {thinking_iterations}"
    assert (
        content_iterations > 1
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

    # Verify interleaved thinking: BOTH pre-tool AND post-tool thinking must occur
    # This validates that the interleaved-thinking beta header is working correctly
    assert pre_tool_thinking_iterations > 0, (
        f"No thinking before tool call. Pre: {pre_tool_thinking_iterations}, "
        f"Post: {post_tool_thinking_iterations}. "
        "The model should think before deciding to call a tool."
    )

    assert post_tool_thinking_iterations > 0, (
        f"No thinking after tool call. Pre: {pre_tool_thinking_iterations}, "
        f"Post: {post_tool_thinking_iterations}. "
        "Interleaved thinking requires the anthropic-beta header and Claude 4.5+ model. "
        "The model should think about the tool result before generating the final response."
    )

    # Also verify the final message has thinking blocks stored
    assert (
        final_chat.latest_message.has_thinking
    ), "Final message should have thinking blocks from the post-tool thinking phase"
