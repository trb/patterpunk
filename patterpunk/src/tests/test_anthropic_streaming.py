"""
Tests for async streaming support with Anthropic.

These tests validate the complete_stream() interface and streaming behavior
with the Anthropic provider. They test both content streaming and thinking
block streaming for reasoning models.
"""

import pytest
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.streaming import StreamIncompleteError


@pytest.mark.asyncio
async def test_stream_content_basic():
    """
    Test basic content streaming without thinking blocks.

    Validates that:
    1. complete_stream() returns an async context manager
    2. stream.content yields accumulated content strings
    3. stream.chat is available after context exit with complete response
    """
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=256,
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

    # After context exit, stream.chat should be available
    final_chat = stream.chat

    # Final chat should have the response
    assert final_chat is not None
    assert final_chat.latest_message is not None
    assert final_chat.latest_message.content is not None
    assert len(final_chat.latest_message.content) > 0

    # The accumulated content should match the final message
    assert accumulated_content == final_chat.latest_message.content


@pytest.mark.asyncio
async def test_stream_thinking_and_content():
    """
    Test streaming with thinking blocks enabled (reasoning model).

    Validates that:
    1. Thinking content streams first in the thinking phase
    2. Content streams second in the content phase
    3. Both thinking and content are captured in the final message
    """
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
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

    # CRITICAL: Verify streaming actually occurred
    assert (
        thinking_iterations > 1
    ), f"Expected multiple thinking iterations, got {thinking_iterations}"
    assert (
        content_iterations > 1
    ), f"Expected multiple content iterations, got {content_iterations}"

    # Verify final chat
    final_chat = stream.chat
    assert final_chat is not None
    assert final_chat.latest_message is not None

    # Content should contain the answer (391)
    assert "391" in accumulated_content

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
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
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
    final_chat = stream.chat
    assert final_chat is not None
    assert "12" in accumulated_content


@pytest.mark.asyncio
async def test_stream_chat_access_before_exit_raises():
    """
    Test that accessing stream.chat before context exit raises StreamIncompleteError.

    This ensures consumers don't accidentally use incomplete data.
    """
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=100,
        )
    )

    chat = chat.add_message(SystemMessage("Be brief.")).add_message(
        UserMessage("Say hi")
    )

    async with chat.complete_stream() as stream:
        # Try to access chat before iteration completes
        with pytest.raises(StreamIncompleteError) as exc_info:
            _ = stream.chat
        assert "still in progress" in str(exc_info.value)

        # Drain the stream
        async for _ in stream.content:
            pass

    # After exit, it should work
    final_chat = stream.chat
    assert final_chat is not None


@pytest.mark.asyncio
async def test_stream_delta_iterators():
    """
    Test the delta iterators (content_delta vs content).

    Delta iterators yield only new text, not accumulated text.
    Useful for manual accumulation or logging.
    """
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=256,
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
    final_chat = stream.chat
    assert full_content == final_chat.latest_message.content
