"""
Async streaming support for patterpunk.

This module provides the types and classes for async streaming responses from LLM providers.
The design follows Python idioms with async context managers and async iterators.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    TypeVar,
)

if TYPE_CHECKING:
    from patterpunk.llm.chat.core import Chat
    from patterpunk.llm.messages.assistant import AssistantMessage


class StreamEventType(Enum):
    """Types of events that can occur during streaming."""

    # Message lifecycle
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_END = "message_end"

    # Content block lifecycle
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"

    # Content deltas
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"

    # Tool use
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_STOP = "tool_use_stop"


class StreamIncompleteError(Exception):
    """
    Raised when accessing stream.chat before the stream has completed.

    This can happen in two scenarios:
    1. Accessing .chat while iteration is still in progress
    2. Accessing .chat after early exit/cancellation from the stream
    """

    def __init__(self, message: str = "Stream has not completed"):
        self.message = message
        super().__init__(self.message)


@dataclass
class StreamChunk:
    """
    A single chunk from the streaming response.

    The event_type discriminates what kind of chunk this is and which fields are populated.
    """

    event_type: StreamEventType

    # For TEXT_DELTA and THINKING_DELTA events
    text: Optional[str] = None

    # For content block tracking
    index: Optional[int] = None
    block_type: Optional[str] = None

    # For tool use events
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments_delta: Optional[str] = None

    # For MESSAGE_END event - usage statistics
    usage: Optional[Dict[str, int]] = None

    # Raw provider data (for debugging/advanced use)
    raw: Optional[Any] = None


@dataclass
class StreamAccumulator:
    """
    Internal accumulator for building up content during streaming.

    Tracks both thinking and content phases, accumulating text and managing state.
    """

    thinking_text: str = ""
    content_text: str = ""
    thinking_blocks: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    current_tool_arguments: Dict[str, str] = field(default_factory=dict)
    usage: Optional[Dict[str, int]] = None
    is_complete: bool = False
    was_cancelled: bool = False

    def append_thinking(self, text: str) -> None:
        """Append text to the thinking accumulator."""
        self.thinking_text += text

    def append_content(self, text: str) -> None:
        """Append text to the content accumulator."""
        self.content_text += text


class ChatStream:
    """
    Async context manager for streaming LLM responses.

    Usage:
        async with chat.complete_stream() as stream:
            # Optional: iterate thinking first
            async for thinking in stream.thinking:
                print(f"Thinking: {thinking}")

            # Then iterate content
            async for content in stream.content:
                print(content, end="")

        # After context exit, access final chat
        final_chat = stream.chat

    Four iterators are available:
    - stream.thinking: Yields accumulated thinking string (for display replacement)
    - stream.thinking_delta: Yields thinking deltas (for manual accumulation)
    - stream.content: Yields accumulated content string (for display replacement)
    - stream.content_delta: Yields content deltas (for manual accumulation)
    """

    def __init__(
        self,
        raw_stream: AsyncIterator[StreamChunk],
        chat_factory: "ChatStreamFactory",
    ):
        self._raw_stream = raw_stream
        self._chat_factory = chat_factory
        self._accumulator = StreamAccumulator()
        self._thinking_consumed = False
        self._content_consumed = False
        self._final_chat: Optional["Chat"] = None
        self._entered = False
        self._exited = False

    async def __aenter__(self) -> "ChatStream":
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._exited = True
        # If we exited early (exception or break), mark as cancelled
        if not self._accumulator.is_complete:
            self._accumulator.was_cancelled = True
        return False  # Don't suppress exceptions

    @property
    def chat(self) -> "Chat":
        """
        Get the final Chat object with the complete response.

        Raises StreamIncompleteError if:
        - Called before the context manager exits
        - Called after early exit/cancellation
        """
        if not self._exited:
            raise StreamIncompleteError("Stream is still in progress")

        if self._accumulator.was_cancelled:
            raise StreamIncompleteError("Stream was cancelled before completion")

        if self._final_chat is None:
            # Build final chat from accumulated content
            self._final_chat = self._chat_factory.build_final_chat(self._accumulator)

        return self._final_chat

    @property
    async def thinking(self) -> AsyncIterator[str]:
        """
        Async iterator yielding accumulated thinking content.

        Each yield contains the full thinking text accumulated so far,
        suitable for display replacement in a UI.
        """
        self._thinking_consumed = True

        async for chunk in self._raw_stream:
            if chunk.event_type == StreamEventType.THINKING_DELTA and chunk.text:
                self._accumulator.append_thinking(chunk.text)
                yield self._accumulator.thinking_text
            elif chunk.event_type == StreamEventType.TEXT_DELTA:
                # Thinking phase is over, content has started
                if chunk.text:
                    self._accumulator.append_content(chunk.text)
                break
            elif chunk.event_type == StreamEventType.MESSAGE_END:
                self._accumulator.is_complete = True
                if chunk.usage:
                    self._accumulator.usage = chunk.usage
                break

    @property
    async def thinking_delta(self) -> AsyncIterator[str]:
        """
        Async iterator yielding thinking deltas.

        Each yield contains only the new thinking text since the last yield,
        suitable for manual accumulation.
        """
        self._thinking_consumed = True

        async for chunk in self._raw_stream:
            if chunk.event_type == StreamEventType.THINKING_DELTA and chunk.text:
                self._accumulator.append_thinking(chunk.text)
                yield chunk.text
            elif chunk.event_type == StreamEventType.TEXT_DELTA:
                # Thinking phase is over, content has started
                if chunk.text:
                    self._accumulator.append_content(chunk.text)
                break
            elif chunk.event_type == StreamEventType.MESSAGE_END:
                self._accumulator.is_complete = True
                if chunk.usage:
                    self._accumulator.usage = chunk.usage
                break

    @property
    async def content(self) -> AsyncIterator[str]:
        """
        Async iterator yielding accumulated content.

        Each yield contains the full content text accumulated so far,
        suitable for display replacement in a UI.

        If thinking was not consumed, it will be auto-drained first.
        """
        # Auto-drain thinking if not consumed
        if not self._thinking_consumed:
            await self._drain_thinking()

        self._content_consumed = True

        # If we already have content from draining thinking
        if self._accumulator.content_text:
            yield self._accumulator.content_text

        async for chunk in self._raw_stream:
            if chunk.event_type == StreamEventType.TEXT_DELTA and chunk.text:
                self._accumulator.append_content(chunk.text)
                yield self._accumulator.content_text
            elif chunk.event_type == StreamEventType.MESSAGE_END:
                self._accumulator.is_complete = True
                if chunk.usage:
                    self._accumulator.usage = chunk.usage
                break

    @property
    async def content_delta(self) -> AsyncIterator[str]:
        """
        Async iterator yielding content deltas.

        Each yield contains only the new content text since the last yield,
        suitable for manual accumulation.

        If thinking was not consumed, it will be auto-drained first.
        """
        # Auto-drain thinking if not consumed
        if not self._thinking_consumed:
            await self._drain_thinking()

        self._content_consumed = True

        # If we already have content from draining thinking, yield it
        if self._accumulator.content_text:
            yield self._accumulator.content_text

        async for chunk in self._raw_stream:
            if chunk.event_type == StreamEventType.TEXT_DELTA and chunk.text:
                self._accumulator.append_content(chunk.text)
                yield chunk.text
            elif chunk.event_type == StreamEventType.MESSAGE_END:
                self._accumulator.is_complete = True
                if chunk.usage:
                    self._accumulator.usage = chunk.usage
                break

    async def _drain_thinking(self) -> None:
        """
        Internal method to consume all thinking events without yielding.

        Called when consumer iterates .content without first iterating .thinking.
        """
        self._thinking_consumed = True

        async for chunk in self._raw_stream:
            if chunk.event_type == StreamEventType.THINKING_DELTA and chunk.text:
                self._accumulator.append_thinking(chunk.text)
            elif chunk.event_type == StreamEventType.TEXT_DELTA:
                # Thinking phase is over, content has started
                if chunk.text:
                    self._accumulator.append_content(chunk.text)
                return
            elif chunk.event_type == StreamEventType.MESSAGE_END:
                self._accumulator.is_complete = True
                if chunk.usage:
                    self._accumulator.usage = chunk.usage
                return


class ChatStreamFactory:
    """
    Factory for building the final Chat from accumulated stream content.

    This is passed to ChatStream and called when stream.chat is accessed.
    Provider-specific implementations can subclass this to customize behavior.
    """

    def __init__(self, original_chat: "Chat"):
        self._original_chat = original_chat

    def build_final_chat(self, accumulator: StreamAccumulator) -> "Chat":
        """
        Build the final Chat object from the accumulated stream content.

        Subclasses can override this to add provider-specific metadata.
        """
        from patterpunk.llm.messages.assistant import AssistantMessage

        message = AssistantMessage(
            content=accumulator.content_text,
            thinking_blocks=(
                accumulator.thinking_blocks if accumulator.thinking_blocks else None
            ),
        )

        # If we accumulated thinking text but no blocks, create one
        if accumulator.thinking_text and not accumulator.thinking_blocks:
            message = AssistantMessage(
                content=accumulator.content_text,
                thinking_blocks=[
                    {"type": "thinking", "thinking": accumulator.thinking_text}
                ],
            )

        return self._original_chat.add_message(message)


class StreamingNotSupported(Exception):
    """Raised when streaming is not supported by a model."""

    pass
