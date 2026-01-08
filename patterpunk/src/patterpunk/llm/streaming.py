"""
Async streaming support for patterpunk.

This module provides the types and classes for async streaming responses from LLM providers.
The design follows Python idioms with async context managers and async iterators.
"""

import asyncio
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

from patterpunk.llm.tool_types import ToolCall, ToolCallList

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

    # For MESSAGE_END event - complete thinking blocks (with signatures if applicable)
    thinking_blocks: Optional[List[Dict[str, Any]]] = None

    # Raw provider data (for debugging/advanced use)
    raw: Optional[Any] = None


@dataclass
class StreamAccumulator:
    """
    Internal accumulator for building up content during streaming.

    Tracks both thinking and content phases, accumulating text and managing state.
    Also tracks tool calls for automatic execution.
    """

    thinking_text: str = ""
    content_text: str = ""
    thinking_blocks: List[Dict[str, Any]] = field(
        default_factory=list
    )  # TODO: Convert to ThinkingBlockList in Phase 4
    tool_calls: ToolCallList = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    is_complete: bool = False
    was_cancelled: bool = False

    # Event signaled when streaming completes (for await stream.chat)
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Current tool call being accumulated
    _current_tool_id: Optional[str] = None
    _current_tool_name: Optional[str] = None
    _current_tool_arguments: str = ""

    def append_thinking(self, text: str) -> None:
        """Append text to the thinking accumulator."""
        self.thinking_text += text

    def append_content(self, text: str) -> None:
        """Append text to the content accumulator."""
        self.content_text += text

    def start_tool_call(self, tool_id: str, tool_name: str) -> None:
        """Start accumulating a new tool call."""
        self._current_tool_id = tool_id
        self._current_tool_name = tool_name
        self._current_tool_arguments = ""

    def append_tool_arguments(self, delta: str) -> None:
        """Append to the current tool call's arguments."""
        self._current_tool_arguments += delta

    def finish_tool_call(self) -> None:
        """Finish the current tool call and add it to the list."""
        if self._current_tool_id and self._current_tool_name:
            self.tool_calls.append(
                ToolCall(
                    id=self._current_tool_id,
                    name=self._current_tool_name,
                    arguments=self._current_tool_arguments,
                )
            )
        self._current_tool_id = None
        self._current_tool_name = None
        self._current_tool_arguments = ""

    def has_tool_calls(self) -> bool:
        """Check if any tool calls were accumulated."""
        return len(self.tool_calls) > 0

    def reset_for_continuation(self) -> None:
        """Reset accumulator for a new streaming round after tool execution."""
        # Keep accumulated content, clear tool calls for next round
        self.tool_calls = []
        self.is_complete = False

    def mark_complete(self) -> None:
        """Mark the stream as complete and signal any waiters."""
        self.is_complete = True
        self._completion_event.set()


class ChatAwaitable:
    """
    Awaitable wrapper for getting the final Chat from a stream.

    This allows `await stream.chat` to:
    - Return immediately if the stream is already complete
    - Wait for completion if iteration is still in progress
    - Raise StreamIncompleteError if the stream was cancelled

    Usage:
        async with chat.complete_stream() as stream:
            async for content in stream.content:
                print(content)

        final_chat = await stream.chat  # Waits if needed
    """

    def __init__(self, stream: "ChatStream"):
        self._stream = stream

    def __await__(self):
        return self._get_chat().__await__()

    async def _get_chat(self) -> "Chat":
        """Get the final chat, waiting for completion if necessary."""
        # If cancelled, raise immediately - there's no complete chat to give
        if self._stream._accumulator.was_cancelled:
            raise StreamIncompleteError(
                "Stream was cancelled before completion. "
                "Use the original chat object instead."
            )

        # Wait for completion if not yet complete
        if not self._stream._accumulator.is_complete:
            await self._stream._accumulator._completion_event.wait()

        # Build and return the final chat
        if self._stream._final_chat is None:
            self._stream._final_chat = self._stream._chat_factory.build_final_chat(
                self._stream._accumulator
            )

        return self._stream._final_chat


class ChatStream:
    """
    Async context manager for streaming LLM responses with automatic tool execution.

    Usage:
        async with chat.complete_stream() as stream:
            # Optional: iterate thinking first
            async for thinking in stream.thinking:
                print(f"Thinking: {thinking}")

            # Then iterate content
            async for content in stream.content:
                print(content, end="")

        # Get the final chat (awaitable - waits for completion if needed)
        final_chat = await stream.chat

    Four iterators are available:
    - stream.thinking: Yields accumulated thinking string (for display replacement)
    - stream.thinking_delta: Yields thinking deltas (for manual accumulation)
    - stream.content: Yields accumulated content string (for display replacement)
    - stream.content_delta: Yields content deltas (for manual accumulation)

    Tool calls are handled automatically:
    - When the model calls a tool, the stream executes it and continues
    - Consumers see a seamless stream of thinking/content
    - ToolExecutionAbortError from a tool stops the stream immediately
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
    def chat(self) -> ChatAwaitable:
        """
        Get the final Chat object with the complete response.

        This property returns an awaitable that:
        - Returns immediately if the stream is already complete
        - Waits for completion if iteration is still in progress
        - Raises StreamIncompleteError if the stream was cancelled

        Usage:
            async with chat.complete_stream() as stream:
                async for content in stream.content:
                    print(content)

            final_chat = await stream.chat
        """
        return ChatAwaitable(self)

    def _handle_tool_event(self, chunk: StreamChunk) -> None:
        """Handle tool use events by accumulating tool call data."""
        if chunk.event_type == StreamEventType.TOOL_USE_START:
            self._accumulator.start_tool_call(
                chunk.tool_call_id or "",
                chunk.tool_name or "",
            )
        elif chunk.event_type == StreamEventType.TOOL_USE_DELTA:
            if chunk.tool_arguments_delta:
                self._accumulator.append_tool_arguments(chunk.tool_arguments_delta)
        elif chunk.event_type == StreamEventType.TOOL_USE_STOP:
            self._accumulator.finish_tool_call()
        elif chunk.event_type == StreamEventType.CONTENT_BLOCK_STOP:
            # CONTENT_BLOCK_STOP may indicate end of a tool use block
            # If we have an active tool call, finish it
            if self._accumulator._current_tool_id:
                self._accumulator.finish_tool_call()

    async def _execute_tools_and_continue(self) -> AsyncIterator[StreamChunk]:
        """
        Execute accumulated tool calls and start a new stream with results.

        Returns a new raw stream to continue iteration.
        """
        # Use complete thinking blocks (with signatures) if available,
        # otherwise fall back to creating blocks from accumulated text
        thinking_blocks = self._accumulator.thinking_blocks
        if not thinking_blocks and self._accumulator.thinking_text:
            thinking_blocks = [
                {"type": "thinking", "thinking": self._accumulator.thinking_text}
            ]

        # Execute tool calls
        new_stream = await self._chat_factory.execute_tools_and_create_stream(
            self._accumulator.tool_calls, thinking_blocks
        )

        # Reset accumulator for next round
        self._accumulator.reset_for_continuation()

        return new_stream

    async def _iter_chunks_with_tool_continuation(self) -> AsyncIterator[StreamChunk]:
        """
        Iterate over stream chunks with automatic tool execution and continuation.

        This is the core streaming loop that:
        1. Handles tool events internally (accumulating tool call data)
        2. Executes tools on MESSAGE_END when tool calls are present
        3. Seamlessly continues with new streams after tool execution
        4. Yields all non-tool chunks for the caller to process

        Callers iterate this once and filter for the event types they care about.
        When a caller breaks/returns early, the underlying stream position is preserved
        for subsequent callers (since they share self._raw_stream).
        """
        tool_events = (
            StreamEventType.TOOL_USE_START,
            StreamEventType.TOOL_USE_DELTA,
            StreamEventType.TOOL_USE_STOP,
        )

        while True:
            async for chunk in self._raw_stream:
                # Handle tool events internally - don't yield these
                if chunk.event_type in tool_events:
                    self._handle_tool_event(chunk)
                    continue

                # CONTENT_BLOCK_STOP may indicate end of a tool use block
                if chunk.event_type == StreamEventType.CONTENT_BLOCK_STOP:
                    if self._accumulator._current_tool_id:
                        self._accumulator.finish_tool_call()
                    continue

                # Handle MESSAGE_END - either continue with tools or complete
                if chunk.event_type == StreamEventType.MESSAGE_END:
                    if chunk.usage:
                        self._accumulator.usage = chunk.usage
                    if chunk.thinking_blocks:
                        self._accumulator.thinking_blocks = chunk.thinking_blocks

                    if self._accumulator.has_tool_calls():
                        # Execute tools and get new stream for continuation
                        self._raw_stream = await self._execute_tools_and_continue()
                        break  # Continue outer while loop with new stream
                    else:
                        self._accumulator.mark_complete()
                        return

                # Yield all other chunks (THINKING_DELTA, TEXT_DELTA, etc.)
                yield chunk
            else:
                # Stream exhausted without MESSAGE_END - shouldn't happen normally
                return

    @property
    async def thinking(self) -> AsyncIterator[str]:
        """
        Async iterator yielding accumulated thinking content.

        Each yield contains the full thinking text accumulated so far,
        suitable for display replacement in a UI.
        """
        self._thinking_consumed = True

        async for chunk in self._iter_chunks_with_tool_continuation():
            if chunk.event_type == StreamEventType.THINKING_DELTA and chunk.text:
                self._accumulator.append_thinking(chunk.text)
                yield self._accumulator.thinking_text
            elif chunk.event_type == StreamEventType.TEXT_DELTA:
                # Thinking phase is over, content has started
                if chunk.text:
                    self._accumulator.append_content(chunk.text)
                return

    @property
    async def thinking_delta(self) -> AsyncIterator[str]:
        """
        Async iterator yielding thinking deltas.

        Each yield contains only the new thinking text since the last yield,
        suitable for manual accumulation.
        """
        self._thinking_consumed = True

        async for chunk in self._iter_chunks_with_tool_continuation():
            if chunk.event_type == StreamEventType.THINKING_DELTA and chunk.text:
                self._accumulator.append_thinking(chunk.text)
                yield chunk.text
            elif chunk.event_type == StreamEventType.TEXT_DELTA:
                # Thinking phase is over, content has started
                if chunk.text:
                    self._accumulator.append_content(chunk.text)
                return

    @property
    async def content(self) -> AsyncIterator[str]:
        """
        Async iterator yielding accumulated content.

        Each yield contains the full content text accumulated so far,
        suitable for display replacement in a UI.

        If thinking was not consumed, it will be auto-drained first.
        Tool calls are executed automatically between streaming rounds.
        """
        # Auto-drain thinking if not consumed
        if not self._thinking_consumed:
            await self._drain_thinking()

        self._content_consumed = True

        # If we already have content from draining thinking
        if self._accumulator.content_text:
            yield self._accumulator.content_text

        async for chunk in self._iter_chunks_with_tool_continuation():
            if chunk.event_type == StreamEventType.TEXT_DELTA and chunk.text:
                self._accumulator.append_content(chunk.text)
                yield self._accumulator.content_text

    @property
    async def content_delta(self) -> AsyncIterator[str]:
        """
        Async iterator yielding content deltas.

        Each yield contains only the new content text since the last yield,
        suitable for manual accumulation.

        If thinking was not consumed, it will be auto-drained first.
        Tool calls are executed automatically between streaming rounds.
        """
        # Auto-drain thinking if not consumed
        if not self._thinking_consumed:
            await self._drain_thinking()

        self._content_consumed = True

        # If we already have content from draining thinking, yield it
        if self._accumulator.content_text:
            yield self._accumulator.content_text

        async for chunk in self._iter_chunks_with_tool_continuation():
            if chunk.event_type == StreamEventType.TEXT_DELTA and chunk.text:
                self._accumulator.append_content(chunk.text)
                yield chunk.text

    async def _drain_thinking(self) -> None:
        """
        Internal method to consume all thinking and tool events without yielding.

        Called when consumer iterates .content without first iterating .thinking.
        """
        self._thinking_consumed = True

        async for chunk in self._iter_chunks_with_tool_continuation():
            if chunk.event_type == StreamEventType.THINKING_DELTA and chunk.text:
                self._accumulator.append_thinking(chunk.text)
            elif chunk.event_type == StreamEventType.TEXT_DELTA:
                # Thinking phase is over, content has started
                if chunk.text:
                    self._accumulator.append_content(chunk.text)
                return


class ChatStreamFactory:
    """
    Factory for building final Chat and creating continuation streams after tool execution.

    This is passed to ChatStream and handles:
    - Building the final Chat from accumulated content
    - Executing tool calls and creating new streams for continuation
    """

    def __init__(self, original_chat: "Chat"):
        self._original_chat = original_chat
        self._current_chat = (
            original_chat  # Tracks chat state during tool execution rounds
        )

    def build_final_chat(self, accumulator: StreamAccumulator) -> "Chat":
        """
        Build the final Chat object from the accumulated stream content.
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

        return self._current_chat.add_message(message)

    async def execute_tools_and_create_stream(
        self,
        tool_calls: List[Dict[str, Any]],
        thinking_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Execute tool calls and create a new stream for the model's continuation.

        This method:
        1. Adds a ToolCallMessage to the chat (with thinking blocks if present)
        2. Executes each tool call
        3. Adds ToolResultMessages for each result
        4. Creates a new stream for the model to continue

        Raises ToolExecutionAbortError if a tool explicitly aborts.
        """
        from patterpunk.llm.messages.tool_call import ToolCallMessage
        from patterpunk.llm.chat.tools import execute_tool_calls_async

        # Add the ToolCallMessage to the conversation (with thinking blocks if present)
        tool_call_message = ToolCallMessage(tool_calls, thinking_blocks=thinking_blocks)
        self._current_chat = self._current_chat.add_message(tool_call_message)

        # Get tool execution context from the chat
        tool_functions = getattr(self._current_chat, "_tool_functions", {})
        mcp_client = getattr(self._current_chat, "_mcp_client", None)
        all_tools = self._current_chat.tools

        # Execute tool calls - this may raise ToolExecutionAbortError
        results, _ = await execute_tool_calls_async(
            tool_calls, tool_functions, mcp_client, all_tools
        )

        # Add tool results to the chat
        for result in results:
            self._current_chat = self._current_chat.add_message(result)

        # Create a new stream for continuation
        message = self._current_chat.latest_message
        model = (
            message.model
            if hasattr(message, "model") and message.model
            else self._current_chat.model
        )

        tools_to_use = None
        if self._current_chat.tools and getattr(message, "allow_tool_calls", True):
            tools_to_use = self._current_chat.tools

        raw_stream = model.stream_assistant_message(
            self._current_chat.messages,
            tools_to_use,
            structured_output=getattr(message, "structured_output", None),
        )

        return raw_stream


class StreamingNotSupported(Exception):
    """Raised when streaming is not supported by a model."""

    pass


class ToolExecutionAbortError(Exception):
    """
    Raise this from a tool to immediately abort the streaming process.

    Unlike regular exceptions (which are sent to the model as error results),
    this exception stops the stream completely and propagates to the caller.

    Use this when:
    - A critical error occurs that the model cannot recover from
    - The user explicitly requests to cancel the operation
    - A security or safety condition is violated

    Example:
        def my_tool(action: str) -> str:
            if action == "dangerous":
                raise ToolExecutionAbortError("Dangerous action not allowed")
            return "Success"
    """

    def __init__(self, message: str = "Tool execution aborted"):
        self.message = message
        super().__init__(self.message)
