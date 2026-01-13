"""
Core Chat class with conversation management only.

This module contains the essential Chat class focused purely on conversation
management, delegating complex operations to specialized modules.
"""

import copy
import re
from typing import List, Optional, Set, Union

from patterpunk.lib.extract_json import extract_json
from patterpunk.llm.defaults import default_model
from patterpunk.llm.tool_types import ToolDefinition
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.streaming import ChatStream, ChatStreamFactory
from .tools import (
    configure_tools,
    configure_mcp_servers,
    execute_mcp_tool_calls,
    execute_all_tool_calls,
)
from .structured_output import get_parsed_output_with_retry


class Chat:
    """
    Core Chat class for conversation management.

    Handles conversation flow, message sequences, and model coordination
    while delegating specialized functionality to focused modules.
    """

    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        model: Optional[Model] = None,
        tools: Optional[ToolDefinition] = None,
    ):
        if messages is None:
            messages = []
        self.messages = messages

        self.model = default_model() if model is None else model
        self.tools = tools
        self._mcp_client = None
        self._tool_functions = {}  # Maps function names to callables

    def add_message(self, message: Message):
        new_chat = self.copy()
        new_chat.messages.append(message)
        return new_chat

    def with_tools(self, tools):
        return configure_tools(self, tools)

    def with_mcp_servers(self, server_configs):
        return configure_mcp_servers(self, server_configs)

    def complete(
        self,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
        execute_tools: bool = True,
    ):
        """
        Complete the conversation by generating a response from the LLM.

        Handles the core completion flow while delegating specialized operations
        to appropriate modules.

        Args:
            output_types: Optional output type constraints
            execute_tools: If True (default), automatically execute tool calls
                          and continue the conversation. If False, return the
                          Chat with the ToolCallMessage for manual handling.
        """
        message = self.latest_message
        model = message.model if message.model else self.model

        tools_to_use = None
        if self.tools and getattr(message, "allow_tool_calls", True):
            tools_to_use = self.tools

        response_message = model.generate_assistant_message(
            self.messages,
            tools_to_use,
            structured_output=getattr(message, "structured_output", None),
            output_types=output_types,
        )

        new_chat = self.add_message(response_message)

        # Auto-execute tool calls if enabled
        if new_chat.is_latest_message_tool_call and execute_tools:
            new_chat = execute_all_tool_calls(new_chat)

        return new_chat

    async def complete_async(
        self, output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None
    ) -> "Chat":
        """
        Async version of complete().

        Completes the conversation by generating a response from the LLM asynchronously.
        Returns a new Chat with the response message appended.
        """
        message = self.latest_message
        model = message.model if message.model else self.model

        tools_to_use = None
        if self.tools and getattr(message, "allow_tool_calls", True):
            tools_to_use = self.tools

        response_message = await model.generate_assistant_message_async(
            self.messages,
            tools_to_use,
            structured_output=getattr(message, "structured_output", None),
            output_types=output_types,
        )

        new_chat = self.add_message(response_message)

        if new_chat.is_latest_message_tool_call and new_chat._mcp_client:
            new_chat = execute_mcp_tool_calls(new_chat)

        return new_chat

    def complete_stream(
        self, output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None
    ) -> ChatStream:
        """
        Stream the conversation completion.

        Returns a ChatStream async context manager for streaming responses.

        Usage:
            async with chat.complete_stream() as stream:
                async for thinking in stream.thinking:
                    print(f"Thinking: {thinking}")
                async for content in stream.content:
                    print(content, end="")

            final_chat = stream.chat
        """
        message = self.latest_message
        model = message.model if message.model else self.model

        tools_to_use = None
        if self.tools and getattr(message, "allow_tool_calls", True):
            tools_to_use = self.tools

        raw_stream = model.stream_assistant_message(
            self.messages,
            tools_to_use,
            structured_output=getattr(message, "structured_output", None),
            output_types=output_types,
        )

        factory = ChatStreamFactory(self)
        return ChatStream(raw_stream, factory)

    def extract_json(self) -> Optional[List[str]]:
        chat_text = "\n\n".join(
            [
                f"{message.role}:\n{message.content}"
                for message in self.messages
                if isinstance(message, AssistantMessage)
            ]
        )

        if re.search(r"\{.*}", chat_text, re.IGNORECASE | re.DOTALL) or re.search(
            r"\[.*]", chat_text, re.IGNORECASE | re.DOTALL
        ):
            jsons = extract_json(chat_text)
            if jsons:
                return jsons

        return None

    @property
    def parsed_output(self):
        return get_parsed_output_with_retry(self)

    @property
    def latest_message(self):
        return self.messages[-1]

    @property
    def is_latest_message_tool_call(self):
        return isinstance(self.latest_message, ToolCallMessage)

    def copy(self):
        return copy.deepcopy(self)

    def count_tokens(self, include_tools: bool = True) -> int:
        """
        Count total tokens for all messages in this chat.

        For API-based providers (Anthropic, Google, Bedrock), this makes a single
        API call for all messages, which is much more efficient than counting
        messages individually.

        Args:
            include_tools: If True, includes tool definitions in count

        Returns:
            Total token count for the conversation
        """
        import json

        # Pass all messages at once - providers handle batch efficiently
        total = self.model.count_tokens(self.messages)

        # Count tool definitions if present and requested
        if include_tools and self.tools:
            tools_json = json.dumps(self.tools)
            total += self.model.count_tokens(tools_json)

        return total

    async def count_tokens_async(self, include_tools: bool = True) -> int:
        """
        Async version of count_tokens.

        For providers with native async support (Anthropic, Google), this uses
        their async APIs for better concurrency.

        Args:
            include_tools: If True, includes tool definitions in count

        Returns:
            Total token count for the conversation
        """
        import json

        total = await self.model.count_tokens_async(self.messages)

        if include_tools and self.tools:
            tools_json = json.dumps(self.tools)
            total += await self.model.count_tokens_async(tools_json)

        return total
