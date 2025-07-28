"""
Core Chat class with conversation management only.

This module contains the essential Chat class focused purely on conversation
management, delegating complex operations to specialized modules.
"""

import copy
import re
from typing import List, Optional

from patterpunk.lib.extract_json import extract_json
from patterpunk.llm.defaults import default_model
from patterpunk.llm.tool_types import ToolDefinition
from patterpunk.llm.messages import AssistantMessage, ToolCallMessage, Message
from patterpunk.llm.models.openai import Model
from .tools import configure_tools, configure_mcp_servers, execute_mcp_tool_calls
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

    def add_message(self, message: Message):
        """Add a message to the conversation and return a new Chat instance."""
        new_chat = self.copy()
        new_chat.messages.append(message)
        return new_chat

    def with_tools(self, tools):
        """Configure tools for this chat, delegating to tools module."""
        return configure_tools(self, tools)

    def with_mcp_servers(self, server_configs):
        """Configure MCP servers for this chat, delegating to tools module."""
        return configure_mcp_servers(self, server_configs)

    def complete(self):
        """
        Complete the conversation by generating a response from the LLM.
        
        Handles the core completion flow while delegating specialized operations
        to appropriate modules.
        """
        message = self.latest_message
        model = message.model if message.model else self.model

        # Only pass tools if the latest message allows tool calls
        tools_to_use = None
        if self.tools and getattr(message, "allow_tool_calls", True):
            tools_to_use = self.tools

        response_message = model.generate_assistant_message(
            self.messages,
            tools_to_use,
            structured_output=getattr(message, "structured_output", None),
        )

        new_chat = self.add_message(response_message)
        
        # Handle MCP tool execution if needed
        if new_chat.is_latest_message_tool_call and new_chat._mcp_client:
            new_chat = execute_mcp_tool_calls(new_chat)
            
        return new_chat

    def extract_json(self) -> Optional[List[str]]:
        """Extract any JSON from assistant messages in the conversation."""
        chat_text = "\n\n".join(
            [
                f"{message.role}:\n{message.content}"
                for message in self.messages
                # we don't want to extract JSON the user send
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
        """Get parsed output with retry logic, delegating to structured_output module."""
        return get_parsed_output_with_retry(self)

    @property
    def latest_message(self):
        """Get the most recent message in the conversation."""
        return self.messages[-1]

    @property
    def is_latest_message_tool_call(self):
        """Check if the latest message is a tool call."""
        return isinstance(self.latest_message, ToolCallMessage)

    def copy(self):
        """Create a deep copy of the chat instance."""
        return copy.deepcopy(self)