import copy
import re
from typing import Dict, List, Optional, _GenericAlias, Union, Callable

from patterpunk.lib.extract_json import extract_json
from patterpunk.lib.function_to_tool import functions_to_tools
from patterpunk.llm.defaults import default_model
from patterpunk.llm.types import ToolDefinition
from patterpunk.llm.messages import (
    AssistantMessage,
    ToolCallMessage,
    Message,
    StructuredOutputFailedToParseError,
    UserMessage,
)
from patterpunk.llm.models.openai import Model
from patterpunk.logger import logger


class StructuredOutputParsingError(Exception):
    pass


class Chat:
    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        model: Optional[Model] = None,
        tools: Optional["ToolDefinition"] = None,
    ):
        if messages is None:
            messages = []
        self.messages = messages

        self.model = default_model() if model is None else model
        self.tools = tools
        self._mcp_client = None

    def add_message(self, message: Message):
        new_chat = self.copy()
        new_chat.messages.append(message)
        return new_chat

    def with_tools(self, tools: Union["ToolDefinition", List[Callable]]):
        """
        Set tools available for this chat.

        :param tools: Either a list of tool definitions in OpenAI format, or a list of annotated Python functions
        :return: New Chat instance with tools set

        Examples:
            # Using annotated functions (recommended)
            def get_weather(location: str, unit: str = "fahrenheit") -> str:
                '''Get current weather for a location.'''
                return f"Weather in {location}"

            chat = chat.with_tools([get_weather])

            # Using manual tool definitions (legacy)
            tools = [{"type": "function", "function": {...}}]
            chat = chat.with_tools(tools)
        """
        new_chat = self.copy()

        # Check if tools is a list of functions or tool definitions
        if tools and isinstance(tools[0], dict):
            # Legacy format: list of tool definitions
            new_chat.tools = tools
        else:
            # New format: list of functions - convert them
            new_chat.tools = functions_to_tools(tools)

        return new_chat

    def with_mcp_servers(self, server_configs):
        """
        Set MCP servers available for this chat.

        :param server_configs: List of MCPServerConfig instances
        :return: New Chat instance with MCP servers configured

        Examples:
            from patterpunk.lib.mcp import MCPServerConfig
            
            weather_server = MCPServerConfig(
                name="weather",
                url="http://localhost:8000/mcp"
            )
            
            chat = chat.with_mcp_servers([weather_server])
        """
        try:
            from patterpunk.lib.mcp import MCPClient
            from patterpunk.lib.mcp.tool_converter import mcp_tools_to_patterpunk_tools
        except ImportError as e:
            raise ImportError(f"MCP functionality requires missing dependencies: {e}")

        new_chat = self.copy()
        new_chat._mcp_client = MCPClient(server_configs)
        
        try:
            mcp_tools = new_chat._mcp_client.get_available_tools()
            patterpunk_tools = mcp_tools_to_patterpunk_tools(mcp_tools)
            
            if new_chat.tools:
                new_chat.tools.extend(patterpunk_tools)
            else:
                new_chat.tools = patterpunk_tools
                
        except Exception as e:
            logger.warning(f"Failed to initialize MCP servers: {e}")
            
        return new_chat

    def complete(self):
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
        
        if new_chat.is_latest_message_tool_call and new_chat._mcp_client:
            new_chat = new_chat._execute_mcp_tool_calls()
            
        return new_chat

    def _execute_mcp_tool_calls(self):
        """Execute MCP tool calls from the latest ToolCallMessage."""
        if not isinstance(self.latest_message, ToolCallMessage):
            return self
            
        new_chat = self
        
        for tool_call in self.latest_message.tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"]["arguments"]
                tool_call_id = tool_call["id"]
                
                import json
                arguments = json.loads(arguments_str) if arguments_str else {}
                
                from patterpunk.lib.mcp.tool_converter import extract_mcp_server_from_tool_call
                server_name = extract_mcp_server_from_tool_call(function_name, self.tools or [])
                
                result = self._mcp_client.call_tool(function_name, arguments, server_name)
                
                tool_result_message = UserMessage(
                    f"Tool '{function_name}' returned: {result}",
                    allow_tool_calls=True
                )
                
                new_chat = new_chat.add_message(tool_result_message)
                
            except Exception as e:
                logger.error(f"Failed to execute MCP tool call '{function_name}': {e}")
                error_message = UserMessage(
                    f"Tool '{function_name}' failed with error: {str(e)}",
                    allow_tool_calls=True
                )
                new_chat = new_chat.add_message(error_message)
        
        return new_chat.complete()

    def extract_json(self) -> Optional[List[str]]:
        """Extracts any json in any non-user message"""
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
        if not getattr(self.latest_message, "structured_output", None):
            return None

        retry = 0
        max_reties = 2

        chat = self

        while retry < max_reties:
            try:
                obj = chat.latest_message.parsed_output
                return obj
            except StructuredOutputFailedToParseError as error:
                logger.debug(
                    "[CHAT] Failed to parse structured_output from latest message",
                    exc_info=error,
                )
                chat = chat.add_message(
                    UserMessage(
                        "You did not generate valid JSON! YOUR RESPONSE HAS TO BE A VALID JSON OBJECT THAT CONFORMS TO THE JSON SCHEMA!",
                        structured_output=chat.latest_message.structured_output,
                    )
                ).complete()
                retry += 1

        raise StructuredOutputParsingError(
            f"[CHAT] Failed to parse structured_output from latest message, latest message:\n{self.latest_message.content}"
        )

    @property
    def latest_message(self):
        return self.messages[-1]

    @property
    def is_latest_message_tool_call(self):
        return isinstance(self.latest_message, ToolCallMessage)

    def copy(self):
        return copy.deepcopy(self)
