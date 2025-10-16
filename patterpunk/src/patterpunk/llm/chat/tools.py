"""
Tool integration for both function tools and MCP servers.

This module handles all tool-related functionality for the Chat class,
including function tool conversion, MCP server integration, and tool execution.
"""

import json
from typing import Union, List, Callable, Optional

from patterpunk.lib.function_to_tool.converter import functions_to_tools
from patterpunk.llm.tool_types import ToolDefinition
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.logger import logger


def configure_tools(chat_instance, tools: Union[ToolDefinition, List[Callable]]):
    """
    Configure tools for a chat instance.

    :param chat_instance: The Chat instance to configure
    :param tools: Either a list of tool definitions in OpenAI format, or a list of annotated Python functions
    :return: New Chat instance with tools set

    Examples:
        # Using annotated functions (recommended)
        def get_weather(location: str, unit: str = "fahrenheit") -> str:
            '''Get current weather for a location.'''
            return f"Weather in {location}"

        chat = configure_tools(chat, [get_weather])

        # Using manual tool definitions (legacy)
        tools = [{"type": "function", "function": {...}}]
        chat = configure_tools(chat, tools)
    """
    new_chat = chat_instance.copy()

    if tools and isinstance(tools[0], dict):
        new_chat.tools = tools
    else:
        new_chat.tools = functions_to_tools(tools)

    return new_chat


def configure_mcp_servers(chat_instance, server_configs):
    """
    Configure MCP servers for a chat instance.

    :param chat_instance: The Chat instance to configure
    :param server_configs: List of MCPServerConfig instances
    :return: New Chat instance with MCP servers configured

    Examples:
        from patterpunk.lib.mcp.server_config import MCPServerConfig

        weather_server = MCPServerConfig(
            name="weather",
            url="http://localhost:8000/mcp"
        )

        chat = configure_mcp_servers(chat, [weather_server])
    """
    try:
        from patterpunk.lib.mcp.client import MCPClient
        from patterpunk.lib.mcp.tool_converter import mcp_tools_to_patterpunk_tools
    except ImportError as e:
        raise ImportError(f"MCP functionality requires missing dependencies: {e}")

    new_chat = chat_instance.copy()
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


def execute_mcp_tool_calls(chat_instance):
    """
    Execute MCP tool calls from the latest ToolCallMessage.

    :param chat_instance: The Chat instance with tool calls to execute
    :return: New Chat instance with tool execution results
    """
    if not isinstance(chat_instance.latest_message, ToolCallMessage):
        return chat_instance

    if not chat_instance._mcp_client:
        return chat_instance

    new_chat = chat_instance.copy()

    for tool_call in chat_instance.latest_message.tool_calls:
        try:
            function_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

            arguments = json.loads(arguments_str) if arguments_str else {}

            from patterpunk.lib.mcp.tool_converter import (
                extract_mcp_server_from_tool_call,
            )

            server_name = extract_mcp_server_from_tool_call(
                function_name, chat_instance.tools or []
            )

            result = chat_instance._mcp_client.call_tool(
                function_name, arguments, server_name
            )

            # Create ToolResultMessage with proper linkage
            tool_result_message = ToolResultMessage(
                content=str(result),
                call_id=tool_call_id,
                function_name=function_name,
                is_error=False,
            )

            new_chat = new_chat.add_message(tool_result_message)

        except Exception as e:
            logger.error(f"Failed to execute MCP tool call '{function_name}': {e}")

            # Create ToolResultMessage for errors with is_error=True
            error_message = ToolResultMessage(
                content=f"Tool '{function_name}' failed with error: {str(e)}",
                call_id=tool_call_id,
                function_name=function_name,
                is_error=True,
            )
            new_chat = new_chat.add_message(error_message)

    return new_chat.complete()
