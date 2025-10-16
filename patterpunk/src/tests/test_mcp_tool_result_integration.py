"""
MCP (Model Context Protocol) integration tests for ToolResultMessage.

Tests that execute_mcp_tool_calls() correctly creates ToolResultMessage instances
with proper linkage (call_id and function_name) instead of generic UserMessage.
"""

import pytest
from unittest.mock import Mock, MagicMock
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.chat.tools import execute_mcp_tool_calls


class TestMCPToolResultCreation:
    """Test that MCP execution creates proper ToolResultMessage instances."""

    def test_execute_mcp_tool_calls_creates_tool_result_message(self):
        """Test that execute_mcp_tool_calls creates ToolResultMessage, not UserMessage."""
        # Create mock chat instance with ToolCallMessage
        mock_chat = Mock()
        mock_chat.latest_message = ToolCallMessage(
            [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "weather__get_weather",
                        "arguments": '{"location": "Paris"}'
                    }
                }
            ]
        )
        mock_chat.tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather__get_weather",
                    "description": "Get weather for a location",
                    "parameters": {}
                }
            }
        ]

        # Mock MCP client
        mock_mcp_client = Mock()
        mock_mcp_client.call_tool.return_value = "sunny, 22°C"
        mock_chat._mcp_client = mock_mcp_client

        # Mock the copy and add_message methods
        new_chat = Mock()
        new_chat.add_message = Mock(return_value=new_chat)
        new_chat.complete = Mock(return_value=new_chat)
        mock_chat.copy = Mock(return_value=new_chat)

        # Execute MCP tool calls
        result_chat = execute_mcp_tool_calls(mock_chat)

        # Verify add_message was called with ToolResultMessage
        assert new_chat.add_message.called
        added_message = new_chat.add_message.call_args[0][0]

        # CRITICAL: Should be ToolResultMessage, not UserMessage
        assert isinstance(added_message, ToolResultMessage)
        assert not isinstance(added_message, UserMessage) or isinstance(added_message, ToolResultMessage)

        # Verify proper linkage
        assert added_message.content == "sunny, 22°C"
        assert added_message.call_id == "call_abc123"
        assert added_message.function_name == "weather__get_weather"
        assert added_message.is_error is False

    def test_execute_mcp_tool_calls_with_error(self):
        """Test that MCP execution errors create ToolResultMessage with is_error=True."""
        # Create mock chat with ToolCallMessage
        mock_chat = Mock()
        mock_chat.latest_message = ToolCallMessage(
            [
                {
                    "id": "call_error123",
                    "type": "function",
                    "function": {
                        "name": "weather__get_weather",
                        "arguments": '{"location": "InvalidCity"}'
                    }
                }
            ]
        )
        mock_chat.tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather__get_weather",
                    "description": "Get weather",
                    "parameters": {}
                }
            }
        ]

        # Mock MCP client that raises error
        mock_mcp_client = Mock()
        mock_mcp_client.call_tool.side_effect = Exception("Location not found")
        mock_chat._mcp_client = mock_mcp_client

        # Mock copy and add_message
        new_chat = Mock()
        new_chat.add_message = Mock(return_value=new_chat)
        new_chat.complete = Mock(return_value=new_chat)
        mock_chat.copy = Mock(return_value=new_chat)

        # Execute MCP tool calls (should handle error)
        result_chat = execute_mcp_tool_calls(mock_chat)

        # Verify error message was added
        assert new_chat.add_message.called
        error_message = new_chat.add_message.call_args[0][0]

        # Should be ToolResultMessage with is_error=True
        assert isinstance(error_message, ToolResultMessage)
        assert error_message.call_id == "call_error123"
        assert error_message.function_name == "weather__get_weather"
        assert error_message.is_error is True
        assert "Location not found" in error_message.content

    def test_execute_mcp_tool_calls_with_multiple_calls(self):
        """Test MCP execution with multiple tool calls creates multiple ToolResultMessages."""
        # Create mock chat with multiple ToolCallMessages
        mock_chat = Mock()
        mock_chat.latest_message = ToolCallMessage(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather__get_weather",
                        "arguments": '{"location": "Paris"}'
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "weather__get_weather",
                        "arguments": '{"location": "London"}'
                    }
                }
            ]
        )
        mock_chat.tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather__get_weather",
                    "description": "Get weather",
                    "parameters": {}
                }
            }
        ]

        # Mock MCP client with different responses
        mock_mcp_client = Mock()
        mock_mcp_client.call_tool.side_effect = ["sunny, 22°C", "rainy, 15°C"]
        mock_chat._mcp_client = mock_mcp_client

        # Mock copy and add_message to track calls
        call_count = [0]
        added_messages = []

        def mock_add_message(msg):
            added_messages.append(msg)
            call_count[0] += 1
            new_mock = Mock()
            new_mock.add_message = mock_add_message
            new_mock.complete = Mock(return_value=new_mock)
            return new_mock

        new_chat = Mock()
        new_chat.add_message = mock_add_message
        new_chat.complete = Mock(return_value=new_chat)
        mock_chat.copy = Mock(return_value=new_chat)

        # Execute MCP tool calls
        result_chat = execute_mcp_tool_calls(mock_chat)

        # Should have added 2 ToolResultMessages
        assert len(added_messages) == 2

        # First message
        assert isinstance(added_messages[0], ToolResultMessage)
        assert added_messages[0].call_id == "call_1"
        assert added_messages[0].content == "sunny, 22°C"

        # Second message
        assert isinstance(added_messages[1], ToolResultMessage)
        assert added_messages[1].call_id == "call_2"
        assert added_messages[1].content == "rainy, 15°C"

    def test_execute_mcp_tool_calls_preserves_function_name(self):
        """Test that function_name is properly extracted and preserved."""
        mock_chat = Mock()
        mock_chat.latest_message = ToolCallMessage(
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "filesystem__read_file",
                        "arguments": '{"path": "/test.txt"}'
                    }
                }
            ]
        )
        mock_chat.tools = [
            {
                "type": "function",
                "function": {
                    "name": "filesystem__read_file",
                    "description": "Read file",
                    "parameters": {}
                }
            }
        ]

        mock_mcp_client = Mock()
        mock_mcp_client.call_tool.return_value = "file contents"
        mock_chat._mcp_client = mock_mcp_client

        new_chat = Mock()
        new_chat.add_message = Mock(return_value=new_chat)
        new_chat.complete = Mock(return_value=new_chat)
        mock_chat.copy = Mock(return_value=new_chat)

        execute_mcp_tool_calls(mock_chat)

        added_message = new_chat.add_message.call_args[0][0]
        assert added_message.function_name == "filesystem__read_file"

    def test_execute_mcp_tool_calls_does_nothing_without_tool_calls(self):
        """Test that execute_mcp_tool_calls returns chat unchanged if no tool calls."""
        from patterpunk.llm.messages.assistant import AssistantMessage

        mock_chat = Mock()
        mock_chat.latest_message = AssistantMessage("Regular response")
        mock_chat._mcp_client = Mock()

        result = execute_mcp_tool_calls(mock_chat)

        # Should return the original chat unchanged
        assert result == mock_chat

    def test_execute_mcp_tool_calls_without_mcp_client(self):
        """Test that execute_mcp_tool_calls returns chat unchanged if no MCP client."""
        mock_chat = Mock()
        mock_chat.latest_message = ToolCallMessage(
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "some_tool",
                        "arguments": '{}'
                    }
                }
            ]
        )
        mock_chat._mcp_client = None

        result = execute_mcp_tool_calls(mock_chat)

        # Should return the original chat unchanged
        assert result == mock_chat
