"""
Unit tests for ToolResultMessage class.

Tests the ToolResultMessage class independently of any provider implementation.
"""

import pytest
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.messages.roles import ROLE_TOOL_RESULT


class TestToolResultMessageCreation:
    """Test ToolResultMessage instantiation with different parameter combinations."""

    def test_create_with_content_only(self):
        """Test creating ToolResultMessage with only content."""
        msg = ToolResultMessage(content="Result: 42")

        assert msg.content == "Result: 42"
        assert msg.role == ROLE_TOOL_RESULT
        assert msg.call_id is None
        assert msg.function_name is None
        assert msg.is_error is False

    def test_create_with_call_id(self):
        """Test creating ToolResultMessage with call_id."""
        msg = ToolResultMessage(content="Result: 42", call_id="call_abc123")

        assert msg.content == "Result: 42"
        assert msg.call_id == "call_abc123"
        assert msg.function_name is None
        assert msg.is_error is False

    def test_create_with_function_name(self):
        """Test creating ToolResultMessage with function_name."""
        msg = ToolResultMessage(content="Result: 42", function_name="get_weather")

        assert msg.content == "Result: 42"
        assert msg.call_id is None
        assert msg.function_name == "get_weather"
        assert msg.is_error is False

    def test_create_with_both_call_id_and_function_name(self):
        """Test creating ToolResultMessage with both call_id and function_name."""
        msg = ToolResultMessage(
            content="Result: 42", call_id="call_abc123", function_name="get_weather"
        )

        assert msg.content == "Result: 42"
        assert msg.call_id == "call_abc123"
        assert msg.function_name == "get_weather"
        assert msg.is_error is False

    def test_create_with_error_flag(self):
        """Test creating ToolResultMessage with is_error=True."""
        msg = ToolResultMessage(
            content="Tool execution failed: Connection timeout",
            call_id="call_abc123",
            is_error=True,
        )

        assert msg.content == "Tool execution failed: Connection timeout"
        assert msg.call_id == "call_abc123"
        assert msg.is_error is True

    def test_create_with_all_parameters(self):
        """Test creating ToolResultMessage with all parameters."""
        msg = ToolResultMessage(
            content="Error: Invalid location",
            call_id="call_abc123",
            function_name="get_weather",
            is_error=True,
        )

        assert msg.content == "Error: Invalid location"
        assert msg.call_id == "call_abc123"
        assert msg.function_name == "get_weather"
        assert msg.is_error is True


class TestToolResultMessageToDict:
    """Test ToolResultMessage.to_dict() method."""

    def test_to_dict_with_minimal_data(self):
        """Test to_dict() with only content."""
        msg = ToolResultMessage(content="Result: 42")
        result = msg.to_dict()

        assert result == {
            "role": ROLE_TOOL_RESULT,
            "content": "Result: 42",
        }

    def test_to_dict_with_call_id(self):
        """Test to_dict() includes call_id when present."""
        msg = ToolResultMessage(content="Result: 42", call_id="call_abc123")
        result = msg.to_dict()

        assert result == {
            "role": ROLE_TOOL_RESULT,
            "content": "Result: 42",
            "call_id": "call_abc123",
        }

    def test_to_dict_with_function_name(self):
        """Test to_dict() includes function_name when present."""
        msg = ToolResultMessage(content="Result: 42", function_name="get_weather")
        result = msg.to_dict()

        assert result == {
            "role": ROLE_TOOL_RESULT,
            "content": "Result: 42",
            "function_name": "get_weather",
        }

    def test_to_dict_with_error_flag(self):
        """Test to_dict() includes is_error when True."""
        msg = ToolResultMessage(content="Error occurred", is_error=True)
        result = msg.to_dict()

        assert result == {
            "role": ROLE_TOOL_RESULT,
            "content": "Error occurred",
            "is_error": True,
        }

    def test_to_dict_excludes_error_when_false(self):
        """Test to_dict() excludes is_error when False."""
        msg = ToolResultMessage(content="Result: 42", is_error=False)
        result = msg.to_dict()

        assert result == {
            "role": ROLE_TOOL_RESULT,
            "content": "Result: 42",
        }
        assert "is_error" not in result

    def test_to_dict_with_all_fields(self):
        """Test to_dict() with all fields populated."""
        msg = ToolResultMessage(
            content="Error: Invalid input",
            call_id="call_abc123",
            function_name="process_data",
            is_error=True,
        )
        result = msg.to_dict()

        assert result == {
            "role": ROLE_TOOL_RESULT,
            "content": "Error: Invalid input",
            "call_id": "call_abc123",
            "function_name": "process_data",
            "is_error": True,
        }


class TestToolResultMessageRepr:
    """Test ToolResultMessage.__repr__() method."""

    def test_repr_with_no_linkage(self):
        """Test __repr__() with no call_id or function_name."""
        msg = ToolResultMessage(content="Short result")
        repr_str = repr(msg)

        assert "ToolResultMessage" in repr_str
        assert "no_linkage" in repr_str
        assert "Short result" in repr_str

    def test_repr_with_call_id(self):
        """Test __repr__() includes call_id."""
        msg = ToolResultMessage(content="Result", call_id="call_123")
        repr_str = repr(msg)

        assert "ToolResultMessage" in repr_str
        assert "call_id=call_123" in repr_str
        assert "Result" in repr_str

    def test_repr_with_function_name(self):
        """Test __repr__() includes function_name."""
        msg = ToolResultMessage(content="Result", function_name="get_weather")
        repr_str = repr(msg)

        assert "ToolResultMessage" in repr_str
        assert "function=get_weather" in repr_str
        assert "Result" in repr_str

    def test_repr_with_error_flag(self):
        """Test __repr__() includes is_error flag."""
        msg = ToolResultMessage(
            content="Error occurred", call_id="call_123", is_error=True
        )
        repr_str = repr(msg)

        assert "ToolResultMessage" in repr_str
        assert "call_id=call_123" in repr_str
        assert "is_error=True" in repr_str

    def test_repr_truncates_long_content(self):
        """Test __repr__() truncates content longer than 30 characters."""
        long_content = "This is a very long result that should be truncated for display"
        msg = ToolResultMessage(content=long_content, call_id="call_123")
        repr_str = repr(msg)

        assert "ToolResultMessage" in repr_str
        assert "..." in repr_str
        # Should show first 30 chars
        assert long_content[:30] in repr_str
        # Should NOT show full content
        assert long_content not in repr_str

    def test_repr_no_truncation_with_truncate_false(self):
        """Test __repr__() with truncate=False shows full content."""
        long_content = "This is a very long result that should NOT be truncated"
        msg = ToolResultMessage(content=long_content)
        repr_str = msg.__repr__(truncate=False)

        assert "ToolResultMessage" in repr_str
        assert long_content in repr_str
        assert "..." not in repr_str

    def test_repr_with_all_metadata(self):
        """Test __repr__() with all metadata fields."""
        msg = ToolResultMessage(
            content="Result", call_id="call_123", function_name="process", is_error=True
        )
        repr_str = repr(msg)

        assert "ToolResultMessage" in repr_str
        assert "call_id=call_123" in repr_str
        assert "function=process" in repr_str
        assert "is_error=True" in repr_str


class TestToolResultMessageInheritance:
    """Test ToolResultMessage inherits correctly from Message base class."""

    def test_inherits_from_message(self):
        """Test ToolResultMessage has Message base class."""
        msg = ToolResultMessage(content="Result")

        # Check inheritance
        from patterpunk.llm.messages.base import Message

        assert isinstance(msg, Message)

    def test_has_role_attribute(self):
        """Test ToolResultMessage has role attribute from base class."""
        msg = ToolResultMessage(content="Result")

        assert hasattr(msg, "role")
        assert msg.role == ROLE_TOOL_RESULT

    def test_has_content_attribute(self):
        """Test ToolResultMessage has content attribute from base class."""
        msg = ToolResultMessage(content="Result: 42")

        assert hasattr(msg, "content")
        assert msg.content == "Result: 42"


class TestToolResultMessageEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_content(self):
        """Test creating ToolResultMessage with empty content."""
        msg = ToolResultMessage(content="")

        assert msg.content == ""
        assert msg.role == ROLE_TOOL_RESULT

    def test_content_with_special_characters(self):
        """Test content with special characters is preserved."""
        special_content = 'Result: {"key": "value", "number": 123, "special": "\\n\\t"}'
        msg = ToolResultMessage(content=special_content, call_id="call_123")

        assert msg.content == special_content
        assert special_content in msg.to_dict()["content"]

    def test_very_long_content(self):
        """Test ToolResultMessage with very long content."""
        long_content = "X" * 10000
        msg = ToolResultMessage(content=long_content)

        assert msg.content == long_content
        assert len(msg.content) == 10000

        # Verify repr truncates it
        repr_str = repr(msg)
        assert len(repr_str) < 200  # Should be much shorter than 10000

    def test_unicode_content(self):
        """Test ToolResultMessage with Unicode content."""
        unicode_content = "Result: 天気 is 晴れ (sunny) 🌞"
        msg = ToolResultMessage(content=unicode_content, function_name="get_weather")

        assert msg.content == unicode_content
        assert msg.to_dict()["content"] == unicode_content

    def test_multiline_content(self):
        """Test ToolResultMessage with multiline content."""
        multiline_content = """Line 1: Result
Line 2: Details
Line 3: More info"""
        msg = ToolResultMessage(content=multiline_content)

        assert msg.content == multiline_content
        assert "\n" in msg.content
