"""Tests for message serialization and deserialization."""

import json
import pytest
from datetime import timedelta
from pydantic import BaseModel, Field

from patterpunk.llm.chunks import TextChunk, CacheChunk, MultimodalChunk
from patterpunk.llm.messages import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCallMessage,
    ToolResultMessage,
    message_from_dict,
    serialize_message,
    DynamicStructuredOutput,
)
from patterpunk.llm.tool_types import ToolCall


# Test Pydantic model for structured output tests
class TestResponseModel(BaseModel):
    """A simple Pydantic model for testing structured output serialization."""

    name: str = Field(description="A name field")
    value: int = Field(description="A numeric value")


class TestTextChunkSerialization:
    def test_text_chunk_round_trip(self):
        chunk = TextChunk(content="Hello, world!")
        data = chunk.serialize()

        assert data["type"] == "text"
        assert data["content"] == "Hello, world!"

        restored = TextChunk.from_dict(data)
        assert restored.content == chunk.content

    def test_text_chunk_empty_content(self):
        chunk = TextChunk(content="")
        data = chunk.serialize()
        restored = TextChunk.from_dict(data)
        assert restored.content == ""


class TestCacheChunkSerialization:
    def test_cache_chunk_round_trip_basic(self):
        chunk = CacheChunk(content="Cached content", cacheable=True)
        data = chunk.serialize()

        assert data["type"] == "cache"
        assert data["content"] == "Cached content"
        assert data["cacheable"] is True
        assert "ttl_seconds" not in data

        restored = CacheChunk.from_dict(data)
        assert restored.content == chunk.content
        assert restored.cacheable == chunk.cacheable
        assert restored.ttl is None

    def test_cache_chunk_round_trip_with_ttl(self):
        chunk = CacheChunk(
            content="Cached content", cacheable=True, ttl=timedelta(hours=2)
        )
        data = chunk.serialize()

        assert data["ttl_seconds"] == 7200.0

        restored = CacheChunk.from_dict(data)
        assert restored.content == chunk.content
        assert restored.cacheable == chunk.cacheable
        assert restored.ttl == timedelta(hours=2)

    def test_cache_chunk_non_cacheable(self):
        chunk = CacheChunk(content="Not cached", cacheable=False)
        data = chunk.serialize()
        restored = CacheChunk.from_dict(data)
        assert restored.cacheable is False


class TestMultimodalChunkSerialization:
    def test_multimodal_chunk_from_base64_round_trip(self):
        # Create a simple base64-encoded "image" (just some bytes)
        import base64

        original_bytes = b"fake image data for testing"
        base64_data = base64.b64encode(original_bytes).decode("utf-8")

        chunk = MultimodalChunk.from_base64(base64_data, media_type="image/png")
        chunk.filename = "test.png"

        data = chunk.serialize()

        assert data["type"] == "multimodal"
        assert data["media_type"] == "image/png"
        assert data["data"] == base64_data
        assert data["filename"] == "test.png"

        restored = MultimodalChunk.from_dict(data)
        assert restored.to_base64() == base64_data
        assert restored.media_type == "image/png"
        assert restored.filename == "test.png"

    def test_multimodal_chunk_bytes_round_trip(self):
        original_bytes = b"binary content for testing"
        chunk = MultimodalChunk.from_bytes(
            original_bytes, media_type="application/octet-stream", filename="data.bin"
        )

        data = chunk.serialize()
        restored = MultimodalChunk.from_dict(data)

        assert restored.to_bytes() == original_bytes
        assert restored.media_type == "application/octet-stream"


class TestSystemMessageSerialization:
    def test_system_message_string_round_trip(self):
        msg = SystemMessage(content="You are a helpful assistant.")
        data = msg.serialize()

        assert data["type"] == "system"
        assert data["content"]["type"] == "string"
        assert data["content"]["value"] == "You are a helpful assistant."

        restored = SystemMessage.from_dict(data)
        assert restored.content == msg.content
        assert restored.role == "system"

    def test_system_message_chunks_round_trip(self):
        chunks = [
            CacheChunk(content="System prompt part 1", cacheable=True),
            CacheChunk(content="System prompt part 2", cacheable=False),
        ]
        msg = SystemMessage(content=chunks)
        data = msg.serialize()

        assert data["content"]["type"] == "chunks"
        assert len(data["content"]["chunks"]) == 2

        restored = SystemMessage.from_dict(data)
        assert len(restored.content) == 2
        assert restored.content[0].content == "System prompt part 1"
        assert restored.content[0].cacheable is True


class TestUserMessageSerialization:
    def test_user_message_string_round_trip(self):
        msg = UserMessage(content="What is the weather?")
        data = msg.serialize()

        assert data["type"] == "user"
        assert data["allow_tool_calls"] is True

        restored = UserMessage.from_dict(data)
        assert restored.content == msg.content
        assert restored.allow_tool_calls is True

    def test_user_message_allow_tool_calls_false(self):
        msg = UserMessage(content="Just answer directly", allow_tool_calls=False)
        data = msg.serialize()

        assert data["allow_tool_calls"] is False

        restored = UserMessage.from_dict(data)
        assert restored.allow_tool_calls is False

    def test_user_message_with_multimodal_content(self):
        import base64

        image_bytes = b"fake image bytes"
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        chunks = [
            TextChunk(content="What is in this image?"),
            MultimodalChunk.from_base64(image_base64, media_type="image/jpeg"),
        ]
        msg = UserMessage(content=chunks)
        data = msg.serialize()

        assert data["content"]["type"] == "chunks"
        assert len(data["content"]["chunks"]) == 2
        assert data["content"]["chunks"][0]["type"] == "text"
        assert data["content"]["chunks"][1]["type"] == "multimodal"

        restored = UserMessage.from_dict(data)
        assert len(restored.content) == 2
        assert isinstance(restored.content[0], TextChunk)
        assert isinstance(restored.content[1], MultimodalChunk)

    def test_user_message_with_structured_output(self):
        msg = UserMessage(
            content="Give me a name and value", structured_output=TestResponseModel
        )
        data = msg.serialize()

        assert "structured_output" in data
        assert "schema" in data["structured_output"]
        assert "class_ref" in data["structured_output"]
        assert "TestResponseModel" in data["structured_output"]["class_ref"]

        # Round trip - should successfully import the class
        restored = UserMessage.from_dict(data)
        assert restored.structured_output is TestResponseModel


class TestAssistantMessageSerialization:
    def test_assistant_message_string_round_trip(self):
        msg = AssistantMessage(content="Here is the answer.")
        data = msg.serialize()

        assert data["type"] == "assistant"

        restored = AssistantMessage.from_dict(data)
        assert restored.content == msg.content

    def test_assistant_message_with_thinking_blocks(self):
        thinking_blocks = [
            {"type": "thinking", "thinking": "Let me consider this..."},
            {"type": "thinking", "thinking": "The answer should be..."},
        ]
        msg = AssistantMessage(
            content="The answer is 42.", thinking_blocks=thinking_blocks
        )
        data = msg.serialize()

        assert data["thinking_blocks"] == thinking_blocks

        restored = AssistantMessage.from_dict(data)
        assert restored.thinking_blocks == thinking_blocks
        assert restored.has_thinking is True
        assert "Let me consider this" in restored.thinking_text

    def test_assistant_message_with_structured_output(self):
        msg = AssistantMessage(
            content='{"name": "test", "value": 42}',
            structured_output=TestResponseModel,
        )
        data = msg.serialize()

        assert "structured_output" in data

        restored = AssistantMessage.from_dict(data)
        assert restored.structured_output is TestResponseModel


class TestToolCallMessageSerialization:
    def test_tool_call_message_single_call(self):
        tool_calls = [
            ToolCall(
                id="call_123",
                name="get_weather",
                arguments='{"location": "NYC"}',
            )
        ]
        msg = ToolCallMessage(tool_calls=tool_calls)
        data = msg.serialize()

        assert data["type"] == "tool_call"
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["id"] == "call_123"
        assert data["tool_calls"][0]["function"]["name"] == "get_weather"

        restored = ToolCallMessage.from_dict(data)
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "call_123"
        assert restored.tool_calls[0].name == "get_weather"
        assert restored.tool_calls[0].arguments == '{"location": "NYC"}'

    def test_tool_call_message_multiple_calls(self):
        tool_calls = [
            ToolCall(id="call_1", name="func_a", arguments="{}"),
            ToolCall(id="call_2", name="func_b", arguments='{"x": 1}'),
        ]
        msg = ToolCallMessage(tool_calls=tool_calls)
        data = msg.serialize()

        assert len(data["tool_calls"]) == 2

        restored = ToolCallMessage.from_dict(data)
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].name == "func_a"
        assert restored.tool_calls[1].name == "func_b"

    def test_tool_call_message_with_thinking_blocks(self):
        tool_calls = [ToolCall(id="call_1", name="search", arguments="{}")]
        thinking_blocks = [{"type": "thinking", "thinking": "I should search..."}]
        msg = ToolCallMessage(tool_calls=tool_calls, thinking_blocks=thinking_blocks)
        data = msg.serialize()

        assert data["thinking_blocks"] == thinking_blocks

        restored = ToolCallMessage.from_dict(data)
        assert restored.thinking_blocks == thinking_blocks


class TestToolResultMessageSerialization:
    def test_tool_result_message_basic(self):
        msg = ToolResultMessage(content="The weather is sunny.")
        data = msg.serialize()

        assert data["type"] == "tool_result"
        assert data["content"] == "The weather is sunny."
        assert "call_id" not in data
        assert "function_name" not in data
        assert "is_error" not in data

        restored = ToolResultMessage.from_dict(data)
        assert restored.content == "The weather is sunny."
        assert restored.call_id is None
        assert restored.is_error is False

    def test_tool_result_message_with_call_id(self):
        msg = ToolResultMessage(content="Result", call_id="call_123")
        data = msg.serialize()

        assert data["call_id"] == "call_123"

        restored = ToolResultMessage.from_dict(data)
        assert restored.call_id == "call_123"

    def test_tool_result_message_with_function_name(self):
        msg = ToolResultMessage(content="Result", function_name="get_weather")
        data = msg.serialize()

        assert data["function_name"] == "get_weather"

        restored = ToolResultMessage.from_dict(data)
        assert restored.function_name == "get_weather"

    def test_tool_result_message_error(self):
        msg = ToolResultMessage(
            content="Error: API failed",
            call_id="call_456",
            is_error=True,
        )
        data = msg.serialize()

        assert data["is_error"] is True

        restored = ToolResultMessage.from_dict(data)
        assert restored.is_error is True

    def test_tool_result_message_all_fields(self):
        msg = ToolResultMessage(
            content="Success",
            call_id="call_789",
            function_name="process_data",
            is_error=False,
        )
        data = msg.serialize()

        restored = ToolResultMessage.from_dict(data)
        assert restored.content == "Success"
        assert restored.call_id == "call_789"
        assert restored.function_name == "process_data"
        assert restored.is_error is False


class TestMessageFromDict:
    def test_message_from_dict_system(self):
        data = {"type": "system", "content": {"type": "string", "value": "Hello"}}
        msg = message_from_dict(data)
        assert isinstance(msg, SystemMessage)
        assert msg.content == "Hello"

    def test_message_from_dict_user(self):
        data = {
            "type": "user",
            "content": {"type": "string", "value": "Hi"},
            "allow_tool_calls": False,
        }
        msg = message_from_dict(data)
        assert isinstance(msg, UserMessage)
        assert msg.allow_tool_calls is False

    def test_message_from_dict_assistant(self):
        data = {
            "type": "assistant",
            "content": {"type": "string", "value": "Response"},
        }
        msg = message_from_dict(data)
        assert isinstance(msg, AssistantMessage)

    def test_message_from_dict_tool_call(self):
        data = {
            "type": "tool_call",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "fn", "arguments": "{}"},
                }
            ],
        }
        msg = message_from_dict(data)
        assert isinstance(msg, ToolCallMessage)

    def test_message_from_dict_tool_result(self):
        data = {"type": "tool_result", "content": "Result"}
        msg = message_from_dict(data)
        assert isinstance(msg, ToolResultMessage)

    def test_message_from_dict_unknown_type(self):
        data = {"type": "unknown", "content": "test"}
        with pytest.raises(ValueError, match="Unknown message type"):
            message_from_dict(data)


class TestDynamicStructuredOutput:
    def test_dynamic_structured_output_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        dso = DynamicStructuredOutput(schema)

        assert dso.model_json_schema() == schema

    def test_dynamic_structured_output_validate_json(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        dso = DynamicStructuredOutput(schema)

        result = dso.model_validate_json('{"name": "test"}')
        assert result == {"name": "test"}

    def test_dynamic_structured_output_validate(self):
        schema = {"type": "object"}
        dso = DynamicStructuredOutput(schema)

        result = dso.model_validate({"key": "value"})
        assert result == {"key": "value"}


class TestStructuredOutputDynamicImport:
    def test_structured_output_import_success(self):
        """Test that structured_output can be dynamically imported."""
        msg = UserMessage(content="Test", structured_output=TestResponseModel)
        data = msg.serialize()

        # Verify class_ref is stored
        assert (
            "tests.test_serialization.TestResponseModel"
            in data["structured_output"]["class_ref"]
        )

        # Deserialize and verify import worked
        restored = UserMessage.from_dict(data)
        assert restored.structured_output is TestResponseModel

    def test_structured_output_fallback_to_dynamic(self):
        """Test fallback to DynamicStructuredOutput when import fails."""
        data = {
            "type": "user",
            "content": {"type": "string", "value": "Test"},
            "structured_output": {
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
                "class_ref": "nonexistent.module.FakeModel",
            },
            "allow_tool_calls": True,
        }

        restored = UserMessage.from_dict(data)

        # Should fall back to DynamicStructuredOutput
        assert isinstance(restored.structured_output, DynamicStructuredOutput)
        assert restored.structured_output.model_json_schema() == {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }


class TestFullConversationRoundTrip:
    def test_complete_conversation(self):
        """Test serializing and deserializing a complete conversation."""
        messages = [
            SystemMessage("You are a helpful assistant."),
            UserMessage("What is 2+2?"),
            AssistantMessage(
                "The answer is 4.",
                thinking_blocks=[
                    {"type": "thinking", "thinking": "Simple addition..."}
                ],
            ),
            UserMessage("Thanks! Now call a tool."),
            ToolCallMessage(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="calculator",
                        arguments='{"op": "multiply", "a": 3, "b": 4}',
                    )
                ]
            ),
            ToolResultMessage(
                content="12", call_id="call_1", function_name="calculator"
            ),
            AssistantMessage("The result of 3 times 4 is 12."),
        ]

        # Serialize all messages
        serialized = [msg.serialize() for msg in messages]

        # Verify it's JSON-serializable
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        # Deserialize all messages
        restored = [message_from_dict(data) for data in parsed]

        # Verify types and content
        assert len(restored) == 7
        assert isinstance(restored[0], SystemMessage)
        assert isinstance(restored[1], UserMessage)
        assert isinstance(restored[2], AssistantMessage)
        assert isinstance(restored[3], UserMessage)
        assert isinstance(restored[4], ToolCallMessage)
        assert isinstance(restored[5], ToolResultMessage)
        assert isinstance(restored[6], AssistantMessage)

        # Verify specific content
        assert restored[0].content == "You are a helpful assistant."
        assert restored[2].thinking_blocks[0]["thinking"] == "Simple addition..."
        assert restored[4].tool_calls[0].name == "calculator"
        assert restored[5].call_id == "call_1"
