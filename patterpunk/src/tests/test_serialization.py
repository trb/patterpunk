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
    deserialize_message,
    serialize_message,
    DynamicStructuredOutput,
)
from patterpunk.llm.messages.serialization import deserialize_content
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

        restored = TextChunk.deserialize(data)
        assert restored.content == chunk.content

    def test_text_chunk_empty_content(self):
        chunk = TextChunk(content="")
        data = chunk.serialize()
        restored = TextChunk.deserialize(data)
        assert restored.content == ""


class TestCacheChunkSerialization:
    def test_cache_chunk_round_trip_basic(self):
        chunk = CacheChunk(content="Cached content", cacheable=True)
        data = chunk.serialize()

        assert data["type"] == "cache"
        assert data["content"] == "Cached content"
        assert data["cacheable"] is True
        assert "ttl_seconds" not in data

        restored = CacheChunk.deserialize(data)
        assert restored.content == chunk.content
        assert restored.cacheable == chunk.cacheable
        assert restored.ttl is None

    def test_cache_chunk_round_trip_with_ttl(self):
        chunk = CacheChunk(
            content="Cached content", cacheable=True, ttl=timedelta(hours=2)
        )
        data = chunk.serialize()

        assert data["ttl_seconds"] == 7200.0

        restored = CacheChunk.deserialize(data)
        assert restored.content == chunk.content
        assert restored.cacheable == chunk.cacheable
        assert restored.ttl == timedelta(hours=2)

    def test_cache_chunk_non_cacheable(self):
        chunk = CacheChunk(content="Not cached", cacheable=False)
        data = chunk.serialize()
        restored = CacheChunk.deserialize(data)
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

        restored = MultimodalChunk.deserialize(data)
        assert restored.to_base64() == base64_data
        assert restored.media_type == "image/png"
        assert restored.filename == "test.png"

    def test_multimodal_chunk_bytes_round_trip(self):
        original_bytes = b"binary content for testing"
        chunk = MultimodalChunk.from_bytes(
            original_bytes, media_type="application/octet-stream", filename="data.bin"
        )

        data = chunk.serialize()
        restored = MultimodalChunk.deserialize(data)

        assert restored.to_bytes() == original_bytes
        assert restored.media_type == "application/octet-stream"


class TestSystemMessageSerialization:
    def test_system_message_string_round_trip(self):
        msg = SystemMessage(content="You are a helpful assistant.")
        data = msg.serialize()

        assert data["type"] == "system"
        assert data["content"]["type"] == "string"
        assert data["content"]["value"] == "You are a helpful assistant."

        restored = SystemMessage.deserialize(data)
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

        restored = SystemMessage.deserialize(data)
        # Validate ALL chunks, not just the first
        assert len(restored.content) == 2
        assert restored.content[0].content == "System prompt part 1"
        assert restored.content[0].cacheable is True
        assert restored.content[1].content == "System prompt part 2"
        assert restored.content[1].cacheable is False


class TestUserMessageSerialization:
    def test_user_message_string_round_trip(self):
        msg = UserMessage(content="What is the weather?")
        data = msg.serialize()

        assert data["type"] == "user"
        assert data["allow_tool_calls"] is True

        restored = UserMessage.deserialize(data)
        assert restored.content == msg.content
        assert restored.allow_tool_calls is True

    def test_user_message_allow_tool_calls_false(self):
        msg = UserMessage(content="Just answer directly", allow_tool_calls=False)
        data = msg.serialize()

        assert data["allow_tool_calls"] is False

        restored = UserMessage.deserialize(data)
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

        restored = UserMessage.deserialize(data)
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
        restored = UserMessage.deserialize(data)
        assert restored.structured_output is TestResponseModel


class TestAssistantMessageSerialization:
    def test_assistant_message_string_round_trip(self):
        msg = AssistantMessage(content="Here is the answer.")
        data = msg.serialize()

        assert data["type"] == "assistant"

        restored = AssistantMessage.deserialize(data)
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

        restored = AssistantMessage.deserialize(data)
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

        restored = AssistantMessage.deserialize(data)
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

        restored = ToolCallMessage.deserialize(data)
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "call_123"
        assert restored.tool_calls[0].name == "get_weather"
        assert restored.tool_calls[0].arguments == '{"location": "NYC"}'

    def test_tool_call_message_multiple_calls(self):
        tool_calls = [
            ToolCall(id="call_1", name="func_a", arguments='{"a": 1}'),
            ToolCall(id="call_2", name="func_b", arguments='{"b": 2}'),
        ]
        msg = ToolCallMessage(tool_calls=tool_calls)
        data = msg.serialize()

        assert len(data["tool_calls"]) == 2

        restored = ToolCallMessage.deserialize(data)
        # Validate ALL tool calls completely
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].id == "call_1"
        assert restored.tool_calls[0].name == "func_a"
        assert restored.tool_calls[0].arguments == '{"a": 1}'
        assert restored.tool_calls[1].id == "call_2"
        assert restored.tool_calls[1].name == "func_b"
        assert restored.tool_calls[1].arguments == '{"b": 2}'

    def test_tool_call_message_with_thinking_blocks(self):
        tool_calls = [ToolCall(id="call_1", name="search", arguments="{}")]
        thinking_blocks = [{"type": "thinking", "thinking": "I should search..."}]
        msg = ToolCallMessage(tool_calls=tool_calls, thinking_blocks=thinking_blocks)
        data = msg.serialize()

        assert data["thinking_blocks"] == thinking_blocks

        restored = ToolCallMessage.deserialize(data)
        assert restored.thinking_blocks == thinking_blocks


class TestToolResultMessageSerialization:
    def test_tool_result_message_basic(self):
        msg = ToolResultMessage(content="The weather is sunny.")
        data = msg.serialize()

        assert data["type"] == "tool_result"
        assert data["content"] == "The weather is sunny."
        # Use explicit None/absence checks
        assert data.get("call_id") is None
        assert data.get("function_name") is None
        assert data.get("is_error") in (None, False)

        restored = ToolResultMessage.deserialize(data)
        assert restored.content == "The weather is sunny."
        assert restored.call_id is None
        assert restored.function_name is None
        assert restored.is_error is False

    def test_tool_result_message_with_call_id(self):
        msg = ToolResultMessage(content="Result", call_id="call_123")
        data = msg.serialize()

        assert data["call_id"] == "call_123"

        restored = ToolResultMessage.deserialize(data)
        assert restored.call_id == "call_123"

    def test_tool_result_message_with_function_name(self):
        msg = ToolResultMessage(content="Result", function_name="get_weather")
        data = msg.serialize()

        assert data["function_name"] == "get_weather"

        restored = ToolResultMessage.deserialize(data)
        assert restored.function_name == "get_weather"

    def test_tool_result_message_error(self):
        msg = ToolResultMessage(
            content="Error: API failed",
            call_id="call_456",
            is_error=True,
        )
        data = msg.serialize()

        assert data["is_error"] is True

        restored = ToolResultMessage.deserialize(data)
        assert restored.is_error is True

    def test_tool_result_message_all_fields(self):
        msg = ToolResultMessage(
            content="Success",
            call_id="call_789",
            function_name="process_data",
            is_error=False,
        )
        data = msg.serialize()

        restored = ToolResultMessage.deserialize(data)
        assert restored.content == "Success"
        assert restored.call_id == "call_789"
        assert restored.function_name == "process_data"
        assert restored.is_error is False


class TestDeserializeMessage:
    def test_deserialize_message_system(self):
        data = {"type": "system", "content": {"type": "string", "value": "Hello"}}
        msg = deserialize_message(data)
        assert isinstance(msg, SystemMessage)
        assert msg.content == "Hello"

    def test_deserialize_message_user(self):
        data = {
            "type": "user",
            "content": {"type": "string", "value": "Hi"},
            "allow_tool_calls": False,
        }
        msg = deserialize_message(data)
        assert isinstance(msg, UserMessage)
        assert msg.allow_tool_calls is False

    def test_deserialize_message_assistant(self):
        data = {
            "type": "assistant",
            "content": {"type": "string", "value": "Response"},
        }
        msg = deserialize_message(data)
        assert isinstance(msg, AssistantMessage)

    def test_deserialize_message_tool_call(self):
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
        msg = deserialize_message(data)
        assert isinstance(msg, ToolCallMessage)

    def test_deserialize_message_tool_result(self):
        data = {"type": "tool_result", "content": "Result"}
        msg = deserialize_message(data)
        assert isinstance(msg, ToolResultMessage)

    def test_deserialize_message_unknown_type(self):
        data = {"type": "unknown", "content": "test"}
        with pytest.raises(ValueError, match="Unknown message type"):
            deserialize_message(data)


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
        restored = UserMessage.deserialize(data)
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

        restored = UserMessage.deserialize(data)

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
        restored = [deserialize_message(data) for data in parsed]

        # Verify types
        assert len(restored) == 7
        assert isinstance(restored[0], SystemMessage)
        assert isinstance(restored[1], UserMessage)
        assert isinstance(restored[2], AssistantMessage)
        assert isinstance(restored[3], UserMessage)
        assert isinstance(restored[4], ToolCallMessage)
        assert isinstance(restored[5], ToolResultMessage)
        assert isinstance(restored[6], AssistantMessage)

        # Verify ALL messages content comprehensively
        # Message 0: SystemMessage
        assert restored[0].content == "You are a helpful assistant."
        assert restored[0].role == "system"

        # Message 1: UserMessage
        assert restored[1].content == "What is 2+2?"
        assert restored[1].allow_tool_calls is True

        # Message 2: AssistantMessage with thinking
        assert restored[2].content == "The answer is 4."
        assert len(restored[2].thinking_blocks) == 1
        assert restored[2].thinking_blocks[0]["thinking"] == "Simple addition..."
        assert restored[2].has_thinking is True

        # Message 3: UserMessage
        assert restored[3].content == "Thanks! Now call a tool."

        # Message 4: ToolCallMessage
        assert len(restored[4].tool_calls) == 1
        assert restored[4].tool_calls[0].id == "call_1"
        assert restored[4].tool_calls[0].name == "calculator"
        assert (
            restored[4].tool_calls[0].arguments == '{"op": "multiply", "a": 3, "b": 4}'
        )

        # Message 5: ToolResultMessage
        assert restored[5].content == "12"
        assert restored[5].call_id == "call_1"
        assert restored[5].function_name == "calculator"
        assert restored[5].is_error is False

        # Message 6: AssistantMessage
        assert restored[6].content == "The result of 3 times 4 is 12."


class TestSerializationErrorHandling:
    """Tests for error handling in serialization/deserialization."""

    def test_deserialize_content_missing_type(self):
        """Content without 'type' field should raise ValueError."""
        with pytest.raises(ValueError, match="missing required 'type' field"):
            deserialize_content({"value": "test"})

    def test_deserialize_content_unknown_type(self):
        """Unknown content type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown content type"):
            deserialize_content({"type": "unknown", "value": "test"})

    def test_deserialize_content_missing_value(self):
        """String content without 'value' should raise ValueError."""
        with pytest.raises(ValueError, match="missing required 'value' field"):
            deserialize_content({"type": "string"})

    def test_deserialize_content_missing_chunks(self):
        """Chunks content without 'chunks' should raise ValueError."""
        with pytest.raises(ValueError, match="missing required 'chunks' field"):
            deserialize_content({"type": "chunks"})

    def test_deserialize_content_unknown_chunk_type(self):
        """Unknown chunk type in list should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown chunk type"):
            deserialize_content(
                {"type": "chunks", "chunks": [{"type": "unknown", "content": "test"}]}
            )

    def test_deserialize_message_missing_type(self):
        """Message without type or role should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown message type"):
            deserialize_message({"content": "test"})


class TestSerializationEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_content(self):
        """Empty string should serialize and deserialize correctly."""
        msg = UserMessage(content="")
        data = msg.serialize()
        restored = UserMessage.deserialize(data)
        assert restored.content == ""

    def test_unicode_content(self):
        """Unicode characters should be preserved."""
        content = "Hello 世界! 🎉 Ñoño"
        msg = UserMessage(content=content)
        data = msg.serialize()
        restored = UserMessage.deserialize(data)
        assert restored.content == content

    def test_empty_tool_calls_list(self):
        """Empty tool_calls list should serialize correctly."""
        msg = ToolCallMessage(tool_calls=[])
        data = msg.serialize()
        assert data["tool_calls"] == []
        restored = ToolCallMessage.deserialize(data)
        assert len(restored.tool_calls) == 0

    def test_mixed_chunk_types(self):
        """Mixed chunk types should all be preserved."""
        import base64

        chunks = [
            TextChunk(content="Text part"),
            CacheChunk(content="Cached part", cacheable=True),
            MultimodalChunk.from_base64(
                base64.b64encode(b"image").decode(), media_type="image/png"
            ),
        ]
        msg = UserMessage(content=chunks)
        data = msg.serialize()
        restored = UserMessage.deserialize(data)

        assert len(restored.content) == 3
        assert isinstance(restored.content[0], TextChunk)
        assert restored.content[0].content == "Text part"
        assert isinstance(restored.content[1], CacheChunk)
        assert restored.content[1].content == "Cached part"
        assert restored.content[1].cacheable is True
        assert isinstance(restored.content[2], MultimodalChunk)

    def test_legacy_role_field_fallback(self):
        """Legacy 'role' field should work for deserialization."""
        data = {"role": "system", "content": {"type": "string", "value": "test"}}
        msg = deserialize_message(data)
        assert isinstance(msg, SystemMessage)
        assert msg.content == "test"

    def test_hyphenated_type_names_tool_call(self):
        """Hyphenated type name 'tool-call' should work."""
        data = {
            "type": "tool-call",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "fn", "arguments": "{}"},
                }
            ],
        }
        msg = deserialize_message(data)
        assert isinstance(msg, ToolCallMessage)

    def test_hyphenated_type_names_tool_result(self):
        """Hyphenated type name 'tool-result' should work."""
        data = {"type": "tool-result", "content": "result"}
        msg = deserialize_message(data)
        assert isinstance(msg, ToolResultMessage)

    def test_cache_chunk_zero_ttl(self):
        """Zero TTL is treated as no TTL (limitation: timedelta(0) is falsy)."""
        chunk = CacheChunk(content="test", cacheable=True, ttl=timedelta(seconds=0))
        data = chunk.serialize()
        # Note: Zero TTL is not serialized because timedelta(0) is falsy
        assert "ttl_seconds" not in data or data["ttl_seconds"] == 0
        restored = CacheChunk.deserialize(data)
        # Zero TTL deserializes as None due to falsy check
        assert restored.ttl is None

    def test_cache_chunk_small_ttl(self):
        """Small non-zero TTL should be preserved."""
        chunk = CacheChunk(content="test", cacheable=True, ttl=timedelta(seconds=1))
        data = chunk.serialize()
        assert data["ttl_seconds"] == 1.0
        restored = CacheChunk.deserialize(data)
        assert restored.ttl == timedelta(seconds=1)

    def test_multimodal_preserves_bytes(self):
        """Multimodal chunk should preserve exact bytes through round-trip."""
        original_bytes = b"\x00\x01\x02\xff\xfe\xfd"
        chunk = MultimodalChunk.from_bytes(
            original_bytes, media_type="application/octet-stream"
        )
        data = chunk.serialize()
        restored = MultimodalChunk.deserialize(data)
        assert restored.to_bytes() == original_bytes

    def test_whitespace_only_content(self):
        """Whitespace-only content should be preserved."""
        msg = UserMessage(content="   \n\t  ")
        data = msg.serialize()
        restored = UserMessage.deserialize(data)
        assert restored.content == "   \n\t  "

    def test_very_long_content(self):
        """Very long content should serialize correctly."""
        long_content = "x" * 100000
        msg = UserMessage(content=long_content)
        data = msg.serialize()
        restored = UserMessage.deserialize(data)
        assert restored.content == long_content
        assert len(restored.content) == 100000


class TestStructuredOutputEdgeCases:
    """Tests for structured output serialization edge cases."""

    def test_structured_output_malformed_class_ref(self):
        """Malformed class_ref should fall back to DynamicStructuredOutput."""
        data = {
            "type": "user",
            "content": {"type": "string", "value": "test"},
            "structured_output": {
                "schema": {"type": "object"},
                "class_ref": "no_dot_separator",  # Invalid format
            },
            "allow_tool_calls": True,
        }
        restored = UserMessage.deserialize(data)
        assert isinstance(restored.structured_output, DynamicStructuredOutput)

    def test_structured_output_empty_class_ref(self):
        """Empty class_ref should fall back to DynamicStructuredOutput."""
        data = {
            "type": "user",
            "content": {"type": "string", "value": "test"},
            "structured_output": {"schema": {"type": "object"}, "class_ref": ""},
            "allow_tool_calls": True,
        }
        restored = UserMessage.deserialize(data)
        assert isinstance(restored.structured_output, DynamicStructuredOutput)

    def test_structured_output_no_schema_no_class_ref(self):
        """structured_output with neither schema nor class_ref returns None."""
        data = {
            "type": "user",
            "content": {"type": "string", "value": "test"},
            "structured_output": {},
            "allow_tool_calls": True,
        }
        restored = UserMessage.deserialize(data)
        assert restored.structured_output is None

    def test_structured_output_only_schema(self):
        """structured_output with only schema should create DynamicStructuredOutput."""
        data = {
            "type": "user",
            "content": {"type": "string", "value": "test"},
            "structured_output": {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
            },
            "allow_tool_calls": True,
        }
        restored = UserMessage.deserialize(data)
        assert isinstance(restored.structured_output, DynamicStructuredOutput)
        assert restored.structured_output.model_json_schema() == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }


class TestMessageUUID:
    """Tests for message UUID auto-generation and persistence."""

    def test_message_gets_auto_generated_id(self):
        """Messages should auto-generate a UUID when none is provided."""
        msg = UserMessage(content="Hello")
        assert msg.id is not None
        assert len(msg.id) == 36  # UUID format: 8-4-4-4-12

    def test_different_messages_get_different_ids(self):
        """Each message should get a unique ID."""
        msg1 = UserMessage(content="Hello")
        msg2 = UserMessage(content="Hello")  # Same content
        msg3 = UserMessage(content="World")
        assert msg1.id != msg2.id
        assert msg2.id != msg3.id

    def test_provided_id_is_used(self):
        """When an ID is provided, it should be used instead of auto-generating."""
        custom_id = "custom-id-12345"
        msg = UserMessage(content="Hello", id=custom_id)
        assert msg.id == custom_id

    def test_id_preserved_through_serialize_deserialize(self):
        """Message ID should be preserved through serialization round-trip."""
        msg = UserMessage(content="Hello")
        original_id = msg.id

        data = msg.serialize()
        assert data["id"] == original_id

        restored = UserMessage.deserialize(data)
        assert restored.id == original_id

    def test_all_message_types_have_id(self):
        """All message types should have auto-generated IDs."""
        from patterpunk.llm.tool_types import ToolCall

        messages = [
            SystemMessage(content="System"),
            UserMessage(content="User"),
            AssistantMessage(content="Assistant"),
            ToolCallMessage(
                tool_calls=[ToolCall(id="call_1", name="fn", arguments="{}")]
            ),
            ToolResultMessage(content="Result", call_id="call_1"),
        ]

        ids = [msg.id for msg in messages]

        # All should have IDs
        assert all(id is not None for id in ids)
        # All IDs should be unique
        assert len(set(ids)) == len(ids)

    def test_id_in_serialized_output(self):
        """All message types should include ID in serialized output."""
        from patterpunk.llm.tool_types import ToolCall

        messages = [
            SystemMessage(content="System"),
            UserMessage(content="User"),
            AssistantMessage(content="Assistant"),
            ToolCallMessage(
                tool_calls=[ToolCall(id="call_1", name="fn", arguments="{}")]
            ),
            ToolResultMessage(content="Result", call_id="call_1"),
        ]

        for msg in messages:
            data = msg.serialize()
            assert (
                "id" in data
            ), f"{type(msg).__name__} should include 'id' in serialize()"
            assert data["id"] == msg.id

    def test_deserialize_without_id_generates_new_one(self):
        """Deserializing data without an ID should generate a new one."""
        data = {
            "type": "user",
            "content": {"type": "string", "value": "test"},
            "allow_tool_calls": True,
        }
        # No "id" field in data

        restored = UserMessage.deserialize(data)
        assert restored.id is not None
        assert len(restored.id) == 36

    def test_copy_preserves_id(self):
        """Copying a message should preserve the ID."""
        msg = UserMessage(content="Hello")
        copied = msg.copy()
        assert copied.id == msg.id
