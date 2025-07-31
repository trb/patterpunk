import pytest
from datetime import timedelta

from patterpunk.llm.types import CacheChunk
from patterpunk.llm.messages import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
)


class TestCacheChunk:
    def test_cache_chunk_creation(self):
        chunk = CacheChunk(
            content="test content", cacheable=True, ttl=timedelta(hours=1)
        )
        assert chunk.content == "test content"
        assert chunk.cacheable is True
        assert chunk.ttl == timedelta(hours=1)

    def test_cache_chunk_defaults(self):
        chunk = CacheChunk(content="test content")
        assert chunk.content == "test content"
        assert chunk.cacheable is False
        assert chunk.ttl is None

    def test_cache_chunk_repr(self):
        chunk = CacheChunk(
            content="test content", cacheable=True, ttl=timedelta(hours=1)
        )
        repr_str = repr(chunk)
        assert "test content" in repr_str
        assert "cacheable=True" in repr_str
        assert "ttl=1:00:00" in repr_str


class TestMessagePolymorphicContent:
    def test_message_string_content(self):
        message = Message(content="simple string")
        assert isinstance(message.content, str)
        assert message.get_content_as_string() == "simple string"
        assert not message.has_cacheable_content()

    def test_message_chunk_content(self):
        chunks = [
            CacheChunk(content="cacheable", cacheable=True),
            CacheChunk(content="not cacheable", cacheable=False),
        ]
        message = Message(content=chunks)
        assert isinstance(message.content, list)
        assert message.get_content_as_string() == "cacheablenot cacheable"
        assert message.has_cacheable_content()

    def test_get_cache_chunks_from_string(self):
        message = Message(content="simple string")
        chunks = message.get_cache_chunks()
        assert len(chunks) == 1
        assert chunks[0].content == "simple string"
        assert chunks[0].cacheable is False

    def test_get_cache_chunks_from_list(self):
        original_chunks = [
            CacheChunk(content="cacheable", cacheable=True),
            CacheChunk(content="not cacheable", cacheable=False),
        ]
        message = Message(content=original_chunks)
        chunks = message.get_cache_chunks()
        assert chunks == original_chunks


class TestMessageSubclasses:
    def test_system_message_with_chunks(self):
        chunks = [CacheChunk(content="system instructions", cacheable=True)]
        message = SystemMessage(content=chunks)
        assert message.role == "system"
        assert message.has_cacheable_content()

    def test_user_message_with_chunks(self):
        chunks = [CacheChunk(content="user query", cacheable=False)]
        message = UserMessage(content=chunks)
        assert message.role == "user"
        assert not message.has_cacheable_content()

    def test_assistant_message_string_only(self):
        message = AssistantMessage(content="assistant response")
        assert message.role == "assistant"
        assert isinstance(message.content, str)
        assert message.get_content_as_string() == "assistant response"


class TestBackwardCompatibility:
    def test_existing_string_messages_still_work(self):
        system_msg = SystemMessage(content="System instructions")
        user_msg = UserMessage(content="User query")
        assistant_msg = AssistantMessage(content="Assistant response")

        assert system_msg.get_content_as_string() == "System instructions"
        assert user_msg.get_content_as_string() == "User query"
        assert assistant_msg.get_content_as_string() == "Assistant response"

        system_dict = system_msg.to_dict()
        assert system_dict["role"] == "system"
        assert system_dict["content"] == "System instructions"

    def test_message_repr_with_chunks(self):
        chunks = [
            CacheChunk(content="long content that should be truncated", cacheable=True)
        ]
        message = UserMessage(content=chunks)
        repr_str = repr(message)
        assert "UserMessage" in repr_str
        assert "long content that should be truncated" in repr_str


class TestCacheChunkUsagePatterns:
    def test_mixed_cacheable_chunks(self):
        chunks = [
            CacheChunk(content="Static instructions: ", cacheable=False),
            CacheChunk(
                content="Large document content...",
                cacheable=True,
                ttl=timedelta(hours=1),
            ),
            CacheChunk(content=f"Query: What is the main theme?", cacheable=False),
        ]
        message = UserMessage(content=chunks)

        assert message.has_cacheable_content()
        assert (
            message.get_content_as_string()
            == "Static instructions: Large document content...Query: What is the main theme?"
        )

        cache_chunks = message.get_cache_chunks()
        assert len(cache_chunks) == 3
        assert cache_chunks[1].cacheable is True
        assert cache_chunks[1].ttl == timedelta(hours=1)

    def test_all_non_cacheable_chunks(self):
        chunks = [
            CacheChunk(content="Dynamic content 1", cacheable=False),
            CacheChunk(content="Dynamic content 2", cacheable=False),
        ]
        message = UserMessage(content=chunks)

        assert not message.has_cacheable_content()
        assert message.get_content_as_string() == "Dynamic content 1Dynamic content 2"

    def test_empty_content_handling(self):
        chunk = CacheChunk(content="", cacheable=True)
        message = UserMessage(content=[chunk])

        assert message.has_cacheable_content()
        assert message.get_content_as_string() == ""
