import pytest

from patterpunk.llm.chunks import TextChunk
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.types import TextChunk as TextChunkFromTypes


def test_text_chunk_basic():
    chunk = TextChunk("Hello world")
    assert chunk.content == "Hello world"


def test_text_chunk_repr():
    chunk = TextChunk("Hello world")
    assert repr(chunk) == 'TextChunk("Hello world")'


def test_text_chunk_repr_long():
    long_text = "A" * 100
    chunk = TextChunk(long_text)
    expected = f'TextChunk("{long_text[:50]}...")'
    assert repr(chunk) == expected


def test_text_chunk_in_user_message():
    message = UserMessage(
        [TextChunk("You are a helpful assistant."), TextChunk("What is Python?")]
    )

    content_str = message.get_content_as_string()
    assert content_str == "You are a helpful assistant.What is Python?"


def test_text_chunk_with_cache_chunk():
    from patterpunk.llm.chunks import CacheChunk

    message = UserMessage(
        [
            TextChunk("Instructions: "),
            CacheChunk("Large context data", cacheable=True),
            TextChunk("Question: What is the answer?"),
        ]
    )

    content_str = message.get_content_as_string()
    assert (
        content_str == "Instructions: Large context dataQuestion: What is the answer?"
    )


def test_text_chunk_import_from_types():
    chunk1 = TextChunk("test")
    chunk2 = TextChunkFromTypes("test")

    assert type(chunk1) == type(chunk2)
    assert chunk1.content == chunk2.content


def test_text_chunk_templating():
    message = UserMessage(
        [TextChunk("Hello {{name}}, "), TextChunk("your age is {{age}}")]
    )

    prepared = message.prepare({"name": "Alice", "age": 30})
    content_str = prepared.get_content_as_string()
    assert content_str == "Hello Alice, your age is 30"


def test_text_chunk_cache_chunks_conversion():
    from patterpunk.llm.chunks import CacheChunk

    message = UserMessage([TextChunk("Regular text"), TextChunk("More text")])

    chunks = message.get_cache_chunks()
    assert len(chunks) == 2
    assert all(isinstance(chunk, CacheChunk) for chunk in chunks)
    assert chunks[0].content == "Regular text"
    assert chunks[0].cacheable is False
    assert chunks[1].content == "More text"
    assert chunks[1].cacheable is False


def test_text_chunk_no_cacheable_content():
    message = UserMessage([TextChunk("Text 1"), TextChunk("Text 2")])

    assert not message.has_cacheable_content()


def test_text_chunk_mixed_with_multimodal():
    from patterpunk.llm.chunks import MultimodalChunk
    from tests.test_utils import get_resource

    message = UserMessage(
        [
            TextChunk("Analyze this image: "),
            MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
            TextChunk("What do you see?"),
        ]
    )

    content_str = message.get_content_as_string()
    assert content_str == "Analyze this image: What do you see?"
