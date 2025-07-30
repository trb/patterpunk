import pytest
from patterpunk.llm.text import TextChunk
from patterpunk.llm.cache import CacheChunk
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.messages import UserMessage, SystemMessage
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.openai import OpenAiModel
from tests.test_utils import get_resource


def test_mixed_content_types_get_content_as_string():
    message = UserMessage([
        TextChunk("Hello "),
        CacheChunk("world", cacheable=True),
        TextChunk("! How are you?")
    ])
    
    content_str = message.get_content_as_string()
    assert content_str == "Hello world! How are you?"


def test_mixed_content_types_with_multimodal():
    """Test that TextChunk works alongside multimodal content."""
    message = UserMessage([
        TextChunk("Please analyze this image: "),
        MultimodalChunk.from_file(get_resource('ducks_pond.jpg')),
        TextChunk(" What do you see?")
    ])
    
    content_str = message.get_content_as_string()
    assert content_str == "Please analyze this image:  What do you see?"


def test_text_chunk_cache_conversion():
    """Test that TextChunk gets converted to CacheChunk for cache processing."""
    message = UserMessage([
        TextChunk("Non-cacheable text"),
        CacheChunk("Cacheable text", cacheable=True),
        TextChunk("More non-cacheable text")
    ])
    
    cache_chunks = message.get_cache_chunks()
    assert len(cache_chunks) == 3
    
    assert all(isinstance(chunk, CacheChunk) for chunk in cache_chunks)
    
    assert cache_chunks[0].content == "Non-cacheable text"
    assert cache_chunks[0].cacheable is False
    
    assert cache_chunks[1].content == "Cacheable text"
    assert cache_chunks[1].cacheable is True
    
    assert cache_chunks[2].content == "More non-cacheable text"
    assert cache_chunks[2].cacheable is False


def test_text_chunk_templating_with_mixed_types():
    message = UserMessage([
        TextChunk("Hello {{name}}, "),
        CacheChunk("you are {{age}} years old", cacheable=True),
        TextChunk(" and you live in {{city}}")
    ])
    
    prepared = message.prepare({
        "name": "Alice",
        "age": 30,
        "city": "New York"
    })
    
    content_str = prepared.get_content_as_string()
    assert content_str == "Hello Alice, you are 30 years old and you live in New York"


@pytest.mark.parametrize("model_class", [AnthropicModel, OpenAiModel])
def test_text_chunk_provider_conversion(model_class):
    if model_class == AnthropicModel:
        model = model_class(model="claude-3-5-sonnet-20240620")
        
        content = [
            TextChunk("Hello "),
            CacheChunk("world", cacheable=True),
            TextChunk("!")
        ]
        
        anthropic_content = model._convert_content_to_anthropic_format(content)
        
        assert len(anthropic_content) == 3
        assert all(item["type"] == "text" for item in anthropic_content)
        assert anthropic_content[0]["text"] == "Hello "
        assert "cache_control" not in anthropic_content[0]
        
        assert anthropic_content[1]["text"] == "world"
        assert "cache_control" in anthropic_content[1]
        
        assert anthropic_content[2]["text"] == "!"
        assert "cache_control" not in anthropic_content[2]  # TextChunk has no cache control
        
    elif model_class == OpenAiModel:
        model = model_class(model="gpt-4o")
        
        # Test mixed content conversion for responses API
        content = [
            TextChunk("Hello "),
            CacheChunk("world", cacheable=False),
            TextChunk("!")
        ]
        
        openai_content = model._convert_message_content_for_openai_responses(content)
        
        assert len(openai_content) == 3
        assert all(item["type"] == "input_text" for item in openai_content)
        assert openai_content[0]["text"] == "Hello "
        assert openai_content[1]["text"] == "world"
        assert openai_content[2]["text"] == "!"


def test_backward_compatibility():
    old_message = UserMessage([
        CacheChunk("Hello", cacheable=False),
        CacheChunk("world", cacheable=True)
    ])
    
    new_message = UserMessage([
        TextChunk("Hello"),
        CacheChunk("world", cacheable=True)
    ])
    
    assert old_message.get_content_as_string() == new_message.get_content_as_string()
    
    old_chunks = old_message.get_cache_chunks()
    new_chunks = new_message.get_cache_chunks()
    
    assert len(old_chunks) == len(new_chunks)
    assert old_chunks[0].content == new_chunks[0].content
    assert old_chunks[0].cacheable == new_chunks[0].cacheable
    assert old_chunks[1].content == new_chunks[1].content
    assert old_chunks[1].cacheable == new_chunks[1].cacheable