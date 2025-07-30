"""
Cache chunk handling and content conversion logic for messages.

This module focuses specifically on how messages interact with cache chunks,
including content conversion and cache-related utility functions.
"""

from typing import Union, List

from ..cache import CacheChunk
from ..multimodal import MultimodalChunk
from ..text import TextChunk
from ..types import ContentType


def get_content_as_string(content: ContentType) -> str:
    """
    Get content as string, handling text, cache, and multimodal chunks.

    :param content: Message content (string or list of text/cache/multimodal chunks)
    :return: String representation of the content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Only include text content from chunks
        text_parts = []
        for chunk in content:
            if isinstance(chunk, (TextChunk, CacheChunk)):
                text_parts.append(chunk.content)
            # MultimodalChunk doesn't contribute to text representation
        return "".join(text_parts)
    else:
        return str(content)


def has_cacheable_content(content: ContentType) -> bool:
    """
    Check if content contains any cacheable chunks.

    :param content: Message content to check
    :return: True if any cache chunks are marked as cacheable
    """
    if isinstance(content, list):
        return any(isinstance(chunk, CacheChunk) and chunk.cacheable for chunk in content)
    return False


def get_cache_chunks(content: ContentType) -> List[Union[CacheChunk, MultimodalChunk]]:
    """
    Get content chunks, converting string content to CacheChunk if needed.
    
    This function is specifically for cache-related processing where everything
    needs to be in CacheChunk format for cache logic to work properly.

    :param content: Message content
    :return: List of chunks representing the content in cache-aware format
    """
    if isinstance(content, str):
        return [CacheChunk(content=content, cacheable=False)]
    elif isinstance(content, list):
        # Convert TextChunk to CacheChunk for cache processing
        cache_chunks = []
        for chunk in content:
            if isinstance(chunk, TextChunk):
                cache_chunks.append(CacheChunk(content=chunk.content, cacheable=False))
            elif isinstance(chunk, (CacheChunk, MultimodalChunk)):
                cache_chunks.append(chunk)
        return cache_chunks
    else:
        return [CacheChunk(content=str(content), cacheable=False)]


def has_multimodal_content(content: ContentType) -> bool:
    """Check if content contains any multimodal chunks."""
    if isinstance(content, str):
        return False
    return any(isinstance(chunk, MultimodalChunk) for chunk in content)


def get_multimodal_chunks(content: ContentType) -> List[MultimodalChunk]:
    """Extract only multimodal chunks from content."""
    if isinstance(content, str):
        return []
    return [chunk for chunk in content if isinstance(chunk, MultimodalChunk)]
