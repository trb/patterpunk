"""
Cache chunk handling and content conversion logic for messages.

This module focuses specifically on how messages interact with cache chunks,
including content conversion and cache-related utility functions.
"""

from typing import Union, List

from ..cache import CacheChunk


def get_content_as_string(content: Union[str, List[CacheChunk]]) -> str:
    """
    Helper method to get content as string for backward compatibility.
    
    :param content: Message content (string or list of cache chunks)
    :return: String representation of the content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "".join(chunk.content for chunk in content)
    else:
        return str(content)


def has_cacheable_content(content: Union[str, List[CacheChunk]]) -> bool:
    """
    Check if content contains any cacheable chunks.
    
    :param content: Message content to check
    :return: True if any cache chunks are marked as cacheable
    """
    if isinstance(content, list):
        return any(chunk.cacheable for chunk in content)
    return False


def get_cache_chunks(content: Union[str, List[CacheChunk]]) -> List[CacheChunk]:
    """
    Get cache chunks, converting string content to non-cacheable chunk if needed.
    
    :param content: Message content
    :return: List of cache chunks representing the content
    """
    if isinstance(content, str):
        return [CacheChunk(content=content, cacheable=False)]
    elif isinstance(content, list):
        return content
    else:
        return [CacheChunk(content=str(content), cacheable=False)]