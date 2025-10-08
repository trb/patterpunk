from typing import Union, List

from ..chunks import CacheChunk, MultimodalChunk, TextChunk
from ..types import ContentType


def get_content_as_string(content: ContentType) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for chunk in content:
            if isinstance(chunk, (TextChunk, CacheChunk)):
                text_parts.append(chunk.content)
        return "".join(text_parts)
    else:
        return str(content)


def has_cacheable_content(content: ContentType) -> bool:
    if isinstance(content, list):
        return any(
            isinstance(chunk, CacheChunk) and chunk.cacheable for chunk in content
        )
    return False


def get_cache_chunks(content: ContentType) -> List[Union[CacheChunk, MultimodalChunk]]:
    if isinstance(content, str):
        return [CacheChunk(content=content, cacheable=False)]
    elif isinstance(content, list):
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
    if isinstance(content, str):
        return False
    return any(isinstance(chunk, MultimodalChunk) for chunk in content)


def get_multimodal_chunks(content: ContentType) -> List[MultimodalChunk]:
    if isinstance(content, str):
        return []
    return [chunk for chunk in content if isinstance(chunk, MultimodalChunk)]
