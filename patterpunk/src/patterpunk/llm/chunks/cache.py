"""
Cache functionality and utilities for LLM content optimization.

This module contains the CacheChunk class and cache-related utilities
for managing cacheable content across different LLM providers.
"""

from datetime import timedelta
from typing import Optional


class CacheChunk:
    """
    Represents a chunk of content that can be cached by LLM providers.

    This class allows fine-grained control over which parts of a message
    should be cached, enabling cost optimization across different providers.
    """

    def __init__(
        self, content: str, cacheable: bool = False, ttl: Optional[timedelta] = None
    ):
        """
        Initialize a cache chunk.

        :param content: The text content of this chunk
        :param cacheable: Whether this chunk should be cached by the provider
        :param ttl: Time-to-live for the cache (provider-specific interpretation)
        """
        self.content = content
        self.cacheable = cacheable
        self.ttl = ttl

    def serialize(self) -> dict:
        """Serialize to dict for persistence."""
        result = {"type": "cache", "content": self.content, "cacheable": self.cacheable}
        if self.ttl:
            result["ttl_seconds"] = self.ttl.total_seconds()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "CacheChunk":
        """Deserialize from dict."""
        ttl = (
            timedelta(seconds=data["ttl_seconds"]) if data.get("ttl_seconds") else None
        )
        return cls(
            content=data["content"], cacheable=data.get("cacheable", False), ttl=ttl
        )

    def __repr__(self):
        cache_info = f", cacheable={self.cacheable}"
        if self.ttl:
            cache_info += f", ttl={self.ttl}"
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return f'CacheChunk("{content_preview}"{cache_info})'
