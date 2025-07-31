"""
Text chunk handling for regular text content in messages.

This module provides the TextChunk class for handling plain text content
with a unified interface alongside CacheChunk and MultimodalChunk.
"""


class TextChunk:
    """
    Represents a chunk of plain text content.

    Provides a consistent interface for text content alongside CacheChunk
    and MultimodalChunk, making message content lists visually scannable.
    """

    def __init__(self, content: str):
        """
        Initialize a text chunk.

        :param content: The text content of this chunk
        """
        self.content = content

    def __repr__(self):
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return f'TextChunk("{content_preview}")'
