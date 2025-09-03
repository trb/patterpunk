from typing import Optional, Any, Union, List

from ..cache import CacheChunk
from ..multimodal import MultimodalChunk
from ..text import TextChunk
from ..types import ContentType
from .base import Message
from .roles import ROLE_ASSISTANT
from .cache import get_content_as_string


class AssistantMessage(Message):

    def __init__(
        self,
        content: ContentType,
        structured_output: Optional[Any] = None,
        parsed_output: Optional[Any] = None,
    ):
        super().__init__(content, ROLE_ASSISTANT)
        self.structured_output = structured_output
        self._parsed_output = parsed_output
        self._raw_content = content

    @property
    def content(self) -> str:
        return get_content_as_string(self._raw_content)

    @content.setter
    def content(self, value: ContentType) -> None:
        self._raw_content = value

    @property
    def chunks(self) -> Optional[List[Union[TextChunk, CacheChunk, MultimodalChunk]]]:
        if isinstance(self._raw_content, str):
            return None
        elif isinstance(self._raw_content, list):
            return self._raw_content
        return None

    @property
    def texts(self) -> List[Union[TextChunk, CacheChunk]]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, (TextChunk, CacheChunk))
            ]
        return []

    @property
    def images(self) -> List[MultimodalChunk]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, MultimodalChunk)
                and chunk.media_type
                and chunk.media_type.startswith("image/")
            ]
        return []

    @property
    def videos(self) -> List[MultimodalChunk]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, MultimodalChunk)
                and chunk.media_type
                and chunk.media_type.startswith("video/")
            ]
        return []

    @property
    def audios(self) -> List[MultimodalChunk]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, MultimodalChunk)
                and chunk.media_type
                and chunk.media_type.startswith("audio/")
            ]
        return []
