from typing import Union, List

from ..chunks import CacheChunk
from .base import Message
from .roles import ROLE_SYSTEM


class SystemMessage(Message):

    def __init__(self, content: Union[str, List[CacheChunk]]):
        super().__init__(content, ROLE_SYSTEM)
