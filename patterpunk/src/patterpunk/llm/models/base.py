import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Optional, Set, Union

from patterpunk.llm.messages.base import Message
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.types import ToolDefinition
from patterpunk.llm.streaming import StreamChunk, StreamingNotSupported


class ModelNotImplemented(Exception):
    pass


class TokenCountingError(Exception):
    """
    Raised when token counting cannot be performed accurately.

    Common causes:
    - Unsupported content type (e.g., PDFs for OpenAI local counting)
    - Missing optional dependencies (e.g., transformers for Llama on Bedrock)
    - Unsupported model (e.g., Amazon Titan has proprietary tokenizer)
    """

    pass


# Shared executor for sync-to-async wrapping
_executor = ThreadPoolExecutor(max_workers=4)


class Model(ABC):
    @abstractmethod
    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        raise ModelNotImplemented("You need to use a LLM-specific model")

    async def generate_assistant_message_async(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        """
        Async version of generate_assistant_message.

        Default implementation wraps the sync method in an executor.
        Providers with native async support should override this.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.generate_assistant_message(
                messages, tools, structured_output, output_types
            ),
        )

    async def stream_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream the assistant message response.

        Default implementation raises StreamingNotSupported.
        Providers with streaming support should override this.
        """
        raise StreamingNotSupported(
            f"{self.get_name()} does not support streaming. "
            "Use generate_assistant_message_async() instead."
        )
        # Make this an async generator by yielding nothing after the raise
        yield  # type: ignore  # This makes it an AsyncIterator

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise ModelNotImplemented("Models need to implement the get_name() method")

    @staticmethod
    @abstractmethod
    def get_available_models() -> List[str]:
        raise ModelNotImplemented(
            "Models need to implement the get_available_models() method"
        )

    @abstractmethod
    def count_tokens(self, content: Union[str, Message, List[Message]]) -> int:
        """
        Count tokens for a string, message, or list of messages.

        For API-based providers (Anthropic, Google, Bedrock), passing a List[Message]
        makes a single API call rather than one per message - much more efficient.

        Args:
            content: A string, single Message, or list of Messages

        Returns:
            Number of tokens

        Raises:
            TokenCountingError: If counting cannot be performed accurately
        """
        raise ModelNotImplemented("Models need to implement count_tokens()")

    async def count_tokens_async(
        self, content: Union[str, Message, List[Message]]
    ) -> int:
        """
        Async version of count_tokens.

        Default implementation wraps the sync method in an executor.
        Providers with native async support should override this.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.count_tokens(content),
        )
