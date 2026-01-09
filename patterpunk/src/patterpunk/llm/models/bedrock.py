import asyncio
import json
import random
from abc import ABC
from typing import AsyncIterator, List, Optional, Set, Union
import time

from patterpunk.config.defaults import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GENERATE_STRUCTURED_OUTPUT_PROMPT,
    MAX_RETRIES,
)
from patterpunk.config.providers.bedrock import (
    boto3,
    get_bedrock_client_by_region,
    create_bedrock_client_for_streaming,
)
from patterpunk.lib.structured_output import get_model_schema, has_model_schema

if boto3:
    from botocore.exceptions import ClientError
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.messages.roles import ROLE_SYSTEM, ROLE_ASSISTANT, ROLE_USER
from patterpunk.llm.models.base import Model
from patterpunk.llm.thinking import ThinkingConfig as UnifiedThinkingConfig
from patterpunk.llm.types import ToolDefinition, CacheChunk, ToolCall
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.chunks import MultimodalChunk, TextChunk
from patterpunk.llm.messages.cache import get_multimodal_chunks, has_multimodal_content
from patterpunk.logger import logger, logger_llm


class BedrockMissingCredentialsError(Exception):
    pass


def get_bedrock_conversation_content(message: Message):
    content_str = message.get_content_as_string()
    if message.structured_output and has_model_schema(message.structured_output):
        return f"{content_str}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(message.structured_output)}"

    return content_str


class BedrockModel(Model, ABC):
    def __init__(
        self,
        model_id: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        thinking_config: Optional[UnifiedThinkingConfig] = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p  # None means don't specify (some models don't allow both temp and top_p)
        self.max_tokens = max_tokens
        self.thinking_config = thinking_config

        self.client = get_bedrock_client_by_region(
            client_type="bedrock-runtime",
            region=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def _convert_tools_to_bedrock_format(self, tools: ToolDefinition) -> dict:
        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                bedrock_tool = {
                    "toolSpec": {
                        "name": func["name"],
                        "description": func["description"],
                        "inputSchema": {"json": func["parameters"]},
                    }
                }
                bedrock_tools.append(bedrock_tool)

        return {"tools": bedrock_tools}

    def _get_thinking_params(self) -> dict:
        if not self.thinking_config:
            return {}

        additional_fields = {}

        if "deepseek" in self.model_id.lower():
            return {}

        if self.thinking_config.effort is not None:
            additional_fields["reasoning_effort"] = self.thinking_config.effort

        if self.thinking_config.token_budget is not None:
            budget_tokens = max(1024, self.thinking_config.token_budget)
            additional_fields["reasoning_config"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

        return additional_fields

    def _convert_content_to_bedrock_format(self, content) -> List[dict]:
        if isinstance(content, str):
            return [{"text": content}]

        bedrock_content = []
        session = None

        for chunk in content:
            if isinstance(chunk, TextChunk):
                bedrock_content.append({"text": chunk.content})

            elif isinstance(chunk, CacheChunk):
                content_block = {"text": chunk.content}
                if chunk.cacheable:
                    content_block["cachePoint"] = {}
                bedrock_content.append(content_block)

            elif isinstance(chunk, MultimodalChunk):
                if chunk.source_type == "url":
                    if session is None:
                        try:
                            import requests

                            session = requests.Session()
                        except ImportError:
                            raise ImportError(
                                "requests library required for URL support with Bedrock"
                            )

                    chunk = chunk.download(session)

                media_type = chunk.media_type or "application/octet-stream"

                if media_type.startswith("image/"):
                    format_map = {
                        "image/jpeg": "jpeg",
                        "image/jpg": "jpeg",
                        "image/png": "png",
                        "image/gif": "gif",
                        "image/webp": "webp",
                    }

                    format = format_map.get(media_type, "jpeg")

                    content_block = {
                        "image": {
                            "format": format,
                            "source": {"bytes": chunk.to_bytes()},
                        }
                    }
                    bedrock_content.append(content_block)
                elif media_type in [
                    "application/pdf",
                    "text/plain",
                    "text/html",
                    "text/markdown",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ]:
                    format_map = {
                        "application/pdf": "pdf",
                        "text/plain": "txt",
                        "text/html": "html",
                        "text/markdown": "md",
                        "application/msword": "doc",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                        "application/vnd.ms-excel": "xls",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                    }

                    document_format = format_map.get(media_type, "pdf")
                    raw_filename = (
                        getattr(chunk, "filename", None)
                        or f"document.{document_format}"
                    )
                    import re

                    document_name = re.sub(r"[^\w\s\-\(\)\[\]]", "", raw_filename)
                    document_name = re.sub(r"\s+", " ", document_name).strip()

                    content_block = {
                        "document": {
                            "format": document_format,
                            "name": document_name,
                            "source": {"bytes": chunk.to_bytes()},
                        }
                    }
                    bedrock_content.append(content_block)
                else:
                    raise ValueError(
                        f"Bedrock does not support media type: {media_type}. "
                        f"Supported types are: text, images (jpeg/png/gif/webp), and documents (pdf/csv/doc/docx/xls/xlsx/html/txt/md)."
                    )

        return bedrock_content

    def _convert_messages_for_bedrock(self, messages: List[Message]) -> List[dict]:
        bedrock_messages = []

        for message in messages:
            if message.role == "tool_call":
                # Serialize ToolCallMessage as assistant message with toolUse content blocks
                # When extended thinking is enabled, thinking blocks must come FIRST
                content_blocks = []

                # Add thinking blocks first (required by Bedrock when thinking is enabled)
                # Bedrock expects: {"reasoningContent": {"reasoningText": {"text": "...", "signature": "..."}}}
                if hasattr(message, "thinking_blocks") and message.thinking_blocks:
                    for block in message.thinking_blocks:
                        # Only add blocks that have actual content (text or signature)
                        thinking_text = block.get("thinking", "")
                        signature = block.get("signature", "")
                        if block.get("type") == "thinking" and (
                            thinking_text or signature
                        ):
                            reasoning_content = {
                                "reasoningContent": {
                                    "reasoningText": {
                                        "text": thinking_text,
                                        "signature": signature,
                                    }
                                }
                            }
                            content_blocks.append(reasoning_content)

                for tool_call in message.tool_calls:
                    # Parse arguments from JSON string
                    try:
                        arguments = json.loads(tool_call.arguments)
                    except (json.JSONDecodeError, KeyError):
                        arguments = {}

                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tool_call.id,
                                "name": tool_call.name,
                                "input": arguments,
                            }
                        }
                    )

                bedrock_messages.append(
                    {"role": "assistant", "content": content_blocks}
                )

            elif message.role == "tool_result":
                # Validate required field for Bedrock
                if not message.call_id:
                    raise ValueError(
                        "AWS Bedrock requires call_id (as toolUseId) in ToolResultMessage. "
                        "Ensure ToolResultMessage is created with call_id from the original ToolCallMessage."
                    )

                # Create toolResult block
                # Bedrock uses status field instead of is_error boolean
                tool_result_block = {
                    "toolResult": {
                        "toolUseId": message.call_id,
                        "content": [{"text": message.content}],
                        "status": "error" if message.is_error else "success",
                    }
                }

                # Bedrock requires consecutive tool results to be merged into a SINGLE user message
                # Check if the previous message is also a user message with toolResult blocks
                if (
                    bedrock_messages
                    and bedrock_messages[-1]["role"] == "user"
                    and bedrock_messages[-1]["content"]
                    and "toolResult" in bedrock_messages[-1]["content"][0]
                ):
                    # Append to existing user message with toolResult blocks
                    bedrock_messages[-1]["content"].append(tool_result_block)
                else:
                    # Create new user message
                    bedrock_messages.append(
                        {"role": "user", "content": [tool_result_block]}
                    )

            else:
                # Handle regular messages (user, assistant)
                if isinstance(message.content, list):
                    content = self._convert_content_to_bedrock_format(message.content)

                    has_text = any("text" in chunk for chunk in content)
                    has_document = any("document" in chunk for chunk in content)

                    if has_document and not has_text:
                        raise ValueError(
                            "Bedrock requires at least one text block when documents are present. "
                            "Please include text content along with your documents."
                        )
                else:
                    content_str = message.get_content_as_string()
                    if message.structured_output and has_model_schema(
                        message.structured_output
                    ):
                        content_str = f"{content_str}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(message.structured_output)}"
                    content = [{"text": content_str}]

                bedrock_messages.append({"role": message.role, "content": content})

        return bedrock_messages

    def _convert_system_messages_for_bedrock(
        self, messages: List[Message]
    ) -> List[dict]:
        bedrock_system = []

        for message in messages:
            if message.role == ROLE_SYSTEM:
                if isinstance(message.content, list):
                    content_str = "".join(chunk.content for chunk in message.content)
                    bedrock_system.append({"text": content_str})
                else:
                    bedrock_system.append({"text": message.content})

        return bedrock_system

    def _prepare_messages_for_converse(
        self, messages: List[Message]
    ) -> tuple[List[dict], Optional[List[dict]]]:
        system_messages = [
            message for message in messages if message.role == ROLE_SYSTEM
        ]
        system_content = self._convert_system_messages_for_bedrock(system_messages)

        user_assistant_messages = [
            message
            for message in messages
            if message.role in [ROLE_USER, ROLE_ASSISTANT, "tool_call", "tool_result"]
        ]
        conversation = self._convert_messages_for_bedrock(user_assistant_messages)

        return conversation, system_content if system_content else None

    def _build_inference_config(
        self, structured_output: Optional[object] = None
    ) -> dict:
        inference_config = {
            "temperature": self.temperature,
        }

        # Only add topP if explicitly specified (some models like Claude 4.5 don't allow both)
        if self.top_p is not None:
            inference_config["topP"] = self.top_p

        # Add max_tokens if specified
        if self.max_tokens:
            inference_config["maxTokens"] = self.max_tokens

        thinking_params = self._get_thinking_params()
        if thinking_params.get("reasoning_config"):
            inference_config.pop("topP", None)
            # Extended thinking requires max_tokens > budget_tokens
            budget = thinking_params["reasoning_config"].get("budget_tokens", 1024)
            if not self.max_tokens or self.max_tokens <= budget:
                inference_config["maxTokens"] = budget + 2000

        return inference_config

    def _prepare_tool_config(self, tools: Optional[ToolDefinition]) -> Optional[dict]:
        if not tools:
            return None

        tool_config = self._convert_tools_to_bedrock_format(tools)
        if tool_config["tools"]:
            return tool_config
        return None

    def _build_converse_params(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
    ) -> dict:
        conversation, system_content = self._prepare_messages_for_converse(messages)
        inference_config = self._build_inference_config(structured_output)

        converse_params = {
            "modelId": self.model_id,
            "messages": conversation,
            "inferenceConfig": inference_config,
        }

        if system_content:
            converse_params["system"] = system_content

        thinking_params = self._get_thinking_params()
        if thinking_params:
            converse_params["additionalModelRequestFields"] = thinking_params

        tool_config = self._prepare_tool_config(tools)
        if tool_config:
            converse_params["toolConfig"] = tool_config

        return converse_params

    def _process_converse_response(
        self, output: dict, structured_output: Optional[object] = None
    ) -> Union[AssistantMessage, ToolCallMessage]:
        # Check for tool use in content blocks regardless of stopReason
        # Some models return tool calls in content without setting stopReason to "tool_use"
        tool_calls = []
        for content_block in output.get("message", {}).get("content", []):
            if "toolUse" in content_block:
                tool_use = content_block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=tool_use["toolUseId"],
                        name=tool_use["name"],
                        arguments=json.dumps(tool_use["input"]),
                    )
                )

        if tool_calls:
            return ToolCallMessage(tool_calls)

        response_content = output["message"]["content"]
        response_text = ""
        reasoning_text = ""

        for content_block in response_content:
            if "text" in content_block:
                response_text += content_block["text"]
            elif "reasoningContent" in content_block:
                reasoning_content = content_block["reasoningContent"]
                if "reasoningText" in reasoning_content:
                    reasoning_text += reasoning_content["reasoningText"]["text"]

        if (
            reasoning_text
            and self.thinking_config
            and self.thinking_config.include_thoughts
        ):
            full_response = (
                f"<thinking>\n{reasoning_text}\n</thinking>\n\n{response_text}"
            )
        else:
            full_response = response_text

        logger_llm.info(f"[Assistant]\n{full_response}")

        return AssistantMessage(full_response, structured_output=structured_output)

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> Union[AssistantMessage, "ToolCallMessage"]:
        logger.info("Request to AWS Bedrock made")
        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )
        logger_llm.info(
            f"Model params: {self.model_id}, temp: {self.temperature}, top_p: {self.top_p}, tools: {tools}"
        )

        converse_params = self._build_converse_params(
            messages, tools, structured_output
        )

        try:
            retry = 0
            retry_sleep = random.randint(30, 60)
            while True:
                try:
                    response = self.client.converse(**converse_params)
                    break
                except ClientError as client_exception:
                    if (
                        client_exception.response["Error"]["Code"]
                        == "ThrottlingException"
                    ):
                        if retry > MAX_RETRIES:
                            logger.error(
                                f"ERROR: AWS Bedrock is throttling and max retries reached. Retries: {retry}, max retries: {MAX_RETRIES} "
                            )
                            raise
                        else:
                            logger.warning(
                                "AWS Bedrock throttling detected, backing off and retrying"
                            )
                            retry += 1
                            time.sleep(retry_sleep)
                            retry_sleep += random.randint(30, 60)
                    else:
                        logger.error(
                            "AWS Bedrock client exception detected",
                            exc_info=client_exception,
                        )
                        raise
            logger.info("AWS Bedrock response received")
        except (ClientError, Exception) as e:
            logger.error(
                f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}", exc_info=e
            )
            raise

        return self._process_converse_response(response["output"], structured_output)

    @staticmethod
    def get_name():
        return "Bedrock"

    @staticmethod
    def get_available_models(
        region: str = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> List[str]:
        client = get_bedrock_client_by_region(client_type="bedrock", region=region)

        return [
            model["modelName"]
            for model in client.list_foundation_models(byOutputModality="TEXT")[
                "modelSummaries"
            ]
        ]

    async def stream_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> AsyncIterator["StreamChunk"]:
        """
        Stream the assistant message response from AWS Bedrock.

        Yields StreamChunk objects for each streaming event.

        Since boto3 doesn't have native async support, this method runs the
        synchronous converse_stream call in a thread pool executor and yields
        chunks asynchronously.
        """
        from patterpunk.llm.streaming import StreamChunk, StreamEventType

        logger.info("Request to AWS Bedrock (streaming) made")
        logger_llm.info(
            f"Model params: {self.model_id}, temp: {self.temperature}, top_p: {self.top_p}, tools: {tools}"
        )

        converse_params = self._build_converse_params(
            messages, tools, structured_output
        )

        # Get region from existing client for creating streaming client
        region_name = self.client.meta.region_name

        # Create a fresh client for this streaming operation
        # This avoids thread-safety issues with sharing clients
        streaming_client = create_bedrock_client_for_streaming(region=region_name)

        # Run the synchronous API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: streaming_client.converse_stream(**converse_params)
        )

        stream = response.get("stream")
        if not stream:
            raise ValueError("No stream returned from converse_stream")

        # Yield chunks from the EventStream
        # Use a queue to pass events from sync iteration to async context
        async for chunk in self._iterate_stream_events(stream):
            yield chunk

    async def _iterate_stream_events(self, stream) -> AsyncIterator["StreamChunk"]:
        """
        Iterate over boto3 EventStream and yield StreamChunks.

        Since EventStream iteration is synchronous, we wrap each iteration
        in run_in_executor to avoid blocking the event loop.
        """
        from patterpunk.llm.streaming import StreamChunk, StreamEventType

        loop = asyncio.get_event_loop()
        iterator = iter(stream)
        usage = None

        # Track thinking blocks with signatures for MESSAGE_END
        # Key: contentBlockIndex, Value: {"text": accumulated_text, "signature": signature}
        thinking_block_state: Dict[int, Dict[str, str]] = {}

        # Sentinel to detect end of iteration - StopIteration can't propagate through run_in_executor
        _STREAM_END = object()

        def _next_event():
            try:
                return next(iterator)
            except StopIteration:
                return _STREAM_END

        while True:
            event = await loop.run_in_executor(None, _next_event)
            if event is _STREAM_END:
                break

            # Extract metadata/usage from metadata event
            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    usage = {
                        "input_tokens": metadata["usage"].get("inputTokens", 0),
                        "output_tokens": metadata["usage"].get("outputTokens", 0),
                    }
                continue

            # Track reasoning content for building complete thinking blocks
            if "contentBlockDelta" in event:
                block_delta = event["contentBlockDelta"]
                index = block_delta.get("contentBlockIndex", 0)
                delta = block_delta.get("delta", {})

                if "reasoningContent" in delta:
                    reasoning = delta["reasoningContent"]
                    if index not in thinking_block_state:
                        thinking_block_state[index] = {"text": "", "signature": ""}

                    if "text" in reasoning:
                        thinking_block_state[index]["text"] += reasoning["text"]
                    if "signature" in reasoning:
                        thinking_block_state[index]["signature"] = reasoning[
                            "signature"
                        ]

            chunk = self._convert_stream_event_to_chunk(event)
            if chunk is not None:
                yield chunk

        # Build complete thinking blocks with signatures
        thinking_blocks = []
        for index in sorted(thinking_block_state.keys()):
            block = thinking_block_state[index]
            if block["text"] or block["signature"]:
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block["text"],
                        "signature": block["signature"],
                    }
                )

        # Yield MESSAGE_END with usage and thinking blocks
        yield StreamChunk(
            event_type=StreamEventType.MESSAGE_END,
            usage=usage or {},
            thinking_blocks=thinking_blocks if thinking_blocks else None,
        )

    def _convert_stream_event_to_chunk(self, event: dict) -> Optional["StreamChunk"]:
        """
        Convert a Bedrock streaming event to a StreamChunk.

        Returns None for events we don't need to expose.
        """
        from patterpunk.llm.streaming import StreamChunk, StreamEventType

        # messageStart - start of message
        if "messageStart" in event:
            return None  # We don't expose message start

        # contentBlockStart - start of a content block
        if "contentBlockStart" in event:
            block_start = event["contentBlockStart"]
            index = block_start.get("contentBlockIndex", 0)
            start = block_start.get("start", {})

            # Tool use start
            if "toolUse" in start:
                tool_use = start["toolUse"]
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_START,
                    index=index,
                    tool_call_id=tool_use.get("toolUseId"),
                    tool_name=tool_use.get("name"),
                )
            # Text content start
            elif "text" in start:
                return StreamChunk(
                    event_type=StreamEventType.CONTENT_BLOCK_START,
                    index=index,
                    block_type="text",
                )

            return None

        # contentBlockDelta - incremental content
        if "contentBlockDelta" in event:
            block_delta = event["contentBlockDelta"]
            index = block_delta.get("contentBlockIndex", 0)
            delta = block_delta.get("delta", {})

            # Text delta
            if "text" in delta:
                return StreamChunk(
                    event_type=StreamEventType.TEXT_DELTA,
                    text=delta["text"],
                    index=index,
                )
            # Tool use input delta
            elif "toolUse" in delta:
                tool_delta = delta["toolUse"]
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_DELTA,
                    tool_arguments_delta=tool_delta.get("input", ""),
                    index=index,
                )
            # Reasoning/thinking content delta (for Claude models with extended thinking)
            elif "reasoningContent" in delta:
                reasoning_content = delta["reasoningContent"]
                if "text" in reasoning_content:
                    return StreamChunk(
                        event_type=StreamEventType.THINKING_DELTA,
                        text=reasoning_content["text"],
                        index=index,
                    )

            return None

        # contentBlockStop - end of a content block
        if "contentBlockStop" in event:
            block_stop = event["contentBlockStop"]
            return StreamChunk(
                event_type=StreamEventType.CONTENT_BLOCK_STOP,
                index=block_stop.get("contentBlockIndex", 0),
            )

        # messageStop - end of message (stop reason but not final)
        if "messageStop" in event:
            return StreamChunk(
                event_type=StreamEventType.MESSAGE_DELTA,
            )

        return None

    def __deepcopy__(self, memo_dict):
        return BedrockModel(
            model_id=self.model_id,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            region_name=self.client.meta.region_name,
            thinking_config=self.thinking_config,
        )
