import json
import random
from abc import ABC
from typing import List, Optional, Union
import time

from patterpunk.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    boto3,
    GENERATE_STRUCTURED_OUTPUT_PROMPT,
    get_bedrock_client_by_region,
    MAX_RETRIES,
)
from patterpunk.lib.structured_output import get_model_schema, has_model_schema

if boto3:
    from botocore.exceptions import ClientError
from patterpunk.llm.messages import (
    Message,
    AssistantMessage,
    ToolCallMessage,
    ROLE_SYSTEM,
    ROLE_ASSISTANT,
    ROLE_USER,
)
from patterpunk.llm.models.base import Model
from patterpunk.llm.thinking import ThinkingConfig as UnifiedThinkingConfig
from patterpunk.llm.types import ToolDefinition, CacheChunk
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
        top_p: float = DEFAULT_TOP_P,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        thinking_config: Optional[UnifiedThinkingConfig] = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_config = thinking_config

        self.client = get_bedrock_client_by_region(
            client_type="bedrock-runtime",
            region=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def _convert_tools_to_bedrock_format(self, tools: ToolDefinition) -> dict:
        """Convert Patterpunk standard tools to Bedrock toolConfig format"""
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
        """Convert unified thinking config to Bedrock additionalModelRequestFields format."""
        if not self.thinking_config:
            return {}

        additional_fields = {}

        if self.thinking_config.effort is not None:
            # For OpenAI/DeepSeek models that use reasoning_effort
            additional_fields["reasoning_effort"] = self.thinking_config.effort

        if self.thinking_config.token_budget is not None:
            # For Anthropic models that use reasoning_config with budget_tokens
            budget_tokens = max(1024, self.thinking_config.token_budget)  # Minimum 1024
            additional_fields["reasoning_config"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

        return additional_fields

    def _convert_content_to_bedrock_format(self, chunks: List[CacheChunk]) -> List[dict]:
        """Convert cache chunks to Bedrock content format with cache points."""
        bedrock_content = []
        
        for chunk in chunks:
            content_block = {
                "text": chunk.content
            }
            
            if chunk.cacheable:
                content_block["cachePoint"] = {}
            
            bedrock_content.append(content_block)
        
        return bedrock_content

    def _convert_messages_for_bedrock(self, messages: List[Message]) -> List[dict]:
        """Convert patterpunk messages to Bedrock format with cache handling."""
        bedrock_messages = []
        
        for message in messages:
            if isinstance(message.content, list):
                content = self._convert_content_to_bedrock_format(message.content)
            else:
                # Add structured output prompt if needed
                content_str = message.get_content_as_string()
                if message.structured_output and has_model_schema(message.structured_output):
                    content_str = f"{content_str}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(message.structured_output)}"
                content = [{"text": content_str}]
            
            bedrock_messages.append({
                "role": message.role,
                "content": content
            })
        
        return bedrock_messages

    def _convert_system_messages_for_bedrock(self, messages: List[Message]) -> List[dict]:
        """Convert system messages to Bedrock format with cache handling."""
        bedrock_system = []
        
        for message in messages:
            if message.role == ROLE_SYSTEM:
                if isinstance(message.content, list):
                    # For system messages with cache chunks, concatenate them
                    # Bedrock system parameter expects simple text format
                    content_str = "".join(chunk.content for chunk in message.content)
                    bedrock_system.append({"text": content_str})
                else:
                    bedrock_system.append({"text": message.content})
        
        return bedrock_system

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
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

        # Bedrock needs system messages to be handled with a special call parameter, not as a message
        system_messages = [
            message for message in messages if message.role == ROLE_SYSTEM
        ]

        # Convert system messages with cache handling
        system_content = self._convert_system_messages_for_bedrock(system_messages)

        # We'll have to investigate tool usage for bedrock to ensure that it's handled via regular messages
        user_assistant_messages = [
            message
            for message in messages
            if message.role in [ROLE_USER, ROLE_ASSISTANT]
        ]

        # Convert messages with cache handling
        conversation = self._convert_messages_for_bedrock(user_assistant_messages)

        try:
            retry = 0
            retry_sleep = random.randint(30, 60)
            while True:
                try:
                    # Build inference config
                    inference_config = {
                        "temperature": self.temperature,
                        "topP": self.top_p,
                    }

                    # Get thinking parameters
                    thinking_params = self._get_thinking_params()

                    # For Anthropic models with reasoning_config, top_p must be disabled
                    if thinking_params.get("reasoning_config"):
                        inference_config.pop("topP", None)

                    converse_params = {
                        "modelId": self.model_id,
                        "system": system_content,
                        "messages": conversation,
                        "inferenceConfig": inference_config,
                    }

                    # Add thinking parameters if present
                    if thinking_params:
                        converse_params["additionalModelRequestFields"] = (
                            thinking_params
                        )

                    # Add tools if provided
                    if tools:
                        tool_config = self._convert_tools_to_bedrock_format(tools)
                        if tool_config["tools"]:
                            converse_params["toolConfig"] = tool_config

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

        # Handle tool use responses
        if response.get("stopReason") == "tool_use":
            tool_calls = []
            for content_block in response["output"]["message"]["content"]:
                if "toolUse" in content_block:
                    tool_use = content_block["toolUse"]
                    tool_call = {
                        "id": tool_use["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": tool_use["name"],
                            "arguments": json.dumps(tool_use["input"]),
                        },
                    }
                    tool_calls.append(tool_call)

            if tool_calls:
                return ToolCallMessage(tool_calls)

        # Handle regular text responses
        response_content = response["output"]["message"]["content"]
        response_text = ""
        reasoning_text = ""

        # Extract both regular text and reasoning content
        for content_block in response_content:
            if "text" in content_block:
                response_text += content_block["text"]
            elif "reasoningContent" in content_block:
                reasoning_content = content_block["reasoningContent"]
                if "reasoningText" in reasoning_content:
                    reasoning_text += reasoning_content["reasoningText"]["text"]

        # Include reasoning content if enabled
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

    def __deepcopy__(self, memo_dict):
        return BedrockModel(
            model_id=self.model_id,
            temperature=self.temperature,
            top_p=self.top_p,
            region_name=self.client.meta.region_name,
            thinking_config=self.thinking_config,
        )
