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
from patterpunk.llm.types import ToolDefinition
from patterpunk.logger import logger, logger_llm


class BedrockMissingCredentialsError(Exception):
    pass


def get_bedrock_conversation_content(message: Message):
    if message.structured_output and has_model_schema(message.structured_output):
        return f"{message.content}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(message.structured_output)}"

    return message.content


class BedrockModel(Model, ABC):
    def __init__(
        self,
        model_id: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p

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
                        "inputSchema": {
                            "json": func["parameters"]
                        }
                    }
                }
                bedrock_tools.append(bedrock_tool)
        
        return {"tools": bedrock_tools}

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

        # We'll have to investigate tool usage for bedrock to ensure that it's handled via regular messages
        messages = [
            message
            for message in messages
            if message.role in [ROLE_USER, ROLE_ASSISTANT]
        ]

        conversation = [
            {
                "role": message.role,
                "content": [{"text": get_bedrock_conversation_content(message)}],
            }
            for message in messages
        ]

        try:
            retry = 0
            retry_sleep = random.randint(30, 60)
            while True:
                try:
                    converse_params = {
                        "modelId": self.model_id,
                        "system": [
                            {"text": message.content} for message in system_messages
                        ],
                        "messages": conversation,
                        "inferenceConfig": {
                            "temperature": self.temperature,
                            "topP": self.top_p,
                        },
                    }

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
        if response.get('stopReason') == 'tool_use':
            tool_calls = []
            for content_block in response['output']['message']['content']:
                if 'toolUse' in content_block:
                    tool_use = content_block['toolUse']
                    tool_call = {
                        "id": tool_use['toolUseId'],
                        "type": "function",
                        "function": {
                            "name": tool_use['name'],
                            "arguments": json.dumps(tool_use['input'])
                        }
                    }
                    tool_calls.append(tool_call)
            
            if tool_calls:
                return ToolCallMessage(tool_calls)

        # Handle regular text responses
        response_text = response["output"]["message"]["content"][0]["text"]
        logger_llm.info(f"[Assistant]\n{response_text}")

        return AssistantMessage(response_text, structured_output=structured_output)

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
        )
