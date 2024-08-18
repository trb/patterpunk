from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from patterpunk.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
)
from patterpunk.llm.messages import (
    Message,
    AssistantMessage,
    ROLE_SYSTEM,
    ROLE_ASSISTANT,
    ROLE_USER,
)
from patterpunk.llm.models import Model
from patterpunk.logger import logger, logger_llm


class BedrockMissingCredentialsError(Exception):
    pass


class BedrockModel(Model):
    def __init__(
        self,
        model_id: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        region_name: Optional[str] = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p

        aws_access_key_id = AWS_ACCESS_KEY_ID
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        aws_region = AWS_REGION

        if not aws_access_key_id or not aws_secret_access_key:
            raise BedrockMissingCredentialsError(
                "Both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY have to be set - check the environment variables PP_AWS_ACCESS_KEY_ID and PP_AWS_SECRET_ACCESS_KEY"
            )

        if region_name is not None:
            aws_region = region_name

        if aws_access_key_id and aws_secret_access_key:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            self.client = boto3.client("bedrock-runtime", region_name=aws_region)

    def generate_assistant_message(
        self, messages: List[Message], functions: list | None = None
    ) -> AssistantMessage:
        logger.info("Request to AWS Bedrock made")
        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )
        logger_llm.info(
            f"Model params: {self.model_id}, temp: {self.temperature}, top_p: {self.top_p}, functions: {functions}"
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
            {"role": message.role, "content": [{"text": message.content}]}
            for message in messages
        ]

        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": message.content} for message in system_messages],
                messages=conversation,
                inferenceConfig={"temperature": self.temperature, "topP": self.top_p},
            )
            logger.info("AWS Bedrock response received")
        except (ClientError, Exception) as e:
            logger.error(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            raise

        response_text = response["output"]["message"]["content"][0]["text"]
        logger_llm.info(f"[Assistant]\n{response_text}")

        return AssistantMessage(response_text)

    def __deepcopy__(self, memo_dict):
        return BedrockModel(
            model_id=self.model_id,
            temperature=self.temperature,
            top_p=self.top_p,
            region_name=self.client.meta.region_name,
        )
