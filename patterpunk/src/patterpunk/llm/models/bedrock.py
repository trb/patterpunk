import random
from abc import ABC
from typing import List, Optional
import time

from patterpunk.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    boto3,
    get_bedrock_client_by_region,
    MAX_RETRIES,
)

if boto3:
    from botocore.exceptions import ClientError
from patterpunk.llm.messages import (
    Message,
    AssistantMessage,
    ROLE_SYSTEM,
    ROLE_ASSISTANT,
    ROLE_USER,
)
from patterpunk.llm.models.base import Model
from patterpunk.logger import logger, logger_llm


class BedrockMissingCredentialsError(Exception):
    pass


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
            retry = 0
            retry_sleep = random.randint(30, 60)
            while True:
                try:
                    response = self.client.converse(
                        modelId=self.model_id,
                        system=[
                            {"text": message.content} for message in system_messages
                        ],
                        messages=conversation,
                        inferenceConfig={
                            "temperature": self.temperature,
                            "topP": self.top_p,
                        },
                    )
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

        response_text = response["output"]["message"]["content"][0]["text"]
        logger_llm.info(f"[Assistant]\n{response_text}")

        return AssistantMessage(response_text)

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
