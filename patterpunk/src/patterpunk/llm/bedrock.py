
from typing import List
import boto3
from botocore.exceptions import ClientError
import os

from patterpunk.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from patterpunk.llm.messages import Message, AssistantMessage
from patterpunk.llm.models import Model
from patterpunk.logger import logger, logger_llm


class BedrockModel(Model):
    def __init__(
        self,
        model_id: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        region_name: str = "us-east-1"
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p

        # Use environment variables for AWS credentials
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
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

        conversation = [
            {
                "role": message.role,
                "content": [{"text": message.content}]
            }
            for message in messages
        ]

        try:
            response = self.client.converse(
                modelId=self.model_id,
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
