import os
from typing import Optional, Literal, Union

DEFAULT_MODEL = os.getenv("PP_DEFAULT_MODEL") or "gpt-4o-mini"
DEFAULT_TEMPERATURE = os.getenv("PP_DEFAULT_TEMPERATURE") or 1.0
DEFAULT_TOP_P = os.getenv("PP_DEFAULT_TOP_P") or 1.0
DEFAULT_FREQUENCY_PENALTY = os.getenv("PP_DEFAULT_FREQUENCY_PENALTY") or 0.0
DEFAULT_PRESENCE_PENALTY = os.getenv("PP_DEFAULT_PRESENCE_PENALTY") or 0.0
MAX_RETRIES = os.getenv("PP_MAX_RETRIES") or 3
OPENAI_MAX_RETRIES = os.getenv("PP_OPENAI_MAX_RETRIES") or MAX_RETRIES

OPENAI_API_KEY = os.getenv("PP_OPENAI_API_KEY", None)

if OPENAI_API_KEY:
    from openai import OpenAI

    openai = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai = None


OLLAMA_API_ENDPOINT = os.getenv("PP_OLLAMA_API_ENDPOINT", None)

if OLLAMA_API_ENDPOINT:
    from ollama import Client

    ollama = Client(host=OLLAMA_API_ENDPOINT)
else:
    ollama = None


ANTHROPIC_API_KEY = os.getenv("PP_ANTHROPIC_API_KEY", None)
ANTHROPIC_DEFAULT_TEMPERATURE = os.getenv("PP_ANTHROPIC_DEFAULT_TEMPERATURE", 0.7)
ANTHROPIC_DEFAULT_TOP_P = os.getenv("PP_ANTHROPIC_DEFAULT_TOP_P", 1.0)
ANTHROPIC_DEFAULT_TOP_K = os.getenv("PP_ANTHROPIC_DEFAULT_TOP_K", 200)
ANTHROPIC_DEFAULT_MAX_TOKENS = os.getenv("PP_ANTHROPIC_DEFAULT_MAX_TOKENS", 4096)

if ANTHROPIC_API_KEY:
    from anthropic import Anthropic

    anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    anthropic = None

AWS_REGION = os.getenv("PP_AWS_REGION", None)
AWS_ACCESS_KEY_ID = os.getenv("PP_AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("PP_AWS_SECRET_ACCESS_KEY", None)

if AWS_REGION:
    import boto3

    def get_bedrock_client_by_region(
        client_type: Union[Literal["bedrock"], Literal["bedrock-runtime"]],
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        if aws_access_key_id and aws_secret_access_key:
            access_key_id = aws_access_key_id
            secret_access_key = aws_secret_access_key
        else:
            access_key_id = AWS_ACCESS_KEY_ID
            secret_access_key = AWS_SECRET_ACCESS_KEY

        if region is None:
            region = AWS_REGION

        if access_key_id and secret_access_key:
            return boto3.client(
                client_type,
                region_name=region,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
        else:
            return boto3.client(
                client_type,
                region_name=region,
            )

else:
    get_bedrock_client_by_region = None
    boto3 = None
