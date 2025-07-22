import json
import time
from abc import ABC
from typing import List, Optional, Union

from patterpunk.config import (
    GOOGLE_ACCOUNT_CREDENTIALS,
    GOOGLE_DEFAULT_MAX_TOKENS,
    GOOGLE_DEFAULT_TEMPERATURE,
    GOOGLE_DEFAULT_TOP_K,
    GOOGLE_DEFAULT_TOP_P,
    GOOGLE_LOCATION,
    MAX_RETRIES,
)

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request

    google_genai_available = True
except ImportError:
    google_genai_available = False

from patterpunk.llm.messages import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    AssistantMessage,
)
from patterpunk.llm.models.base import Model
from patterpunk.logger import logger


class GoogleAuthenticationError(Exception):
    pass


class GoogleRateLimitError(Exception):
    """Raised when all retry attempts for rate limit errors are exhausted"""

    pass


class GoogleMaxTokensError(Exception):
    """Raised when the model hits the maximum token limit"""

    pass


class GoogleNotImplemented(Exception):
    """Raised for features not yet implemented"""

    pass


class GoogleAPIError(Exception):
    """Raised for general API errors"""

    pass


class GoogleModel(Model, ABC):
    client: Optional[genai.Client] = None

    @staticmethod
    def get_client(
        location: str,
        google_account_credentials: Optional[GOOGLE_ACCOUNT_CREDENTIALS] = None,
    ):
        if google_account_credentials is None:
            google_account_credentials = GOOGLE_ACCOUNT_CREDENTIALS
        if not google_account_credentials:
            raise GoogleAuthenticationError(
                "No Google account credentials provided. Please pass `google_account_credentials` to constructor or set `PP_GOOGLE_ACCOUNT_CREDENTIALS` environment variable."
            )

        from json import JSONDecodeError

        try:
            credentials_info = json.loads(google_account_credentials)
        except JSONDecodeError:
            raise GoogleAuthenticationError(
                f"Provided credentials were not in JSON format, please provide the account key .json file as a one-line string in json format"
            )

        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        # Refresh right away to detect errors early
        credentials.refresh(Request())
        project_id = credentials_info["project_id"]

        return genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            credentials=credentials,
        )

    # Remember to make sure to update __deepcopy__ if you change the __init__ parameters
    def __init__(
        self,
        model: str,
        location: str = GOOGLE_LOCATION,
        temperature: float = GOOGLE_DEFAULT_TEMPERATURE,
        top_p: float = GOOGLE_DEFAULT_TOP_P,
        top_k: int = GOOGLE_DEFAULT_TOP_K,
        max_tokens: int = GOOGLE_DEFAULT_MAX_TOKENS,
        # Either google_account_credentials OR client have to be set, but not both
        google_account_credentials: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ):
        if not google_genai_available:
            raise ImportError(
                "The google-genai package is not installed. "
                "Please install it with `pip install google-genai`"
            )
        if google_account_credentials and client:
            raise ValueError(
                "Cannot set both `google_account_credentials` and `client`"
            )

        if client:
            self.client = client
        else:
            if not GoogleModel.client:
                GoogleModel.client = GoogleModel.get_client(
                    location, google_account_credentials
                )
            self.client = (
                GoogleModel.get_client(location, google_account_credentials)
                if google_account_credentials
                else GoogleModel.client
            )

        self.model = model
        self.location = location
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools=None,
        structured_output: Optional[object] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        system_prompt = "\n\n".join(
            [message.content for message in messages if message.role == ROLE_SYSTEM]
        )

        contents = []

        for message in messages:
            if message.role == ROLE_SYSTEM:
                continue
            elif message.role == ROLE_USER:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=message.content)],
                    )
                )
            elif message.role == ROLE_ASSISTANT:
                contents.append(
                    types.Content(
                        role="model", parts=[types.Part.from_text(text=message.content)]
                    )
                )

        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if system_prompt:
            config.system_instruction = system_prompt

        if structured_output:
            config.response_mime_type = "application/json"
            if hasattr(structured_output, "__annotations__"):
                config.response_schema = structured_output
            else:
                config.response_schema = structured_output

        retry_count = 0
        wait_time = 60

        while retry_count < MAX_RETRIES:
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )

                if hasattr(response, "text"):
                    content = response.text
                    return AssistantMessage(
                        content, structured_output=structured_output
                    )
                else:
                    raise GoogleAPIError("Unexpected response format from Vertex AI")

            except genai_errors.APIError as error:
                if error.code == 429:
                    if retry_count >= MAX_RETRIES - 1:
                        raise GoogleRateLimitError(
                            f"Rate limit exceeded after {retry_count + 1} retries"
                        ) from error

                    logger.warning(
                        f"VertexAI: Rate limit hit, attempt {retry_count + 1}/{MAX_RETRIES}. "
                        f"Waiting {wait_time} seconds before retry."
                    )

                    time.sleep(wait_time)
                    retry_count += 1
                    wait_time = int(wait_time * 1.5)
                    continue
                else:
                    logger.error(f"VertexAI: Unexpected Api error {error}")
                    raise error
            except Exception as e:
                raise GoogleAPIError(f"Error generating content: {str(e)}") from e

        raise GoogleAPIError(
            "Unexpected outcome - out of retries, but neither error raised or message returned"
        )

    @staticmethod
    def get_available_models(location: Optional[str] = None) -> List[str]:
        default_models = [
            "gemini-1.5-pro-002",
            "gemini-1.5-pro-001",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-001",
            "gemini-1.0-pro-001",
            "gemini-1.0-pro",
        ]

        if not google_genai_available:
            print("no vertex")
            return []

        try:
            if not GoogleModel.client:
                GoogleModel.client = GoogleModel.get_client(location=GOOGLE_LOCATION)

            client = (
                GoogleModel.get_client(location=location)
                if location
                else GoogleModel.client
            )

            # Filter for Gemini models only
            gemini_models = []
            # For whatever reason, some regions won't return any models (like northamerica-northeast1), even though
            # gemini-1.0*/gemini-1.5* models are available. If I'd have to guess it's because google is deprecating
            # those models without making the gemini-2* models available, so we just get no gemini models *sigh*
            # if we truly receive no models, return a list of relatively safe (barring the deprecation...) models
            for model in client.models.list(config={"page_size": 100}):
                model_name = model.name.lower()
                if "gemini" in model_name:
                    if model_name.startswith("publishers/google/models/"):
                        model_name = model_name[len("publishers/google/models/") :]
                    gemini_models.append(model_name)

            if not gemini_models:
                logger.warning(
                    "VertexAI: No gemini models found, substituting with pre-set default models, which may or may not work"
                )
                return default_models
            return gemini_models
        except Exception as e:
            logger.error(f"Error listing Vertex AI models: {str(e)}")

            # Return a snapshot of known models if we can't fetch them
            return default_models

    @staticmethod
    def get_name():
        return "Vertex AI"

    def __deepcopy__(self, memo_dict):
        return GoogleModel(
            model=self.model,
            location=self.location,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            client=self.client,
        )
