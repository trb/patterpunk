import json
import random
import time
from abc import ABC
from typing import List, Optional, Union

from patterpunk.config import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_DEFAULT_MAX_TOKENS,
    GOOGLE_DEFAULT_TEMPERATURE,
    GOOGLE_DEFAULT_TOP_K,
    GOOGLE_DEFAULT_TOP_P,
    GEMINI_REGION,
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

print('has it?', google_genai_available)

from patterpunk.llm.messages import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    AssistantMessage,
    ToolCallMessage,
)
from patterpunk.llm.models.base import Model
from patterpunk.llm.types import ToolDefinition
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
        google_account_credentials: Optional[str] = None,
    ):
        if google_account_credentials is None:
            google_account_credentials = GOOGLE_APPLICATION_CREDENTIALS
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
        location: str = GEMINI_REGION,
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

    def _convert_tools_to_google_format(self, tools: ToolDefinition) -> List:
        """Convert Patterpunk standard tools to Google genai format"""
        if not google_genai_available:
            return []
        
        # Type mapping from JSON Schema to Google genai types
        type_mapping = {
            "string": "STRING",
            "number": "NUMBER", 
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT"
        }
            
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                
                # Convert properties to Google Schema format
                properties = {}
                for prop_name, prop_def in func["parameters"].get("properties", {}).items():
                    json_type = prop_def.get("type", "string")
                    google_type = type_mapping.get(json_type.lower(), "STRING")
                    
                    properties[prop_name] = types.Schema(
                        type=google_type,
                        description=prop_def.get("description", "")
                    )
                
                function_declaration = types.FunctionDeclaration(
                    name=func["name"],
                    description=func["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=func["parameters"].get("required", [])
                    )
                )
                function_declarations.append(function_declaration)
        
        # Wrap function declarations in a Tool object
        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return []

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
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

        if tools:
            google_tools = self._convert_tools_to_google_format(tools)
            if google_tools:
                config.tools = google_tools
                # Disable automatic function calling to handle manually
                config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)

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

                # Process response candidates and parts
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        tool_calls = []
                        text_parts = []
                        
                        # Iterate through all parts to collect both text and function calls
                        for part in candidate.content.parts:
                            # Check for function calls with proper None handling
                            if hasattr(part, 'function_call') and part.function_call is not None:
                                # Convert Google function call to standard format
                                tool_call = {
                                    "id": f"call_{part.function_call.name}_{random.randint(1000, 9999)}",
                                    "type": "function",
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": json.dumps(dict(part.function_call.args))
                                    }
                                }
                                tool_calls.append(tool_call)
                            
                            # Check for text content (changed elif to if to handle parts with both text and function_call)
                            if hasattr(part, 'text') and part.text and part.text != 'None':
                                text_parts.append(part.text)
                        
                        # Return ToolCallMessage if function calls were found
                        if tool_calls:
                            return ToolCallMessage(tool_calls)
                        
                        # Return AssistantMessage with text content
                        if text_parts:
                            content = "\n".join(text_parts)
                            return AssistantMessage(
                                content, structured_output=structured_output
                            )
                
                # Fallback: try using response.text for simple responses
                try:
                    if hasattr(response, "text") and response.text:
                        content = response.text
                        return AssistantMessage(
                            content, structured_output=structured_output
                        )
                except Exception:
                    # response.text failed, likely because it's a complex response
                    pass
                
                # If we get here, something went wrong
                raise GoogleAPIError("No content found in Vertex AI response")

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
                GoogleModel.client = GoogleModel.get_client(location=GEMINI_REGION)

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
