import json
import random
import time
from abc import ABC
from typing import List, Optional, Set, Union

from patterpunk.config.providers.google import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_DEFAULT_MAX_TOKENS,
    GOOGLE_DEFAULT_TEMPERATURE,
    GOOGLE_DEFAULT_TOP_K,
    GOOGLE_DEFAULT_TOP_P,
    GEMINI_REGION,
)
from patterpunk.config.defaults import MAX_RETRIES

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request

    google_genai_available = True
except ImportError:
    google_genai_available = False

print("has it?", google_genai_available)

from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.roles import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.types import ToolDefinition, CacheChunk
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.chunks import MultimodalChunk, TextChunk
from patterpunk.llm.messages.cache import get_multimodal_chunks, has_multimodal_content
from patterpunk.logger import logger


class GoogleAuthenticationError(Exception):
    pass


class GoogleRateLimitError(Exception):
    pass


class GoogleMaxTokensError(Exception):
    pass


class GoogleNotImplemented(Exception):
    pass


class GoogleAPIError(Exception):
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
        credentials.refresh(Request())
        project_id = credentials_info["project_id"]

        return genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            credentials=credentials,
        )

    def __init__(
        self,
        model: str,
        location: str = GEMINI_REGION,
        temperature: float = GOOGLE_DEFAULT_TEMPERATURE,
        top_p: float = GOOGLE_DEFAULT_TOP_P,
        top_k: int = GOOGLE_DEFAULT_TOP_K,
        max_tokens: int = GOOGLE_DEFAULT_MAX_TOKENS,
        google_account_credentials: Optional[str] = None,
        client: Optional[genai.Client] = None,
        thinking_config: Optional[ThinkingConfig] = None,
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

        thinking_budget = None
        include_thoughts = False
        if thinking_config is not None:
            if thinking_config.token_budget is not None:
                thinking_budget = min(thinking_config.token_budget, 24576)
            else:
                effort_to_tokens = {"low": 1500, "medium": 4000, "high": 12000}
                thinking_budget = effort_to_tokens[thinking_config.effort]
            include_thoughts = thinking_config.include_thoughts

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
        self.thinking_budget = thinking_budget
        self.include_thoughts = include_thoughts
        self.thinking_config = thinking_config

    def _translate_output_types_to_google_modalities(
        self, output_types: Optional[Union[List[OutputType], Set[OutputType]]]
    ) -> Optional[List[str]]:
        if not output_types:
            return None

        modality_mapping = {
            OutputType.TEXT: "TEXT",
            OutputType.IMAGE: "IMAGE",
        }

        modalities = []
        for output_type in output_types:
            if output_type in modality_mapping:
                modalities.append(modality_mapping[output_type])

        return modalities if modalities else None

    def _convert_tools_to_google_format(self, tools: ToolDefinition) -> List:
        if not google_genai_available:
            return []

        type_mapping = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]

                properties = {}
                for prop_name, prop_def in (
                    func["parameters"].get("properties", {}).items()
                ):
                    json_type = prop_def.get("type", "string")
                    google_type = type_mapping.get(json_type.lower(), "STRING")

                    properties[prop_name] = types.Schema(
                        type=google_type, description=prop_def.get("description", "")
                    )

                function_declaration = types.FunctionDeclaration(
                    name=func["name"],
                    description=func["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=func["parameters"].get("required", []),
                    ),
                )
                function_declarations.append(function_declaration)

        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return []

    def _create_cache_objects_for_chunks(
        self, chunks: List[Union[CacheChunk, TextChunk]]
    ) -> dict[str, str]:
        cache_mappings = {}

        if not google_genai_available:
            return cache_mappings

        for i, chunk in enumerate(chunks):
            if (
                isinstance(chunk, CacheChunk)
                and chunk.cacheable
                and len(chunk.content) > 32000
            ):
                cache_id = f"cache_chunk_{i}_{hash(chunk.content)}"

                try:
                    cached_content = self.client.caches.create(
                        config=types.CreateCachedContentConfig(
                            model=self.model,
                            contents=[
                                types.Content(
                                    parts=[types.Part.from_text(chunk.content)]
                                )
                            ],
                            ttl=chunk.ttl or types.Timedelta(hours=1),
                        )
                    )
                    cache_mappings[cache_id] = cached_content.name
                except Exception as e:
                    logger.warning(f"Failed to create cache for chunk {i}: {e}")

        return cache_mappings

    def _convert_message_content_for_google(self, content) -> List:
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        parts = []

        for chunk in content:
            if isinstance(chunk, TextChunk):
                parts.append(types.Part.from_text(text=chunk.content))
            elif isinstance(chunk, CacheChunk):
                parts.append(types.Part.from_text(text=chunk.content))
            elif isinstance(chunk, MultimodalChunk):
                if chunk.source_type == "gcs_uri":
                    parts.append(
                        types.Part.from_uri(
                            uri=chunk.get_url(),
                            mime_type=chunk.media_type or "application/octet-stream",
                        )
                    )
                elif hasattr(chunk, "file_id"):
                    parts.append(chunk.file_id)
                else:
                    media_type = chunk.media_type or "application/octet-stream"

                    if chunk.source_type == "url":
                        chunk = chunk.download()

                    parts.append(
                        types.Part.from_bytes(
                            data=chunk.to_bytes(), mime_type=media_type
                        )
                    )

        return parts

    def upload_file_to_google(self, chunk: MultimodalChunk):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            suffix=f".{chunk.filename.split('.')[-1] if chunk.filename else 'bin'}",
            delete=False,
        ) as tmp_file:
            tmp_file.write(chunk.to_bytes())
            tmp_file_path = tmp_file.name

        try:
            uploaded_file = self.client.files.upload(file=tmp_file_path)
            return uploaded_file
        finally:
            os.unlink(tmp_file_path)

    def _convert_messages_for_google_with_cache(
        self, messages: List[Message]
    ) -> tuple[List[types.Content], dict[str, str]]:
        contents = []
        all_cache_mappings = {}

        for message in messages:
            if message.role == ROLE_SYSTEM:
                continue

            if message.role == "tool_call":
                # Serialize ToolCallMessage as model role with functionCall parts
                parts = []
                for tool_call in message.tool_calls:
                    # Parse arguments from JSON string
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        arguments = {}

                    # Google uses FunctionCall for tool requests
                    parts.append(
                        types.Part.from_function_call(
                            name=tool_call["function"]["name"], args=arguments
                        )
                    )

                contents.append(types.Content(role="model", parts=parts))
                continue

            if message.role == "tool_result":
                # Validate required field for Google
                if not message.function_name:
                    raise ValueError(
                        "Google Vertex AI requires function_name in ToolResultMessage. "
                        "Ensure ToolResultMessage is created with function_name from the original ToolCallMessage."
                    )

                # Try to parse content as JSON; fall back to wrapping in result object
                try:
                    response_data = json.loads(message.content)
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": message.content}

                # Google uses functionResponse for tool results
                parts = [
                    types.Part.from_function_response(
                        name=message.function_name, response=response_data
                    )
                ]

                contents.append(types.Content(role="user", parts=parts))
                continue

            if isinstance(message.content, list):
                if has_multimodal_content(message.content):
                    parts = self._convert_message_content_for_google(message.content)
                    has_text = any(
                        isinstance(chunk, (TextChunk, CacheChunk))
                        for chunk in message.content
                    )
                    if not has_text:
                        raise ValueError(
                            "Google requires at least one text block when multimodal content is present. "
                            "Please include text content along with your images or other media."
                        )
                else:
                    cache_mappings = self._create_cache_objects_for_chunks(
                        message.content
                    )
                    all_cache_mappings.update(cache_mappings)

                    parts = []
                    for chunk in message.content:
                        parts.append(types.Part.from_text(text=chunk.content))
            else:
                parts = [types.Part.from_text(text=message.get_content_as_string())]

            if message.role == ROLE_USER:
                contents.append(
                    types.Content(
                        role="user",
                        parts=parts,
                    )
                )
            elif message.role == ROLE_ASSISTANT:
                contents.append(types.Content(role="model", parts=parts))

        return contents, all_cache_mappings

    def _convert_system_messages_for_google_with_cache(
        self, messages: List[Message]
    ) -> str:
        system_parts = []

        for message in messages:
            if message.role == ROLE_SYSTEM:
                if isinstance(message.content, list):
                    content_str = "\n\n".join(
                        chunk.content for chunk in message.content
                    )
                    system_parts.append(content_str)
                else:
                    system_parts.append(message.content)

        return "\n\n".join(system_parts)

    def _build_generation_config(
        self,
        tools: Optional[ToolDefinition],
        structured_output: Optional[object],
        output_types: Optional[Union[List[OutputType], Set[OutputType]]],
        system_instruction: Optional[str],
    ) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        response_modalities = self._translate_output_types_to_google_modalities(
            output_types
        )
        if response_modalities:
            config.response_modalities = response_modalities

        if self.thinking_budget is not None or self.include_thoughts:
            thinking_config_kwargs = {}
            if self.thinking_budget is not None:
                thinking_config_kwargs["thinking_budget"] = self.thinking_budget
            if self.include_thoughts:
                thinking_config_kwargs["include_thoughts"] = self.include_thoughts
            config.thinking_config = types.ThinkingConfig(**thinking_config_kwargs)

        if system_instruction:
            config.system_instruction = system_instruction

        if tools:
            google_tools = self._convert_tools_to_google_format(tools)
            if google_tools:
                config.tools = google_tools
                config.automatic_function_calling = (
                    types.AutomaticFunctionCallingConfig(disable=True)
                )

        if structured_output:
            config.response_mime_type = "application/json"
            if hasattr(structured_output, "__annotations__"):
                config.response_schema = structured_output
            else:
                config.response_schema = structured_output

        return config

    def _prepare_generation_request(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> tuple[List[types.Content], types.GenerateContentConfig, dict[str, str]]:
        system_instruction = self._convert_system_messages_for_google_with_cache(
            messages
        )

        contents, cache_mappings = self._convert_messages_for_google_with_cache(
            messages
        )

        config = self._build_generation_config(
            tools, structured_output, output_types, system_instruction
        )

        return contents, config, cache_mappings

    def _process_generation_response(
        self, response, structured_output: Optional[object]
    ) -> Union[Message, "ToolCallMessage"]:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content.parts:
                tool_calls = []
                chunks = []

                for part in candidate.content.parts:
                    if (
                        hasattr(part, "function_call")
                        and part.function_call is not None
                    ):
                        tool_call = {
                            "id": f"call_{part.function_call.name}_{random.randint(1000, 9999)}",
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args)),
                            },
                        }
                        tool_calls.append(tool_call)

                    elif hasattr(part, "text") and part.text and part.text != "None":
                        chunks.append(TextChunk(part.text))

                    elif hasattr(part, "inline_data") and part.inline_data:
                        mime_type = getattr(part.inline_data, "mime_type", None)
                        data = getattr(part.inline_data, "data", None)
                        if data and mime_type:
                            if isinstance(data, bytes):
                                image_chunk = MultimodalChunk.from_bytes(
                                    data, media_type=mime_type
                                )
                            else:
                                image_chunk = MultimodalChunk.from_base64(
                                    data, media_type=mime_type
                                )
                            chunks.append(image_chunk)

                if tool_calls:
                    return ToolCallMessage(tool_calls)

                if chunks:
                    if len(chunks) == 1 and isinstance(chunks[0], TextChunk):
                        return AssistantMessage(
                            chunks[0].content,
                            structured_output=structured_output,
                        )
                    else:
                        return AssistantMessage(
                            chunks, structured_output=structured_output
                        )

        try:
            if hasattr(response, "text") and response.text:
                content = response.text
                return AssistantMessage(content, structured_output=structured_output)
        except Exception:
            pass

        raise GoogleAPIError("No content found in Vertex AI response")

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        contents, config, cache_mappings = self._prepare_generation_request(
            messages, tools, structured_output, output_types
        )

        retry_count = 0
        wait_time = 60

        while retry_count < MAX_RETRIES:
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )

                return self._process_generation_response(response, structured_output)

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

            gemini_models = []
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
            thinking_config=self.thinking_config,
        )
