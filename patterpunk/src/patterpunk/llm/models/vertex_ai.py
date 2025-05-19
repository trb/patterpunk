
import time
from abc import ABC
from typing import List, Optional, Callable, Dict, Any, Union

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    vertexai_available = True
except ImportError:
    vertexai_available = False

from patterpunk.config import (
    VERTEXAI_DEFAULT_TEMPERATURE,
    VERTEXAI_DEFAULT_TOP_P,
    VERTEXAI_DEFAULT_TOP_K,
    VERTEXAI_DEFAULT_MAX_TOKENS,
    MAX_RETRIES,
)
from patterpunk.llm.messages import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    AssistantMessage,
)
from patterpunk.llm.models.base import Model
from patterpunk.logger import logger


class VertexAIRateLimitError(Exception):
    """Raised when all retry attempts for rate limit errors are exhausted"""
    pass


class VertexAIMaxTokensError(Exception):
    """Raised when the model hits the maximum token limit"""
    pass


class VertexAINotImplemented(Exception):
    """Raised for features not yet implemented"""
    pass


class VertexAIAPIError(Exception):
    """Raised for general API errors"""
    pass


class VertexAIModel(Model, ABC):
    def __init__(
        self,
        model: str,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        temperature: float = VERTEXAI_DEFAULT_TEMPERATURE,
        top_p: float = VERTEXAI_DEFAULT_TOP_P,
        top_k: int = VERTEXAI_DEFAULT_TOP_K,
        max_tokens: int = VERTEXAI_DEFAULT_MAX_TOKENS,
    ):
        if not vertexai_available:
            raise ImportError(
                "The google-genai package is not installed. "
                "Please install it with `pip install google-genai`"
            )
        
        self.model = model
        self.project_id = project_id
        self.location = location
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        
        # Initialize the client
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )

    def generate_assistant_message(
        self, messages: List[Message], functions: Optional[List[Callable]] = None, structured_output: Optional[object] = None
    ) -> Message:
        system_prompt = "\n\n".join(
            [message.content for message in messages if message.role == ROLE_SYSTEM]
        )

        # Convert patterpunk messages to GenAI Content objects
        contents = []
        
        # Add system instruction separately as it's handled differently in GenAI
        # The actual messages will be added as a conversation history
        
        # Process all non-system messages in order to maintain conversation flow
        for message in messages:
            if message.role == ROLE_SYSTEM:
                # System messages are handled via system_instruction parameter
                continue
            elif message.role == ROLE_USER:
                if not message.is_function_call:
                    # Regular user message
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(message.content)]
                    ))
                else:
                    # Function call messages are not directly supported in this format
                    # Could be implemented with tool calls in the future
                    logger.warning("Function call messages are not fully supported in VertexAI yet")
            elif message.role == ROLE_ASSISTANT:
                # Assistant messages
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(message.content)]
                ))
        
        # Configure the generation parameters
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        
        # Add system instruction if available
        if system_prompt:
            config.system_instruction = system_prompt
        
        # Add structured output configuration if needed
        if structured_output:
            config.response_mime_type = "application/json"
            if hasattr(structured_output, "__annotations__"):
                # If it's a Pydantic-like model with type annotations
                config.response_schema = structured_output
            else:
                # Otherwise, assume it's a dictionary schema
                config.response_schema = structured_output
        
        # Add function calling support if functions are provided
        if functions:
            config.tools = functions
        
        retry_count = 0
        wait_time = 60  # Initial wait time in seconds
        
        while retry_count < MAX_RETRIES:
            try:
                # Send the entire conversation history to the model
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                
                # Extract the content from the response
                if hasattr(response, "text"):
                    content = response.text
                    return AssistantMessage(content, structured_output=structured_output)
                else:
                    # Handle case where response doesn't have text attribute
                    raise VertexAIAPIError("Unexpected response format from Vertex AI")
                
            except genai_errors.RateLimitError as e:
                if retry_count >= MAX_RETRIES - 1:  # -1 because we increment after this check
                    raise VertexAIRateLimitError(
                        f"Rate limit exceeded after {retry_count + 1} retries"
                    ) from e
                
                logger.warning(
                    f"Rate limit hit, attempt {retry_count + 1}/{MAX_RETRIES}. "
                    f"Waiting {wait_time} seconds before retry."
                )
                
                time.sleep(wait_time)
                retry_count += 1
                wait_time = int(wait_time * 1.5)  # Increase wait time by 50%
                continue
                
            except genai_errors.StopCandidateError as e:
                if "max_tokens" in str(e).lower():
                    logger.warning("Vertex AI response was cut off as the model hit MAX_TOKENS")
                    # Try to extract partial content if available
                    if hasattr(e, "candidate") and hasattr(e.candidate, "content"):
                        content = e.candidate.content
                        return AssistantMessage(content, structured_output=structured_output)
                    raise VertexAIMaxTokensError("Response exceeded maximum token limit") from e
                raise VertexAIAPIError(f"Error generating content: {str(e)}") from e
                
            except Exception as e:
                # Handle any other exceptions
                raise VertexAIAPIError(f"Error generating content: {str(e)}") from e
        
        # This should never be reached, but added for completeness
        raise VertexAIAPIError("Unexpected outcome - out of retries, but neither error raised or message returned")

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Returns a list of available Gemini models on Vertex AI.
        """
        if not vertexai_available:
            return []
        
        try:
            # Create a temporary client to list models
            client = genai.Client(vertexai=True)
            
            # Filter for Gemini models only
            gemini_models = []
            for model in client.models.list():
                if "gemini" in model.name.lower():
                    gemini_models.append(model.name)
            
            return gemini_models
        except Exception as e:
            logger.error(f"Error listing Vertex AI models: {str(e)}")
            
            # Return a snapshot of known models if we can't fetch them
            return [
                "gemini-2.5-pro-001",
                "gemini-2.5-flash-001",
                "gemini-2.0-pro-001",
                "gemini-2.0-flash-001",
                "gemini-1.5-pro-001",
                "gemini-1.5-flash-001",
                "gemini-1.0-pro-001",
                "gemini-1.0-pro",
            ]

    @staticmethod
    def get_name():
        return "Vertex AI"
