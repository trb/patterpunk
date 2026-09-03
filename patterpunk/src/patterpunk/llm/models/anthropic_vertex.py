from typing import List, Optional

from patterpunk.config.providers.anthropic import (
    ANTHROPIC_DEFAULT_TEMPERATURE,
    ANTHROPIC_DEFAULT_TOP_P,
    ANTHROPIC_DEFAULT_TOP_K,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_TIMEOUT,
)
from patterpunk.config.providers.anthropic_vertex import (
    ANTHROPIC_VERTEX_REGION,
    get_anthropic_vertex_async_client,
    get_anthropic_vertex_client,
    get_anthropic_vertex_project,
)
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.retry_config import RetryConfig
from patterpunk.llm.thinking import ThinkingConfig as UnifiedThinkingConfig


class AnthropicVertexMissingConfigurationError(Exception):
    pass


class AnthropicVertexModel(AnthropicModel):
    """
    Vertex names Claude models with an "@"-date form (claude-sonnet-4-5@20250929)
    or a bare id (claude-opus-5). A dash-date Anthropic id such as
    claude-sonnet-4-5-20250929 is not a Vertex model and the API returns 404.
    """

    def __init__(
        self,
        model: str,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        temperature: float = ANTHROPIC_DEFAULT_TEMPERATURE,
        top_p: float = ANTHROPIC_DEFAULT_TOP_P,
        top_k: int = ANTHROPIC_DEFAULT_TOP_K,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
        timeout: int = ANTHROPIC_DEFAULT_TIMEOUT,
        thinking_config: Optional[UnifiedThinkingConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.project_id = project_id or get_anthropic_vertex_project()
        self.region = region or ANTHROPIC_VERTEX_REGION
        if not self.project_id:
            raise AnthropicVertexMissingConfigurationError(
                "Anthropic on Vertex AI was not initialized correctly. Set "
                "PP_ANTHROPIC_VERTEX_PROJECT (or point "
                "PP_GOOGLE_APPLICATION_CREDENTIALS at a service-account file "
                "containing project_id), or pass project_id explicitly."
            )
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            timeout=timeout,
            thinking_config=thinking_config,
            retry_config=retry_config,
        )

    def _get_sync_client(self):
        return get_anthropic_vertex_client(self.project_id, self.region)

    def _get_async_client(self):
        return get_anthropic_vertex_async_client(self.project_id, self.region)

    @staticmethod
    def get_name():
        return "Anthropic Vertex"

    @staticmethod
    def get_available_models() -> List[str]:
        # The Vertex client exposes no models resource; Claude availability
        # is managed per project in the Vertex Model Garden.
        return []
