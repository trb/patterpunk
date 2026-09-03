from typing import Callable, List, Optional

from patterpunk.config.providers.anthropic import (
    ANTHROPIC_DEFAULT_TEMPERATURE,
    ANTHROPIC_DEFAULT_TOP_P,
    ANTHROPIC_DEFAULT_TOP_K,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_TIMEOUT,
)
from patterpunk.config.providers.anthropic_foundry import (
    ANTHROPIC_FOUNDRY_API_KEY,
    get_anthropic_foundry_async_client,
    get_anthropic_foundry_client,
    has_foundry_endpoint,
)
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.claude_capabilities import parse_claude_version
from patterpunk.llm.retry_config import RetryConfig
from patterpunk.llm.thinking import ThinkingConfig as UnifiedThinkingConfig
from patterpunk.logger import logger


class AnthropicFoundryMissingConfigurationError(Exception):
    pass


class AnthropicFoundryModel(AnthropicModel):
    """
    Foundry's "model" request field carries a deployment name that users can
    rename freely. model_id therefore exists to carry the real Claude id for
    version detection.
    """

    def __init__(
        self,
        deployment_name: str,
        *,
        model_id: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        temperature: float = ANTHROPIC_DEFAULT_TEMPERATURE,
        top_p: float = ANTHROPIC_DEFAULT_TOP_P,
        top_k: int = ANTHROPIC_DEFAULT_TOP_K,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
        timeout: int = ANTHROPIC_DEFAULT_TIMEOUT,
        thinking_config: Optional[UnifiedThinkingConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        if not has_foundry_endpoint():
            raise AnthropicFoundryMissingConfigurationError(
                "Anthropic on Microsoft Foundry was not initialized correctly. "
                "Set PP_ANTHROPIC_FOUNDRY_RESOURCE or PP_ANTHROPIC_FOUNDRY_BASE_URL."
            )
        if azure_ad_token_provider is None and not ANTHROPIC_FOUNDRY_API_KEY:
            raise AnthropicFoundryMissingConfigurationError(
                "Anthropic on Microsoft Foundry has no credentials. Set "
                "PP_ANTHROPIC_FOUNDRY_API_KEY or pass azure_ad_token_provider."
            )
        self.model_id = model_id or deployment_name
        self.azure_ad_token_provider = azure_ad_token_provider
        super().__init__(
            model=deployment_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            timeout=timeout,
            thinking_config=thinking_config,
            retry_config=retry_config,
        )

    def __deepcopy__(self, memo_dict):
        # A token provider may close over unpicklable credential objects, so
        # the copy shares the provider reference instead of deep-copying it.
        return AnthropicFoundryModel(
            deployment_name=self.model,
            model_id=self.model_id,
            azure_ad_token_provider=self.azure_ad_token_provider,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            thinking_config=self.thinking_config,
            retry_config=self.retry_config,
        )

    def _parse_model_version(self) -> tuple[int, int]:
        # Every Foundry deployment serves a Claude model. An unrecognisable
        # deployment name falls back to the newest family's rules. The base
        # class's (0, 0) path would send sampling params that current models
        # reject with a 400.
        if parse_claude_version(self.model_id) is None:
            logger.warning(
                f"[ANTHROPIC FOUNDRY] Cannot detect a Claude version from "
                f"'{self.model_id}'. Applying newest-family (Claude 5) parameter "
                f"rules. Pass model_id with the deployment's real Claude id for "
                f"exact capability detection."
            )
            return (5, 0)
        return super()._parse_model_version()

    def _get_sync_client(self):
        return get_anthropic_foundry_client(self.azure_ad_token_provider)

    def _get_async_client(self):
        return get_anthropic_foundry_async_client(self.azure_ad_token_provider)

    def _capability_model_id(self) -> str:
        return self.model_id

    @staticmethod
    def get_name():
        return "Anthropic Foundry"

    @staticmethod
    def get_available_models() -> List[str]:
        # Foundry serves no Models API endpoint, and deployment names are
        # per resource, so enumeration is impossible from the client side.
        return []
