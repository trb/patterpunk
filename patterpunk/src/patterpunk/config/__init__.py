# Re-export global defaults for backward compatibility
from .defaults import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_PRESENCE_PENALTY,
    MAX_RETRIES,
    GENERATE_STRUCTURED_OUTPUT_PROMPT,
)

# Re-export OpenAI configuration for backward compatibility
from .providers.openai import (
    OPENAI_API_KEY,
    OPENAI_MAX_RETRIES,
    openai,
)

# Re-export Anthropic configuration for backward compatibility
from .providers.anthropic import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_DEFAULT_TEMPERATURE,
    ANTHROPIC_DEFAULT_TOP_P,
    ANTHROPIC_DEFAULT_TOP_K,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_TIMEOUT,
    anthropic,
)

# Re-export Bedrock configuration for backward compatibility
from .providers.bedrock import (
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    get_bedrock_client_by_region,
    boto3,
)

# Re-export Google configuration for backward compatibility
from .providers.google import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GEMINI_REGION,
    GEMINI_PROJECT,
    GOOGLE_DEFAULT_TEMPERATURE,
    GOOGLE_DEFAULT_TOP_P,
    GOOGLE_DEFAULT_TOP_K,
    GOOGLE_DEFAULT_MAX_TOKENS,
)

# Re-export Ollama configuration for backward compatibility
from .providers.ollama import (
    OLLAMA_API_ENDPOINT,
    ollama,
)

# Export provider registry functions
from .providers import (
    get_available_providers,
    is_provider_available,
    get_provider_module,
)