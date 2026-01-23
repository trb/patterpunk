import os

DEFAULT_MODEL = os.getenv("PP_DEFAULT_MODEL") or "gpt-4o-mini"
DEFAULT_TEMPERATURE = os.getenv("PP_DEFAULT_TEMPERATURE") or 1.0
DEFAULT_TOP_P = os.getenv("PP_DEFAULT_TOP_P") or 1.0
DEFAULT_FREQUENCY_PENALTY = os.getenv("PP_DEFAULT_FREQUENCY_PENALTY") or 0.0
DEFAULT_PRESENCE_PENALTY = os.getenv("PP_DEFAULT_PRESENCE_PENALTY") or 0.0
MAX_RETRIES = os.getenv("PP_MAX_RETRIES") or 3

# Retry backoff configuration for rate limits
# These defaults are tuned for per-minute rate limits (OpenAI/Azure OpenAI)
RETRY_BASE_DELAY = float(
    os.getenv("PP_RETRY_BASE_DELAY", "60")
)  # Base delay in seconds
RETRY_MAX_DELAY = float(os.getenv("PP_RETRY_MAX_DELAY", "300"))  # Maximum 5 minute cap
RETRY_MIN_DELAY = float(os.getenv("PP_RETRY_MIN_DELAY", "45"))  # Hard minimum delay
RETRY_JITTER_FACTOR = float(os.getenv("PP_RETRY_JITTER_FACTOR", "0.5"))  # ±50% jitter
SDK_MAX_RETRIES = int(
    os.getenv("PP_SDK_MAX_RETRIES", "0")
)  # SDK internal retries (0=disabled)

# Minimum token budget for extended thinking/reasoning features
# Used by providers that support reasoning (e.g., Bedrock with Claude)
MIN_THINKING_BUDGET_TOKENS = 1024

GENERATE_STRUCTURED_OUTPUT_PROMPT = "YOUR RESPONSE HAS TO INCLUDE A VALID JSON OBJECT THAT IMPLEMENTS THIS JSON SCHEMA: "
