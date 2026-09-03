from ..defaults import resolve_timeout_default

# Endpoints and clients are constructor-only because one process may talk to
# several OpenAI-compatible endpoints at once. Only the timeout default comes
# from the environment.
OPENAI_COMPATIBLE_DEFAULT_TIMEOUT = resolve_timeout_default(
    "PP_OPENAI_COMPATIBLE_DEFAULT_TIMEOUT"
)
