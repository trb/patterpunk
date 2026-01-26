import logging

# Libraries should only create loggers, not configure handlers.
# The application controls how logs are formatted and where they go.
# See: https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library

logger = logging.getLogger("patterpunk")
logger_llm = logging.getLogger("patterpunk.llm")

# Add NullHandler to prevent "No handler found" warnings when
# the application doesn't configure logging for this namespace.
logger.addHandler(logging.NullHandler())
