"""
Structured output orchestration with retry mechanisms.

This module handles chat-level structured output parsing with sophisticated
retry logic and error recovery mechanisms.
"""

from patterpunk.llm.messages.exceptions import StructuredOutputFailedToParseError
from patterpunk.llm.messages.roles import ROLE_USER
from patterpunk.llm.messages.user import UserMessage
from patterpunk.logger import logger
from .exceptions import StructuredOutputParsingError


def get_parsed_output_with_retry(chat_instance):
    """
    Get parsed output from the latest message with retry logic.

    Implements sophisticated retry mechanism that prompts the LLM to fix
    invalid JSON responses when structured output parsing fails.

    :param chat_instance: The Chat instance to parse output from
    :return: Parsed structured output object
    :raises StructuredOutputParsingError: If parsing fails after all retry attempts
    """
    if not getattr(chat_instance.latest_message, "structured_output", None):
        if _requested_structured_output(chat_instance) is None:
            return None
        # A silent None here turns into AttributeErrors far away from the cause,
        # e.g. when the model answered a structured request with a real tool call.
        raise StructuredOutputParsingError(
            f"[CHAT] A structured output was requested but the latest message "
            f"({type(chat_instance.latest_message).__name__}) carries none, "
            f"latest message:\n{chat_instance.latest_message.content}"
        )

    retry = 0
    max_retries = 2

    chat = chat_instance

    while retry < max_retries:
        try:
            obj = chat.latest_message.parsed_output
            return obj
        except StructuredOutputFailedToParseError as error:
            logger.debug(
                "[CHAT] Failed to parse structured_output from latest message",
                exc_info=error,
            )
            chat = chat.add_message(
                UserMessage(
                    "You did not generate valid JSON! YOUR RESPONSE HAS TO BE A VALID JSON OBJECT THAT CONFORMS TO THE JSON SCHEMA!",
                    structured_output=chat.latest_message.structured_output,
                )
            ).complete()
            retry += 1

    raise StructuredOutputParsingError(
        f"[CHAT] Failed to parse structured_output from latest message, latest message:\n{chat_instance.latest_message.content}"
    )


def _requested_structured_output(chat_instance):
    for message in reversed(chat_instance.messages):
        if message.role == ROLE_USER:
            return getattr(message, "structured_output", None)
    return None
