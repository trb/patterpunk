from typing import Union, List, Optional, Any

from patterpunk.lib.extract_json import extract_json
from patterpunk.logger import logger
from ..chunks import CacheChunk
from .exceptions import (
    StructuredOutputNotPydanticLikeError,
    StructuredOutputFailedToParseError,
)
from .cache import get_content_as_string


def parse_structured_output(
    content: Union[str, List[CacheChunk]],
    structured_output: Any,
    role: str,
    cached_parsed_output: Optional[Any] = None,
) -> Any:
    if cached_parsed_output is not None:
        return cached_parsed_output

    if not structured_output:
        return None

    if not getattr(structured_output, "parse_raw", None) and not getattr(
        structured_output, "model_validate_json", None
    ):
        raise StructuredOutputNotPydanticLikeError(
            f"[MESSAGE][{role}] The provided structured_output is not a pydantic model (missing parse_raw or model_validate_json)"
        )

    content_str = get_content_as_string(content)
    json_messages = extract_json(content_str)

    for json_message in json_messages:
        try:
            if getattr(structured_output, "model_validate_json", None):
                obj = structured_output.model_validate_json(json_message)
            else:
                obj = structured_output.parse_raw(json_message)

            return obj
        except Exception as error:
            logger.debug(
                f"[MESSAGE][{role}][STRUCTURED_OUTPUT] Failed to parse response {json_message}: {error}",
                exc_info=error,
            )

    raise StructuredOutputFailedToParseError(
        f"[MESSAGE][{role}] Structured output could not be parsed, message: \n{content_str}"
    )
