import pytest

from patterpunk.llm.output_limits import (
    forget_output_limits,
    learned_output_limit,
    parse_output_limit,
    record_output_limit,
)

BEDROCK_MESSAGE = (
    "An error occurred (ValidationException) when calling the Converse operation: "
    "The maximum tokens you requested exceeds the model limit of 2048. Try again "
    "with a maximum tokens value that is lower than 2048."
)
ANTHROPIC_MESSAGE = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'max_tokens: 200000 > 128000, which is the maximum allowed number "
    "of output tokens for claude-sonnet-5'}}"
)
GEMINI_MESSAGE = (
    "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Unable to submit "
    "request because it has a maxOutputTokens value of 200000 but the supported "
    "range is from 1 (inclusive) to 65537 (exclusive). Update the value and try "
    "again.', 'status': 'INVALID_ARGUMENT'}}"
)


@pytest.mark.parametrize(
    "message,expected",
    [
        (BEDROCK_MESSAGE, 2048),
        (ANTHROPIC_MESSAGE, 128000),
        (GEMINI_MESSAGE, 65536),
        ("The provided model identifier is invalid.", None),
        (
            "thinking.budget_tokens: supported range is from 1024 (inclusive) "
            "to 65537 (exclusive)",
            None,
        ),
    ],
)
def test_parse_output_limit(message, expected):
    assert parse_output_limit(message) == expected


def test_learned_limits_are_recorded_per_model_and_forgettable():
    forget_output_limits()
    assert learned_output_limit("meta.llama3-70b-instruct-v1:0") is None
    record_output_limit("meta.llama3-70b-instruct-v1:0", 2048)
    assert learned_output_limit("meta.llama3-70b-instruct-v1:0") == 2048
    assert learned_output_limit("mistral.mistral-large-2402-v1:0") is None
    forget_output_limits()
    assert learned_output_limit("meta.llama3-70b-instruct-v1:0") is None
