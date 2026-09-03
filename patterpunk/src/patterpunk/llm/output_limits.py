"""Output-token ceilings learned from provider rejections.

Anthropic, Bedrock and Vertex reject an over-large max_tokens with a 400 that
names the exact ceiling. Recording it per model id costs one failed round
trip per process, once. Later instances for the same id start from it.
"""

import re
from typing import Dict, Optional, Tuple

# Message shapes observed live. Bedrock: "exceeds the model limit of 64000".
# Anthropic: "max_tokens: 200000 > 128000, which is the maximum allowed
# number of output tokens for claude-sonnet-5". Vertex names an exclusive
# bound: "supported range is from 1 (inclusive) to 65537 (exclusive)".
_LIMIT_PATTERNS: Tuple[Tuple[re.Pattern, int], ...] = (
    (re.compile(r"exceeds the model limit of (\d+)"), 0),
    (
        re.compile(
            r"max_tokens: \d+ > (\d+), which is the maximum allowed number of "
            r"output tokens"
        ),
        0,
    ),
    (
        re.compile(
            r"maxOutputTokens value of \d+ but the supported range is from 1 "
            r"\(inclusive\) to (\d+) \(exclusive\)"
        ),
        1,
    ),
)

_learned_limits: Dict[str, int] = {}


def parse_output_limit(error_message: str) -> Optional[int]:
    for pattern, exclusive_bound_offset in _LIMIT_PATTERNS:
        match = pattern.search(error_message)
        if match:
            return int(match.group(1)) - exclusive_bound_offset
    return None


def learned_output_limit(model_id: str) -> Optional[int]:
    return _learned_limits.get(model_id)


def record_output_limit(model_id: str, limit: int) -> None:
    _learned_limits[model_id] = limit


def forget_output_limits() -> None:
    _learned_limits.clear()
