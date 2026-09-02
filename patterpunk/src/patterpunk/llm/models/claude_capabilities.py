"""Version parsing and sampling-parameter rules shared by every platform that
hosts Claude models (direct API, Bedrock, Vertex AI, Microsoft Foundry).

Anthropic enforces these rules server-side on all platforms identically:
- Models released after Opus 4.6 reject temperature, top_p and top_k outright.
- Claude 4.0-4.6 rejects temperature and top_p together; keep temperature.
- Extended thinking requires temperature=1.0 and rejects top_p/top_k.
- Claude 3.x and older accept all sampling parameters.

This module is stdlib-only and log-free: it returns unprefixed warning strings.
Each provider logs them under its own [ANTHROPIC]/[BEDROCK] prefix.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

_BEDROCK_REGION_PREFIX_RE = re.compile(r"^(us|eu|apac|ca|jp|au|global)\.")
_DATED_RELEASE_SUFFIX_RE = re.compile(r"(?<=\d{8})-v\d+$")

_CLAUDE3_MINOR_RE = re.compile(r"claude-3-(\d+)-(?:opus|sonnet|haiku)")
_CLAUDE3_BASE_RE = re.compile(r"claude-3-(?:opus|sonnet|haiku)[-@]\d{8}")
_CLAUDE4PLUS_DATED_RE = re.compile(
    r"claude-(?:fable|mythos|opus|sonnet|haiku)-(\d+)(?:-(\d+))?[-@]\d{8}$"
)
_CLAUDE4PLUS_BARE_RE = re.compile(
    r"claude-(?:fable|mythos|opus|sonnet|haiku)-(\d+)(?:-(\d+))?$"
)
_CLAUDE_LEGACY_V_RE = re.compile(r"^claude-v(\d+)$")
_CLAUDE_LEGACY_DOTTED_RE = re.compile(r"^claude-(\d)(?:\.\d+)?$")


@dataclass(frozen=True)
class ClaudeVersion:
    major: int
    minor: int
    recognized: bool = True

    def at_least(self, major: int, minor: int) -> bool:
        return (self.major, self.minor) >= (major, minor)


@dataclass(frozen=True)
class SamplingParams:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


@dataclass(frozen=True)
class SamplingResolution:
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    warnings: Tuple[str, ...]


def normalize_claude_model_id(model_id: str) -> str:
    # Bedrock wraps ids in ARNs ("arn:...:inference-profile/us.anthropic.<id>"),
    # regional routing prefixes, the "anthropic." vendor prefix, and a "-v1:0"
    # release suffix. Vertex uses "<id>@<date>". Reduce all forms to the bare
    # Anthropic id. The release-suffix strip requires a preceding date so the
    # legacy ids "claude-v2" and "claude-instant-v1" keep their version marker.
    normalized = model_id.rsplit("/", 1)[-1]
    normalized = _BEDROCK_REGION_PREFIX_RE.sub("", normalized)
    if normalized.startswith("anthropic."):
        normalized = normalized[len("anthropic.") :]
    normalized = normalized.split(":", 1)[0]
    return _DATED_RELEASE_SUFFIX_RE.sub("", normalized)


def parse_claude_version(model_id: str) -> Optional[ClaudeVersion]:
    model = normalize_claude_model_id(model_id)

    match = _CLAUDE3_MINOR_RE.search(model)
    if match:
        return ClaudeVersion(3, int(match.group(1)))

    if _CLAUDE3_BASE_RE.search(model):
        return ClaudeVersion(3, 0)

    match = _CLAUDE4PLUS_DATED_RE.search(model)
    if match:
        return ClaudeVersion(int(match.group(1)), int(match.group(2) or 0))

    match = _CLAUDE4PLUS_BARE_RE.search(model)
    if match:
        return ClaudeVersion(int(match.group(1)), int(match.group(2) or 0))

    if model.startswith("claude-instant"):
        return ClaudeVersion(1, 0)

    match = _CLAUDE_LEGACY_V_RE.match(model)
    if match:
        return ClaudeVersion(int(match.group(1)), 0)

    match = _CLAUDE_LEGACY_DOTTED_RE.match(model)
    if match:
        return ClaudeVersion(int(match.group(1)), 0)

    if model.startswith("claude-"):
        # Models released after Opus 4.6 reject temperature, top_p and top_k.
        # Assuming the strictest rules keeps a new, unknown id from sending them.
        return ClaudeVersion(5, 0, recognized=False)

    return None


def resolve_claude_sampling(
    version: ClaudeVersion,
    thinking_enabled: bool,
    requested: SamplingParams,
    defaults: SamplingParams,
) -> SamplingResolution:
    def user_set(name: str):
        value = getattr(requested, name)
        if value is None or value == getattr(defaults, name):
            return None
        return value

    if version.at_least(4, 7):
        dropped = [
            f"{name}={value}"
            for name in ("temperature", "top_p", "top_k")
            if (value := user_set(name)) is not None
        ]
        warnings = ()
        if dropped:
            warnings = (
                f"Claude Opus 4.7+ removed sampling params from the API. "
                f"Dropping user-set value(s): {', '.join(dropped)}. "
                f"Use prompting to guide model behavior instead.",
            )
        return SamplingResolution(None, None, None, warnings)

    if thinking_enabled:
        warnings = []
        temperature = user_set("temperature")
        if temperature is not None and temperature != 1.0:
            warnings.append(
                f"Extended thinking requires temperature=1.0. "
                f"Coercing user-set temperature={temperature} to 1.0."
            )
        top_p = user_set("top_p")
        if top_p is not None:
            warnings.append(
                f"Extended thinking rejects 'top_p'. Dropping user-set top_p={top_p}."
            )
        top_k = user_set("top_k")
        if top_k is not None:
            warnings.append(
                f"Extended thinking rejects 'top_k'. Dropping user-set top_k={top_k}."
            )
        return SamplingResolution(1.0, None, None, tuple(warnings))

    if version.at_least(4, 0):
        warnings = []
        top_p = user_set("top_p")
        if top_p is not None:
            warnings.append(
                f"Claude 4+ rejects both 'temperature' and 'top_p' simultaneously. "
                f"Dropping top_p={top_p} (keeping temperature={requested.temperature}). "
                f"Anthropic recommends using 'temperature' for most use cases."
            )
        top_k = user_set("top_k")
        if top_k is not None:
            warnings.append(
                f"Claude 4+ restricts sampling to 'temperature'. "
                f"Dropping user-set top_k={top_k}."
            )
        return SamplingResolution(requested.temperature, None, None, tuple(warnings))

    return SamplingResolution(
        requested.temperature, requested.top_p, requested.top_k, ()
    )
