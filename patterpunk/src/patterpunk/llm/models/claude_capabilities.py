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

_BEDROCK_REGION_PREFIX_RE = re.compile(r"^(us|us-gov|eu|apac|ca|jp|au|in|global)\.")
_RELEASE_SUFFIX_RE = re.compile(r"(?<=\d)-v\d+$")

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
    # release suffix. Vertex appends "@<date>", which stays in place for the
    # version regexes. Bedrock appends "-v1" to dated and undated ids alike,
    # as in "claude-opus-4-6-v1". The strip therefore keys on a preceding
    # digit rather than a date. The legacy ids "claude-v2" and
    # "claude-instant-v1" have a letter before "-v" and keep their marker.
    normalized = model_id.rsplit("/", 1)[-1]
    normalized = _BEDROCK_REGION_PREFIX_RE.sub("", normalized)
    if normalized.startswith("anthropic."):
        normalized = normalized[len("anthropic.") :]
    normalized = normalized.split(":", 1)[0]
    return _RELEASE_SUFFIX_RE.sub("", normalized)


def thinking_cannot_be_disabled(model_id: str) -> bool:
    # Fable and Mythos models run adaptive thinking permanently. They answer
    # thinking type "disabled" with a 400, unlike Opus 4.7+ and Sonnet 5.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-adaptive-thinking.html
    model = normalize_claude_model_id(model_id)
    return model.startswith("claude-fable") or model.startswith("claude-mythos")


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


def resolve_max_output_tokens(version: ClaudeVersion, model_id: str) -> Optional[int]:
    # The API rejects a max_tokens above the model's cap with a 400.
    # The 3.7 cap rises to 128000 when the output-128k beta header is sent.
    if not version.recognized:
        return None
    release = (version.major, version.minor)
    if release >= (4, 6):
        return None
    if release >= (4, 5):
        return 64000
    if release >= (4, 0):
        return 32000 if "opus" in normalize_claude_model_id(model_id) else 64000
    if release >= (3, 7):
        return 64000
    if release >= (3, 5):
        return 8192
    return 4096


def _clamp_temperature(value: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    if value is None or 0.0 <= value <= 1.0:
        return value, None
    clamped = 1.0 if value > 1.0 else 0.0
    return clamped, (
        f"Claude accepts temperature between 0.0 and 1.0; other providers allow "
        f"up to 2.0. Clamping temperature={value} to {clamped}."
    )


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

    temperature, clamp_warning = _clamp_temperature(requested.temperature)

    if version.at_least(4, 0):
        warnings = [clamp_warning] if clamp_warning else []
        top_p = user_set("top_p")
        if top_p is not None:
            warnings.append(
                f"Claude 4+ rejects both 'temperature' and 'top_p' simultaneously. "
                f"Dropping top_p={top_p} (keeping temperature={temperature}). "
                f"Anthropic recommends using 'temperature' for most use cases."
            )
        top_k = user_set("top_k")
        if top_k is not None:
            warnings.append(
                f"Claude 4+ restricts sampling to 'temperature'. "
                f"Dropping user-set top_k={top_k}."
            )
        return SamplingResolution(temperature, None, None, tuple(warnings))

    return SamplingResolution(
        temperature,
        requested.top_p,
        requested.top_k,
        (clamp_warning,) if clamp_warning else (),
    )
