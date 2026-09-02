import pytest

from patterpunk.llm.models.claude_capabilities import (
    ClaudeVersion,
    SamplingParams,
    normalize_claude_model_id,
    parse_claude_version,
    resolve_claude_sampling,
)


def version_tuple(model_id):
    version = parse_claude_version(model_id)
    assert version is not None, f"Expected a Claude version for {model_id!r}"
    return (version.major, version.minor)


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("claude-3-haiku-20240307", (3, 0)),
        ("claude-3-opus-20240229", (3, 0)),
        ("claude-3-5-haiku-20241022", (3, 5)),
        ("claude-3-7-sonnet-20250219", (3, 7)),
        ("claude-opus-4-20250514", (4, 0)),
        ("claude-sonnet-4-5-20250929", (4, 5)),
        ("claude-haiku-4-5", (4, 5)),
        ("claude-opus-4-7", (4, 7)),
        ("claude-opus-4-7-20260416", (4, 7)),
        ("claude-fable-5", (5, 0)),
        ("claude-sonnet-5-1-20250814", (5, 1)),
    ],
)
def test_parse_direct_api_ids(model_id, expected):
    assert version_tuple(model_id) == expected


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("anthropic.claude-3-sonnet-20240229-v1:0", (3, 0)),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", (4, 5)),
        ("eu.anthropic.claude-haiku-4-5-20251001-v1:0", (4, 5)),
        ("apac.anthropic.claude-3-5-sonnet-20241022-v2:0", (3, 5)),
        ("ca.anthropic.claude-3-7-sonnet-20250219-v1:0", (3, 7)),
        ("us.anthropic.claude-opus-5-20260120-v1:0", (5, 0)),
    ],
)
def test_parse_bedrock_ids(model_id, expected):
    assert version_tuple(model_id) == expected


@pytest.mark.parametrize(
    "model_id,expected",
    [
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            (4, 5),
        ),
        (
            "arn:aws:bedrock:us-east-1::foundation-model/"
            "anthropic.claude-3-sonnet-20240229-v1:0",
            (3, 0),
        ),
    ],
)
def test_parse_bedrock_arns(model_id, expected):
    assert version_tuple(model_id) == expected


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("anthropic.claude-v2", (2, 0)),
        ("anthropic.claude-v2:1", (2, 0)),
        ("anthropic.claude-instant-v1", (1, 0)),
        ("claude-2.1", (2, 0)),
        ("claude-instant-1.2", (1, 0)),
    ],
)
def test_parse_legacy_ids(model_id, expected):
    assert version_tuple(model_id) == expected


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("claude-sonnet-4-5@20250929", (4, 5)),
        ("claude-opus-4-1@20250805", (4, 1)),
        ("claude-3-5-sonnet-v2@20241022", (3, 5)),
        ("claude-3-haiku@20240307", (3, 0)),
        ("claude-opus-5", (5, 0)),
    ],
)
def test_parse_vertex_ids(model_id, expected):
    assert version_tuple(model_id) == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-4o",
        "gemini-2.5-pro",
        "meta.llama3-70b-instruct-v1:0",
        "mistral.mistral-large-2402-v1:0",
        "amazon.nova-pro-v1:0",
        "deepseek.r1-v1:0",
        "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123",
    ],
)
def test_parse_non_claude_returns_none(model_id):
    assert parse_claude_version(model_id) is None


def test_parse_unknown_claude_id_treated_as_newest_family():
    version = parse_claude_version("claude-nonsense-xyz")
    assert version == ClaudeVersion(5, 0, recognized=False)
    assert version.at_least(4, 7)


def test_normalize_strips_all_bedrock_wrapping():
    assert (
        normalize_claude_model_id(
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        == "claude-sonnet-4-5-20250929"
    )
    assert normalize_claude_model_id("anthropic.claude-v2:1") == "claude-v2"


ANTHROPIC_DEFAULTS = SamplingParams(temperature=0.7, top_p=1.0, top_k=200)
BEDROCK_DEFAULTS = SamplingParams(temperature=1.0, top_p=None, top_k=None)


def test_resolver_post_46_drops_all_params_with_combined_warning():
    resolution = resolve_claude_sampling(
        ClaudeVersion(4, 7),
        thinking_enabled=False,
        requested=SamplingParams(temperature=0.2, top_p=0.5, top_k=40),
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert resolution.temperature is None
    assert resolution.top_p is None
    assert resolution.top_k is None
    assert len(resolution.warnings) == 1
    assert "temperature=0.2" in resolution.warnings[0]
    assert "top_p=0.5" in resolution.warnings[0]
    assert "top_k=40" in resolution.warnings[0]


def test_resolver_post_46_silent_at_defaults():
    resolution = resolve_claude_sampling(
        ClaudeVersion(5, 0),
        thinking_enabled=False,
        requested=ANTHROPIC_DEFAULTS,
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert (resolution.temperature, resolution.top_p, resolution.top_k) == (
        None,
        None,
        None,
    )
    assert resolution.warnings == ()


def test_resolver_thinking_coerces_temperature_and_drops_top_p_top_k():
    resolution = resolve_claude_sampling(
        ClaudeVersion(4, 5),
        thinking_enabled=True,
        requested=SamplingParams(temperature=0.3, top_p=0.9, top_k=40),
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert resolution.temperature == 1.0
    assert resolution.top_p is None
    assert resolution.top_k is None
    assert any("temperature=0.3" in warning for warning in resolution.warnings)
    assert any("top_p=0.9" in warning for warning in resolution.warnings)
    assert any("top_k=40" in warning for warning in resolution.warnings)


def test_resolver_thinking_silent_at_defaults():
    resolution = resolve_claude_sampling(
        ClaudeVersion(3, 7),
        thinking_enabled=True,
        requested=ANTHROPIC_DEFAULTS,
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert resolution.temperature == 1.0
    assert resolution.top_p is None
    assert resolution.top_k is None
    assert resolution.warnings == ()


def test_resolver_thinking_user_set_temperature_of_one_is_silent():
    resolution = resolve_claude_sampling(
        ClaudeVersion(4, 5),
        thinking_enabled=True,
        requested=SamplingParams(temperature=1.0, top_p=1.0, top_k=200),
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert resolution.temperature == 1.0
    assert not any("temperature" in warning for warning in resolution.warnings)


def test_resolver_claude4_keeps_temperature_drops_top_p_and_top_k():
    resolution = resolve_claude_sampling(
        ClaudeVersion(4, 0),
        thinking_enabled=False,
        requested=SamplingParams(temperature=0.2, top_p=0.5, top_k=40),
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert resolution.temperature == 0.2
    assert resolution.top_p is None
    assert resolution.top_k is None
    assert any("Dropping top_p=0.5" in warning for warning in resolution.warnings)
    assert any("keeping temperature=0.2" in warning for warning in resolution.warnings)
    assert any("top_k=40" in warning for warning in resolution.warnings)


def test_resolver_claude4_silent_at_defaults():
    resolution = resolve_claude_sampling(
        ClaudeVersion(4, 5),
        thinking_enabled=False,
        requested=ANTHROPIC_DEFAULTS,
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert resolution.temperature == 0.7
    assert resolution.top_p is None
    assert resolution.top_k is None
    assert resolution.warnings == ()


def test_resolver_claude3_passes_everything_through():
    requested = SamplingParams(temperature=0.2, top_p=0.5, top_k=40)
    resolution = resolve_claude_sampling(
        ClaudeVersion(3, 0),
        thinking_enabled=False,
        requested=requested,
        defaults=ANTHROPIC_DEFAULTS,
    )
    assert resolution.temperature == 0.2
    assert resolution.top_p == 0.5
    assert resolution.top_k == 40
    assert resolution.warnings == ()


def test_resolver_unrequested_params_stay_unrequested():
    resolution = resolve_claude_sampling(
        ClaudeVersion(4, 5),
        thinking_enabled=False,
        requested=SamplingParams(temperature=0.4, top_p=None, top_k=None),
        defaults=BEDROCK_DEFAULTS,
    )
    assert resolution.temperature == 0.4
    assert resolution.top_p is None
    assert resolution.top_k is None
    assert resolution.warnings == ()


def test_resolver_bedrock_style_user_top_p_warns_even_at_one():
    resolution = resolve_claude_sampling(
        ClaudeVersion(4, 5),
        thinking_enabled=False,
        requested=SamplingParams(temperature=1.0, top_p=1.0, top_k=None),
        defaults=BEDROCK_DEFAULTS,
    )
    assert resolution.top_p is None
    assert any("Dropping top_p=1.0" in warning for warning in resolution.warnings)
