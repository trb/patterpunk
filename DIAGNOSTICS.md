# Response Diagnostics & Safety Filtering

Patterpunk exposes a unified, provider-agnostic surface for inspecting *why* a model stopped generating, plus controls for disabling safety filters where supported. This is the toolkit for triaging non-text outcomes (empty responses, content blocks, max-token cutoffs) and for working with corpora that trip default safety filters.

## When to reach for these features

- **You got an empty or unexpected response and need to know why.** Use `AssistantMessage.finish_reason` and `AssistantMessage._provider.raw_finish_reason` to triage.
- **A model is refusing benign content** (legal corpora, medical text, security research). Use `Chat(disable_safety_filters=True)` for the coarse knob, or `GoogleModel(safety_settings=[...])` for per-category control.
- **You're processing a batch and don't want one empty response to abort the whole pipeline.** Use `GoogleModel(allow_empty_response=True)` to receive an empty `AssistantMessage` instead of a raised exception.
- **Your conversation persistence layer needs to record outcomes for downstream analytics.** Both `finish_reason` and `_provider` round-trip through `serialize()` / `deserialize()`.

## Unified `finish_reason`

Every `AssistantMessage` returned by a provider model carries a normalized `finish_reason` enum. The vocabulary is deliberately small (5 values) so consumer code can branch on it without learning each provider's native enum:

| Value | Meaning |
|-------|---------|
| `FinishReason.STOP` | Natural completion. Provider returned `end_turn` / `stop` / `STOP` / `completed`. |
| `FinishReason.MAX_TOKENS` | Token limit hit. Provider returned `max_tokens` / `length` / `MAX_TOKENS` / `max_output_tokens`. |
| `FinishReason.TOOL_USE` | Model emitted a tool call alongside content. (Most provider/patterpunk paths route pure tool calls to `ToolCallMessage` — this value mainly appears for Anthropic/OpenAI/Bedrock when content + tool call coexist.) |
| `FinishReason.SAFETY` | Content blocked by a safety filter. Includes Google's `SAFETY`/`PROHIBITED_CONTENT`/`BLOCKLIST`/`SPII`/`IMAGE_*`, OpenAI's `content_filter`, Anthropic's `refusal`, Bedrock's `guardrail_intervened`/`content_filtered`. |
| `FinishReason.OTHER` | Anything not in the above. Includes Google's `RECITATION`/`MALFORMED_FUNCTION_CALL`, Anthropic's `model_context_window_exceeded`, Bedrock's `malformed_*`, Ollama's `load`/`unload`, and any future enum value the SDK adds. |

`FinishReason` is a `StrEnum`, so equality with raw strings still works — both styles are valid:

```python
from patterpunk.llm.finish_reason import FinishReason

if response.finish_reason == FinishReason.SAFETY:
    ...

# Equivalent — works because FinishReason is str-backed:
if response.finish_reason == "safety":
    ...
```

### Reading the normalized value

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages import UserMessage

result = Chat().add_message(UserMessage("Hello")).complete().latest_message

if result.finish_reason == FinishReason.SAFETY:
    log.warning("Model refused — content blocked by safety filter")
elif result.finish_reason == FinishReason.MAX_TOKENS:
    log.warning("Output truncated — increase max_tokens")
elif result.finish_reason == FinishReason.OTHER:
    log.info("Unusual finish reason; check raw value below")
```

### Why a small vocabulary

The native enums vary widely (Google has 15+ values; Bedrock has 9; OpenAI has 4). Mapping each provider's full vocabulary into a normalized set forces hard choices about what to fold together. Patterpunk's design folds aggressively into 5 categories — covering the high-frequency outcomes — and leaves the rest of the SDK's signal accessible via the escape hatch below.

## Provider-specific escape hatch: `_provider`

For values that don't cleanly fit the unified vocabulary, every `AssistantMessage` carries a `_provider: ProviderData` attribute. Access fields by attribute; missing fields return `None`:

```python
result = chat.complete().latest_message

# Always-available across all providers:
result._provider.raw_finish_reason  # provider-native string, e.g., "RECITATION"

# Google-only (None for other providers):
result._provider.prompt_block_reason  # e.g., "SAFETY" when the prompt itself was blocked
```

### Per-provider populated fields

| Provider | `raw_finish_reason` example values | Other populated fields |
|----------|------------------------------------|------------------------|
| Google | `STOP`, `SAFETY`, `RECITATION`, `MALFORMED_FUNCTION_CALL`, etc. | `prompt_block_reason` |
| Anthropic | `end_turn`, `stop_sequence`, `max_tokens`, `tool_use`, `refusal`, `model_context_window_exceeded` | — |
| OpenAI | `completed`, `failed`, `max_output_tokens`, `content_filter`, etc. | — |
| Bedrock | `end_turn`, `tool_use`, `guardrail_intervened`, `content_filtered`, `malformed_model_output`, etc. | — |
| Ollama | `stop`, `length`, `load`, `unload` | — |

The leading underscore on `_provider` is a convention: it signals "advanced / non-portable". Code that accesses `_provider` is making a deliberate trade — the value is provider-specific and won't necessarily be populated when you swap providers later.

### When to reach into `_provider`

- **Distinguishing Google's `RECITATION` from `SAFETY`.** Both fold to `FinishReason.OTHER` and `FinishReason.SAFETY` respectively, but `RECITATION` has different remediation (rephrase the prompt) than safety blocks.
- **Detecting Google's prompt-level blocks.** If `prompt_feedback.block_reason` was set, the prompt itself was rejected before generation — distinct from a candidate that ran and produced safety-blocked output.
- **Triaging Bedrock guardrail interventions vs. intrinsic content filters.** `guardrail_intervened` and `content_filtered` fold to the same `SAFETY` bucket but have different operational meanings.
- **Recording the raw value in logs and analytics**, so future analysis isn't constrained by patterpunk's mapping decisions.

## Chat-level `disable_safety_filters`

```python
chat = Chat(model=GoogleModel(model="gemini-2.5-pro"), disable_safety_filters=True)
```

This is a coarse, provider-agnostic toggle. Each provider translates it according to what its API actually supports.

| Provider | Effect |
|----------|--------|
| Google | All 10 safety categories set to `OFF` (`HarmBlockThreshold.OFF`). |
| Anthropic | No-op + debug log. Anthropic does not expose API parameters that weaken intrinsic safety. |
| OpenAI | No-op + debug log. The OpenAI Python SDK has no parameter that disables content filtering. **Azure OpenAI** deployments configure content filters at the deployment level (severity profiles + the gated "Modified Content Filters" approval workflow); no SDK request parameter can disable them. |
| Bedrock | No-op + debug log. Bedrock guardrails are *additive* (they ADD filtering when configured); there is no API parameter that weakens a model's intrinsic safety. |
| Ollama | No-op + debug log. Ollama serves local models with no API-level safety filtering layer. |

### The asymmetry is intentional

Today, only Google ships an API-level safety-disable toggle. Patterpunk keeps the Chat-level abstraction anyway so that:

1. Consumer code is provider-portable: switching from Google to Anthropic doesn't require removing the `disable_safety_filters` flag — it just becomes a no-op.
2. If other providers later expose similar controls, the abstraction is already in place — no consumer migration needed.
3. The intent ("treat this conversation as a corpus where safety filters are not appropriate") is captured at the call site even when no provider currently honors it for non-Google calls.

The debug logs make the asymmetry visible: enable `patterpunk` logger at DEBUG to see which providers honored the flag and which silently no-op'd.

## Google-specific: `safety_settings` (per-category)

For fine-grained Google control beyond the all-or-nothing chat-level toggle, pass a list of `types.SafetySetting` to `GoogleModel`:

```python
from google.genai import types
from patterpunk.llm.models.google import GoogleModel

model = GoogleModel(
    model="gemini-2.5-pro",
    safety_settings=[
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ],
)
```

### Composition with `disable_safety_filters`

If you set both, **model-level `safety_settings` wins**. Rationale: a consumer who explicitly chose per-category settings on `GoogleModel` knows what they want; the chat-level flag is the coarse default. There is no merging — the model-level list replaces the chat-level "all OFF" list entirely.

```python
# disable_safety_filters=True is overridden by the explicit safety_settings:
chat = Chat(
    model=GoogleModel(
        model="gemini-2.5-pro",
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ],
    ),
    disable_safety_filters=True,  # ignored — model-level is more specific
)
```

### When to use which

- **Coarse "disable all safety"**: `Chat(disable_safety_filters=True)`. Quickest path; portable to non-Google providers (as a no-op).
- **Per-category Google control**: `GoogleModel(safety_settings=[...])`. Use when some categories should fire and others shouldn't (e.g., disable harassment filters but keep dangerous-content filters intact).

## Google-specific: `allow_empty_response`

Some Gemini calls — particularly on dense source material — return an empty candidate even with safety filters off. By default, `GoogleModel` raises `GoogleAPIError("No content found in Vertex AI response")` in that case. This forces try/except control flow at the call site, which obscures intent when empty responses are a *routine* outcome (e.g., batch classification where ~2.5% of inputs legitimately produce no signal).

Setting `allow_empty_response=True` opts into a sentinel `AssistantMessage("")` instead:

```python
model = GoogleModel(model="gemini-2.5-pro", allow_empty_response=True)
result = Chat(model=model).add_message(UserMessage(...)).complete().latest_message

if not result.content:
    # Empty response — finish_reason and _provider still populated:
    log.warning(
        "Empty response: finish=%s prompt_block=%s",
        result.finish_reason,
        result._provider.prompt_block_reason,
    )
    # Treat as a default outcome and continue the batch:
    return DefaultOutput()

return parse(result.content)
```

The diagnostic fields are populated even on empty responses, so you can still distinguish "model emitted nothing" (`finish_reason` is `STOP`) from "safety filter triggered" (`finish_reason` is `SAFETY`) from "max-tokens cut off" (`finish_reason` is `MAX_TOKENS`).

`allow_empty_response=False` (the default) preserves the original `raise GoogleAPIError(...)` behavior — existing consumers that catch this exception continue to work unchanged.

### When to enable

- You're processing a batch where empty results are tolerable and shouldn't abort the batch.
- The empty response carries diagnostic value (e.g., "this page was probably blank — emit a default classification").
- You'd otherwise wrap every call in try/except just for control flow.

### When to leave disabled

- Empty responses are unexpected and should escalate to an alert.
- You haven't quantified your empty-response rate yet — leave the default and let exceptions surface so you can measure first.

## Persistence

Both `finish_reason` and `_provider` round-trip through `serialize()` / `deserialize()`:

```python
from patterpunk.llm.messages.assistant import AssistantMessage

msg = chat.latest_message
data = msg.serialize()
# data["finish_reason"] == "safety"
# data["provider_data"] == {"raw_finish_reason": "PROHIBITED_CONTENT", "prompt_block_reason": None}

restored = AssistantMessage.deserialize(data)
assert restored.finish_reason == FinishReason.SAFETY
assert restored._provider.raw_finish_reason == "PROHIBITED_CONTENT"
```

Backwards-compatible: legacy payloads without these fields deserialize cleanly with `finish_reason=None` and an empty `ProviderData`.

A corrupt `finish_reason` value (one not in the enum) raises `ValueError` at deserialization — failing fast at the boundary rather than silently propagating a bad value.

See [SERIALIZATION.md](SERIALIZATION.md) for the full serialization contract.

## End-to-end example

A pattern from a legal e-discovery pipeline that exercises all four features together:

```python
from google.genai import types
from patterpunk.llm.chat import Chat
from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages import SystemMessage, UserMessage
from patterpunk.llm.models.google import GoogleModel

# Granular safety control: disable harassment + dangerous content categories,
# but keep CSAM filter intact (Google's safety filters block more than
# patterpunk's normalized FinishReason captures).
SAFETY_FOR_LEGAL_CORPUS = [
    types.SafetySetting(category=cat, threshold=types.HarmBlockThreshold.OFF)
    for cat in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    )
]

model = GoogleModel(
    model="gemini-2.5-pro",
    temperature=0.0,
    safety_settings=SAFETY_FOR_LEGAL_CORPUS,
    allow_empty_response=True,
)

result = (
    Chat(model=model)
    .add_message(SystemMessage(prompt))
    .add_message(UserMessage(document_text))
    .complete()
    .latest_message
)

if not result.content:
    metrics.increment(
        "empty_response",
        tags={
            "finish_reason": result.finish_reason or "none",
            "raw": result._provider.raw_finish_reason or "none",
            "prompt_block": result._provider.prompt_block_reason or "none",
        },
    )
    return DefaultClassification()

if result.finish_reason == FinishReason.MAX_TOKENS:
    log.warning("Document too long — output truncated")

return parse(result.content)
```

The four features compose: model-level fine control via `safety_settings`, opt-in to empty-response sentinels via `allow_empty_response`, normalized triage via `finish_reason`, and full provider detail via `_provider.raw_finish_reason`.
