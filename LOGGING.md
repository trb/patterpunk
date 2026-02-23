# Logging

Patterpunk provides detailed logging for debugging LLM interactions. This is essential during development to inspect exactly what's being sent to and received from LLM providers.

## Quick Start

To see all LLM interactions at DEBUG level:

```python
import logging

# Configure patterpunk loggers
logging.basicConfig(
    level=logging.WARNING,  # Default level for all loggers
    format="%(levelname)s %(name)s: %(message)s"
)
logging.getLogger("patterpunk").setLevel(logging.DEBUG)
```

This shows the full conversation history, model parameters, and responses without noise from HTTP libraries.

## Patterpunk Loggers

Patterpunk uses two loggers:

| Logger | Purpose |
|--------|---------|
| `patterpunk` | General operational events (request start, warnings, errors) |
| `patterpunk.llm` | Detailed LLM interactions (messages, parameters, responses) |

### Log Levels

**DEBUG level** - Full conversation context with clear structure:

```
────────────────────────────────────────────────────────────
 Google Vertex AI Request
────────────────────────────────────────────────────────────
[system]
    You are a weather assistant.
[user]
    What is the weather in Paris?
[params] model=gemini-2.5-flash, temp=0.5, tools=[get_weather]
[response] Tool calls: get_weather({"location": "Paris"})
────────────────────────────────────────────────────────────
 Google Vertex AI Request
────────────────────────────────────────────────────────────
[system]
    You are a weather assistant.
[user]
    What is the weather in Paris?
[tool_call]
    get_weather({"location": "Paris"})
[tool_result]
    get_weather → 22 degrees celsius
[params] model=gemini-2.5-flash, temp=0.5, tools=[get_weather]
[response]
    The weather in Paris is 22 degrees celsius.
```

Each API request is clearly separated with a header. Messages show their full content indented under the role label.

### What Gets Logged

| Event | Level | Logger | Content |
|-------|-------|--------|---------|
| Request header | DEBUG | `patterpunk.llm` | Visual separator and "Google Vertex AI Request" |
| Message history | DEBUG | `patterpunk.llm` | `[role]` followed by full content on indented lines |
| Model parameters | DEBUG | `patterpunk.llm` | `[params] model=..., temp=..., tools=[...]` |
| Text response | DEBUG | `patterpunk.llm` | `[response]` followed by full content on indented lines |
| Tool calls | DEBUG | `patterpunk.llm` | `[response] Tool calls: func_name({"arg": "value"})` |
| Thinking blocks | DEBUG | `patterpunk.llm` | `[thinking] N block(s)` |
| Usage stats | DEBUG | `patterpunk.llm` | `[usage] input=N, output=N, thinking=N` |
| Rate limits | WARNING | `patterpunk` | Retry attempts and wait times |
| API errors | ERROR | `patterpunk` | Error details |

### Log Modes

Control how much of the conversation history is logged on each request using the `PP_LLM_LOG_MODE` environment variable:

| Mode | Description |
|------|-------------|
| `full` (default) | Log all messages on every request |
| `incremental` | Log only new messages since the last request |

**Full mode** is useful for investigating specific interactions—you see the complete conversation context sent to the API on every call.

**Incremental mode** is useful when working with long system prompts or multi-turn conversations. Instead of repeating the same 4K token system message on every request, you only see what's new:

```
────────────────────────────────────────────────────────────
 Google Vertex AI Request
────────────────────────────────────────────────────────────
[system]
    You are a weather assistant with expertise in...
    (long system prompt)
[user]
    What is the weather in Paris?
[params] model=gemini-2.5-flash, temp=0.5, tools=[get_weather]
[response] Tool calls: get_weather({"location": "Paris"})
────────────────────────────────────────────────────────────
 Google Vertex AI Request
────────────────────────────────────────────────────────────
  ... 2 earlier messages ...
[tool_call]
    get_weather({"location": "Paris"})
[tool_result]
    get_weather → 22 degrees celsius
[params] model=gemini-2.5-flash, temp=0.5, tools=[get_weather]
[response]
    The weather in Paris is 22 degrees celsius.
```

Set the mode via environment variable:

```bash
export PP_LLM_LOG_MODE=incremental  # or 'full'
```

Or in Python before making requests:

```python
import os
os.environ["PP_LLM_LOG_MODE"] = "incremental"
```

The mode can be changed at runtime—it's checked on each request.

## Silencing Noisy Libraries

When you enable DEBUG logging globally, HTTP client libraries produce verbose connection-level output that obscures the LLM interactions you care about. Here's what each library logs and when you might want to see it:

### urllib3

**What it is:** Low-level HTTP client library used by many Python packages.

**What it logs at DEBUG:**
```
DEBUG urllib3.connectionpool: Starting new HTTPS connection (1): oauth2.googleapis.com:443
DEBUG urllib3.connectionpool: https://oauth2.googleapis.com:443 "POST /token HTTP/1.1" 200 None
```

**When useful:** Debugging OAuth token refresh issues or connection problems.

**To silence:**
```python
logging.getLogger("urllib3").setLevel(logging.WARNING)
```

### httpcore

**What it is:** Low-level HTTP transport library that httpx builds on. Handles raw TCP connections, TLS handshakes, and HTTP/1.1 or HTTP/2 protocol details.

**What it logs at DEBUG:**
```
DEBUG httpcore.connection: connect_tcp.started host='us-central1-aiplatform.googleapis.com' port=443
DEBUG httpcore.connection: start_tls.started ssl_context=<ssl.SSLContext object at 0x...>
DEBUG httpcore.http11: send_request_headers.started request=<Request [b'POST']>
DEBUG httpcore.http11: receive_response_headers.complete return_value=(b'HTTP/1.1', 200, b'OK', ...)
```

**When useful:** Debugging TLS/SSL issues, HTTP/2 upgrade problems, or connection timeouts at the transport layer.

**To silence:**
```python
logging.getLogger("httpcore").setLevel(logging.WARNING)
```

### httpx

**What it is:** Modern HTTP client library used by the Google GenAI SDK. Provides the high-level request/response interface.

**What it logs at INFO:**
```
INFO httpx: HTTP Request: POST https://us-central1-aiplatform.googleapis.com/v1beta1/.../gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
```

**When useful:** Seeing which API endpoints are being called and their HTTP status codes without the low-level connection details.

**To silence:**
```python
logging.getLogger("httpx").setLevel(logging.WARNING)
```

### google_genai

**What it is:** Google's GenAI SDK internal logger.

**What it logs at INFO:**
```
INFO google_genai._api_client: The user provided Google Cloud credentials will take precedence...
INFO google_genai.models: AFC is enabled with max remote calls: 10.
INFO google_genai.models: AFC remote call 1 is done.
```

**When useful:** Understanding Google SDK behavior like automatic function calling (AFC) status or credential handling.

**To silence:**
```python
logging.getLogger("google_genai").setLevel(logging.WARNING)
```

## Recommended Configurations

### Development - LLM Debugging

See exactly what's sent to and received from the LLM:

```python
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s"
)

# Enable patterpunk DEBUG logging
logging.getLogger("patterpunk").setLevel(logging.DEBUG)
```

### Development - Full HTTP Visibility

Debug connection issues or see exact API endpoints:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s"
)

# Keep patterpunk at DEBUG, silence low-level transport
logging.getLogger("httpcore").setLevel(logging.WARNING)
```

### Production

Only see warnings and errors:

```python
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
```

### Production with Request Tracking

For production monitoring, the WARNING level captures rate limits and errors. Full request logging is only available at DEBUG level to avoid accidental exposure of sensitive data in production logs.

## Complete Example

```python
import logging

# Configure logging before any patterpunk imports
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s"
)

# Enable detailed LLM logging
logging.getLogger("patterpunk").setLevel(logging.DEBUG)

# Silence noisy HTTP libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Now import and use patterpunk
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.google import GoogleModel

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny, 22°C in {location}"

model = GoogleModel(model="gemini-2.5-flash", location="us-central1")

chat = (
    Chat(model=model)
    .add_message(SystemMessage("You are a weather assistant."))
    .add_message(UserMessage("What's the weather in Paris?"))
    .with_tools([get_weather])
    .complete()
)
```

**Output:**
```
────────────────────────────────────────────────────────────
 Google Vertex AI Request
────────────────────────────────────────────────────────────
[system]
    You are a weather assistant.
[user]
    What's the weather in Paris?
[params] model=gemini-2.5-flash, temp=1.0, tools=[get_weather]
[response] Tool calls: get_weather({"location": "Paris"})
────────────────────────────────────────────────────────────
 Google Vertex AI Request
────────────────────────────────────────────────────────────
[system]
    You are a weather assistant.
[user]
    What's the weather in Paris?
[tool_call]
    get_weather({"location": "Paris"})
[tool_result]
    get_weather → Sunny, 22°C in Paris
[params] model=gemini-2.5-flash, temp=1.0, tools=[get_weather]
[response]
    The weather in Paris is sunny with a temperature of 22°C.
```

## Provider Support

| Provider | Logging Support |
|----------|-----------------|
| Google Vertex AI | Full support (messages, parameters, responses, streaming) |
| OpenAI | Full support |
| Azure OpenAI | Full support |
| AWS Bedrock | Full support |
| Anthropic | Partial (warnings and errors only) |
| Ollama | Partial (warnings and errors only) |

Providers with "partial" support log operational events but not detailed message history. This is a known inconsistency being addressed.
