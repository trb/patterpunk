# Streaming API

Stream LLM responses token-by-token for real-time display. The streaming API works uniformly across all providers (OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, Azure OpenAI).

## Quick Start

```python
import asyncio
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage

async def main():
    chat = Chat().add_message(UserMessage("Explain quantum entanglement briefly"))

    async with chat.complete_stream() as stream:
        async for content in stream.content:
            print(content, end="\r")  # Overwrite with accumulated text

    final_chat = await stream.chat
    print(f"\n\nFinal response: {final_chat.latest_message.content}")

asyncio.run(main())
```

The `complete_stream()` method returns an async context manager. Each iteration of `stream.content` yields the full accumulated response. After streaming completes, `await stream.chat` returns the final `Chat` object.

## Core Concepts

### The ChatStream Context Manager

`chat.complete_stream()` returns a `ChatStream` that manages the streaming lifecycle:

```python
async with chat.complete_stream() as stream:
    # Iterate over content
    async for content in stream.content:
        display(content)

# After the context exits, get the final chat
final_chat = await stream.chat
```

### Accumulated vs Delta Iterators

Four async iterators are available, offering different trade-offs:

| Iterator | Yields | Use Case |
|----------|--------|----------|
| `stream.content` | Full accumulated text | UI display (replace mode) |
| `stream.content_delta` | New chunks only | Manual accumulation, logging |
| `stream.thinking` | Full accumulated thinking | Reasoning display |
| `stream.thinking_delta` | New thinking chunks | Thinking logging |

**Accumulated iterators** (`content`, `thinking`) yield the complete text so far on each iteration. This is ideal for UIs that replace the displayed text:

```python
async for content in stream.content:
    terminal.clear_line()
    terminal.write(content)  # Always shows complete response
```

**Delta iterators** (`content_delta`, `thinking_delta`) yield only new text since the last iteration. This is useful for manual accumulation or streaming to external systems:

```python
accumulated = ""
async for delta in stream.content_delta:
    accumulated += delta
    log(f"Received chunk: {delta}")
```

### Stream Lifecycle

1. **Enter context** - `async with chat.complete_stream() as stream`
2. **Iterate thinking** (optional) - `async for thinking in stream.thinking`
3. **Iterate content** - `async for content in stream.content`
4. **Exit context** - Context manager cleanup
5. **Get final chat** - `await stream.chat`

## Content Streaming

### Accumulated Content

The most common pattern - each iteration yields the complete response so far:

```python
async with chat.complete_stream() as stream:
    async for content in stream.content:
        # content grows with each iteration:
        # "Hello"
        # "Hello, I"
        # "Hello, I can"
        # "Hello, I can help"
        # ...
        print(f"\r{content}", end="")
```

### Delta Content

For when you need individual chunks:

```python
async with chat.complete_stream() as stream:
    chunks = []
    async for delta in stream.content_delta:
        # delta is only the new text:
        # "Hello"
        # ", I"
        # " can"
        # " help"
        chunks.append(delta)

full_response = "".join(chunks)
```

## Thinking/Reasoning Streaming

When using models with reasoning capabilities (via `ThinkingConfig`), you can stream the model's thinking process before the final response.

### Enabling Thinking

Configure thinking on your model:

```python
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.models.anthropic import AnthropicModel

model = AnthropicModel(
    model="claude-sonnet-4-20250514",
    thinking_config=ThinkingConfig(token_budget=2000)
)

chat = Chat(model=model).add_message(
    UserMessage("What's 127 * 389?")
)
```

### Streaming Both Thinking and Content

```python
async with chat.complete_stream() as stream:
    # First, stream thinking
    print("Thinking:")
    async for thinking in stream.thinking:
        print(thinking)

    # Then, stream content
    print("\nResponse:")
    async for content in stream.content:
        print(content)

final_chat = await stream.chat
```

### Auto-Draining Behavior

If you only care about content and skip the thinking iterator, thinking is automatically drained in the background:

```python
async with chat.complete_stream() as stream:
    # Thinking is auto-drained internally
    async for content in stream.content:
        print(content)

# The final chat still contains the complete thinking
final_chat = await stream.chat
thinking = final_chat.latest_message.thinking_blocks
```

## Getting the Final Result

### Using await stream.chat

After iterating, get the complete `Chat` object with the final response:

```python
async with chat.complete_stream() as stream:
    async for content in stream.content:
        pass  # Process content

final_chat = await stream.chat
print(final_chat.latest_message.content)

# Continue the conversation
next_chat = final_chat.add_message(
    UserMessage("Tell me more")
).complete()
```

The `await stream.chat` behavior:
- Returns immediately if the stream has already completed
- Waits for completion if iteration is still in progress
- Raises `StreamIncompleteError` if the stream was cancelled early

### Handling Early Exit

If you break from iteration early, the stream is marked as cancelled:

```python
async with chat.complete_stream() as stream:
    count = 0
    async for content in stream.content:
        count += 1
        if count >= 5:
            break  # Early exit cancels the stream

# This raises StreamIncompleteError
try:
    final_chat = await stream.chat
except StreamIncompleteError:
    print("Stream was cancelled before completion")
```

## Tool Calling with Streaming

Tools work seamlessly with streaming. When the model calls a tool, the stream automatically executes it and continues.

### Automatic Tool Execution

Like `chat.complete()`, streaming automatically executes tool calls by default (both function tools and MCP tools):

```python
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny, 72F in {location}"

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

chat = Chat().with_tools([get_weather, calculate]).add_message(
    UserMessage("What's the weather in Paris and what's 25 * 4?")
)

async with chat.complete_stream() as stream:
    async for content in stream.content:
        print(content)  # Seamless response including tool results

final_chat = await stream.chat
# Messages include: ToolCallMessage, ToolResultMessage, AssistantMessage
```

The streaming flow:
1. Model streams initial response
2. Model calls tools → stream executes them automatically
3. Tool results sent back to model
4. Model continues streaming with tool results incorporated
5. Process repeats if model calls more tools

### Multiple Tool Rounds

The stream handles multiple rounds of tool calls transparently:

```python
# Model might:
# 1. Call get_weather("Paris") -> get result
# 2. Call calculate("25 * 4") -> get result
# 3. Generate final response incorporating both results

async with chat.complete_stream() as stream:
    async for content in stream.content:
        # You see one seamless stream of content
        print(content)
```

### Tool Error Handling

Regular exceptions in tools are sent to the model as error results:

```python
def risky_tool(action: str) -> str:
    """Perform a risky action."""
    if not validate(action):
        raise ValueError("Invalid action")
    return "Success"

# The model receives the error and can respond appropriately
async with chat.complete_stream() as stream:
    async for content in stream.content:
        # Model might apologize or ask for clarification
        print(content)
```

### Aborting Streams with ToolExecutionAbortError

For critical errors that should stop the stream immediately:

```python
from patterpunk.llm.streaming import ToolExecutionAbortError

def dangerous_tool(action: str) -> str:
    """Perform a potentially dangerous action."""
    if action == "delete_everything":
        raise ToolExecutionAbortError("Dangerous action blocked")
    return "Action completed"

try:
    async with chat.complete_stream() as stream:
        async for content in stream.content:
            print(content)
except ToolExecutionAbortError as e:
    print(f"Stream aborted: {e.message}")
```

Use `ToolExecutionAbortError` when:
- A critical error occurs that the model cannot recover from
- The user explicitly requests cancellation
- A security or safety condition is violated

## Error Handling

### Exception Types

```python
from patterpunk.llm.streaming import (
    StreamingError,
    StreamIncompleteError,
    ToolExecutionAbortError,
    StreamingNotSupported,
)
```

| Exception | When Raised | How to Handle |
|-----------|-------------|---------------|
| `StreamingError` | Timeout, connection error, invalid state | Retry or fall back to non-streaming |
| `StreamIncompleteError` | Accessing `stream.chat` after early exit | Use original chat or don't break early |
| `ToolExecutionAbortError` | Tool explicitly aborts the stream | Handle the abort condition |
| `StreamingNotSupported` | Model doesn't support streaming | Use `complete()` instead |

### Example Error Handling

```python
from patterpunk.llm.streaming import (
    StreamingError,
    StreamIncompleteError,
    ToolExecutionAbortError,
    StreamingNotSupported,
)

try:
    async with chat.complete_stream() as stream:
        async for content in stream.content:
            print(content)
    final_chat = await stream.chat

except StreamingNotSupported:
    # Fall back to non-streaming
    final_chat = chat.complete()

except StreamingError as e:
    # Handle streaming failure
    print(f"Streaming failed: {e.message}")

except ToolExecutionAbortError as e:
    # Handle tool abort
    print(f"Tool aborted: {e.message}")
```

## Provider-Specific Notes

### Thinking Support by Provider

| Provider | Thinking Support | Configuration |
|----------|-----------------|---------------|
| Anthropic | Full (Claude 4.5+) | `ThinkingConfig(token_budget=N)` |
| OpenAI | End-only (o3-mini) | `ThinkingConfig(effort="medium")` |
| AWS Bedrock | Full (Claude models) | `ThinkingConfig(token_budget=N)` |
| Google | Best-effort | `ThinkingConfig(token_budget=N, include_thoughts=True)` |
| Azure OpenAI | End-only | `ThinkingConfig(effort="medium")` |

### Interleaved Thinking (Anthropic)

Anthropic's Claude 4.5+ models support interleaved thinking with tool calling - the model can think before calling tools and again after receiving results:

```python
async with chat.complete_stream() as stream:
    # Pre-tool thinking
    async for thinking in stream.thinking:
        print(f"Pre-tool thinking: {thinking}")

    # Tool execution happens automatically

    # Post-tool thinking (if model thinks again)
    async for thinking in stream.thinking:
        print(f"Post-tool thinking: {thinking}")

    # Final response
    async for content in stream.content:
        print(content)
```

## Complete Examples

### Real-Time Terminal Display

```python
import asyncio
import sys
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage

async def stream_to_terminal():
    chat = Chat().add_message(
        UserMessage("Write a haiku about programming")
    )

    async with chat.complete_stream() as stream:
        async for content in stream.content:
            sys.stdout.write(f"\r{content}")
            sys.stdout.flush()

    print()  # Newline after streaming
    final_chat = await stream.chat
    return final_chat

asyncio.run(stream_to_terminal())
```

### Streaming with Tool Calling

```python
import asyncio
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage

def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol."""
    prices = {"AAPL": 178.50, "GOOGL": 141.25, "MSFT": 378.90}
    return f"${prices.get(symbol, 'Unknown')}"

async def stock_assistant():
    chat = Chat().with_tools([get_stock_price]).add_message(
        UserMessage("What are the prices of AAPL, GOOGL, and MSFT?")
    )

    async with chat.complete_stream() as stream:
        async for content in stream.content:
            print(f"\r{content}", end="")

    print()
    final_chat = await stream.chat

    # Show the conversation history
    for msg in final_chat.messages:
        print(f"{type(msg).__name__}: {getattr(msg, 'content', msg)}")

asyncio.run(stock_assistant())
```

### Reasoning Model with Streaming

```python
import asyncio
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.models.anthropic import AnthropicModel

async def reasoning_stream():
    model = AnthropicModel(
        model="claude-sonnet-4-20250514",
        thinking_config=ThinkingConfig(token_budget=4000)
    )

    chat = Chat(model=model).add_message(
        UserMessage("Solve: If 3x + 7 = 22, what is x?")
    )

    async with chat.complete_stream() as stream:
        print("=== Thinking ===")
        async for thinking in stream.thinking:
            print(thinking)

        print("\n=== Response ===")
        async for content in stream.content:
            print(content)

    final_chat = await stream.chat

asyncio.run(reasoning_stream())
```
