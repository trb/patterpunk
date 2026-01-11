# Message Serialization

Patterpunk provides built-in serialization for storing conversations in databases and resuming them later. All message types serialize to JSON-compatible dictionaries suitable for PostgreSQL JSONB columns or document stores.

## Quick Start

Serialize a conversation and restore it later:

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import (
    SystemMessage, UserMessage, message_from_dict
)

# Build a conversation
chat = Chat().add_message(
    SystemMessage("You are a helpful assistant")
).add_message(
    UserMessage("What's the capital of France?")
).complete()

# Serialize all messages for storage
serialized = [msg.serialize() for msg in chat.messages]

# Store in database (serialized is a list of dicts)
# db.execute("INSERT INTO conversations (data) VALUES (%s)", [json.dumps(serialized)])

# Later: restore the conversation
# row = db.fetchone("SELECT data FROM conversations WHERE id = %s", [conv_id])
# serialized = json.loads(row["data"])

restored_messages = [message_from_dict(data) for data in serialized]
restored_chat = Chat().add_messages(restored_messages)

# Continue the conversation
continued = restored_chat.add_message(
    UserMessage("And what about Germany?")
).complete()
```

Each message's `serialize()` method returns a self-contained dictionary. The `message_from_dict()` function reconstructs the appropriate message type from that dictionary.

## Database Storage

Serialized messages are plain dictionaries, compatible with most database storage patterns.

### PostgreSQL with JSONB

For PostgreSQL, store messages directly in JSONB columns:

```python
import json
import psycopg2
from psycopg2.extras import Json

# With psycopg2, wrap dicts in Json() for JSONB columns
conn = psycopg2.connect(...)
cursor = conn.cursor()

# Store conversation
serialized = [msg.serialize() for msg in chat.messages]
cursor.execute(
    "INSERT INTO conversations (id, messages) VALUES (%s, %s)",
    [conversation_id, Json(serialized)]
)

# Retrieve conversation
cursor.execute("SELECT messages FROM conversations WHERE id = %s", [conversation_id])
row = cursor.fetchone()
messages = [message_from_dict(data) for data in row[0]]
```

With psycopg3, dictionaries convert automatically without the `Json()` wrapper. SQLAlchemy with JSONB columns also accepts dictionaries directly.

### Document Databases

For MongoDB, Redis, or other document stores, serialize to JSON strings:

```python
import json

# Serialize
data = json.dumps([msg.serialize() for msg in chat.messages])

# Restore
messages = [message_from_dict(d) for d in json.loads(data)]
```

## Supported Message Types

All message types implement `serialize()` and `from_dict()`:

| Message Type | Serialized Fields |
|-------------|-------------------|
| `SystemMessage` | content |
| `UserMessage` | content, structured_output, allow_tool_calls |
| `AssistantMessage` | content, thinking_blocks, structured_output |
| `ToolCallMessage` | tool_calls, thinking_blocks |
| `ToolResultMessage` | content, call_id, function_name, is_error |

### Message with Simple Content

```python
from patterpunk.llm.messages import SystemMessage, message_from_dict

msg = SystemMessage("You are a helpful assistant")
data = msg.serialize()
# {'type': 'system', 'content': {'type': 'string', 'value': 'You are a helpful assistant'}}

restored = message_from_dict(data)
assert restored.content == msg.content
```

### Message with Multimodal Content

Messages containing images or other media serialize to base64 for self-contained storage:

```python
from patterpunk.llm.messages import UserMessage, message_from_dict
from patterpunk.llm.chunks import TextChunk, MultimodalChunk

msg = UserMessage([
    TextChunk("Describe this image:"),
    MultimodalChunk.from_file("photo.jpg")
])

data = msg.serialize()
# Image is converted to base64 in the serialized data

restored = message_from_dict(data)
# Image data is preserved and can be sent to the LLM
```

URL-based images are downloaded during serialization to ensure the stored data is self-contained and doesn't depend on external URLs remaining available.

### Message with Thinking Blocks

Extended thinking (reasoning) content serializes alongside the message:

```python
from patterpunk.llm.messages import AssistantMessage, message_from_dict

# After a completion with thinking enabled
thinking_msg = chat.latest_message  # AssistantMessage with thinking_blocks

data = thinking_msg.serialize()
# {'type': 'assistant', 'content': {...}, 'thinking_blocks': [...]}

restored = message_from_dict(data)
assert restored.thinking_blocks == thinking_msg.thinking_blocks
```

## Structured Output Handling

When messages include structured output (Pydantic models), serialization stores both the JSON schema and a class reference for reconstruction.

### Basic Structured Output

```python
from pydantic import BaseModel
from patterpunk.llm.messages import UserMessage, message_from_dict

class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str

msg = UserMessage(
    "What's the weather in Tokyo?",
    structured_output=WeatherReport
)

data = msg.serialize()
# Includes schema and class reference:
# {'type': 'user', 'content': {...}, 'structured_output': {
#     'schema': {...},
#     'class_ref': 'myapp.models.WeatherReport'
# }}
```

### Deserialization with Original Class Available

When the original Pydantic model class is importable, deserialization restores it:

```python
# If myapp.models.WeatherReport is importable
restored = message_from_dict(data)
assert restored.structured_output is WeatherReport  # Original class
```

### Fallback with DynamicStructuredOutput

When the original class cannot be imported (different environment, refactored code), deserialization falls back to `DynamicStructuredOutput`:

```python
from patterpunk.llm.messages import DynamicStructuredOutput

# If original class is not importable
restored = message_from_dict(data)
assert isinstance(restored.structured_output, DynamicStructuredOutput)

# DynamicStructuredOutput satisfies patterpunk's interface
schema = restored.structured_output.model_json_schema()  # Returns stored schema
parsed = restored.structured_output.model_validate_json('{"location": "Tokyo"}')  # Returns dict
```

`DynamicStructuredOutput` parses JSON without validation. For full schema validation, ensure the original Pydantic model is importable in the environment where deserialization occurs.

## Tool Call Serialization

Tool calls serialize with their function names, arguments, and IDs:

```python
from patterpunk.llm.messages import ToolCallMessage, message_from_dict

# After a completion with tool calls
tool_msg = chat.latest_message  # ToolCallMessage

data = tool_msg.serialize()
# {'type': 'tool_call', 'tool_calls': [
#     {'id': 'call_123', 'type': 'function', 'function': {
#         'name': 'get_weather', 'arguments': '{"city": "Paris"}'
#     }}
# ]}

restored = message_from_dict(data)
for tc in restored.tool_calls:
    print(f"{tc.function_name}: {tc.arguments}")
```

Tool results also round-trip correctly:

```python
from patterpunk.llm.messages import ToolResultMessage, message_from_dict

result = ToolResultMessage(
    content="72°F, sunny",
    call_id="call_123",
    function_name="get_weather"
)

data = result.serialize()
restored = message_from_dict(data)
assert restored.call_id == "call_123"
```

## Serialization Format Reference

### Content Types

Content serializes with a discriminator field:

```python
# String content
{"type": "string", "value": "Hello, world"}

# Chunk list content
{"type": "chunks", "chunks": [
    {"type": "text", "content": "Describe this:"},
    {"type": "multimodal", "media_type": "image/png", "data": "base64...", "filename": "image.png"},
    {"type": "cache", "content": "...", "cacheable": True, "ttl_seconds": 300}
]}
```

### Chunk Types

| Chunk Type | Fields |
|-----------|--------|
| TextChunk | `type`, `content` |
| MultimodalChunk | `type`, `media_type`, `data` (base64), `filename` |
| CacheChunk | `type`, `content`, `cacheable`, `ttl_seconds` (optional) |

### Message Types

Each message type includes a `type` discriminator:

```python
# SystemMessage
{"type": "system", "content": {...}}

# UserMessage
{"type": "user", "content": {...}, "allow_tool_calls": True, "structured_output": {...}}

# AssistantMessage
{"type": "assistant", "content": {...}, "thinking_blocks": [...], "structured_output": {...}}

# ToolCallMessage
{"type": "tool_call", "tool_calls": [...], "thinking_blocks": [...]}

# ToolResultMessage
{"type": "tool_result", "content": "...", "call_id": "...", "function_name": "...", "is_error": False}
```

## API Reference

### message_from_dict

Deserialize any message type from a dictionary.

```python
from patterpunk.llm.messages import message_from_dict

message = message_from_dict(data)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | `dict` | Yes | Serialized message with `type` field |

**Returns:** The appropriate message subclass instance (`SystemMessage`, `UserMessage`, etc.)

**Raises:** `ValueError` if the message type is unknown.

### serialize_message

Convenience function to serialize any message.

```python
from patterpunk.llm.messages import serialize_message

data = serialize_message(message)
```

Equivalent to calling `message.serialize()` directly.

### DynamicStructuredOutput

Fallback class for structured_output when the original Pydantic model cannot be imported.

```python
from patterpunk.llm.messages import DynamicStructuredOutput

dso = DynamicStructuredOutput(schema)
dso.model_json_schema()      # Returns the stored JSON schema
dso.model_validate_json(s)   # Parses JSON string, returns dict
dso.model_validate(d)        # Returns dict as-is
```

This class satisfies patterpunk's structured_output interface but does not perform schema validation. Use it when you need to restore conversations but don't have access to the original Pydantic model classes.

## Complete Example: Chat Persistence

A full example showing conversation storage and retrieval:

```python
import json
from datetime import datetime, timezone
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import (
    SystemMessage, UserMessage, message_from_dict
)

class ConversationStore:
    """Simple conversation persistence layer."""

    def __init__(self, db_connection):
        self.db = db_connection

    def save_conversation(self, conversation_id: str, chat: Chat) -> None:
        """Persist a chat's messages to the database."""
        serialized = [msg.serialize() for msg in chat.messages]

        self.db.execute(
            """INSERT INTO conversations (id, messages, updated_at)
               VALUES (%s, %s, %s)
               ON CONFLICT (id) DO UPDATE
               SET messages = EXCLUDED.messages, updated_at = EXCLUDED.updated_at""",
            [conversation_id, json.dumps(serialized), datetime.now(timezone.utc)]
        )

    def load_conversation(self, conversation_id: str) -> Chat:
        """Restore a chat from stored messages."""
        row = self.db.fetchone(
            "SELECT messages FROM conversations WHERE id = %s",
            [conversation_id]
        )

        if not row:
            return Chat()  # New conversation

        messages = [message_from_dict(d) for d in json.loads(row[0])]
        return Chat().add_messages(messages)

    def continue_conversation(self, conversation_id: str, user_input: str) -> Chat:
        """Load a conversation, add a message, complete, and save."""
        chat = self.load_conversation(conversation_id)

        updated_chat = chat.add_message(
            UserMessage(user_input)
        ).complete()

        self.save_conversation(conversation_id, updated_chat)
        return updated_chat


# Usage
store = ConversationStore(db)

# Start a new conversation
chat = Chat().add_message(
    SystemMessage("You are a helpful travel advisor")
).add_message(
    UserMessage("I'm planning a trip to Japan")
).complete()

store.save_conversation("trip-planning-123", chat)

# Later: continue the conversation
continued = store.continue_conversation(
    "trip-planning-123",
    "What's the best time to visit Kyoto?"
)

print(continued.latest_message.content)
```

This pattern enables multi-turn conversations across sessions, API requests, or even different services that share access to the same database.
