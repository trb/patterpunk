# Prompt Caching

Patterpunk provides unified prompt caching across providers through `CacheChunk` with provider-specific optimizations. Each provider implements caching differently - understanding these differences is critical for effective usage.

## CacheChunk API

```python
from datetime import timedelta
from patterpunk.llm.chunks import CacheChunk
from patterpunk.llm.messages import UserMessage

# Basic usage with UserMessage
message = UserMessage([
    CacheChunk(content="Large context...", cacheable=True),
    CacheChunk(content="Dynamic query", cacheable=False)
])

# With TTL (provider-specific interpretation)
message = UserMessage([
    CacheChunk(
        content="System instructions...",
        cacheable=True,
        ttl=timedelta(hours=2)
    ),
    CacheChunk(content="Current request", cacheable=False)
])
```

Key behavior: Only chunks with `cacheable=True` are cached. Non-cacheable chunks are processed normally but bypass caching entirely.

## Provider-Specific Implementations

### General

Not all aspects of CacheChunk are supported by all providers. E.g. how providers handle "ttl" varies considerably. Review specifics on prompt caching for the provider you're using if you have any questions.

### OpenAI: Prefix-Based Caching

OpenAI requires cacheable content to be at the **beginning** of messages (prefix pattern). Non-prefix cacheable content triggers warnings and may be ineffective.

```python
from patterpunk.llm.chunks import CacheChunk, TextChunk

# Correct: Cacheable content as prefix
message = UserMessage([
    CacheChunk("Large codebase context...", cacheable=True),
    CacheChunk("Current task: ", cacheable=False),
    TextChunk("analyze this function")
])

# Problematic: Cacheable content not at prefix
message = UserMessage([
    CacheChunk("Instructions: ", cacheable=False),
    CacheChunk("Large context...", cacheable=True),  # Warning logged
    CacheChunk("Query: what is X?", cacheable=False)
])
```

OpenAI implementation validates prefix patterns and logs warnings:
```
[OPENAI_CACHE] Non-prefix cacheable content detected in user message - caching may be ineffective
```

### Anthropic

Needs no specific considerations

### Google

Automatically caches prompts but can make use of explicit caching for chunks >32k tokens (patterpunk automatically handles this)

### Bedrock

Needs no specific considerations


## Content Integration Patterns

### Mixed Content Strategies

```python
from patterpunk.llm.chunks import TextChunk, CacheChunk

# Strategic caching pattern
message = UserMessage([
    CacheChunk("System prompt and large context", cacheable=True),
    TextChunk("Dynamic user input: "),
    CacheChunk(user_query, cacheable=False)
])

# System message caching for persistent context
system = SystemMessage([
    CacheChunk("You are an expert with access to:", cacheable=False),
    CacheChunk(large_knowledge_base, cacheable=True, ttl=timedelta(hours=4)),
    CacheChunk("Current session settings: " + settings, cacheable=False)
])
```

### Multi-Turn Conversation Optimization

```python
from patterpunk.llm.chunks import CacheChunk, TextChunk

# Cache system context across conversation turns
chat = Chat().add_message(SystemMessage([
    CacheChunk(persistent_instructions, cacheable=True)
]))

# Subsequent messages leverage cached system context
for user_input in conversation:
    response = chat.add_message(UserMessage([
        CacheChunk(current_context, cacheable=False),  # Changes each turn
        TextChunk(user_input)
    ])).complete()

    chat = response
```

### Multimodal Content with Caching

```python
from patterpunk.llm.chunks import MultimodalChunk, CacheChunk, TextChunk

# Cache text context, not multimodal content
message = UserMessage([
    CacheChunk("Analysis framework and instructions", cacheable=True),
    TextChunk("Please analyze this image: "),
    MultimodalChunk.from_file("chart.png"),  # Not cached
    CacheChunk("Use the following criteria: " + criteria, cacheable=True)
])
```

## Advanced Caching Strategies

### Content Processing Pipeline

All content goes through automatic processing:

```python
from patterpunk.llm.chunks import TextChunk, CacheChunk

# TextChunk automatically converts to non-cacheable CacheChunk
message = UserMessage([
    TextChunk("Instructions"),  # → CacheChunk(cacheable=False)
    CacheChunk("Context", cacheable=True)
])

# Access processed cache chunks
cache_chunks = message.get_cache_chunks()
has_cacheable = message.has_cacheable_content()
```

### Cost Optimization Patterns

```python
from patterpunk.llm.chunks import CacheChunk

# High-value caching: Large, reused content
system_context = CacheChunk(
    content=load_large_documentation(),
    cacheable=True,
    ttl=timedelta(hours=8)  # Long TTL for stable content
)

# Low-value caching: Avoid for small or dynamic content
user_query = CacheChunk(
    content="What is 2+2?",  # Too small to benefit
    cacheable=False
)

# Session-based caching
conversation_history = CacheChunk(
    content=format_history(messages),
    cacheable=True,
    ttl=timedelta(minutes=30)  # Shorter TTL for dynamic content
)
```

### Cache Effectiveness Analysis

```python
from patterpunk.llm.chunks import CacheChunk

# Analyze cache potential
def analyze_cache_potential(message):
    chunks = message.get_cache_chunks()

    for i, chunk in enumerate(chunks):
        if chunk.cacheable:
            size = len(chunk.content)
            print(f"Chunk {i}: {size} chars, cacheable")

            # Provider-specific considerations
            if provider == "google" and size < 32000:
                print(f"  Warning: Below Google's 32KB threshold")
            elif provider == "openai" and i > 0:
                print(f"  Warning: Non-prefix cacheable content")

# Cache hit optimization
def optimize_cache_layout(chunks):
    # Move cacheable content to beginning for OpenAI
    cacheable = [c for c in chunks if c.cacheable]
    non_cacheable = [c for c in chunks if not c.cacheable]
    return cacheable + non_cacheable
```

## Performance Considerations

### When to Cache

**Good candidates:**
- System prompts and instructions (static, reused)
- Large context documents (>1KB, stable)
- Multi-turn conversation context
- Template-based content with stable portions

**Poor candidates:**
- Small text fragments (<100 chars)
- Highly dynamic content that changes frequently  
- Content used only once
- User queries and variable input

### Provider Cost Implications

```python
from patterpunk.llm.chunks import CacheChunk

# Anthropic: Cache control reduces token costs
anthropic_message = UserMessage([
    CacheChunk(large_context, cacheable=True),  # Cached tokens cost less
    CacheChunk(user_query, cacheable=False)     # Normal token pricing
])

# Google: Cache object creation has overhead
google_message = UserMessage([
    CacheChunk(content * 1000, cacheable=True)  # Explicit caching only over >32k tokens
])

# OpenAI: Prefix caching optimization
openai_message = UserMessage([
    CacheChunk(system_context, cacheable=True),  # Must be first
    CacheChunk(dynamic_content, cacheable=False)
])
```

## Debugging and Troubleshooting

### Common Patterns and Issues

```python
from patterpunk.llm.chunks import CacheChunk

# Debug cache effectiveness
def debug_caching(chat_response):
    for message in chat_response.messages:
        if message.has_cacheable_content():
            chunks = message.get_cache_chunks()
            cacheable_count = sum(1 for c in chunks if c.cacheable)
            print(f"{message.role}: {cacheable_count} cacheable chunks")

# Monitor cache warnings (OpenAI)
import logging
logging.getLogger('patterpunk').setLevel(logging.WARNING)
# Watch for: "[OPENAI_CACHE] Non-prefix cacheable content detected"
```

### Provider-Specific Debugging

```python
from patterpunk.llm.chunks import CacheChunk

# OpenAI: Validate prefix pattern
def validate_openai_cache_pattern(chunks):
    last_cacheable = -1
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, CacheChunk) and chunk.cacheable:
            last_cacheable = i

    # Check if all content before last cacheable is also cacheable
    for i in range(last_cacheable):
        if not (isinstance(chunks[i], CacheChunk) and chunks[i].cacheable):
            return False, f"Non-cacheable content at position {i}"
    return True, "Valid prefix pattern"

# Google: Check size thresholds
def check_google_cache_eligibility(chunks):
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, CacheChunk) and chunk.cacheable:
            if len(chunk.content) < 32000:
                print(f"Chunk {i}: {len(chunk.content)} chars - below 32KB threshold")
```

## Integration with Other Systems

### Agent Workflows

```python
from patterpunk.llm.agent import Agent
from patterpunk.llm.chunks import CacheChunk

class CachedAgent(Agent[str, str]):
    def prepare_chat(self):
        return super().prepare_chat().add_message(SystemMessage([
            CacheChunk(self.system_knowledge, cacheable=True, ttl=timedelta(hours=2)),
            CacheChunk("Current task context: ", cacheable=False)
        ]))
```

### Structured Output with Caching

```python
from pydantic import BaseModel
from patterpunk.llm.chunks import CacheChunk, TextChunk

class Analysis(BaseModel):
    findings: list[str]
    confidence: float

# Cache analysis framework, not the specific request
message = UserMessage([
    CacheChunk("Analysis framework: " + framework_prompt, cacheable=True),
    TextChunk("Analyze this specific case: " + case_data)
], structured_output=Analysis)

response = chat.add_message(message).complete()
analysis = response.parsed_output
```

Caching reduces token costs and improves response times when the same context is reused across multiple requests, but effectiveness varies significantly by provider implementation.
