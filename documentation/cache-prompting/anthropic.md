# Anthropic Claude Prompt Caching

## Overview

Anthropic introduced prompt caching for Claude models in August 2024, with general availability announced on December 17, 2024. Claude's prompt caching implementation offers more granular control than other providers, featuring explicit cache control markers, extended cache durations, and integration with advanced features like token-efficient tool use. The system can reduce costs by up to 90% and latency by up to 85% for long prompts.

## Implementation Architecture

### How Anthropic Implemented Caching

Anthropic's approach differs from automatic caching systems by providing explicit control:
- **Cache Breakpoints**: Developers explicitly mark content for caching using `cache_control` markers
- **Prefix-Based Caching**: Content before cache breakpoints becomes cached prompt prefixes
- **Automatic Optimization**: Updated system automatically reads from longest previously cached prefix
- **Hierarchical Caching**: Support for multiple cache points within a single prompt

### Cache Lifecycle and Management

- **Default TTL**: 5 minutes (ephemeral cache type)
- **Extended TTL**: Up to 1 hour (beta feature)
- **Sliding Window**: TTL resets with each cache hit
- **Account Isolation**: Caches are account-specific for security

## Supported Models (2024-2025)

### Current Model Support
- **Claude 3.5 Sonnet** (all versions including 3.7)
- **Claude 3.5 Haiku**
- **Claude 3 Opus**
- **Claude 3 Haiku**
- **Claude Opus 4** and **Claude Sonnet 4** (latest)

### Token Requirements
- **Most models**: Minimum 1,024 tokens for caching
- **Claude Haiku models**: Minimum 2,048 tokens for caching

## SDK Usage and Implementation

### Basic Python SDK Setup

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# Basic prompt caching with cache control
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is a large document to analyze:",
                },
                {
                    "type": "text", 
                    "text": document_content,
                    "cache_control": {"type": "ephemeral"}  # Mark for caching
                },
                {
                    "type": "text",
                    "text": "Please analyze the key themes in this document."
                }
            ]
        }
    ]
)
```

### System Message Caching

```python
# Caching system instructions
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    system=[
        {
            "type": "text",
            "text": "You are an expert researcher analyzing documents...",
        },
        {
            "type": "text",
            "text": detailed_instructions,
            "cache_control": {"type": "ephemeral"}  # Cache detailed instructions
        }
    ],
    messages=[
        {
            "role": "user", 
            "content": "Analyze this document: " + document
        }
    ]
)
```

### Beta Features Configuration

#### Using Beta Headers

```python
# For generally available features (no longer needs beta header)
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    # No beta header needed for basic prompt caching
    messages=messages
)

# For extended cache TTL (beta feature)
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    extra_headers={
        "anthropic-beta": "extended-cache-ttl-2025-04-11"
    },
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": content_to_cache,
                    "cache_control": {
                        "type": "ephemeral",
                        "ttl": "1h"  # Extended 1-hour cache
                    }
                }
            ]
        }
    ]
)

# Token-efficient tool use
response = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    extra_headers={
        "anthropic-beta": "token-efficient-tools-2025-02-19"
    },
    messages=messages,
    tools=tools
)
```

## Advanced Features

### Extended Cache Duration

```python
# 5-minute cache (default)
cache_control = {"type": "ephemeral"}

# 1-hour cache (beta - requires special header)
cache_control = {
    "type": "ephemeral",
    "ttl": "1h"  # Options: "5m" or "1h"
}
```

### Token-Efficient Tool Use

Claude 3.7 Sonnet supports token-efficient tool calling that reduces output token consumption by up to 70%:

```python
import anthropic

client = anthropic.Anthropic()

# Enable token-efficient tools
response = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    extra_headers={
        "anthropic-beta": "token-efficient-tools-2025-02-19"
    },
    max_tokens=1000,
    tools=[
        {
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": large_context_document,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": "What's the weather like in locations mentioned in this document?"
                }
            ]
        }
    ]
)
```

### Cache-Aware Rate Limits

For Claude 3.7 Sonnet, prompt cache read tokens no longer count against Input Tokens Per Minute (ITPM) limits, providing additional performance benefits.

## Cost Structure and Pricing

### Cache Pricing Model

1. **Cache Writes**: 25% markup over base input token price (5-minute TTL)
2. **Cache Reads**: 90% discount - only 10% of base input token price
3. **Extended Cache**: Higher write costs for 1-hour TTL

### Cost Calculation Example

```python
def calculate_cache_costs(base_tokens, cached_tokens, cache_writes):
    base_cost_per_token = 0.003  # Example rate per 1K tokens
    
    # First request (cache write)
    write_cost = base_tokens * base_cost_per_token * 1.25  # 25% markup
    
    # Subsequent requests (cache reads)
    read_cost = cached_tokens * base_cost_per_token * 0.10  # 90% discount
    
    return {
        'write_cost': write_cost,
        'read_cost_per_request': read_cost,
        'savings_per_read': cached_tokens * base_cost_per_token * 0.90
    }
```

## Best Practices and Optimization

### Strategic Cache Placement

1. **Cache breakpoint positioning**:
```python
# Good: Cache large, stable content
{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "System instructions and context...",
            "cache_control": {"type": "ephemeral"}  # Cache stable content
        },
        {
            "type": "text",
            "text": dynamic_user_query  # Dynamic content after cache
        }
    ]
}

# Bad: Cache small or frequently changing content
{
    "role": "user", 
    "content": [
        {
            "type": "text",
            "text": f"Current time: {datetime.now()}",  # Changes constantly
            "cache_control": {"type": "ephemeral"}
        }
    ]
}
```

2. **Hierarchical caching strategy**:
```python
# Multiple cache points for different content types
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Core system instructions...",
                "cache_control": {"type": "ephemeral"}  # Most stable
            },
            {
                "type": "text", 
                "text": document_content,
                "cache_control": {"type": "ephemeral"}  # Document-specific
            },
            {
                "type": "text",
                "text": "Please analyze with focus on: " + current_focus
            }
        ]
    }
]
```

### Performance Optimization

3. **Maintain cache consistency**:
   - Keep cached content identical across requests
   - Use template systems to ensure consistent formatting
   - Avoid embedding timestamps or random data in cached sections

4. **Monitor cache effectiveness**:
```python
def analyze_anthropic_cache_usage(response):
    usage = response.usage
    
    cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
    cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)
    
    total_input = usage.input_tokens
    cache_hit_rate = cache_read_tokens / total_input if total_input > 0 else 0
    
    return {
        'cache_hit_rate': cache_hit_rate,
        'cache_creation_tokens': cache_creation_tokens,
        'cache_read_tokens': cache_read_tokens,
        'total_savings': cache_read_tokens * 0.90  # 90% savings on reads
    }
```

## Real-World Use Cases and Performance

### High-Impact Applications

1. **Document Analysis Systems**:
```python
# Cache large documents for multiple analysis queries
def analyze_document_with_caching(document, queries):
    base_message = [
        {
            "type": "text",
            "text": "Document to analyze:",
        },
        {
            "type": "text",
            "text": document,
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    results = []
    for query in queries:
        messages = [{
            "role": "user",
            "content": base_message + [{"type": "text", "text": query}]
        }]
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages
        )
        results.append(response)
    
    return results
```

2. **Coding Assistants with Repository Context**:
```python
# Cache entire codebase context
def code_assistant_with_context(codebase_context, user_query):
    return client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system=[
            {
                "type": "text",
                "text": "You are a coding assistant. Here's the codebase context:",
            },
            {
                "type": "text",
                "text": codebase_context,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": user_query}]
    )
```

### Performance Reports from Users

- **Cognition (Devin)**: "Prompt caching allows us to provide more context about the codebase to get higher quality results while reducing cost and latency"
- **Notion**: Using prompt caching for Claude-powered AI assistant features with reduced costs and increased speed
- **Early adopters**: Average 14% reduction in output tokens with token-efficient tool use

## Advanced Integration Patterns

### Multi-Turn Conversations with Caching

```python
class CachedConversation:
    def __init__(self, system_context):
        self.system_context = system_context
        self.conversation_history = []
    
    def add_message(self, role, content):
        if role == "user" and not self.conversation_history:
            # First message - establish cache
            message_content = [
                {
                    "type": "text",
                    "text": "Context for this conversation:",
                },
                {
                    "type": "text", 
                    "text": self.system_context,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": content
                }
            ]
        else:
            message_content = content
            
        self.conversation_history.append({"role": role, "content": message_content})
        return self
    
    def get_response(self):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", 
            max_tokens=1000,
            messages=self.conversation_history
        )
        
        self.add_message("assistant", response.content[0].text)
        return response
```

### Batch Processing with Smart Caching

```python
def batch_process_with_caching(items, base_instruction, extended_cache=False):
    cache_control = {
        "type": "ephemeral",
        "ttl": "1h" if extended_cache else "5m"
    }
    
    headers = {}
    if extended_cache:
        headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
    
    results = []
    for item in items:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            extra_headers=headers,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": base_instruction,
                            "cache_control": cache_control
                        },
                        {
                            "type": "text",
                            "text": f"Item to process: {item}"
                        }
                    ]
                }
            ]
        )
        results.append(response)
    
    return results
```

## Troubleshooting Common Issues

### Cache Miss Debugging

1. **Inconsistent formatting**:
```python
# Bad - inconsistent spacing
content_v1 = "Here is the context:\n\n" + document
content_v2 = "Here is the context:\n" + document  # Different spacing

# Good - consistent formatting
template = "Here is the context:\n\n{document}"
content = template.format(document=document)
```

2. **Dynamic content in cached sections**:
```python
# Bad - timestamp breaks caching
{
    "type": "text",
    "text": f"Analysis at {datetime.now()}: {static_content}",
    "cache_control": {"type": "ephemeral"}
}

# Good - timestamp after cached content
[
    {
        "type": "text", 
        "text": static_content,
        "cache_control": {"type": "ephemeral"}
    },
    {
        "type": "text",
        "text": f"Current analysis time: {datetime.now()}"
    }
]
```

### Performance Monitoring

```python
def monitor_cache_performance(responses):
    total_requests = len(responses)
    cache_hits = sum(1 for r in responses if getattr(r.usage, 'cache_read_input_tokens', 0) > 0)
    
    total_cached_tokens = sum(getattr(r.usage, 'cache_read_input_tokens', 0) for r in responses)
    total_created_tokens = sum(getattr(r.usage, 'cache_creation_input_tokens', 0) for r in responses)
    
    print(f"Cache hit rate: {cache_hits/total_requests:.2%}")
    print(f"Total tokens cached: {total_cached_tokens}")
    print(f"Total cache creation: {total_created_tokens}")
    print(f"Estimated savings: ${total_cached_tokens * 0.003 * 0.90:.2f}")
```

## Migration and Adoption Strategies

### Gradual Implementation

1. **Phase 1**: Identify high-repetition prompts
2. **Phase 2**: Implement basic caching for stable content
3. **Phase 3**: Add extended caching for long-running processes
4. **Phase 4**: Integrate token-efficient tools for advanced use cases

### Legacy Application Updates

```python
# Before: Standard prompt structure
def old_approach(context, query):
    return client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": f"{context}\n\nQuery: {query}"}]
    )

# After: Cache-optimized structure  
def new_approach(context, query):
    return client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": context,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text", 
                    "text": f"Query: {query}"
                }
            ]
        }]
    )
```

## Future Roadmap and Considerations

### Upcoming Features
- Enhanced cache analytics and reporting
- Longer cache TTL options
- Cross-conversation cache sharing (potentially)
- Improved integration with Claude's reasoning capabilities

### Best Practices for Long-term Success
- Design applications with cache-friendly prompt structures from the start
- Implement comprehensive cache monitoring and analytics
- Plan for cache invalidation strategies in dynamic environments
- Consider cache warming for production applications

## Summary

Anthropic's prompt caching implementation offers the most sophisticated control mechanisms among major LLM providers. With explicit cache control, extended TTL options, and integration with advanced features like token-efficient tool use, it enables precise optimization for complex applications. The up to 90% cost reduction and 85% latency improvement make it particularly valuable for document-heavy, context-rich, or repetitive workloads. The combination of fine-grained control and powerful performance benefits makes Claude's prompt caching ideal for enterprise applications requiring both cost optimization and predictable performance.