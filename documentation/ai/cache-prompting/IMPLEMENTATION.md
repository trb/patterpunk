# Patterpunk Prompt Caching Implementation Guide

This document outlines the implementation strategy for unified prompt caching across all supported providers in patterpunk.

## Overview

The implementation extends patterpunk's existing Message architecture with cache-aware content chunks while maintaining backward compatibility and the chainable immutable interface. The approach uses polymorphic message content that can be either a string (legacy) or a list of cache chunks (new caching interface).

## Core Interface Design

### CacheChunk Class

```python
from datetime import timedelta
from typing import Optional

class CacheChunk:
    def __init__(
        self, 
        content: str, 
        cacheable: bool = False, 
        ttl: Optional[timedelta] = None
    ):
        self.content = content
        self.cacheable = cacheable
        self.ttl = ttl
```

### Message Class Updates

Update the base Message class to support polymorphic content:

```python
from typing import Union, List

class Message:
    def __init__(self, content: Union[str, List[CacheChunk]], role: str = ROLE_USER):
        self.content = content
        self.role = role
        # ... existing attributes
```

### Usage Examples

```python
# Legacy string content (unchanged)
UserMessage(content="Simple message without caching")

# New chunked content with caching
UserMessage(content=[
    CacheChunk(content="Large codebase context...", cacheable=True),
    CacheChunk(content="Current task: update file X", cacheable=False)
])

# System message with caching
SystemMessage(content=[
    CacheChunk(content="You are an expert analyst with access to:", cacheable=False),
    CacheChunk(content=large_document_content, cacheable=True, ttl=timedelta(hours=1))
])
```

## Implementation Steps

### 1. Create CacheChunk Type

Add to `patterpunk/llm/types.py`:

```python
from datetime import timedelta
from typing import Optional

class CacheChunk:
    def __init__(
        self, 
        content: str, 
        cacheable: bool = False, 
        ttl: Optional[timedelta] = None
    ):
        self.content = content
        self.cacheable = cacheable
        self.ttl = ttl
    
    def __repr__(self):
        cache_info = f", cacheable={self.cacheable}"
        if self.ttl:
            cache_info += f", ttl={self.ttl}"
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f'CacheChunk("{content_preview}"{cache_info})'
```

### 2. Update Message Base Class

Modify `patterpunk/llm/messages.py`:

```python
from typing import Union, List
from patterpunk.llm.types import CacheChunk

class Message:
    def __init__(self, content: Union[str, List[CacheChunk]], role: str = ROLE_USER):
        self.content = content
        self.role = role
        # ... existing attributes
    
    # User feedback: Let's make this a public method, i.e. don't prefix with "_"
    def _get_content_as_string(self) -> str:
        """Helper method to get content as string for backward compatibility."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return "".join(chunk.content for chunk in self.content)
        else:
            return str(self.content)
    
    def has_cacheable_content(self) -> bool:
        """Check if message contains any cacheable chunks."""
        if isinstance(self.content, list):
            return any(chunk.cacheable for chunk in self.content)
        return False
    
    def get_cache_chunks(self) -> List[CacheChunk]:
        """Get cache chunks, converting string content to non-cacheable chunk if needed."""
        if isinstance(self.content, str):
            return [CacheChunk(content=self.content, cacheable=False)]
        elif isinstance(self.content, list):
            return self.content
        else:
            return [CacheChunk(content=str(self.content), cacheable=False)]
```

### 3. Update Message Subclasses

Update constructors to accept the new content type:

```python
class SystemMessage(Message):
    def __init__(self, content: Union[str, List[CacheChunk]]):
        super().__init__(content, ROLE_SYSTEM)

class UserMessage(Message):
    def __init__(
        self, 
        content: Union[str, List[CacheChunk]], 
        structured_output=None, 
        allow_tool_calls=True
    ):
        super().__init__(content, ROLE_USER)
        self.structured_output = structured_output
        self.allow_tool_calls = allow_tool_calls

class AssistantMessage(Message):
    def __init__(
        self,
        # User feedback: I don't think it makes sense to have this a Union type, this can't be anything but "str" I don't think - consider only allowing "str" for AssistantMessage
        content: Union[str, List[CacheChunk]], 
        structured_output=None, 
        parsed_output=None
    ):
        super().__init__(content, ROLE_ASSISTANT)
        self.structured_output = structured_output
        self._parsed_output = parsed_output
```

### 4. Update Model Base Class

USER FEEDBACK: Ignore any and all metrics for now. We only want support for prompt caching, we'll add metrics later!
Add cache metrics to the base model interface:

```python
from typing import Dict, Any, Optional

class CacheMetrics:
    def __init__(
        self,
        hit_rate: float = 0.0,
        cached_tokens: int = 0,
        total_tokens: int = 0,
        cost_savings_estimate: float = 0.0,
        provider_specific: Optional[Dict[str, Any]] = None
    ):
        self.hit_rate = hit_rate
        self.cached_tokens = cached_tokens
        self.total_tokens = total_tokens
        self.cost_savings_estimate = cost_savings_estimate
        self.provider_specific = provider_specific or {}
```

## Provider-Specific Implementations

### OpenAI Implementation

OpenAI uses automatic prefix caching, so we need to:
1. Concatenate all chunks into a single string
2. Warn if non-prefix cacheable patterns are detected
3. Use the Responses API when available for better caching

```python
class OpenAiModel(Model):
    def _process_cache_chunks_for_openai(self, chunks: List[CacheChunk]) -> tuple[str, bool]:
        """
        Process cache chunks for OpenAI's automatic prefix caching.
        Returns (concatenated_content, should_warn_about_caching)
        """
        content = "".join(chunk.content for chunk in chunks)
        
        # Check for non-prefix cacheable patterns
        cacheable_positions = [i for i, chunk in enumerate(chunks) if chunk.cacheable]
        
        # Warn if we have cacheable content that's not at the prefix
        should_warn = False
        if cacheable_positions:
            # Check if there are non-cacheable chunks before the last cacheable chunk
            last_cacheable = max(cacheable_positions)
            for i in range(last_cacheable):
                if not chunks[i].cacheable:
                    should_warn = True
                    break
        
        return content, should_warn
    
    def _convert_messages_for_openai(self, messages: List[Message]) -> List[Dict]:
        """Convert patterpunk messages to OpenAI format with cache handling."""
        openai_messages = []
        cache_warnings = []
        
        for message in messages:
            if isinstance(message.content, list):
                content, should_warn = self._process_cache_chunks_for_openai(message.content)
                if should_warn:
                    cache_warnings.append(f"Non-prefix cacheable content detected in {message.role} message")
            else:
                content = message.content
            
            openai_messages.append({
                "role": message.role,
                "content": content
            })
        
        # Log warnings about suboptimal caching patterns
        for warning in cache_warnings:
            logger.warning(f"[OPENAI_CACHE] {warning} - caching may be ineffective")
        
        return openai_messages
    
    def _extract_cache_metrics_from_response(self, response) -> CacheMetrics:
        """Extract cache metrics from OpenAI response."""
        usage = getattr(response, 'usage', None)
        if not usage:
            return CacheMetrics()
        
        prompt_tokens_details = getattr(usage, 'prompt_tokens_details', {})
        cached_tokens = prompt_tokens_details.get('cached_tokens', 0)
        total_tokens = getattr(usage, 'prompt_tokens', 0)
        
        hit_rate = cached_tokens / total_tokens if total_tokens > 0 else 0.0
        cost_savings = cached_tokens * 0.5  # 50% savings on cached tokens
        
        return CacheMetrics(
            hit_rate=hit_rate,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            cost_savings_estimate=cost_savings,
            provider_specific={
                'prompt_tokens_details': prompt_tokens_details,
                'completion_tokens': getattr(usage, 'completion_tokens', 0),
                'total_tokens': getattr(usage, 'total_tokens', 0)
            }
        )
```

### Anthropic Implementation

Anthropic uses explicit cache control markers on content blocks:

```python
class AnthropicModel(Model):
    def _convert_content_to_anthropic_format(self, chunks: List[CacheChunk]) -> List[Dict]:
        """Convert cache chunks to Anthropic content format with cache controls."""
        anthropic_content = []
        
        for chunk in chunks:
            content_block = {
                "type": "text",
                "text": chunk.content
            }
            
            if chunk.cacheable:
                cache_control = {"type": "ephemeral"}
                if chunk.ttl:
                    # Convert ttl to appropriate format if extended TTL is needed
                    if chunk.ttl.total_seconds() > 300:  # More than 5 minutes
                        cache_control["ttl"] = "1h"  # Use extended TTL
                content_block["cache_control"] = cache_control
            
            anthropic_content.append(content_block)
        
        return anthropic_content
    
    def _convert_messages_for_anthropic(self, messages: List[Message]) -> List[Dict]:
        """Convert patterpunk messages to Anthropic format."""
        anthropic_messages = []
        
        for message in messages:
            if isinstance(message.content, list):
                content = self._convert_content_to_anthropic_format(message.content)
            else:
                content = [{"type": "text", "text": message.content}]
            
            anthropic_messages.append({
                "role": message.role,
                "content": content
            })
        
        return anthropic_messages
    
    def _extract_cache_metrics_from_response(self, response) -> CacheMetrics:
        """Extract cache metrics from Anthropic response."""
        usage = getattr(response, 'usage', None)
        if not usage:
            return CacheMetrics()
        
        cached_tokens = getattr(usage, 'cache_read_input_tokens', 0)
        total_tokens = getattr(usage, 'input_tokens', 0)
        
        hit_rate = cached_tokens / total_tokens if total_tokens > 0 else 0.0
        cost_savings = cached_tokens * 0.9  # 90% savings on cached tokens
        
        return CacheMetrics(
            hit_rate=hit_rate,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            cost_savings_estimate=cost_savings,
            provider_specific={
                'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', 0),
                'cache_read_input_tokens': cached_tokens,
                'input_tokens': total_tokens,
                'output_tokens': getattr(usage, 'output_tokens', 0)
            }
        )
```

### AWS Bedrock Implementation

Bedrock uses cache points similar to Anthropic but with different syntax:

```python
class BedrockModel(Model):
    def _convert_content_to_bedrock_format(self, chunks: List[CacheChunk]) -> List[Dict]:
        """Convert cache chunks to Bedrock content format with cache points."""
        bedrock_content = []
        
        for chunk in chunks:
            content_block = {
                "text": chunk.content
            }
            
            if chunk.cacheable:
                content_block["cachePoint"] = {}
            
            bedrock_content.append(content_block)
        
        return bedrock_content
    
    def _convert_messages_for_bedrock(self, messages: List[Message]) -> List[Dict]:
        """Convert patterpunk messages to Bedrock format."""
        bedrock_messages = []
        
        for message in messages:
            if isinstance(message.content, list):
                content = self._convert_content_to_bedrock_format(message.content)
            else:
                content = [{"text": message.content}]
            
            bedrock_messages.append({
                "role": message.role,
                "content": content
            })
        
        return bedrock_messages
    
    def _extract_cache_metrics_from_response(self, response) -> CacheMetrics:
        """Extract cache metrics from Bedrock response."""
        usage = response.get('usage', {})
        if not usage:
            return CacheMetrics()
        
        cached_tokens = usage.get('cacheReadInputTokens', 0)
        total_tokens = usage.get('inputTokens', 0)
        
        hit_rate = cached_tokens / total_tokens if total_tokens > 0 else 0.0
        cost_savings = cached_tokens * 0.9  # 90% savings on cached tokens
        
        return CacheMetrics(
            hit_rate=hit_rate,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            cost_savings_estimate=cost_savings,
            provider_specific={
                'cacheReadInputTokens': cached_tokens,
                'cacheWriteInputTokens': usage.get('cacheWriteInputTokens', 0),
                'inputTokens': total_tokens,
                'outputTokens': usage.get('outputTokens', 0)
            }
        )
```

### Google Vertex AI Implementation

Google requires pre-creating cache objects for cacheable content:

```python
class GoogleModel(Model):
    def _create_cache_objects_for_chunks(self, chunks: List[CacheChunk]) -> Dict[str, str]:
        """Pre-create cache objects for cacheable chunks."""
        cache_mappings = {}
        
        for i, chunk in enumerate(chunks):
            if chunk.cacheable and len(chunk.content) > 32000:  # Meet minimum token requirement
                cache_id = f"cache_chunk_{i}_{hash(chunk.content)}"
                
                try:
                    # Create cached content object
                    cached_content = self.client.caches.create(
                        config=CreateCachedContentConfig(
                            model=self.model,
                            contents=[Content(parts=[Part.from_text(chunk.content)])],
                            ttl=chunk.ttl or timedelta(hours=1)
                        )
                    )
                    cache_mappings[cache_id] = cached_content.name
                except Exception as e:
                    logger.warning(f"Failed to create cache for chunk {i}: {e}")
        
        return cache_mappings
    
    def _convert_messages_for_google(self, messages: List[Message]) -> tuple[List[Dict], Dict[str, str]]:
        """Convert patterpunk messages to Google format with pre-cached content."""
        google_messages = []
        all_cache_mappings = {}
        
        for message in messages:
            if isinstance(message.content, list):
                # Pre-create cache objects for cacheable chunks
                cache_mappings = self._create_cache_objects_for_chunks(message.content)
                all_cache_mappings.update(cache_mappings)
                
                # Convert to Google format (concatenate for now, as Google will auto-detect cached content)
                content = "".join(chunk.content for chunk in message.content)
            else:
                content = message.content
            
            google_messages.append({
                "role": message.role,
                "parts": [{"text": content}]
            })
        
        return google_messages, all_cache_mappings
    
    def _extract_cache_metrics_from_response(self, response) -> CacheMetrics:
        """Extract cache metrics from Google response."""
        # Google's cache metrics are less standardized, adapt based on actual response structure
        usage_metadata = getattr(response, 'usage_metadata', {})
        
        # Estimate based on available data
        total_tokens = usage_metadata.get('prompt_token_count', 0)
        # Google doesn't always provide explicit cache hit data in the same way
        
        return CacheMetrics(
            hit_rate=0.0,  # Would need to calculate based on pre-cached content
            cached_tokens=0,  # Would need provider-specific logic
            total_tokens=total_tokens,
            cost_savings_estimate=0.0,
            provider_specific=usage_metadata
        )
```

### Ollama Implementation

Ollama doesn't support prompt caching, so we ignore cache settings:

```python
class OllamaModel(Model):
    def _convert_messages_for_ollama(self, messages: List[Message]) -> List[Dict]:
        """Convert patterpunk messages to Ollama format, ignoring cache settings."""
        ollama_messages = []
        
        for message in messages:
            # Always convert to string, ignoring cache settings
            content = message._get_content_as_string()
            
            ollama_messages.append({
                "role": message.role,
                "content": content
            })
        
        return ollama_messages
    
    def _extract_cache_metrics_from_response(self, response) -> CacheMetrics:
        """Ollama doesn't support caching, return empty metrics."""
        return CacheMetrics()
```

## Testing Strategy

### Unit Tests

1. **CacheChunk creation and properties**
2. **Message polymorphic content handling**
3. **Provider-specific content conversion**
4. **Cache metrics extraction**

### Integration Tests

1. **End-to-end caching with each provider**
2. **Backward compatibility with string content**
3. **Cache performance measurement**
4. **Error handling for invalid cache configurations**

### Test Examples

```python
def test_cache_chunk_creation():
    chunk = CacheChunk(content="test", cacheable=True, ttl=timedelta(hours=1))
    assert chunk.content == "test"
    assert chunk.cacheable is True
    assert chunk.ttl == timedelta(hours=1)

def test_message_polymorphic_content():
    # String content (legacy)
    msg1 = UserMessage(content="simple string")
    assert isinstance(msg1.content, str)
    assert not msg1.has_cacheable_content()
    
    # Chunked content (new)
    msg2 = UserMessage(content=[
        CacheChunk(content="cacheable", cacheable=True),
        CacheChunk(content="not cacheable", cacheable=False)
    ])
    assert isinstance(msg2.content, list)
    assert msg2.has_cacheable_content()

def test_openai_cache_warnings():
    chunks = [
        CacheChunk(content="dynamic", cacheable=False),
        CacheChunk(content="static", cacheable=True)
    ]
    model = OpenAiModel()
    content, should_warn = model._process_cache_chunks_for_openai(chunks)
    assert should_warn is True  # Non-prefix cacheable pattern
```

## Migration Guide

### For Existing Code

Existing code continues to work unchanged:

```python
# This still works exactly as before
UserMessage(content="Simple string message")
SystemMessage(content="System instructions")
```

### For New Cache-Aware Code

Gradually adopt chunked content for caching benefits:

```python
# Start with simple caching
UserMessage(content=[
    CacheChunk(content=large_context, cacheable=True)
])

# Progress to advanced patterns
UserMessage(content=[
    CacheChunk(content="Context: ", cacheable=False),
    CacheChunk(content=large_document, cacheable=True, ttl=timedelta(hours=2)),
    CacheChunk(content=f"Query: {user_query}", cacheable=False)
])
```

## Performance Considerations

1. **Cache chunk overhead**: Minimal memory overhead for CacheChunk objects
2. **Provider conversion cost**: One-time cost during message conversion
3. **Cache metrics extraction**: Standardized across providers for consistent monitoring
4. **Backward compatibility**: Zero performance impact for string content

## Future Extensions

1. **Chat-level cache strategies**: Automatic detection of cacheable patterns
2. **Tool definition caching**: Extend to cache large tool schemas
3. **Cross-conversation caching**: Persistent cache across multiple Chat instances
4. **Cache analytics**: Enhanced metrics and optimization suggestions
