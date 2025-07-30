# Anthropic Claude Multimodal Input Support

## Overview
Anthropic supports multimodal input (text and images) for text generation through Claude models with production-ready vision capabilities.

## SDK Information (2025)
**Version**: 0.59.0 | **Install**: `pip install anthropic==0.59.0` | **Requirements**: Python 3.8+ | **Status**: Production-ready

## Multimodal Input Examples

### Single Image Input
```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# Basic image analysis
response = client.messages.create(
    model="claude-4-sonnet-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what you see in this image"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "base64_encoded_image_data"
                }
            }
        ]
    }]
)
```

### Multiple Images Input
```python
# Multiple images comparison
response = client.messages.create(
    model="claude-4-sonnet-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two images and identify differences"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
        ]
    }]
)
```

### Files API for Repeated Use
```python
# Upload file once, use multiple times
file_response = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="vision"
)

response = client.messages.create(
    model="claude-4-sonnet-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this document"},
            {"type": "document", "source": {"type": "file", "file_id": file_response.id}}
        ]
    }]
)
```

### Streaming Responses
```python
# Stream text responses for image analysis
with client.messages.stream(
    model="claude-4-sonnet-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
        ]
    }]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Supported Input Formats
**Images**: JPEG, PNG, GIF, WebP | **Size**: 5MB max | **Dimensions**: 8,000×8,000px max | **Requests**: Up to 100 images per API request

## Key Features
- **Models**: Claude 4 series (Opus, Sonnet) for vision tasks
- **Input Types**: Base64 encoded images, Files API uploads
- **Context**: 200K tokens
- **Token Usage**: Images use `(width × height) / 750` tokens
- **API**: Use `client.messages.create()` with content array including text and image objects
- **Files API**: Upload once with `client.files.create()`, reference with file_id for repeated use