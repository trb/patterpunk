# Multi-Modal Provider API Analysis

## Executive Summary

This analysis examines multi-modal input capabilities across OpenAI, Anthropic, Google, AWS Bedrock, and Ollama to identify common patterns and key differences for implementing a unified interface in Patterpunk.

## Common Patterns Across All Providers

### 1. Message-Based Conversation Structure
All providers use a role-based message structure with `user`/`assistant` roles:
```python
messages = [{"role": "user", "content": [...]}]
```

### 2. Mixed Content Support  
All providers support combining text and images within a single message using content arrays:
```python
content = [
    {"type": "text", "text": "Analyze this image"},
    {"type": "image", ...}
]
```

### 3. Multiple Image Support
All providers support multiple images per request, enabling comparison and batch analysis workflows.

### 4. Streaming Response Support
Most providers (4/5) support streaming responses for multi-modal inputs:
- ✅ OpenAI, Anthropic, Bedrock, Ollama
- ⚠️ Google (documented but limited examples)

### 5. File Size Limitations
All providers impose file size and/or dimension constraints for practical processing limits.

### 6. Base64 Encoding Support
Most providers (4/5) support base64-encoded image data:
- ✅ OpenAI, Anthropic, Google, Bedrock
- ❌ Ollama (file paths only)

## Critical API Structure Differences

### Input Parameter Names
| Provider | Primary Parameter | Secondary Parameters |
|----------|------------------|---------------------|
| OpenAI | `input` | - |
| Anthropic | `messages` | - |
| Google | `contents` | - |
| Bedrock | `messages` | - |
| Ollama | `messages` | `images` (separate array) |

**Impact**: OpenAI's `input` vs universal `messages` creates structural divergence. Ollama's separate `images` array breaks content array pattern.

### Content Type Naming Schemes
| Provider | Text Type | Image Type | File/Document Type |
|----------|-----------|------------|-------------------|
| OpenAI | `input_text` | `input_image` | `input_file` |
| Anthropic | `text` | `image` | `document` |
| Google | `Part.from_text()` | `Part.from_uri()` / `Part.from_bytes()` | (same methods) |
| Bedrock | `text` | `image` | (not supported) |
| Ollama | (text in content field) | (separate images array) | (not supported) |

**Impact**: OpenAI uses `input_*` prefix pattern. Google uses factory methods. Others use simple type strings.

### Image Data Structures
| Provider | Structure Pattern |
|----------|------------------|
| OpenAI | `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}`
| Anthropic | `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}`
| Google | `Part.from_bytes(data=bytes, mime_type="image/jpeg")`
| Bedrock | `{"image": {"format": "jpeg", "source": {"bytes": bytes}}}`
| Ollama | `"images": ["path/to/image.jpg"]`

**Impact**: Significant structural differences in image data representation require provider-specific transformation logic.

## File Input Method Variations

### 1. **OpenAI**: Triple Method Support
- **URL**: `"file_url": "https://..."`
- **Upload**: `"file_id": "file-abc123"`  
- **Base64**: `"file_data": "data:application/pdf;base64,..."`

### 2. **Anthropic**: Base64 + Files API
- **Base64**: Embedded in source object
- **Upload**: Files API with `file_id` reference

### 3. **Google**: Triple Method Support
- **Cloud Storage**: `Part.from_uri("gs://bucket/file.pdf")`
- **Upload**: `client.files.upload()` then reference
- **Inline**: `Part.from_bytes(data, mime_type)`

### 4. **Bedrock**: Bytes Only
- **Bytes**: Direct byte data with format specification

### 5. **Ollama**: File Paths Only  
- **Local Files**: `"images": ["local/path.jpg"]`

**Impact**: Wide variation in file input methods requires abstraction layer for unified interface.

## Supported Media Type Matrix

| Provider | Images | Video | Audio | PDFs | Other Documents |
|----------|--------|-------|-------|------|----------------|
| OpenAI | ✅ | ❌ | ✅ (via Whisper) | ✅ | ❌ |
| Anthropic | ✅ | ❌ | ❌ | ✅ (via Files API) | ❌ |
| Google | ✅ | ✅ | ✅ | ✅ | ❌ |
| Bedrock | ✅ | ❌ | ❌ | ❌ | ❌ |
| Ollama | ✅ | ❌ | ❌ | ❌ | ❌ |

**Impact**: Google has broadest format support. Audio requires special handling (Whisper preprocessing for OpenAI).

## Size and Dimension Limits

| Provider | Image Size | Image Dimensions | Other Limits |
|----------|------------|------------------|--------------|
| OpenAI | 20MB max | Not specified | PDFs: 100 pages, 32MB total |
| Anthropic | 5MB max | 8,000×8,000px max | 100 images per request |
| Google | 7MB single, 10MB total | Not specified | - |
| Bedrock | 3.75MB max | 8,000×8,000px max | - |
| Ollama | System memory | Not specified | Local processing |

**Impact**: Anthropic and Bedrock have strictest image size limits. File limits vary significantly.

## Token Usage Patterns

| Provider | Token Calculation Method |
|----------|-------------------------|
| OpenAI | PDF text + page images (both counted) |
| Anthropic | `(width × height) / 750` for images |
| Google | Not specified |
| Bedrock | Not specified |
| Ollama | Not applicable (local) |

**Impact**: Only Anthropic provides explicit image token calculation formula.

## Key Architectural Divergences

### 1. **API Endpoint Differences**
- **OpenAI**: New Responses API (`client.responses.create()`) vs traditional Chat API
- **Others**: Standard chat/message completion endpoints

### 2. **Parameter Structure Divergence**  
- **OpenAI**: `input` parameter breaks convention
- **Ollama**: Separate `images` array breaks content array pattern
- **Others**: Consistent `messages`/`contents` approach

### 3. **Content Type Philosophy**
- **Google**: Factory method approach (`Part.from_*()`)
- **OpenAI**: Prefixed types (`input_*`)
- **Others**: Simple string types

### 4. **File Reference Strategy**
- **Cloud-First**: Google (GCS), OpenAI (URLs)
- **API-First**: Anthropic (Files API), OpenAI (Upload)
- **Local-First**: Ollama (file paths)
- **Bytes-Only**: Bedrock

## Recommendations for Unified Interface Design

### 1. **Message Structure**
Adopt universal `messages` parameter with content arrays to maintain consistency with 4/5 providers:
```python
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "source": "..."}
        ]
    }
]
```

### 2. **Content Type Normalization**
Use simplified type names without prefixes:
- `text` for text content
- `image` for image content  
- `file` for document content

### 3. **Image Source Abstraction**
Support multiple input methods via unified `source` parameter:
```python
{
    "type": "image",
    "source": {
        "type": "base64",  # base64 | url | file_path | bytes
        "data": "...",     # base64 string | URL | file path | byte data
        "media_type": "image/jpeg"  # MIME type
    }
}
```

### 4. **Provider-Specific Transformation**
Implement transformation layer in individual provider models (`/llm/models/`) to convert unified format to provider-specific structures.

### 5. **File Type Support Matrix**
Implement capability detection to validate supported media types per provider and gracefully handle unsupported formats.

### 6. **Size Validation**
Implement pre-submission validation against provider-specific size limits to prevent API errors.

## Implementation Priority

1. **High**: Image support (all providers)
2. **Medium**: PDF/document support (OpenAI, Anthropic, Google)  
3. **Low**: Video/audio support (Google only for video, OpenAI audio via Whisper)

This analysis provides the foundation for implementing unified multi-modal interfaces while maintaining full compatibility with provider-specific requirements and capabilities.