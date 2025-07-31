# Multimodal Content

Patterpunk provides unified multimodal content handling across all providers through the `MultimodalChunk` class. Content sources are automatically detected and converted to provider-compatible formats.

## Creation Methods

### From Files

```python
from patterpunk.llm.multimodal import MultimodalChunk

# Local files with automatic media type detection
image = MultimodalChunk.from_file("screenshot.png")
pdf = MultimodalChunk.from_file("/path/to/document.pdf")
video = MultimodalChunk.from_file("demo.mp4")

# Direct constructor (less preferred)
chunk = MultimodalChunk(source="image.jpg")  # Inferred as file_path
```

### From URLs

```python
# Remote files with automatic downloading
web_image = MultimodalChunk.from_url("https://example.com/chart.png")

# Google Cloud Storage URIs
gcs_file = MultimodalChunk.from_url("gs://bucket/document.pdf")

# Download with custom session for authentication
import requests
session = requests.Session()
session.headers.update({"Authorization": "Bearer token"})
downloaded = web_image.download(session=session)
```

### From Raw Data

```python
# From bytes with explicit media type
with open("image.png", "rb") as f:
    data = f.read()
chunk = MultimodalChunk.from_bytes(data, media_type="image/png", filename="image.png")

# From file objects
with open("document.pdf", "rb") as f:
    chunk = MultimodalChunk.from_file_object(f, filename="document.pdf")

# From base64 encoded data
b64_data = "iVBORw0KGgoAAAANSUhEUgAA..."
chunk = MultimodalChunk.from_base64(b64_data, media_type="image/png")

# From data URIs
data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
chunk = MultimodalChunk.from_data_uri(data_uri)
```

## Format Conversion

```python
chunk = MultimodalChunk.from_file("image.jpg")

# Convert to different formats
bytes_data = chunk.to_bytes()           # Raw bytes
b64_string = chunk.to_base64()          # Base64 encoded string
data_uri = chunk.to_data_uri()          # Data URI format
file_path = chunk.get_file_path()       # Path object if from file
url = chunk.get_url()                   # URL if from remote source

# Access metadata
print(chunk.media_type)      # "image/jpeg"
print(chunk.source_type)     # "file_path"
print(chunk.filename)        # "image.jpg"
```

## Message Integration

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.text import TextChunk

# Mixed content types in messages
message = UserMessage([
    TextChunk("Analyze this chart: "),
    MultimodalChunk.from_file("chart.png"),
    TextChunk(" and this document: "),
    MultimodalChunk.from_url("https://example.com/report.pdf").download()
])

response = (Chat(model=OpenAiModel())
    .add_message(message)
    .complete())
```

## Automatic Processing

### Media Type Detection

```python
# Automatic detection from file extensions
pdf_chunk = MultimodalChunk.from_file("report.pdf")
print(pdf_chunk.media_type)  # "application/pdf"

# From HTTP Content-Type headers during download
web_chunk = MultimodalChunk.from_url("https://api.com/image").download()
print(web_chunk.media_type)  # From server response headers

# From data URI metadata
uri_chunk = MultimodalChunk.from_data_uri("data:video/mp4;base64,...")
print(uri_chunk.media_type)  # "video/mp4"
```

### Source Type Inference

```python
# Automatic source type detection
MultimodalChunk("image.jpg")                    # file_path
MultimodalChunk("https://example.com/img.png")  # url
MultimodalChunk("gs://bucket/file.pdf")         # gcs_uri
MultimodalChunk("data:image/png;base64,...")    # data_uri
MultimodalChunk(b"binary_data")                 # bytes
```

## Provider Compatibility

### Document Types by Provider

```python
# PDF support varies by provider
pdf_chunk = MultimodalChunk.from_file("document.pdf")

# OpenAI: Full PDF support
openai_chat = Chat(model=OpenAiModel()).add_message(
    UserMessage([pdf_chunk])
).complete()

# Anthropic: Full PDF support
anthropic_chat = Chat(model=AnthropicModel()).add_message(
    UserMessage([pdf_chunk])
).complete()

# Google: PDF and document support
google_chat = Chat(model=GoogleModel()).add_message(
    UserMessage([pdf_chunk])
).complete()
```

### Image Formats

```python
# Common formats supported across providers
formats = [
    MultimodalChunk.from_file("image.jpg"),    # JPEG
    MultimodalChunk.from_file("image.png"),    # PNG
    MultimodalChunk.from_file("image.gif"),    # GIF
    MultimodalChunk.from_file("image.webp"),   # WebP
]

for image in formats:
    response = chat.add_message(UserMessage([
        TextChunk("Describe this image: "), image
    ])).complete()
```

## Lazy Loading and Caching

```python
# Content is loaded on-demand
url_chunk = MultimodalChunk.from_url("https://example.com/large-file.pdf")
print(url_chunk.source_type)  # "url" - not downloaded yet

# First access triggers download and caching
data = url_chunk.to_bytes()  # Downloads and caches
data2 = url_chunk.to_bytes()  # Uses cached version

# Explicit downloading for control
downloaded_chunk = url_chunk.download()
print(downloaded_chunk.source_type)  # "bytes"
```

## Error Handling

```python
# File not found
try:
    chunk = MultimodalChunk.from_file("nonexistent.jpg")
except FileNotFoundError:
    print("File not found")

# Network errors during download
try:
    chunk = MultimodalChunk.from_url("https://invalid-domain.com/file.pdf")
    data = chunk.download()
except ValueError as e:
    print(f"Download failed: {e}")

# Invalid data formats
try:
    chunk = MultimodalChunk.from_base64("invalid-base64-data")
except ValueError:
    print("Invalid base64 format")

# Missing dependencies
try:
    chunk = MultimodalChunk.from_url("https://example.com/file.txt")
    chunk.download()  # Requires requests library
except ImportError:
    print("Install requests: pip install requests")
```

## Advanced Patterns

### Batch Processing

```python
# Process multiple files efficiently
image_paths = ["img1.jpg", "img2.png", "img3.gif"]
chunks = [MultimodalChunk.from_file(path) for path in image_paths]

# Combined analysis
message = UserMessage([
    TextChunk("Compare these images: ")
] + chunks)

response = chat.add_message(message).complete()
```

### Custom Sessions for Authentication

```python
import requests

# Authenticated downloads
session = requests.Session()
session.headers.update({
    "Authorization": "Bearer your-token",
    "User-Agent": "MyApp/1.0"
})

secure_chunk = MultimodalChunk.from_url("https://api.example.com/secure-file.pdf")
downloaded = secure_chunk.download(session=session)
```

### Mixed Content Workflows

```python
from patterpunk.llm.text import TextChunk
from patterpunk.llm.cache import CacheChunk

# Complex content composition
message = UserMessage([
    TextChunk("Please analyze: "),
    CacheChunk("Context from previous analysis", cacheable=True),
    MultimodalChunk.from_file("chart.png"),
    TextChunk(" and cross-reference with: "),
    MultimodalChunk.from_url("https://example.com/data.pdf").download()
])
```

## Content Type Support

| Type | OpenAI | Anthropic | Google | Bedrock | Ollama |
|------|---------|-----------|---------|---------|---------|
| Images (JPEG, PNG, GIF, WebP) | ✅ | ✅ | ✅ | ✅ | ✅ |
| PDF Documents | ✅ | ✅ | ✅ | ✅ | ❌ |
| Video (MP4, MOV) | ❌ | ❌ | ✅ | ❌ | ❌ |
| Audio (MP3, WAV) | ❌ | ❌ | ✅ | ❌ | ❌ |
| Office Documents | ❌ | ❌ | ✅ | ❌ | ❌ |

Provider-specific validation and conversion happens automatically in model implementations.