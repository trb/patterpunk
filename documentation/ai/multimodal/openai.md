# OpenAI Multimodal Input Support

## Overview
OpenAI supports multimodal input (text, images, audio) for text generation using GPT-4o, GPT-4.1 series and reasoning models (o3, o4) via the Responses API introduced in March 2025.

## SDK Information (2025)
**Version**: 1.97.1+ | **Install**: `pip install openai` | **Requirements**: Python 3.8+ | **Status**: Production-ready

## Responses API Multimodal Format

The Responses API uses a specific format for multimodal inputs with `input` parameter containing role-based content arrays.

### Image Input (URL)
```python
import openai

client = openai.OpenAI(api_key="your-api-key")

# Single image analysis with URL
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analyze this image and describe what you see"},
                {"type": "input_image", "image_url": "https://example.com/image.jpg"}
            ]
        }
    ]
)

# Multiple images comparison
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user", 
            "content": [
                {"type": "input_text", "text": "Compare these two images"},
                {"type": "input_image", "image_url": "https://example.com/image1.jpg"},
                {"type": "input_image", "image_url": "https://example.com/image2.jpg"}
            ]
        }
    ]
)
```

### Image Input (Base64)
```python
import base64

# Read and encode image
with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
            ]
        }
    ]
)
```


### PDF File Input

The Responses API supports direct PDF file input with three methods: file URLs, uploaded files, and base64 encoding.

#### File URL Method
```python
# Direct PDF URL input
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_url": "https://example.com/document.pdf"
                },
                {
                    "type": "input_text",
                    "text": "Analyze this document and provide a summary."
                }
            ]
        }
    ]
)
```

#### File Upload Method
```python
# Upload file first, then reference by ID
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="user_data"
)

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id
                },
                {
                    "type": "input_text",
                    "text": "What are the key findings in this document?"
                }
            ]
        }
    ]
)
```

#### Base64 Encoding Method
```python
import base64

# Base64 encode PDF file
with open("document.pdf", "rb") as f:
    data = f.read()
base64_string = base64.b64encode(data).decode("utf-8")

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "document.pdf",
                    "file_data": f"data:application/pdf;base64,{base64_string}"
                },
                {
                    "type": "input_text",
                    "text": "Summarize the main points in this document."
                }
            ]
        }
    ]
)
```

### Audio Input
```python
# Audio transcription and analysis (use Whisper first, then Responses API)
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

# Use transcribed text with Responses API
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"Analyze this transcription: {transcript.text}"}
            ]
        }
    ]
)
```

### Combined Text + Image Input
```python
# Mixed input types with Responses API
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Based on this document and image, provide analysis:"},
                {"type": "input_text", "text": "Additional context: This is a financial report from Q3 2024"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,{base64_image}"}
            ]
        }
    ]
)

# Tool integration example
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Search for news about this image content"},
                {"type": "input_image", "image_url": "https://example.com/news-image.jpg"}
            ]
        }
    ],
    tools=[{"type": "web_search"}]
)
```

## Supported Input Formats

**Images**: PNG, JPEG, WebP, GIF (20MB max) 
**PDFs**: Direct input support (100 pages max, 32MB total per request)
**Audio**: mp3, mp4, mpeg, mpga, m4a, wav, webm (25MB max via Whisper API first)
**Text**: Direct text input via `input_text` type

### PDF Processing Details
- Extracts both text and images from each page
- Provides full page context to vision-capable models
- Supports file URLs, uploaded files, and base64 encoding

## Key Features

- **Models**: GPT-4o, GPT-4.1 series for general use, o3/o4 for advanced reasoning (vision-capable models required for PDF/image inputs)
- **Context**: Up to 1M tokens depending on model
- **API**: Responses API (`client.responses.create()`) introduced March 2025
- **Format**: Role-based input with `content` arrays containing typed elements
- **Image Types**: `"type": "input_image"` with direct `image_url` field
- **File Types**: `"type": "input_file"` with `file_url`, `file_id`, or `file_data` fields
- **Text Types**: `"type": "input_text"` for text content
- **Tool Integration**: Built-in support for tools like web search
- **Audio Processing**: Use Whisper API first, then include transcription as input_text

## Usage Considerations

### Token Usage
PDF inputs consume tokens for both extracted text AND images of each page, regardless of visual content. Consider pricing implications for production use.

### File Limitations
- **PDF**: 100 pages max, 32MB total per request across all file inputs
- **Images**: 20MB max per image
- **Audio**: 25MB max (via separate Whisper API)

### File Upload Purpose
Use `purpose="user_data"` when uploading files via Files API for model input.

## Responses API Format Structure

```python
{
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Your text prompt"},
                {"type": "input_image", "image_url": "URL or base64 data URL"},
                {"type": "input_file", "file_url": "https://example.com/file.pdf"},
                {"type": "input_file", "file_id": "file-abc123"},
                {"type": "input_file", "filename": "doc.pdf", "file_data": "data:application/pdf;base64,..."}
            ]
        }
    ]
}
```

## Key Differences from Chat Completions API

The Responses API uses a different structure than the traditional Chat Completions API:
- Parameter: `input` (not `messages`)
- Content types: `input_text`, `input_image`, `input_file` (not `text`, `image_url`)
- Image format: Direct `image_url` field (not nested object)
- File format: `input_file` with `file_url`, `file_id`, or `file_data` options
- Enhanced tool integration and state management capabilities