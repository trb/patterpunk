# Google Multimodal Input Support

## Overview
Google supports multimodal input (text, images, video, audio, PDFs) for text generation through Gemini models via the Google GenAI SDK.

## Critical SDK Migration (2025)
⚠️ **VERTEX AI SDK DEPRECATED**: June 24, 2025 → Removed June 24, 2026
**Migrate to**: Google GenAI SDK (`google-genai`)

## SDK Information (2025)
**Version**: 1.24.0 | **Install**: `pip install google-genai==1.24.0` | **Status**: General Availability

## File Input Methods

### Setup
```python
from google import genai
from google.genai import types

# Vertex AI setup
client = genai.Client(vertexai=True, project='your-project-id', location='us-central1')
```

### Image Input
```python
# METHOD 1: Google Cloud Storage URLs
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_text('Describe this image'),
        types.Part.from_uri('gs://bucket/image.jpg', 'image/jpeg')
    ]
)

# METHOD 2: Local file via Files API upload
uploaded_file = client.files.upload(file='local-image.jpg')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_text('Analyze this image'),
        uploaded_file
    ]
)

# METHOD 3: Local file bytes (inline, <20MB total)
with open('image.jpg', 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_text('What do you see?'),
        types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
    ]
)
```

### Video Input
```python
# Video from Google Cloud Storage
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[
        types.Part.from_text('Summarize this video'),
        types.Part.from_uri('gs://bucket/video.mp4', 'video/mp4')
    ]
)

# Local video via Files API upload
uploaded_video = client.files.upload(file='local-video.mp4')
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[
        types.Part.from_text('Describe what happens in this video'),
        uploaded_video
    ]
)
```

### Audio Input
```python
# Audio from Google Cloud Storage
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[
        types.Part.from_text('Transcribe and summarize this audio'),
        types.Part.from_uri('gs://bucket/audio.mp3', 'audio/mp3')
    ]
)

# Local audio via Files API upload
uploaded_audio = client.files.upload(file='audio-file.wav')
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[
        types.Part.from_text('What is discussed in this audio?'),
        uploaded_audio
    ]
)

# Local audio as bytes (smaller files)
with open('recording.mp3', 'rb') as f:
    audio_bytes = f.read()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_text('Transcribe this recording'),
        types.Part.from_bytes(data=audio_bytes, mime_type='audio/mp3')
    ]
)
```

### PDF Input
```python
# PDF from Google Cloud Storage
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_text('Summarize this document'),
        types.Part.from_uri('gs://bucket/document.pdf', 'application/pdf')
    ]
)

# Local PDF as bytes
with open('document.pdf', 'rb') as f:
    pdf_bytes = f.read()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_text('Extract key information from this PDF'),
        types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf')
    ]
)
```

### Multiple Input Types
```python
# Combine different input types
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[
        types.Part.from_text('Analyze these materials together:'),
        types.Part.from_uri('gs://bucket/chart.png', 'image/png'),
        types.Part.from_uri('gs://bucket/report.pdf', 'application/pdf'),
        types.Part.from_text('Focus on financial trends and projections')
    ]
)
```


## Supported Input Formats
**Images**: JPEG, PNG, GIF, WebP, BMP | **Video**: MP4, MOV, AVI, FLV, MKV, WebM | **Audio**: MP3, WAV, FLAC, AAC, OGG | **Documents**: PDF

**File Reference Methods**:
1. **Google Cloud Storage**: `types.Part.from_uri('gs://bucket/file.jpg', 'image/jpeg')`
2. **Files API Upload**: `client.files.upload(file='local-file.jpg')`
3. **Inline Bytes**: `types.Part.from_bytes(data=file_bytes, mime_type='image/jpeg')`

**Limits**: 10MB total (7MB single file) for inline bytes

## Migration Guide
```python
# OLD (Deprecated)
from vertexai.generative_models import GenerativeModel

# NEW (Required by June 24, 2026)
from google import genai
client = genai.Client(vertexai=True, project='your-project-id', location='us-central1')
```

## Key Features
- **Models**: Gemini 2.5 Pro/Flash for maximum capability
- **Input Types**: Images, video, audio, PDFs via three file reference methods
- **API**: Use `client.models.generate_content()` with content array
- **File Handling**: GCS URLs, Files API upload, or inline bytes
- **Context**: Long context support for large files