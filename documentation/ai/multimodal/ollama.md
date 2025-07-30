# Ollama Multimodal Input Support

## Overview
Ollama supports multimodal input (text and images) for text generation through local vision models with complete privacy and offline operation.

## SDK Information (2025)
**Version**: 0.5.1 | **Install**: `pip install ollama==0.5.1` | **Requirements**: Python 3.8+ | **Status**: Production-ready

## Multimodal Input Examples

### Single Image Input
```python
import ollama

# Basic image analysis
response = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{
        'role': 'user',
        'content': 'Describe what you see in this image',
        'images': ['image.jpg']
    }]
)
print(response['message']['content'])
```

### Multiple Images Input
```python
# Multiple images comparison
response = ollama.chat(
    model='qwen2.5vl:7b',
    messages=[{
        'role': 'user',
        'content': 'Compare these two images and identify differences',
        'images': ['image1.jpg', 'image2.jpg']
    }]
)
```

### Different Models
```python
# Document analysis with Qwen
response = ollama.chat(
    model='qwen2.5vl:3b',
    messages=[{
        'role': 'user',
        'content': 'Extract key information from this document',
        'images': ['document.jpg']
    }]
)

# Lightweight analysis with Moondream
response = ollama.chat(
    model='moondream:latest',
    messages=[{
        'role': 'user',
        'content': 'Describe this image briefly',
        'images': ['photo.jpg']
    }],
    options={'temperature': 0.1}
)
```

### Streaming Responses
```python
# Stream text responses for image analysis
for chunk in ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{
        'role': 'user',
        'content': 'Analyze this image in detail',
        'images': ['image.jpg']
    }],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```



## Supported Input Formats
**Images**: JPEG, PNG, GIF, BMP, WebP, TIFF | **Limits**: System memory dependent

## Key Features
- **Models**: Llama 3.2 Vision, Qwen2.5-VL, Moondream2, Granite 3.2 Vision, LLaVA 1.6
- **Input**: Images via `'images': ['path/to/image.jpg']` in messages
- **API**: Use `ollama.chat()` with model and messages including images array
- **Local Processing**: Complete privacy, offline operation, zero API costs
- **Hardware**: Supports CUDA (NVIDIA), Metal (Apple), ROCm (AMD)
- **Requirements**: CPU-only (Moondream2) to 64GB VRAM (Llama 3.2 Vision 90B)