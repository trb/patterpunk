# AWS Bedrock Multimodal Input Support

## Overview
AWS Bedrock supports multimodal input (text and images) for text generation through Claude, Llama, and Mistral models.

## SDK Information (2025)
**Version**: Boto3 1.39.14+ | **Install**: `pip install boto3` | **Requirements**: Python 3.9+ | **Status**: Production-ready

## Multimodal Input Examples

### Single Image Input
```python
import boto3
import json

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Claude 4 image analysis
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

response = bedrock.converse(
    modelId="anthropic.claude-4-sonnet-20250514-v1:0",
    messages=[{
        "role": "user",
        "content": [
            {"text": "Describe what you see in this image"},
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": image_bytes}
                }
            }
        ]
    }]
)
```

### Multiple Images Input
```python
# Multiple images with Llama
with open("chart1.png", "rb") as f:
    chart1_bytes = f.read()
with open("chart2.png", "rb") as f:
    chart2_bytes = f.read()

response = bedrock.converse(
    modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
    messages=[{
        "role": "user",
        "content": [
            {"text": "Compare these two charts"},
            {"image": {"format": "png", "source": {"bytes": chart1_bytes}}},
            {"image": {"format": "png", "source": {"bytes": chart2_bytes}}}
        ]
    }]
)
```

### Different Model Providers
```python
# Mistral Pixtral for technical documents
response = bedrock.converse(
    modelId="mistral.pixtral-large-2411-v1:0",
    messages=[{
        "role": "user",
        "content": [
            {"text": "Analyze this technical diagram"},
            {"image": {"format": "png", "source": {"bytes": diagram_bytes}}}
        ]
    }]
)
```


### Text + Image Combined
```python
# Mixed text and image content
response = bedrock.converse(
    modelId="anthropic.claude-4-sonnet-20250514-v1:0",
    messages=[{
        "role": "user",
        "content": [
            {"text": "Context: This is a quarterly financial report."},
            {"text": "Please analyze this chart:"},
            {"image": {"format": "png", "source": {"bytes": chart_bytes}}},
            {"text": "Focus on revenue trends and growth projections."}
        ]
    }]
)
```

### Streaming Responses
```python
# Stream text responses for image analysis
response = bedrock.converse_stream(
    modelId="anthropic.claude-4-sonnet-20250514-v1:0",
    messages=[{
        "role": "user",
        "content": [
            {"text": "Describe this image in detail"},
            {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}
        ]
    }]
)

for event in response['stream']:
    if 'contentBlockDelta' in event:
        delta = event['contentBlockDelta']['delta']
        if 'text' in delta:
            print(delta['text'], end="", flush=True)
```

## Supported Input Formats
**Images**: JPEG, PNG, GIF, WebP | **Limits**: 3.75MB max, 8000×8000px max (Claude)

## Key Features
- **Models**: Claude 4 (Anthropic), Llama 4 (Meta), Pixtral Large (Mistral)
- **Input**: Images as bytes via `{"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}`
- **API**: Use `bedrock.converse()` with messages containing text and image content
- **Streaming**: Use `bedrock.converse_stream()` for streaming responses
- **Enterprise**: Built-in security, no data retention, AWS ecosystem integration