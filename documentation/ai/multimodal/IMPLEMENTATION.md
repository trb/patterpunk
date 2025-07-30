# Multimodal Implementation Guide for Patterpunk (CORRECTED)

This guide provides **corrected** detailed instructions for implementing multimodal support in patterpunk's Message classes, addressing critical API format errors identified in the original implementation.

## Core Design Principle

The Message classes will accept content as an array of typed chunks, extending the existing `CacheChunk` pattern to support multimodal data. Each chunk can represent text, images, files, or other media types, with flexible input methods and **provider capability awareness**.

## New Type Definitions

### 1. Enhanced MultimodalChunk Class

Create a new file: `/patterpunk/src/patterpunk/llm/multimodal.py`

```python
from typing import Union, Optional
from io import BytesIO
from pathlib import Path
import base64
import mimetypes
import re

class MultimodalChunk:
    """
    Represents a chunk of multimodal content (image, file, audio, video).
    
    Provides unified interface for different file sources without provider-specific logic.
    Provider validation and capabilities are handled by individual provider models.
    """
    
    def __init__(
        self,
        source: Union[str, bytes, BytesIO, Path],
        media_type: Optional[str] = None,
        source_type: Optional[str] = None,
        filename: Optional[str] = None,  # For file uploads
    ):
        self.source = source
        self.source_type = source_type or self._infer_source_type(source)
        self.media_type = media_type or self._detect_media_type()
        self.filename = filename
        self._cached_bytes: Optional[bytes] = None
        
    def _infer_source_type(self, source) -> str:
        """Infer the source type from the input with improved detection."""
        if isinstance(source, str):
            if source.startswith(('http://', 'https://')):
                return "url"
            elif source.startswith('gs://'):
                return "gcs_uri"  # Google Cloud Storage
            elif source.startswith('data:'):
                return "data_uri"
            elif self._is_base64(source):
                return "base64"
            else:
                return "file_path"
        elif isinstance(source, (bytes, BytesIO)):
            return "bytes"
        elif isinstance(source, Path):
            return "file_path"
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def _is_base64(self, s: str) -> bool:
        """Improved base64 detection with proper validation."""
        if len(s) < 4 or len(s) % 4 != 0:
            return False
        
        # Check for valid base64 characters
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        if not base64_pattern.match(s):
            return False
            
        try:
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False
    
    def _detect_media_type(self) -> Optional[str]:
        """Enhanced media type detection."""
        if self.source_type == "file_path":
            path = Path(self.source) if isinstance(self.source, str) else self.source
            mime_type, _ = mimetypes.guess_type(str(path))
            return mime_type
        elif self.source_type == "data_uri":
            header = self.source.split(",", 1)[0]
            return header.split(";")[0].replace("data:", "")
        elif self.source_type == "gcs_uri":
            # Extract from GCS URI path extension
            path = Path(self.source)
            mime_type, _ = mimetypes.guess_type(str(path))
            return mime_type
        
        return None
    
    
    def to_bytes(self) -> bytes:
        """Convert source to bytes with caching."""
        if self._cached_bytes is not None:
            return self._cached_bytes
            
        if self.source_type == "bytes":
            if isinstance(self.source, bytes):
                self._cached_bytes = self.source
            else:  # BytesIO
                self._cached_bytes = self.source.getvalue()
        elif self.source_type == "base64":
            self._cached_bytes = base64.b64decode(self.source)
        elif self.source_type == "data_uri":
            header, data = self.source.split(',', 1)
            self._cached_bytes = base64.b64decode(data)
        elif self.source_type in ["file_path", "gcs_uri"]:
            path = Path(self.source) if isinstance(self.source, str) else self.source
            if self.source_type == "gcs_uri":
                raise ValueError("GCS URIs must be downloaded first. Use provider-specific handling.")
            with open(path, 'rb') as f:
                self._cached_bytes = f.read()
        elif self.source_type == "url":
            raise ValueError("URL content must be downloaded first. Call download() before to_bytes()")
        else:
            raise ValueError(f"Cannot convert {self.source_type} to bytes")
            
        return self._cached_bytes
    
    def download(self, session=None) -> 'MultimodalChunk':
        """Download content from URL with improved error handling."""
        if self.source_type not in ["url", "gcs_uri"]:
            return self
        
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests library required for URL downloading. "
                "Install with: pip install requests"
            )
        
        try:
            if session is None:
                response = requests.get(self.source, timeout=30)
            else:
                response = session.get(self.source, timeout=30)
            
            response.raise_for_status()
            
            # Get media type from response headers if not provided
            media_type = self.media_type
            if not media_type:
                content_type = response.headers.get('content-type', '').split(';')[0]
                if content_type:
                    media_type = content_type
            
            return MultimodalChunk(
                source=response.content,
                media_type=media_type,
                source_type="bytes",
                filename=self.filename or Path(self.source).name
            )
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to download from {self.source}: {str(e)}")
    
    def to_base64(self) -> str:
        """Convert source to base64 string."""
        if self.source_type == "base64":
            return self.source
        elif self.source_type == "data_uri":
            header, data = self.source.split(',', 1)
            return data
        else:
            return base64.b64encode(self.to_bytes()).decode('utf-8')
    
    def to_data_uri(self) -> str:
        """Convert to data URI format."""
        if self.source_type == "data_uri":
            return self.source
        base64_data = self.to_base64()
        media_type = self.media_type or "application/octet-stream"
        return f"data:{media_type};base64,{base64_data}"
    
    def get_file_path(self) -> Optional[Path]:
        """Get file path if source is a file."""
        if self.source_type == "file_path":
            return Path(self.source) if isinstance(self.source, str) else self.source
        return None
    
    def get_url(self) -> Optional[str]:
        """Get URL if source is a URL."""
        if self.source_type in ["url", "gcs_uri"]:
            return self.source
        return None
    
    # Static Factory Methods (enhanced with validation)
    
    @classmethod
    def from_url(cls, url: str, media_type: Optional[str] = None) -> 'MultimodalChunk':
        """Create a MultimodalChunk from a URL with validation."""
        if not url.startswith(('http://', 'https://', 'gs://')):
            raise ValueError(f"Invalid URL format: {url}")
        return cls(source=url, media_type=media_type)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'MultimodalChunk':
        """Create a MultimodalChunk from a file path with existence check."""
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return cls(source=path, filename=path.name)
    
    @classmethod
    def from_bytes(cls, data: bytes, media_type: Optional[str] = None, filename: Optional[str] = None) -> 'MultimodalChunk':
        """Create a MultimodalChunk from raw bytes."""
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        return cls(source=data, media_type=media_type, filename=filename)
    
    @classmethod
    def from_base64(cls, data: str, media_type: Optional[str] = None) -> 'MultimodalChunk':
        """Create a MultimodalChunk from base64-encoded string with validation."""
        chunk = cls(source=data, media_type=media_type, source_type="base64")
        if not chunk._is_base64(data):
            raise ValueError("Invalid base64 data")
        return chunk
    
    @classmethod
    def from_data_uri(cls, data_uri: str) -> 'MultimodalChunk':
        """Create a MultimodalChunk from a data URI with validation."""
        if not data_uri.startswith("data:"):
            raise ValueError("Invalid data URI format - must start with 'data:'")
        
        if ',' not in data_uri:
            raise ValueError("Invalid data URI format - missing comma separator")
        
        return cls(source=data_uri, source_type="data_uri")
    
    @classmethod
    def from_file_object(cls, file_obj, media_type: Optional[str] = None, filename: Optional[str] = None) -> 'MultimodalChunk':
        """Create a MultimodalChunk from an open file object."""
        if not hasattr(file_obj, 'read'):
            raise TypeError("Object must have a 'read' method")
        
        data = file_obj.read()
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        
        return cls(
            source=BytesIO(data), 
            media_type=media_type, 
            filename=filename or getattr(file_obj, 'name', None)
        )
    
    def __repr__(self):
        source_preview = str(self.source)[:50] + "..." if len(str(self.source)) > 50 else str(self.source)
        return f'MultimodalChunk(type="{self.media_type}", source_type="{self.source_type}", source="{source_preview}")'
```

### 2. Updated Content Type Union

Update `/patterpunk/src/patterpunk/llm/types.py`:

```python
# Add after existing imports
from .multimodal import MultimodalChunk

# Content can be a string, or a list of chunks (text/cache/multimodal)
ContentType = Union[str, List[Union[CacheChunk, MultimodalChunk]]]
```

## Message Class Updates

### 1. Update Base Message Class

Modify `/patterpunk/src/patterpunk/llm/messages/base.py`:

```python
# Update imports
from typing import Union, List, Optional, Any
from ..cache import CacheChunk
from ..multimodal import MultimodalChunk
from ..types import ContentType

class Message:
    def __init__(self, content: ContentType, role: str = ROLE_USER):
        self.content = content
        # ... rest remains the same
```

### 2. Update UserMessage Class

Modify `/patterpunk/src/patterpunk/llm/messages/user.py`:

```python
# Update imports and type hint
from ..types import ContentType

class UserMessage(Message):
    def __init__(
        self, 
        content: ContentType,
        structured_output: Optional[Any] = None, 
        allow_tool_calls: bool = True
    ):
        super().__init__(content, ROLE_USER)
        # ... rest remains the same
```

### 3. Enhanced Cache Helper Module

Modify `/patterpunk/src/patterpunk/llm/messages/cache.py`:

```python
from typing import Union, List
from ..cache import CacheChunk
from ..multimodal import MultimodalChunk

def get_content_as_string(content: Union[str, List[Union[CacheChunk, MultimodalChunk]]]) -> str:
    """Get content as string, handling both cache and multimodal chunks."""
    if isinstance(content, str):
        return content
    
    # Only include text content from chunks
    text_parts = []
    for chunk in content:
        if isinstance(chunk, CacheChunk):
            text_parts.append(chunk.content)
        # MultimodalChunk doesn't contribute to text representation
    
    return "".join(text_parts)

def get_content_chunks(content: Union[str, List[Union[CacheChunk, MultimodalChunk]]]) -> List[Union[CacheChunk, MultimodalChunk]]:
    """Get content as chunks, converting string to CacheChunk if needed."""
    if isinstance(content, str):
        return [CacheChunk(content, cacheable=False)]
    return content

def has_multimodal_content(content: Union[str, List[Union[CacheChunk, MultimodalChunk]]]) -> bool:
    """Check if content contains any multimodal chunks."""
    if isinstance(content, str):
        return False
    return any(isinstance(chunk, MultimodalChunk) for chunk in content)

def get_multimodal_chunks(content: Union[str, List[Union[CacheChunk, MultimodalChunk]]]) -> List[MultimodalChunk]:
    """Extract only multimodal chunks from content."""
    if isinstance(content, str):
        return []
    return [chunk for chunk in content if isinstance(chunk, MultimodalChunk)]

```

## Provider Model Updates

### 1. OpenAI Model Updates (RESPONSES API)

Modify `/patterpunk/src/patterpunk/llm/models/openai.py`:

```python
def _convert_message_content_for_openai_responses(self, content: ContentType) -> List[dict]:
    """
    Convert message content to OpenAI Responses API format.
    
    CRITICAL: Uses Responses API format (March 2025) with input_* content types,
    NOT the legacy Chat Completions API format.
    
    Provider handles validation - let OpenAI API return clear error messages for invalid content.
    """
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    
    openai_content = []
    session = None  # For URL downloading
    
    for chunk in content:
        if isinstance(chunk, CacheChunk):
            openai_content.append({
                "type": "input_text",
                "text": chunk.content
            })
        elif isinstance(chunk, MultimodalChunk):
            if chunk.media_type and chunk.media_type.startswith("image/"):
                # Handle image content
                if chunk.source_type == "url":
                    openai_content.append({
                        "type": "input_image",
                        "image_url": chunk.get_url()
                    })
                else:
                    # Convert to data URI for base64/file/bytes
                    openai_content.append({
                        "type": "input_image",
                        "image_url": chunk.to_data_uri()
                    })
            elif chunk.media_type == "application/pdf":
                # Handle PDF files - OpenAI supports three methods
                if chunk.source_type == "url":
                    openai_content.append({
                        "type": "input_file",
                        "file_url": chunk.get_url()
                    })
                elif hasattr(chunk, 'file_id'):  # From Files API upload
                    openai_content.append({
                        "type": "input_file", 
                        "file_id": chunk.file_id
                    })
                else:
                    # Convert to base64 data
                    openai_content.append({
                        "type": "input_file",
                        "filename": chunk.filename or "document.pdf",
                        "file_data": chunk.to_data_uri()
                    })
            else:
                # Try to handle as generic file
                openai_content.append({
                    "type": "input_file",
                    "filename": chunk.filename or "file",
                    "file_data": chunk.to_data_uri()
                })
    
    return openai_content

def _prepare_messages_for_openai_responses(self, messages: List[Message]) -> dict:
    """Prepare messages for OpenAI Responses API (not Chat Completions)."""
    input_messages = []
    
    for message in messages:
        if message.role in ['user', 'assistant']:
            input_messages.append({
                "role": message.role,
                "content": self._convert_message_content_for_openai_responses(message.content)
            })
    
    return {"input": input_messages}  # Note: 'input' not 'messages'

def complete(self, messages: List[Message], **kwargs) -> AssistantMessage:
    """Complete using OpenAI Responses API."""
    request_data = self._prepare_messages_for_openai_responses(messages)
    request_data.update(self._get_completion_params(**kwargs))
    
    try:
        # Use Responses API, not Chat Completions
        response = openai.responses.create(**request_data)
        
        content = response.choices[0].message.content
        return AssistantMessage(content)
        
    except Exception as e:
        logger.error(f"OpenAI Responses API error: {e}")
        raise OpenAiApiError(f"API call failed: {e}")
```

### 2. Anthropic Model Updates (WITH FILES API)

Modify `/patterpunk/src/patterpunk/llm/models/anthropic.py`:

```python
def _convert_content_to_anthropic_format(self, content: ContentType) -> List[dict]:
    """
    Convert content to Anthropic format with cache controls, multimodal, and Files API support.
    
    Provider handles validation - let Anthropic API return clear error messages for invalid content.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    
    anthropic_content = []
    session = None
    
    for chunk in content:
        if isinstance(chunk, CacheChunk):
            content_block = {
                "type": "text",
                "text": chunk.content
            }
            
            if chunk.cacheable:
                cache_control = {"type": "ephemeral"}
                if chunk.ttl:
                    cache_control["ttl"] = int(chunk.ttl.total_seconds())
                content_block["cache_control"] = cache_control
            
            anthropic_content.append(content_block)
            
        elif isinstance(chunk, MultimodalChunk):
            # Check if this is a Files API reference
            if hasattr(chunk, 'file_id'):
                # Use Files API reference for documents
                content_block = {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": chunk.file_id
                    }
                }
                anthropic_content.append(content_block)
                continue
            
            # Handle URL downloading (Anthropic doesn't support URLs directly)
            if chunk.source_type == "url":
                if session is None:
                    try:
                        import requests
                        session = requests.Session()
                    except ImportError:
                        raise ImportError("requests library required for URL support with Anthropic")
                
                chunk = chunk.download(session)
            
            media_type = chunk.media_type or "application/octet-stream"
            
            if media_type.startswith("image/"):
                content_block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": chunk.to_base64()
                    }
                }
                anthropic_content.append(content_block)
            elif media_type == "application/pdf":
                # PDF as document type
                content_block = {
                    "type": "document", 
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": chunk.to_base64()
                    }
                }
                anthropic_content.append(content_block)
    
    return anthropic_content

def upload_file_to_anthropic(self, chunk: MultimodalChunk) -> str:
    """Upload file to Anthropic Files API and return file_id."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Create temporary file for upload
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(
            suffix=f".{chunk.media_type.split('/')[-1] if chunk.media_type else 'bin'}", 
            delete=False
        ) as tmp_file:
            tmp_file.write(chunk.to_bytes())
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as f:
                file_response = client.files.create(
                    file=f,
                    purpose="vision"
                )
            
            return file_response.id
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Failed to upload file to Anthropic: {e}")
        raise
```

### 3. Google Model Updates

Modify `/patterpunk/src/patterpunk/llm/models/google.py`:

```python
def _convert_message_to_google_format(self, message: Message) -> dict:
    """
    Convert message to Google format with corrected SDK usage.
    
    CRITICAL: Uses google-genai SDK (not deprecated vertexai SDK)
    with proper client setup and Part factory methods.
    
    Provider handles validation - let Google API return clear error messages for invalid content.
    """
    role = "user" if message.role == "user" else "model"
    
    if isinstance(message.content, str):
        return {
            "role": role,
            "parts": [{"text": message.content}]
        }
    
    # Import corrected SDK components
    try:
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai SDK required. Install with: pip install google-genai==1.24.0")
    
    parts = []
    
    for chunk in message.content:
        if isinstance(chunk, CacheChunk):
            parts.append(types.Part.from_text(chunk.content))
        elif isinstance(chunk, MultimodalChunk):
            if chunk.source_type == "gcs_uri":
                # Google Cloud Storage URI - native support
                parts.append(types.Part.from_uri(
                    uri=chunk.get_url(),
                    mime_type=chunk.media_type or "application/octet-stream"
                ))
            elif hasattr(chunk, 'file_id'):
                # Files API upload reference
                parts.append(chunk.file_id)  # Direct file object reference
            else:
                # Convert to bytes for inline data
                media_type = chunk.media_type or "application/octet-stream"
                
                # Download URLs first
                if chunk.source_type == "url":
                    chunk = chunk.download()
                
                parts.append(types.Part.from_bytes(
                    data=chunk.to_bytes(),
                    mime_type=media_type
                ))
    
    return {"role": role, "parts": parts}

def _setup_google_client(self):
    """Setup Google GenAI client with proper configuration."""
    try:
        from google import genai
        
        # Vertex AI setup (corrected pattern)
        client = genai.Client(
            vertexai=True, 
            project=self.project_id, 
            location=self.location or 'us-central1'
        )
        return client
    except ImportError:
        raise ImportError("google-genai SDK required. Install with: pip install google-genai==1.24.0")

def upload_file_to_google(self, chunk: MultimodalChunk) -> Any:
    """Upload file to Google Files API."""
    client = self._setup_google_client()
    
    # Create temporary file for upload
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(
        suffix=f".{chunk.filename.split('.')[-1] if chunk.filename else 'bin'}", 
        delete=False
    ) as tmp_file:
        tmp_file.write(chunk.to_bytes())
        tmp_file_path = tmp_file.name
    
    try:
        uploaded_file = client.files.upload(file=tmp_file_path)
        return uploaded_file
    finally:
        os.unlink(tmp_file_path)

def complete(self, messages: List[Message], **kwargs) -> AssistantMessage:
    """Complete using corrected Google GenAI SDK."""
    client = self._setup_google_client()
    
    # Convert messages to Google format
    google_messages = []
    for message in messages:
        google_messages.append(self._convert_message_to_google_format(message))
    
    try:
        # Use corrected API call
        response = client.models.generate_content(
            model=self.model,
            contents=google_messages,
            **self._get_completion_params(**kwargs)
        )
        
        return AssistantMessage(response.text)
        
    except Exception as e:
        logger.error(f"Google GenAI API error: {e}")
        raise
```

### 4. Bedrock Model Updates

Modify `/patterpunk/src/patterpunk/llm/models/bedrock.py`:

```python
def _convert_content_to_bedrock_format(self, content: ContentType) -> List[dict]:
    """
    Convert content to Bedrock format.
    
    Provider handles validation - let Bedrock API return clear error messages for invalid content.
    """
    if isinstance(content, str):
        return [{"text": content}]
    
    bedrock_content = []
    session = None
    
    for chunk in content:
        if isinstance(chunk, CacheChunk):
            content_block = {"text": chunk.content}
            if chunk.cacheable:
                content_block["cachePoint"] = {}
            bedrock_content.append(content_block)
            
        elif isinstance(chunk, MultimodalChunk):
            # Bedrock requires bytes, download URLs if needed
            if chunk.source_type == "url":
                if session is None:
                    try:
                        import requests
                        session = requests.Session()
                    except ImportError:
                        raise ImportError("requests library required for URL support with Bedrock")
                
                chunk = chunk.download(session)
            
            media_type = chunk.media_type or "application/octet-stream"
            
            if media_type.startswith("image/"):
                # Enhanced format mapping for Bedrock
                format_map = {
                    "image/jpeg": "jpeg",
                    "image/jpg": "jpeg", 
                    "image/png": "png",
                    "image/gif": "gif",
                    "image/webp": "webp"
                }
                
                format = format_map.get(media_type, "jpeg")
                
                content_block = {
                    "image": {
                        "format": format,
                        "source": {
                            "bytes": chunk.to_bytes()
                        }
                    }
                }
                bedrock_content.append(content_block)
            else:
                # Bedrock primarily supports images
                logger.warning(f"Bedrock may not support media type: {media_type}")
    
    return bedrock_content
```

### 5. Ollama Model Updates

Modify `/patterpunk/src/patterpunk/llm/models/ollama.py`:

```python
def _prepare_messages_for_ollama(self, messages: List[Message]) -> tuple[List[dict], List[str]]:
    """
    Prepare messages for Ollama API with CORRECTED architecture.
    
    CRITICAL: Ollama uses separate 'images' array, NOT content array embedding.
    This is fundamentally different from other providers.
    
    Provider handles validation - let Ollama API return clear error messages for invalid content.
    """
    import tempfile
    import os
    
    ollama_messages = []
    all_images = []
    session = None
    temp_files = []  # Track temp files for cleanup
    
    try:
        for message in messages:
            # Get text content only
            if isinstance(message.content, str):
                content_text = message.content
            else:
                # Extract only text chunks for content field
                text_parts = []
                for chunk in message.content:
                    if isinstance(chunk, CacheChunk):
                        text_parts.append(chunk.content)
                content_text = "".join(text_parts)
            
            # Extract images separately - Ollama's unique architecture
            message_images = []
            if isinstance(message.content, list):
                for chunk in message.content:
                    if isinstance(chunk, MultimodalChunk) and chunk.media_type and chunk.media_type.startswith("image/"):
                        
                        if chunk.source_type == "file_path":
                            # Direct file path - Ollama's preferred method
                            message_images.append(str(chunk.get_file_path()))
                        else:
                            # Convert other sources to temp files (Ollama requirement)
                            if chunk.source_type == "url":
                                if session is None:
                                    try:
                                        import requests
                                        session = requests.Session()
                                    except ImportError:
                                        raise ImportError("requests library required for URL support")
                                
                                chunk = chunk.download(session)
                            
                            # Save to temporary file with proper extension
                            media_type = chunk.media_type or "image/jpeg"
                            suffix = self._get_file_extension(media_type)
                            
                            with tempfile.NamedTemporaryFile(
                                suffix=suffix, 
                                delete=False
                            ) as tmp_file:
                                tmp_file.write(chunk.to_bytes())
                                temp_files.append(tmp_file.name)
                                message_images.append(tmp_file.name)
            
            # Create Ollama message with separate images array
            ollama_message = {
                "role": message.role,
                "content": content_text
            }
            
            # Add images to THIS message (not global array)
            if message_images:
                ollama_message["images"] = message_images
            
            ollama_messages.append(ollama_message)
            all_images.extend(message_images)
        
        # Store temp files for cleanup after completion
        self._temp_files = temp_files
        
        return ollama_messages, all_images
    
    except Exception:
        # Clean up temp files if error occurs
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise

def complete(self, messages: List[Message], **kwargs) -> AssistantMessage:
    """Complete using Ollama with corrected message format."""
    ollama_messages, all_images = self._prepare_messages_for_ollama(messages)
    
    try:
        import ollama
        
        # Use prepared messages with embedded images arrays
        response = ollama.chat(
            model=self.model,
            messages=ollama_messages,  # Images already embedded per message
            **self._get_completion_params(**kwargs)
        )
        
        content = response['message']['content']
        return AssistantMessage(content)
        
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise
    finally:
        # Clean up temp files
        if hasattr(self, '_temp_files'):
            for temp_file in self._temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            del self._temp_files

def _get_file_extension(self, media_type: str) -> str:
    """Get file extension from media type."""
    extension_map = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png", 
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff"
    }
    return extension_map.get(media_type, ".jpg")
```

## Provider Architecture

Each provider model handles its own capabilities and validation internally, following the core design principle that all provider-specific logic must be isolated within provider model files.

**Provider Responsibilities:**
- Format validation (image types, file sizes, etc.)
- URL downloading if not natively supported  
- Files API integration where available
- Conversion to provider-specific API formats
- Clear error message handling from their APIs

**Error Handling:**
When multimodal content cannot be processed, the provider's API will return clear error messages that are passed through to the user. This eliminates the need for preemptive validation and ensures users receive accurate, up-to-date information about provider limitations.
