"""
Multimodal content handling for LLM interactions.

This module provides the MultimodalChunk class for handling different types
of media content (images, files, audio, video) with a unified interface
across different LLM providers.
"""

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
        filename: Optional[str] = None,
    ):
        self.source = source
        self.source_type = source_type or self._infer_source_type(source)
        self.media_type = media_type or self._detect_media_type()
        self.filename = filename
        self._cached_bytes: Optional[bytes] = None

    def _infer_source_type(self, source) -> str:
        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                return "url"
            elif source.startswith("gs://"):
                return "gcs_uri"
            elif source.startswith("data:"):
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
        if len(s) < 4 or len(s) % 4 != 0:
            return False

        base64_pattern = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
        if not base64_pattern.match(s):
            return False

        try:
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False

    def _detect_media_type(self) -> Optional[str]:
        if self.source_type == "file_path":
            path = Path(self.source) if isinstance(self.source, str) else self.source
            mime_type, _ = mimetypes.guess_type(str(path))
            return mime_type
        elif self.source_type == "data_uri":
            header = self.source.split(",", 1)[0]
            return header.split(";")[0].replace("data:", "")
        elif self.source_type == "gcs_uri":
            path = Path(self.source)
            mime_type, _ = mimetypes.guess_type(str(path))
            return mime_type

        return None

    def to_bytes(self) -> bytes:
        if self._cached_bytes is not None:
            return self._cached_bytes

        if self.source_type == "bytes":
            if isinstance(self.source, bytes):
                self._cached_bytes = self.source
            else:
                self._cached_bytes = self.source.getvalue()
        elif self.source_type == "base64":
            self._cached_bytes = base64.b64decode(self.source)
        elif self.source_type == "data_uri":
            header, data = self.source.split(",", 1)
            self._cached_bytes = base64.b64decode(data)
        elif self.source_type in ["file_path", "gcs_uri"]:
            path = Path(self.source) if isinstance(self.source, str) else self.source
            if self.source_type == "gcs_uri":
                raise ValueError(
                    "GCS URIs must be downloaded first. Use provider-specific handling."
                )
            with open(path, "rb") as f:
                self._cached_bytes = f.read()
        elif self.source_type == "url":
            raise ValueError(
                "URL content must be downloaded first. Call download() before to_bytes()"
            )
        else:
            raise ValueError(f"Cannot convert {self.source_type} to bytes")

        return self._cached_bytes

    def download(self, session=None) -> "MultimodalChunk":
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

            media_type = self.media_type
            if not media_type:
                content_type = response.headers.get("content-type", "").split(";")[0]
                if content_type:
                    media_type = content_type

            return MultimodalChunk(
                source=response.content,
                media_type=media_type,
                source_type="bytes",
                filename=self.filename or Path(self.source).name,
            )

        except requests.RequestException as e:
            raise ValueError(f"Failed to download from {self.source}: {str(e)}")

    def to_base64(self) -> str:
        if self.source_type == "base64":
            return self.source
        elif self.source_type == "data_uri":
            header, data = self.source.split(",", 1)
            return data
        else:
            return base64.b64encode(self.to_bytes()).decode("utf-8")

    def to_data_uri(self) -> str:
        if self.source_type == "data_uri":
            return self.source
        base64_data = self.to_base64()
        media_type = self.media_type or "application/octet-stream"
        return f"data:{media_type};base64,{base64_data}"

    def get_file_path(self) -> Optional[Path]:
        if self.source_type == "file_path":
            return Path(self.source) if isinstance(self.source, str) else self.source
        return None

    def get_url(self) -> Optional[str]:
        if self.source_type in ["url", "gcs_uri"]:
            return self.source
        return None

    @classmethod
    def from_url(cls, url: str, media_type: Optional[str] = None) -> "MultimodalChunk":
        if not url.startswith(("http://", "https://", "gs://")):
            raise ValueError(f"Invalid URL format: {url}")
        return cls(source=url, media_type=media_type)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "MultimodalChunk":
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return cls(source=path, filename=path.name)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        media_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> "MultimodalChunk":
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        return cls(source=data, media_type=media_type, filename=filename)

    @classmethod
    def from_base64(
        cls, data: str, media_type: Optional[str] = None
    ) -> "MultimodalChunk":
        chunk = cls(source=data, media_type=media_type, source_type="base64")
        if not chunk._is_base64(data):
            raise ValueError("Invalid base64 data")
        return chunk

    @classmethod
    def from_data_uri(cls, data_uri: str) -> "MultimodalChunk":
        if not data_uri.startswith("data:"):
            raise ValueError("Invalid data URI format - must start with 'data:'")

        if "," not in data_uri:
            raise ValueError("Invalid data URI format - missing comma separator")

        return cls(source=data_uri, source_type="data_uri")

    @classmethod
    def from_file_object(
        cls, file_obj, media_type: Optional[str] = None, filename: Optional[str] = None
    ) -> "MultimodalChunk":
        if not hasattr(file_obj, "read"):
            raise TypeError("Object must have a 'read' method")

        data = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        return cls(
            source=BytesIO(data),
            media_type=media_type,
            filename=filename or getattr(file_obj, "name", None),
        )

    def __repr__(self):
        source_preview = (
            str(self.source)[:50] + "..."
            if len(str(self.source)) > 50
            else str(self.source)
        )
        return f'MultimodalChunk(type="{self.media_type}", source_type="{self.source_type}", source="{source_preview}")'
