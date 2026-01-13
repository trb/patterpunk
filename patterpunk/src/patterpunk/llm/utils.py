"""
Utility functions for LLM operations.

This module contains helper functions that support the LLM functionality
without fitting into more specific modules.
"""

import io
import struct
from typing import Tuple


def get_image_dimensions(data: bytes) -> Tuple[int, int]:
    """
    Extract image dimensions from raw bytes by parsing headers.

    Uses only Python's standard library (struct module) - no external
    dependencies like Pillow needed. Parses the file header to extract
    dimensions without loading the full image into memory.

    Supports: PNG, JPEG, GIF, BMP, WebP

    Args:
        data: Raw image bytes

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If image format is unsupported or corrupt
    """
    size = len(data)

    # PNG: Magic bytes \x89PNG\r\n\x1a\n, dimensions in IHDR chunk
    if size >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        if data[12:16] == b"IHDR":
            w, h = struct.unpack(">II", data[16:24])
        else:
            # Older PNG without IHDR label at expected position
            w, h = struct.unpack(">II", data[8:16])
        return int(w), int(h)

    # GIF: Magic bytes GIF87a or GIF89a
    if size >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return int(w), int(h)

    # JPEG: Magic bytes \xff\xd8, scan for SOF marker
    if size >= 2 and data[:2] == b"\xff\xd8":
        return _parse_jpeg_dimensions(data)

    # BMP: Magic bytes BM
    if size >= 26 and data[:2] == b"BM":
        header_size = struct.unpack("<I", data[14:18])[0]
        if header_size == 12:  # OS/2 v1
            w, h = struct.unpack("<HH", data[18:22])
        else:  # Windows v3+
            w, h = struct.unpack("<ii", data[18:26])
            h = abs(h)  # Height can be negative (top-down bitmap)
        return int(w), int(h)

    # WebP: RIFF....WEBP
    if size >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _parse_webp_dimensions(data)

    raise ValueError(f"Unsupported image format (first bytes: {data[:8].hex()})")


# JPEG marker constants
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xC4))  # SOF0-SOF3 contain dimensions
_JPEG_SOS_MARKER = 0xDA  # Start of Scan - no more segments after this
_JPEG_STANDALONE_MARKERS = frozenset(
    [0x00, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9]
)


def _find_next_jpeg_marker(stream: io.BytesIO) -> int | None:
    """
    Find the next JPEG marker in the stream.

    Skips non-marker bytes and 0xFF padding bytes to find the actual marker.
    Returns the marker byte (0x00-0xFF) or None if end of stream.
    """
    b = stream.read(1)

    # Skip until we find 0xFF (marker prefix)
    while b and ord(b) != 0xFF:
        b = stream.read(1)

    if not b:
        return None

    # Skip any padding 0xFF bytes
    while b and ord(b) == 0xFF:
        b = stream.read(1)

    if not b:
        return None

    return ord(b)


def _read_jpeg_sof_dimensions(stream: io.BytesIO) -> Tuple[int, int] | None:
    """
    Read dimensions from a JPEG SOF (Start of Frame) marker.

    Assumes stream is positioned right after the SOF marker byte.
    Returns (width, height) tuple or None if read fails.
    """
    stream.read(3)  # Skip length (2 bytes) and precision (1 byte)
    height_bytes = stream.read(2)
    width_bytes = stream.read(2)

    if len(height_bytes) != 2 or len(width_bytes) != 2:
        return None

    h = struct.unpack(">H", height_bytes)[0]
    w = struct.unpack(">H", width_bytes)[0]
    return int(w), int(h)


def _skip_jpeg_segment(stream: io.BytesIO) -> bool:
    """
    Skip a JPEG segment by reading its length and advancing the stream.

    Returns True if skip succeeded, False if read failed.
    """
    length_bytes = stream.read(2)
    if len(length_bytes) != 2:
        return False

    length = struct.unpack(">H", length_bytes)[0]
    stream.read(length - 2)  # Length includes the 2 length bytes
    return True


def _parse_jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    """
    Parse JPEG dimensions by scanning for SOF (Start of Frame) markers.

    JPEG files contain multiple segments. We scan for SOF0-SOF3 markers
    (0xC0-0xC3) which contain the image dimensions.

    Algorithm:
    1. Skip SOI marker
    2. Find next marker (skipping 0xFF padding)
    3. If SOF marker: read and return dimensions
    4. If SOS marker: stop (image data follows, no more metadata)
    5. Otherwise: skip segment and continue
    """
    stream = io.BytesIO(data)
    stream.read(2)  # Skip SOI (Start of Image) marker

    while True:
        marker = _find_next_jpeg_marker(stream)
        if marker is None:
            break

        if marker in _JPEG_SOF_MARKERS:
            dimensions = _read_jpeg_sof_dimensions(stream)
            if dimensions:
                return dimensions
            break

        if marker == _JPEG_SOS_MARKER:
            break  # Start of Scan - no dimensions found before image data

        # Skip variable-length segments (but not standalone markers)
        if marker not in _JPEG_STANDALONE_MARKERS:
            if not _skip_jpeg_segment(stream):
                break

    raise ValueError("Could not find JPEG dimensions - invalid or corrupt file")


def _parse_webp_dimensions(data: bytes) -> Tuple[int, int]:
    """
    Parse WebP dimensions.

    WebP has three variants:
    - VP8 (lossy): Simple format
    - VP8L (lossless): Different dimension encoding
    - VP8X (extended): Canvas dimensions in header
    """
    chunk_type = data[12:16]

    if chunk_type == b"VP8 ":
        # Lossy WebP: dimensions at bytes 26-29 (14-bit each)
        if len(data) < 30:
            raise ValueError("VP8 WebP file too short")
        w = (data[26] | (data[27] << 8)) & 0x3FFF
        h = (data[28] | (data[29] << 8)) & 0x3FFF
        return int(w), int(h)

    elif chunk_type == b"VP8L":
        # Lossless WebP: dimensions packed in bits 21-25
        if len(data) < 25:
            raise ValueError("VP8L WebP file too short")
        bits = struct.unpack("<I", data[21:25])[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return int(w), int(h)

    elif chunk_type == b"VP8X":
        # Extended WebP: canvas size at bytes 24-29
        if len(data) < 30:
            raise ValueError("VP8X WebP file too short")
        # Width and height are stored as 24-bit values minus 1
        w = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
        h = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
        return int(w), int(h)

    raise ValueError(f"Unknown WebP variant: {chunk_type}")
