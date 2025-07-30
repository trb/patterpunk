import base64
import http.server
import socketserver
import threading
import time
from pathlib import Path

import pytest

from patterpunk.llm.multimodal import MultimodalChunk
from tests.test_utils import get_resource


def test_file_object_to_base64():
    with open(get_resource('research.pdf'), 'rb') as f:
        first_bytes = f.read(15)
        f.seek(0)
        encoded = MultimodalChunk.from_file_object(f).to_base64()
        assert encoded[:15] == 'JVBERi0xLjcNJeL'

        decoded = base64.b64decode(encoded)
        assert decoded[:15] == first_bytes, (
            'After decoding, MultimodalChunk base64 encoded file object does not match the first 15 bytes of the original file.'
        )


def test_bytes():
    multimodal_chunk = MultimodalChunk.from_file(get_resource('research.pdf'))
    with open(get_resource('research.pdf'), 'rb') as f:
        assert multimodal_chunk.to_bytes() == f.read(), 'Bytes from MultimodalChunk does not match the original file.'


@pytest.fixture
def http_server():
    resources_dir = Path(get_resource(''))
    port = 0
    
    class ResourceHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(resources_dir), **kwargs)
    
    with socketserver.TCPServer(("", port), ResourceHandler) as httpd:
        actual_port = httpd.server_address[1]
        server_url = f"http://localhost:{actual_port}"
        
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        
        time.sleep(0.1)
        
        yield server_url
        
        httpd.shutdown()
        server_thread.join(timeout=1)


def test_download_from_url_pdf(http_server):
    pdf_url = f"{http_server}/research.pdf"
    
    chunk = MultimodalChunk.from_url(pdf_url)
    assert chunk.source_type == "url"
    assert chunk.get_url() == pdf_url
    
    downloaded_chunk = chunk.download()
    
    assert downloaded_chunk.source_type == "bytes"
    
    with open(get_resource('research.pdf'), 'rb') as f:
        original_content = f.read()
    
    assert downloaded_chunk.to_bytes() == original_content
    assert downloaded_chunk.media_type == "application/pdf"
    assert downloaded_chunk.filename == "research.pdf"


def test_download_from_url_image(http_server):
    image_url = f"{http_server}/ducks_pond.jpg"
    
    chunk = MultimodalChunk.from_url(image_url).download()
    
    with open(get_resource('ducks_pond.jpg'), 'rb') as f:
        original_content = f.read()
    
    assert chunk.to_bytes() == original_content
    assert chunk.media_type == "image/jpeg"
    assert chunk.filename == "ducks_pond.jpg"
    
    base64_data = chunk.to_base64()
    assert len(base64_data) > 0
    assert base64.b64decode(base64_data) == original_content


def test_download_nonexistent_file(http_server):
    nonexistent_url = f"{http_server}/nonexistent.txt"
    chunk = MultimodalChunk.from_url(nonexistent_url)
    
    with pytest.raises(ValueError, match="Failed to download"):
        chunk.download()


def test_download_invalid_url():
    chunk = MultimodalChunk.from_url("http://invalid-host-that-does-not-exist.com/file.txt")
    
    with pytest.raises(ValueError, match="Failed to download"):
        chunk.download()


def test_download_url_without_requests():
    chunk = MultimodalChunk.from_url("http://example.com/file.txt")
    
    import sys
    original_modules = sys.modules.copy()
    
    try:
        if 'requests' in sys.modules:
            del sys.modules['requests']
        
        import unittest.mock
        with unittest.mock.patch.dict('sys.modules', {'requests': None}):
            with pytest.raises(ImportError, match="requests library required"):
                chunk.download()
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


@pytest.fixture
def custom_content_type_server():
    resources_dir = Path(get_resource(''))
    port = 0
    
    class CustomContentTypeHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(resources_dir), **kwargs)
        
        def guess_type(self, path):
            if path.endswith('/research.pdf'):
                return 'application/custom-pdf'
            elif path.endswith('/ducks_pond.jpg'):
                return 'image/custom-jpeg'
            return super().guess_type(path)
    
    with socketserver.TCPServer(("", port), CustomContentTypeHandler) as httpd:
        actual_port = httpd.server_address[1]
        server_url = f"http://localhost:{actual_port}"
        
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.1)
        
        yield server_url
        
        httpd.shutdown()
        server_thread.join(timeout=1)


def test_content_type_from_http_headers(custom_content_type_server):
    pdf_url = f"{custom_content_type_server}/research.pdf"
    chunk = MultimodalChunk.from_url(pdf_url)
    
    assert chunk.media_type is None
    
    downloaded_chunk = chunk.download()
    assert downloaded_chunk.media_type == "application/custom-pdf"
    assert downloaded_chunk.filename == "research.pdf"
    
    image_url = f"{custom_content_type_server}/ducks_pond.jpg"
    downloaded_image = MultimodalChunk.from_url(image_url).download()
    assert downloaded_image.media_type == "image/custom-jpeg"
    assert downloaded_image.filename == "ducks_pond.jpg"


def test_content_type_fallback_when_no_header(http_server):
    pdf_url = f"{http_server}/research.pdf"
    
    chunk = MultimodalChunk(source=pdf_url, media_type=None)
    downloaded_chunk = chunk.download()
    
    assert downloaded_chunk.media_type == "application/pdf"
