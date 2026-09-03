import json
import os
from typing import Optional

try:
    from anthropic import AnthropicVertex, AsyncAnthropicVertex
except ImportError:
    AnthropicVertex = None
    AsyncAnthropicVertex = None

try:
    from google.oauth2 import service_account
except ImportError:
    service_account = None

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("PP_GOOGLE_APPLICATION_CREDENTIALS", None)
ANTHROPIC_VERTEX_PROJECT = os.getenv("PP_ANTHROPIC_VERTEX_PROJECT", None)
# us-east5 is the region with the widest Claude model availability on Vertex.
# https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude
ANTHROPIC_VERTEX_REGION = os.getenv("PP_ANTHROPIC_VERTEX_REGION", "us-east5")

_vertex_clients = {}
_vertex_async_clients = {}


def _project_id_from_credentials_file() -> Optional[str]:
    if not GOOGLE_APPLICATION_CREDENTIALS:
        return None
    try:
        with open(GOOGLE_APPLICATION_CREDENTIALS) as credentials_file:
            return json.load(credentials_file).get("project_id")
    except (OSError, ValueError):
        return None


def get_anthropic_vertex_project() -> Optional[str]:
    return ANTHROPIC_VERTEX_PROJECT or _project_id_from_credentials_file()


def _build_google_credentials():
    # With no credentials the SDK falls back to Application Default
    # Credentials, which is the working setup inside GCP environments.
    # The SDK refreshes expired credentials per request, so clients cached
    # here stay valid indefinitely.
    if not GOOGLE_APPLICATION_CREDENTIALS or service_account is None:
        return None
    return service_account.Credentials.from_service_account_file(
        GOOGLE_APPLICATION_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def get_anthropic_vertex_client(project_id: str, region: str):
    cache_key = (project_id, region)
    if cache_key not in _vertex_clients:
        _vertex_clients[cache_key] = AnthropicVertex(
            project_id=project_id,
            region=region,
            credentials=_build_google_credentials(),
        )
    return _vertex_clients[cache_key]


def get_anthropic_vertex_async_client(project_id: str, region: str):
    cache_key = (project_id, region)
    if cache_key not in _vertex_async_clients:
        _vertex_async_clients[cache_key] = AsyncAnthropicVertex(
            project_id=project_id,
            region=region,
            credentials=_build_google_credentials(),
        )
    return _vertex_async_clients[cache_key]


def is_anthropic_vertex_available() -> bool:
    return (
        AnthropicVertex is not None
        and GOOGLE_APPLICATION_CREDENTIALS is not None
        and get_anthropic_vertex_project() is not None
    )
