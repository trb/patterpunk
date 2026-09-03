import json
from typing import Callable

try:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    google_auth_available = True
except ImportError:
    google_auth_available = False

_CLOUD_PLATFORM_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def make_gcp_bearer_token_provider(credentials_json: str) -> Callable[[], str]:
    # GCP access tokens expire after one hour. A static api_key built from
    # one token starts failing with 401s once the process outlives it.
    if not google_auth_available:
        raise ImportError(
            "google-auth is required for GCP bearer tokens. "
            "Install it with: pip install google-auth"
        )

    raw = credentials_json.strip()
    if raw.startswith("{"):
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(raw), scopes=_CLOUD_PLATFORM_SCOPES
        )
    else:
        credentials = service_account.Credentials.from_service_account_file(
            raw, scopes=_CLOUD_PLATFORM_SCOPES
        )

    def provider() -> str:
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token

    return provider
