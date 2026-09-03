from typing import Callable

try:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    google_auth_available = True
except ImportError:
    google_auth_available = False


def make_gcp_bearer_token_provider(credentials_path: str) -> Callable[[], str]:
    # GCP OAuth tokens expire after roughly an hour. A static api_key built
    # from one token starts failing with 401s once the process outlives it.
    if not google_auth_available:
        raise ImportError(
            "google-auth is required for GCP bearer tokens. "
            "Install it with: pip install google-auth"
        )

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    def provider() -> str:
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token

    return provider
