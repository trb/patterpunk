import os
import functools

DIRECTORY = os.path.dirname(__file__)
RESOURCES = f"{DIRECTORY}/resources"


def get_resource(resource: str) -> str:
    return f"{RESOURCES}/{resource}"


@functools.lru_cache(maxsize=1)
def openai_quota_available() -> bool:
    """Probe OpenAI once per test process. If we get 429/insufficient_quota, return False
    so the test file's pytestmark skips everything cleanly instead of every test
    spinning in retry/backoff for minutes."""
    try:
        from patterpunk.config.providers.openai import openai

        if openai is None:
            return False
        openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True
    except Exception as exc:
        msg = str(exc)
        if "insufficient_quota" in msg or "429" in msg:
            return False
        # Other errors (network, auth, etc.) — let real tests surface them
        return True
