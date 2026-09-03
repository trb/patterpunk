"""
These tests set or clear PP_ANTHROPIC_VERTEX_* environment variables and
call importlib.reload on the Vertex config and model modules. Reloading
rebinds the module globals other tests may hold references to. Run this
file alone:

    pytest tests/test_anthropic_vertex_isolated.py
"""

import importlib
import json

import pytest

from patterpunk.config.providers import anthropic_vertex as vertex_config
from patterpunk.llm.models import anthropic_vertex as vertex_model

_VERTEX_ENV_VARS = (
    "PP_ANTHROPIC_VERTEX_PROJECT",
    "PP_ANTHROPIC_VERTEX_REGION",
    "PP_GOOGLE_APPLICATION_CREDENTIALS",
)


@pytest.fixture
def vertex_env(monkeypatch):
    def apply(**env):
        for name in _VERTEX_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        importlib.reload(vertex_config)
        importlib.reload(vertex_model)

    yield apply
    # monkeypatch's own restore runs after this finalizer, too late for the
    # reloads to see the original environment. Undo it here first.
    monkeypatch.undo()
    importlib.reload(vertex_config)
    importlib.reload(vertex_model)


def test_missing_configuration_raises(vertex_env):
    vertex_env()
    with pytest.raises(vertex_model.AnthropicVertexMissingConfigurationError):
        vertex_model.AnthropicVertexModel(model="claude-opus-5")


def test_project_resolves_from_env(vertex_env):
    vertex_env(PP_ANTHROPIC_VERTEX_PROJECT="env-project")
    model = vertex_model.AnthropicVertexModel(model="claude-opus-5")
    assert model.project_id == "env-project"
    assert model.region == "us-east5"


def test_region_resolves_from_env(vertex_env):
    vertex_env(
        PP_ANTHROPIC_VERTEX_PROJECT="env-project",
        PP_ANTHROPIC_VERTEX_REGION="europe-west1",
    )
    model = vertex_model.AnthropicVertexModel(model="claude-opus-5")
    assert model.region == "europe-west1"


def test_project_falls_back_to_credentials_file(vertex_env, tmp_path):
    credentials_path = tmp_path / "sa.json"
    credentials_path.write_text(json.dumps({"project_id": "file-project"}))
    vertex_env(PP_GOOGLE_APPLICATION_CREDENTIALS=str(credentials_path))
    assert vertex_config.get_anthropic_vertex_project() == "file-project"
    assert vertex_config.is_anthropic_vertex_available() is True


def test_unavailable_without_credentials(vertex_env):
    vertex_env(PP_ANTHROPIC_VERTEX_PROJECT="env-project")
    assert vertex_config.is_anthropic_vertex_available() is False


def test_clients_are_cached_by_project_and_region(vertex_env, monkeypatch):
    vertex_env(PP_ANTHROPIC_VERTEX_PROJECT="env-project")

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vertex_config, "AnthropicVertex", FakeClient)

    first = vertex_config.get_anthropic_vertex_client("p1", "us-east5")
    second = vertex_config.get_anthropic_vertex_client("p1", "us-east5")
    other_region = vertex_config.get_anthropic_vertex_client("p1", "europe-west1")

    assert first is second
    assert other_region is not first
    assert first.kwargs["project_id"] == "p1"
    assert first.kwargs["region"] == "us-east5"
    assert first.kwargs["credentials"] is None
