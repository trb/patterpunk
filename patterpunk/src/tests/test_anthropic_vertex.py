from patterpunk.llm.models.anthropic_vertex import AnthropicVertexModel
from patterpunk.llm.thinking import ThinkingConfig


def make_model(model, **kwargs):
    return AnthropicVertexModel(model=model, project_id="test-project", **kwargs)


def test_vertex_at_date_id_parses_to_45_capabilities():
    model = make_model("claude-sonnet-4-5@20250929")
    assert model._parse_model_version() == (4, 5)
    assert model._uses_adaptive_thinking_api() is False
    assert model._supports_native_structured_output() is True


def test_vertex_bare_opus_5_uses_adaptive_request_shape():
    model = make_model("claude-opus-5", thinking_config=ThinkingConfig(effort="high"))
    assert model._parse_model_version() == (5, 0)
    api_params = model._build_base_api_parameters([], None)
    api_params = model._apply_thinking_configuration(api_params)
    assert "temperature" not in api_params
    assert "top_p" not in api_params
    assert "top_k" not in api_params
    assert api_params["thinking"] == {"type": "adaptive"}
    assert api_params["output_config"] == {"effort": "high"}


def test_vertex_45_id_drops_top_p_keeping_temperature(caplog):
    model = make_model("claude-sonnet-4-5@20250929", temperature=0.2, top_p=0.5)
    api_params = model._build_base_api_parameters([], None)
    with caplog.at_level("WARNING", logger="patterpunk"):
        api_params = model._apply_thinking_configuration(api_params)
    assert api_params["temperature"] == 0.2
    assert "top_p" not in api_params
    assert any("Dropping top_p=0.5" in r.message for r in caplog.records)


def test_vertex_model_identity():
    model = make_model("claude-opus-5")
    assert AnthropicVertexModel.get_name() == "Anthropic Vertex"
    assert AnthropicVertexModel.get_available_models() == []
    assert model.project_id == "test-project"


def test_vertex_explicit_region_wins_over_default():
    model = make_model("claude-opus-5", region="europe-west1")
    assert model.region == "europe-west1"
