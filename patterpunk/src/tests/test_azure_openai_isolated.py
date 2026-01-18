"""
Isolated tests for Azure OpenAI that require environment manipulation.

These tests modify environment variables and reload Python modules, which
corrupts module state for other tests. They MUST be run in isolation:

    pytest tests/test_azure_openai_isolated.py

Do NOT run these alongside other tests in the same pytest invocation.
"""

import os
import pytest


def test_azure_model_requires_credentials():
    """Test that AzureOpenAiModel raises error without credentials."""
    import importlib

    # Save original env vars
    original_azure_endpoint = os.environ.get("PP_AZURE_OPENAI_ENDPOINT")
    original_azure_key = os.environ.get("PP_AZURE_OPENAI_API_KEY")

    try:
        # Clear Azure credentials
        if "PP_AZURE_OPENAI_ENDPOINT" in os.environ:
            del os.environ["PP_AZURE_OPENAI_ENDPOINT"]
        if "PP_AZURE_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_AZURE_OPENAI_API_KEY"]

        # Reload modules to pick up missing credentials
        from patterpunk.config.providers import azure_openai as azure_config

        importlib.reload(azure_config)

        from patterpunk.llm.models import azure_openai as azure_model

        importlib.reload(azure_model)

        # Verify client was NOT created
        assert azure_config.azure_openai is None, (
            "Azure OpenAI client should be None without credentials"
        )

        # Verify model raises error
        AzureOpenAiModelFresh = azure_model.AzureOpenAiModel
        AzureOpenAiMissingConfigurationErrorFresh = (
            azure_model.AzureOpenAiMissingConfigurationError
        )

        class TestAzureModel(AzureOpenAiModelFresh):
            pass

        with pytest.raises(AzureOpenAiMissingConfigurationErrorFresh):
            TestAzureModel(deployment_name="gpt-4")

    finally:
        # Restore original environment
        if original_azure_endpoint is not None:
            os.environ["PP_AZURE_OPENAI_ENDPOINT"] = original_azure_endpoint
        elif "PP_AZURE_OPENAI_ENDPOINT" in os.environ:
            del os.environ["PP_AZURE_OPENAI_ENDPOINT"]

        if original_azure_key is not None:
            os.environ["PP_AZURE_OPENAI_API_KEY"] = original_azure_key
        elif "PP_AZURE_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_AZURE_OPENAI_API_KEY"]

        # Reload modules to restore original state
        from patterpunk.config.providers import azure_openai as azure_config
        from patterpunk.llm.models import azure_openai as azure_model

        importlib.reload(azure_config)
        importlib.reload(azure_model)


def test_azure_endpoint_url_formatting():
    """Test that Azure endpoint URLs are formatted correctly for v1 API.

    This test manipulates environment variables and reloads modules, so it
    must be run in isolation.
    """
    import importlib

    # Save original env vars
    original_azure_endpoint = os.environ.get("PP_AZURE_OPENAI_ENDPOINT")
    original_azure_key = os.environ.get("PP_AZURE_OPENAI_API_KEY")

    try:
        # Reset the global client
        from patterpunk.config.providers import azure_openai

        azure_openai._azure_openai_client = None

        test_cases = [
            (
                "https://test.openai.azure.com",
                "https://test.openai.azure.com/openai/v1/",
            ),
            (
                "https://test.openai.azure.com/",
                "https://test.openai.azure.com/openai/v1/",
            ),
            (
                "https://test.openai.azure.com/openai/v1/",
                "https://test.openai.azure.com/openai/v1/",
            ),
        ]

        for input_endpoint, expected_base_url in test_cases:
            os.environ["PP_AZURE_OPENAI_ENDPOINT"] = input_endpoint
            os.environ["PP_AZURE_OPENAI_API_KEY"] = "test-key"

            # Reset module to force re-initialization
            importlib.reload(azure_openai)

            client = azure_openai.azure_openai
            assert (
                client is not None
            ), f"Client should be created for endpoint: {input_endpoint}"
            assert str(client.base_url) == expected_base_url, (
                f"Base URL mismatch for input '{input_endpoint}': "
                f"expected '{expected_base_url}', got '{client.base_url}'"
            )

    finally:
        # Restore original environment
        if original_azure_endpoint is not None:
            os.environ["PP_AZURE_OPENAI_ENDPOINT"] = original_azure_endpoint
        elif "PP_AZURE_OPENAI_ENDPOINT" in os.environ:
            del os.environ["PP_AZURE_OPENAI_ENDPOINT"]

        if original_azure_key is not None:
            os.environ["PP_AZURE_OPENAI_API_KEY"] = original_azure_key
        elif "PP_AZURE_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_AZURE_OPENAI_API_KEY"]

        # Reload modules to restore original state
        from patterpunk.config.providers import azure_openai as azure_config
        from patterpunk.llm.models import azure_openai as azure_model

        importlib.reload(azure_config)
        importlib.reload(azure_model)


def test_azure_client_not_created_without_credentials():
    """Test that client is not created when credentials are missing.

    This test manipulates environment variables and reloads modules, so it
    must be run in isolation.
    """
    import importlib

    # Save original env vars
    original_azure_endpoint = os.environ.get("PP_AZURE_OPENAI_ENDPOINT")
    original_azure_key = os.environ.get("PP_AZURE_OPENAI_API_KEY")

    try:
        # Clear Azure credentials
        if "PP_AZURE_OPENAI_ENDPOINT" in os.environ:
            del os.environ["PP_AZURE_OPENAI_ENDPOINT"]
        if "PP_AZURE_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_AZURE_OPENAI_API_KEY"]

        from patterpunk.config.providers import azure_openai

        importlib.reload(azure_openai)

        assert (
            azure_openai.azure_openai is None
        ), "Client should be None without credentials"
        assert (
            not azure_openai.is_azure_openai_available()
        ), "Azure should not be available without credentials"

    finally:
        # Restore original environment
        if original_azure_endpoint is not None:
            os.environ["PP_AZURE_OPENAI_ENDPOINT"] = original_azure_endpoint
        elif "PP_AZURE_OPENAI_ENDPOINT" in os.environ:
            del os.environ["PP_AZURE_OPENAI_ENDPOINT"]

        if original_azure_key is not None:
            os.environ["PP_AZURE_OPENAI_API_KEY"] = original_azure_key
        elif "PP_AZURE_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_AZURE_OPENAI_API_KEY"]

        # Reload modules to restore original state
        from patterpunk.config.providers import azure_openai as azure_config
        from patterpunk.llm.models import azure_openai as azure_model

        importlib.reload(azure_config)
        importlib.reload(azure_model)


def test_azure_openai_works_without_openai_api_key():
    """
    Test that Azure OpenAI can be used without setting PP_OPENAI_API_KEY.

    This test verifies that AzureOpenAiModel only requires Azure-specific
    environment variables (PP_AZURE_OPENAI_ENDPOINT, PP_AZURE_OPENAI_API_KEY)
    and does NOT require PP_OPENAI_API_KEY.

    Regression test for: Azure model incorrectly required OpenAI API key
    because it inherited validation from OpenAiModel.
    """
    import importlib

    # Save original env vars
    original_openai_key = os.environ.get("PP_OPENAI_API_KEY")
    original_azure_endpoint = os.environ.get("PP_AZURE_OPENAI_ENDPOINT")
    original_azure_key = os.environ.get("PP_AZURE_OPENAI_API_KEY")

    try:
        # Clear OpenAI API key - this is the key scenario
        if "PP_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_OPENAI_API_KEY"]

        # Set only Azure credentials
        os.environ["PP_AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["PP_AZURE_OPENAI_API_KEY"] = "test-azure-key"

        # Reload the config modules to pick up new environment
        from patterpunk.config.providers import openai as openai_config
        from patterpunk.config.providers import azure_openai as azure_config

        importlib.reload(openai_config)
        importlib.reload(azure_config)

        # Also reload the model modules since they import the clients at module level
        from patterpunk.llm.models import openai as openai_model
        from patterpunk.llm.models import azure_openai as azure_model

        importlib.reload(openai_model)
        importlib.reload(azure_model)

        # Get the fresh AzureOpenAiModel class
        AzureOpenAiModelFresh = azure_model.AzureOpenAiModel

        # Verify Azure client was created
        assert azure_config.azure_openai is not None, (
            "Azure OpenAI client should be created with Azure credentials"
        )

        # Verify OpenAI client was NOT created (no PP_OPENAI_API_KEY)
        assert openai_config.openai is None, (
            "OpenAI client should be None without PP_OPENAI_API_KEY"
        )

        # This is the key assertion: Azure model should work without OpenAI API key
        try:
            # Create a concrete test model (AzureOpenAiModel is ABC, so we test via subclass)
            class TestAzureModel(AzureOpenAiModelFresh):
                pass

            model = TestAzureModel(deployment_name="gpt-4")

            # If we get here, the model was created successfully
            assert model.model == "gpt-4", "Model should have the correct deployment name"

        except openai_model.OpenAiMissingConfigurationError as e:
            pytest.fail(
                f"AzureOpenAiModel incorrectly requires PP_OPENAI_API_KEY: {e}\n"
                "The Azure model should only need Azure credentials, not OpenAI credentials."
            )

    finally:
        # Restore original environment
        if original_openai_key is not None:
            os.environ["PP_OPENAI_API_KEY"] = original_openai_key
        elif "PP_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_OPENAI_API_KEY"]

        if original_azure_endpoint is not None:
            os.environ["PP_AZURE_OPENAI_ENDPOINT"] = original_azure_endpoint
        elif "PP_AZURE_OPENAI_ENDPOINT" in os.environ:
            del os.environ["PP_AZURE_OPENAI_ENDPOINT"]

        if original_azure_key is not None:
            os.environ["PP_AZURE_OPENAI_API_KEY"] = original_azure_key
        elif "PP_AZURE_OPENAI_API_KEY" in os.environ:
            del os.environ["PP_AZURE_OPENAI_API_KEY"]

        # Reload modules to restore original state
        from patterpunk.config.providers import openai as openai_config
        from patterpunk.config.providers import azure_openai as azure_config
        from patterpunk.llm.models import openai as openai_model
        from patterpunk.llm.models import azure_openai as azure_model

        importlib.reload(openai_config)
        importlib.reload(azure_config)
        importlib.reload(openai_model)
        importlib.reload(azure_model)
