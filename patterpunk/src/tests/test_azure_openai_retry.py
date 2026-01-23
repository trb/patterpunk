"""
Tests for Azure OpenAI retry logic and APIError import fix.

This test reproduces the bug where using Azure OpenAI WITHOUT regular OpenAI
configured causes NameError because APIError is conditionally imported based
on the openai client (not azure_openai client) being truthy.

Run with:
    docker compose -p patterpunk run --rm patterpunk -c '/app/bin/test.dev /app/tests/test_azure_openai_retry.py'
"""

import pytest
import os
import importlib


class TestAPIErrorImportFix:
    """Tests verifying the APIError import fix works correctly."""

    def test_apierror_available_in_openai_module(self):
        """APIError should be defined in openai module regardless of client state."""
        from patterpunk.llm.models import openai as openai_module

        # Access the module's globals to check APIError exists
        assert "APIError" in dir(openai_module) or hasattr(openai_module, "APIError")

    def test_apierror_available_in_azure_module(self):
        """APIError should be defined in azure_openai module regardless of client state."""
        from patterpunk.llm.models import azure_openai as azure_module

        assert "APIError" in dir(azure_module) or hasattr(azure_module, "APIError")

    def test_apierror_available_when_only_azure_configured(self):
        """
        CRITICAL TEST: Reproduces the exact bug scenario.

        When ONLY Azure OpenAI is configured (no regular OpenAI API key),
        APIError must still be available in openai.py's namespace because
        AzureOpenAiModel inherits _stream_with_retry from OpenAiModel.
        """
        # Save originals
        saved_azure_key = os.environ.get("PP_AZURE_OPENAI_API_KEY")
        saved_azure_endpoint = os.environ.get("PP_AZURE_OPENAI_ENDPOINT")
        saved_openai_key = os.environ.get("PP_OPENAI_API_KEY")

        try:
            # Simulate Azure-only configuration: Azure configured, OpenAI NOT configured
            os.environ["PP_AZURE_OPENAI_API_KEY"] = "test-key"
            os.environ["PP_AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
            if "PP_OPENAI_API_KEY" in os.environ:
                del os.environ["PP_OPENAI_API_KEY"]

            # Reload all relevant modules to simulate fresh import
            from patterpunk.config.providers import openai as openai_config
            from patterpunk.config.providers import azure_openai as azure_config
            from patterpunk.llm.models import openai as openai_model
            from patterpunk.llm.models import azure_openai as azure_model

            importlib.reload(openai_config)
            importlib.reload(azure_config)
            importlib.reload(openai_model)
            importlib.reload(azure_model)

            # Verify that openai client is None (simulating Azure-only setup)
            assert openai_config.openai is None, "Expected openai client to be None"

            # THE CRITICAL CHECK: APIError must still be available in openai_model
            # This is where the bug occurred - APIError wasn't defined when openai was None
            # With the fix, APIError is imported if the PACKAGE is installed, not if the CLIENT is configured
            assert "APIError" in dir(
                openai_model
            ), "APIError must be available in openai module even when openai client is None"

        finally:
            # Restore environment
            if saved_azure_key:
                os.environ["PP_AZURE_OPENAI_API_KEY"] = saved_azure_key
            elif "PP_AZURE_OPENAI_API_KEY" in os.environ:
                del os.environ["PP_AZURE_OPENAI_API_KEY"]

            if saved_azure_endpoint:
                os.environ["PP_AZURE_OPENAI_ENDPOINT"] = saved_azure_endpoint
            elif "PP_AZURE_OPENAI_ENDPOINT" in os.environ:
                del os.environ["PP_AZURE_OPENAI_ENDPOINT"]

            if saved_openai_key:
                os.environ["PP_OPENAI_API_KEY"] = saved_openai_key

            # Reload to restore state
            importlib.reload(openai_config)
            importlib.reload(azure_config)
            importlib.reload(openai_model)
            importlib.reload(azure_model)


class TestAzureOpenAIRetryIntegration:
    """Integration tests hitting actual Azure API."""

    @pytest.mark.asyncio
    async def test_streaming_completes_successfully(self):
        """Verify streaming works and error handling code is reachable."""
        from patterpunk.llm.chat.core import Chat
        from patterpunk.llm.models.azure_openai import AzureOpenAiModel
        from patterpunk.llm.messages.system import SystemMessage
        from patterpunk.llm.messages.user import UserMessage

        chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.0))

        chat = chat.add_message(SystemMessage("Reply with exactly: PONG")).add_message(
            UserMessage("PING")
        )

        async with chat.complete_stream() as stream:
            async for _ in stream.content:
                pass  # Consume the stream

        final_chat = await stream.chat
        assert "PONG" in final_chat.latest_message.content

    def test_sync_completes_successfully(self):
        """Verify sync completion works and error handling code is reachable."""
        from patterpunk.llm.chat.core import Chat
        from patterpunk.llm.models.azure_openai import AzureOpenAiModel
        from patterpunk.llm.messages.system import SystemMessage
        from patterpunk.llm.messages.user import UserMessage

        chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.0))
        chat = (
            chat.add_message(SystemMessage("Reply with exactly: PONG"))
            .add_message(UserMessage("PING"))
            .complete()
        )

        assert "PONG" in chat.latest_message.content
