import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.tools.decorators import tool


@pytest.mark.asyncio
async def test_embedding_preloading_reduces_latency():
    """
    Tests that calling start_embedding_preload() reduces the latency of the first
    semantic tool selection.
    """
    # 1. Mock the SemanticToolSelector and its embedding initialization
    mock_selector_instance = MagicMock()
    mock_selector_instance.select_relevant_tools_with_context = AsyncMock(return_value=[])

    # Define a dummy tool to register
    @tool
    def dummy_tool():
        """A dummy tool for testing."""
        return "dummy"

    # Simulate a delay in loading embeddings the first time
    load_delay = 0.2

    async def async_sleep_side_effect(*args, **kwargs):
        await asyncio.sleep(load_delay)

    mock_selector_instance.initialize_tool_embeddings = AsyncMock(
        side_effect=async_sleep_side_effect
    )

    with (
        patch(
            "victor.tools.semantic_selector.SemanticToolSelector",
            return_value=mock_selector_instance,
        ),
        patch("victor.core.bootstrap_services.bootstrap_new_services"),
    ):
        settings = Settings(use_semantic_tool_selection=True, embedding_model="test-model")
        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = True
        mock_provider.name = "test_provider"
        mock_provider.get_context_window.return_value = 100000

        # --- Test 1: Without preloading ---
        orchestrator_no_preload = AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )
        orchestrator_no_preload.tools.register(dummy_tool)

        await orchestrator_no_preload.tool_selector.select_semantic("test message")
        # Ensure initialize was called during select (no preload)
        mock_selector_instance.initialize_tool_embeddings.assert_awaited_once()

        # --- Test 2: With preloading ---
        # Reset the mock for the new orchestrator instance
        mock_selector_instance.initialize_tool_embeddings.reset_mock()
        mock_selector_instance.initialize_tool_embeddings.side_effect = async_sleep_side_effect

        orchestrator_with_preload = AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )
        orchestrator_with_preload.tools.register(dummy_tool)

        # Start the preload and wait for it to finish
        orchestrator_with_preload.start_embedding_preload()
        # Give the event loop time to run the task
        await asyncio.sleep(load_delay + 0.1)

        # select_semantic should NOT re-initialize (preload already did it)
        await orchestrator_with_preload.tool_selector.select_semantic("test message")

        # --- Assertions ---
        # Behavioral check: preloading means initialize_tool_embeddings is called
        # once during preload, not again during select_semantic(). This is the
        # property that matters — wall-clock timing comparisons are unreliable
        # on shared CI runners where scheduling jitter exceeds the margin.
        mock_selector_instance.initialize_tool_embeddings.assert_awaited_once()
