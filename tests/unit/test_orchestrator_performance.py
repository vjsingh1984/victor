import asyncio
import time
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

    with patch(
        "victor.agent.orchestrator.SemanticToolSelector", return_value=mock_selector_instance
    ):
        settings = Settings(use_semantic_tool_selection=True, embedding_model="test-model")
        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = True

        # --- Test 1: Without preloading ---
        orchestrator_no_preload = AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )
        orchestrator_no_preload.tools.register(dummy_tool)

        start_time_no_preload = time.monotonic()
        # Use the ToolSelector's select_semantic method
        await orchestrator_no_preload.tool_selector.select_semantic("test message")
        end_time_no_preload = time.monotonic()

        latency_no_preload = end_time_no_preload - start_time_no_preload
        # Ensure the mock was actually called and waited for
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

        # Now, measure the latency of the actual selection call
        start_time_with_preload = time.monotonic()
        # Use the ToolSelector's select_semantic method
        await orchestrator_with_preload.tool_selector.select_semantic("test message")
        end_time_with_preload = time.monotonic()

        latency_with_preload = end_time_with_preload - start_time_with_preload

        # --- Assertions ---
        print(f"Latency without preload: {latency_no_preload:.4f}s")
        print(f"Latency with preload: {latency_with_preload:.4f}s")

        # The latency without preloading should be at least the load delay
        assert latency_no_preload >= load_delay
        # The latency with preloading should be much smaller
        assert latency_with_preload < load_delay
        # The call with preloading should be significantly faster
        assert latency_with_preload < latency_no_preload / 2

        # The mock should have been called once during preload and not again during select
        mock_selector_instance.initialize_tool_embeddings.assert_awaited_once()
