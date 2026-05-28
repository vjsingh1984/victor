"""Hot-path test coverage for ToolPipeline parallel execution and caching (Item 6)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(*, parallel: bool = True):
    """Build a minimal ToolPipeline with all dependencies mocked."""
    from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

    tool_registry = MagicMock()
    tool_registry.is_tool_enabled.return_value = True
    tool_registry.get_tool.return_value = None

    tool_executor = MagicMock()

    config = ToolPipelineConfig(enable_parallel_execution=parallel)
    pipeline = ToolPipeline(tool_registry, tool_executor, config=config)
    return pipeline


def _tool_call(name: str = "read_file", tool_id: str = "c1") -> dict:
    return {"id": tool_id, "name": name, "arguments": {"path": "/tmp/x"}}


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestExecuteToolCallsParallel:
    async def test_force_parallel_bypasses_sequential_fallback(self):
        from victor.agent.parallel_executor import ParallelExecutionResult
        from victor.agent.tool_executor import ToolExecutionResult

        pipeline = _make_pipeline(parallel=True)
        tc1 = _tool_call("read_file", "c1")
        tc2 = _tool_call("write_file", "c2")
        parallel_result = ParallelExecutionResult(
            results=[
                ToolExecutionResult(
                    tool_name="read_file",
                    success=True,
                    result="ok",
                ),
                ToolExecutionResult(
                    tool_name="write_file",
                    success=True,
                    result="ok",
                ),
            ],
            completed_count=2,
            parallel_speedup=2.0,
        )

        with (
            patch.object(
                pipeline, "execute_tool_calls", new_callable=AsyncMock
            ) as sequential,
            patch.object(
                pipeline.parallel_executor,
                "execute_parallel",
                new_callable=AsyncMock,
                return_value=parallel_result,
            ) as execute_parallel,
        ):
            result = await pipeline.execute_tool_calls_parallel(
                [tc1, tc2], force_parallel=True
            )

        sequential.assert_not_awaited()
        execute_parallel.assert_awaited_once()
        assert result.parallel_execution_used is True

    async def test_invalid_tool_name_produces_skip_result(self):
        pipeline = _make_pipeline()
        bad_call = {"id": "bad1", "name": "", "arguments": {}}
        result = await pipeline.execute_tool_calls_parallel([bad_call])
        # Should return a result (skip) rather than raising
        assert result is not None


# ---------------------------------------------------------------------------
# Synthesis checkpoint
# ---------------------------------------------------------------------------


class TestSynthesisCheckpoint:
    async def test_checkpoint_skipped_when_disabled(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(enable_synthesis_checkpoints=False)
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._synthesis_checkpoint is None

    async def test_checkpoint_created_when_enabled(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(enable_synthesis_checkpoints=True)
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._synthesis_checkpoint is not None


# ---------------------------------------------------------------------------
# Cross-turn deduplication
# ---------------------------------------------------------------------------


class TestCrossTurnDedup:
    async def test_cross_turn_dedup_enabled_by_config(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(
            enable_cross_turn_dedup=True, cross_turn_dedup_ttl=60.0
        )
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._cross_turn_enabled is True

    async def test_cross_turn_dedup_disabled_by_config(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(enable_cross_turn_dedup=False)
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._cross_turn_enabled is False
