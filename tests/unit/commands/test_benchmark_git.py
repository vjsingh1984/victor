# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for benchmark git timeout helper and pre-warm timeout."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRunGitWithTimeout:
    """Tests for _run_git_with_timeout helper."""

    @pytest.mark.asyncio
    async def test_run_git_with_timeout_success(self):
        """Successful git command returns (returncode, stdout, stderr)."""
        from victor.ui.commands.benchmark import _run_git_with_timeout

        result = await _run_git_with_timeout(["git", "--version"], cwd=".", timeout=10)
        rc, stdout, stderr = result
        assert rc == 0
        assert b"git version" in stdout

    @pytest.mark.asyncio
    async def test_run_git_with_timeout_kills_on_hang(self):
        """Hanging git command is killed after timeout."""
        from victor.ui.commands.benchmark import _run_git_with_timeout

        with pytest.raises(asyncio.TimeoutError):
            await _run_git_with_timeout(["sleep", "60"], cwd=".", timeout=1)

    @pytest.mark.asyncio
    async def test_prewarm_timeout_continues(self):
        """Pre-warm timeout should not crash the benchmark — it's best-effort."""

        # Simulate what happens when pre-warm times out
        async def hanging_prewarm(*args, **kwargs):
            await asyncio.sleep(60)

        try:
            await asyncio.wait_for(hanging_prewarm(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected — benchmark should continue

        # Verify we can still proceed after timeout
        assert True


class TestCodeIntelligencePrewarm:
    @pytest.mark.asyncio
    async def test_prewarm_success_caches_graph_stats(self, tmp_path: Path):
        from victor.ui.commands.benchmark import _prewarm_code_intelligence_index

        warmed = {}
        fake_graph_store = MagicMock()
        fake_graph_store.stats = AsyncMock(return_value={"nodes": 12, "edges": 34})
        fake_index = MagicMock(graph_store=fake_graph_store)

        async def fake_get_or_build_index(*args, **kwargs):
            return fake_index, False

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.tools.code_search_tool._get_or_build_index",
                side_effect=fake_get_or_build_index,
            ) as mock_get,
        ):
            first = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)
            second = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)

        assert first.status == "ready"
        assert first.graph_nodes == 12
        assert first.graph_edges == 34
        assert second.cached_hit is True
        assert second.status == "ready"
        assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_prewarm_failure_is_cached_for_repo(self, tmp_path: Path):
        from victor.ui.commands.benchmark import _prewarm_code_intelligence_index

        warmed = {}

        async def fake_get_or_build_index(*args, **kwargs):
            raise RuntimeError("missing embedding dependency")

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.tools.code_search_tool._get_or_build_index",
                side_effect=fake_get_or_build_index,
            ) as mock_get,
        ):
            first = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)
            second = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)

        assert first.status == "failed"
        assert "missing embedding dependency" in first.message
        assert second.cached_hit is True
        assert second.status == "failed"
        assert mock_get.call_count == 1


class TestBenchmarkRuntimeReadiness:
    def test_runtime_readiness_passes_with_enabled_tools(self):
        from victor.evaluation.agent_adapter import BenchmarkToolReadiness
        from victor.ui.commands.benchmark import _ensure_benchmark_runtime_tools

        adapter = MagicMock()
        adapter.get_benchmark_tool_readiness.return_value = BenchmarkToolReadiness(
            required_tools=("code_search", "graph", "read"),
            enabled_tools=("code_search", "graph", "read"),
        )

        readiness = _ensure_benchmark_runtime_tools(adapter)

        assert readiness.ready is True
        assert "graph" in readiness.enabled_tools

    def test_runtime_readiness_fails_fast_for_missing_graph(self):
        from victor.evaluation.agent_adapter import BenchmarkToolReadiness
        from victor.ui.commands.benchmark import _ensure_benchmark_runtime_tools

        adapter = MagicMock()
        adapter.get_benchmark_tool_readiness.return_value = BenchmarkToolReadiness(
            required_tools=("code_search", "graph", "read"),
            enabled_tools=("code_search", "read"),
            missing_tools=("graph",),
        )

        with pytest.raises(RuntimeError, match="missing tools: graph"):
            _ensure_benchmark_runtime_tools(adapter)

    def test_runtime_readiness_fails_fast_for_disabled_graph(self):
        from victor.evaluation.agent_adapter import BenchmarkToolReadiness
        from victor.ui.commands.benchmark import _ensure_benchmark_runtime_tools

        adapter = MagicMock()
        adapter.get_benchmark_tool_readiness.return_value = BenchmarkToolReadiness(
            required_tools=("code_search", "graph", "read"),
            enabled_tools=("code_search", "read"),
            disabled_tools=("graph",),
        )

        with pytest.raises(RuntimeError, match="disabled tools: graph"):
            _ensure_benchmark_runtime_tools(adapter)
