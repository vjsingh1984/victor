# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for ExplorationCoordinator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.coordinators.exploration_coordinator import (
    ExplorationCoordinator,
    ExplorationResult,
)


class TestExplorationCoordinator:
    """Tests for parallel exploration."""

    def test_decompose_creates_subtasks(self):
        """Task decomposition generates scoped subtasks."""
        coord = ExplorationCoordinator(MagicMock())
        tasks = coord._decompose_exploration(
            "Fix the bug in separability_matrix for nested models"
        )
        assert len(tasks) >= 2
        # Each task should have read-only tools
        for task in tasks:
            assert "read" in task.allowed_tools
            assert "ls" in task.allowed_tools

    def test_aggregate_extracts_file_paths(self):
        """Aggregation extracts file paths from summaries."""
        coord = ExplorationCoordinator(MagicMock())

        mock_result = MagicMock()
        r1 = MagicMock()
        r1.success = True
        r1.summary = (
            "Found relevant source:\n"
            "  astropy/modeling/separable.py (line 45)\n"
            "  astropy/modeling/core.py (CompoundModel class)"
        )
        r1.tool_calls_used = 5
        r2 = MagicMock()
        r2.success = True
        r2.summary = (
            "Found test files:\n"
            "  astropy/modeling/tests/test_separable.py"
        )
        r2.tool_calls_used = 3
        mock_result.results = [r1, r2]

        result = coord._aggregate(mock_result)
        assert len(result.file_paths) >= 2
        assert any("separable.py" in p for p in result.file_paths)
        assert result.tool_calls_total == 8
        assert result.subagent_count == 2

    def test_aggregate_handles_failures(self):
        """Aggregation handles failed subagents gracefully."""
        coord = ExplorationCoordinator(MagicMock())

        mock_result = MagicMock()
        r1 = MagicMock()
        r1.success = False
        r1.summary = ""
        r1.tool_calls_used = 0
        mock_result.results = [r1]

        result = coord._aggregate(mock_result)
        assert result.file_paths == []
        assert result.summary == ""

    def test_exploration_result_defaults(self):
        """ExplorationResult has sensible defaults."""
        result = ExplorationResult()
        assert result.file_paths == []
        assert result.summary == ""
        assert result.duration_seconds == 0.0
        assert result.subagent_count == 0
