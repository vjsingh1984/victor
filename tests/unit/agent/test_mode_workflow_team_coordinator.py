"""Tests for removed ModeWorkflowTeamCoordinator compatibility surface."""

from __future__ import annotations

import pytest


def test_mode_workflow_team_coordinator_module_removed() -> None:
    """The deprecated mode_workflow_team_coordinator module should stay removed."""
    with pytest.raises(ImportError, match="mode_workflow_team_coordinator"):
        import victor.agent.mode_workflow_team_coordinator  # noqa: F401
