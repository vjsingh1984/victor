"""Tests for vertical runtime diagnostics snapshot helper."""

from __future__ import annotations

from types import SimpleNamespace

from victor.core.verticals.runtime_diagnostics import get_vertical_runtime_diagnostics


def test_runtime_diagnostics_includes_all_telemetry_sections(monkeypatch) -> None:
    """Snapshot should include loader, tool dependency, and framework entry-point stats."""
    fake_loader = SimpleNamespace(
        active_vertical_name="coding",
        get_discovery_stats=lambda: {"vertical": {"calls": 1}},
    )
    monkeypatch.setattr(
        "victor.core.verticals.runtime_diagnostics.get_tool_dependency_resolution_stats",
        lambda: {"total_requests": 3},
    )
    monkeypatch.setattr(
        "victor.core.verticals.runtime_diagnostics.get_entry_point_loader_stats",
        lambda: {"tool_dependency_calls": 2},
    )

    snapshot = get_vertical_runtime_diagnostics(loader=fake_loader)

    assert snapshot["active_vertical"] == "coding"
    assert "timestamp_utc" in snapshot
    assert snapshot["vertical_loader"] == {"vertical": {"calls": 1}}
    assert snapshot["tool_dependency_loader"] == {"total_requests": 3}
    assert snapshot["framework_entry_point_loader"] == {"tool_dependency_calls": 2}


def test_runtime_diagnostics_is_resilient_to_loader_failure(monkeypatch) -> None:
    """Snapshot should still return tool/framework stats when loader access fails."""

    class _BrokenLoader:
        @property
        def active_vertical_name(self):
            raise RuntimeError("loader unavailable")

    monkeypatch.setattr(
        "victor.core.verticals.runtime_diagnostics.get_tool_dependency_resolution_stats",
        lambda: {"total_requests": 1},
    )
    monkeypatch.setattr(
        "victor.core.verticals.runtime_diagnostics.get_entry_point_loader_stats",
        lambda: {"cache_hits": 4},
    )

    snapshot = get_vertical_runtime_diagnostics(loader=_BrokenLoader())

    assert snapshot["active_vertical"] is None
    assert "error" in snapshot["vertical_loader"]
    assert snapshot["tool_dependency_loader"] == {"total_requests": 1}
    assert snapshot["framework_entry_point_loader"] == {"cache_hits": 4}


def test_runtime_diagnostics_is_resilient_to_component_stat_failures(monkeypatch) -> None:
    """Snapshot should report per-section errors when telemetry providers fail."""
    fake_loader = SimpleNamespace(
        active_vertical_name="research",
        get_discovery_stats=lambda: {"vertical": {"calls": 2}},
    )

    def _fail_tool_stats():
        raise RuntimeError("tool stats unavailable")

    def _fail_framework_stats():
        raise RuntimeError("framework stats unavailable")

    monkeypatch.setattr(
        "victor.core.verticals.runtime_diagnostics.get_tool_dependency_resolution_stats",
        _fail_tool_stats,
    )
    monkeypatch.setattr(
        "victor.core.verticals.runtime_diagnostics.get_entry_point_loader_stats",
        _fail_framework_stats,
    )

    snapshot = get_vertical_runtime_diagnostics(loader=fake_loader)

    assert snapshot["active_vertical"] == "research"
    assert snapshot["vertical_loader"] == {"vertical": {"calls": 2}}
    assert "error" in snapshot["tool_dependency_loader"]
    assert "error" in snapshot["framework_entry_point_loader"]
