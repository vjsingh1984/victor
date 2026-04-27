# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pytest

from victor.agent.services.chat_compat_telemetry import (
    get_deprecated_chat_shim_report,
    get_deprecated_chat_shim_telemetry,
    has_deprecated_chat_shim_usage,
    record_deprecated_chat_shim_access,
    reset_deprecated_chat_shim_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_chat_compat_telemetry():
    reset_deprecated_chat_shim_telemetry()
    yield
    reset_deprecated_chat_shim_telemetry()


def test_chat_compat_telemetry_is_empty_after_reset():
    assert get_deprecated_chat_shim_telemetry() == {"total": 0}
    assert get_deprecated_chat_shim_report() == {
        "total": 0,
        "deprecated_surface_count": 0,
        "components": {},
        "route_totals": {},
        "active_components": [],
        "active_routes": [],
        "active_surfaces": [],
    }
    assert has_deprecated_chat_shim_usage() is False


def test_chat_compat_telemetry_report_groups_by_component_surface_and_route():
    record_deprecated_chat_shim_access("chat_coordinator", "chat", "chat_service")
    record_deprecated_chat_shim_access("chat_coordinator", "chat", "chat_service")
    record_deprecated_chat_shim_access("chat_coordinator", "stream_chat", "orchestrator_public")
    record_deprecated_chat_shim_access("orchestration_facade", "chat_coordinator", "lazy_getter")

    report = get_deprecated_chat_shim_report()

    assert report["total"] == 4
    assert report["deprecated_surface_count"] == 3
    assert report["route_totals"] == {
        "chat_service": 2,
        "lazy_getter": 1,
        "orchestrator_public": 1,
    }
    assert report["active_components"] == [
        {"component": "chat_coordinator", "count": 3},
        {"component": "orchestration_facade", "count": 1},
    ]
    assert report["active_routes"] == [
        {"route": "chat_service", "count": 2},
        {"route": "lazy_getter", "count": 1},
        {"route": "orchestrator_public", "count": 1},
    ]
    assert report["components"]["chat_coordinator"] == {
        "total": 3,
        "surfaces": {
            "chat": {
                "total": 2,
                "routes": {"chat_service": 2},
            },
            "stream_chat": {
                "total": 1,
                "routes": {"orchestrator_public": 1},
            },
        },
    }
    assert report["components"]["orchestration_facade"] == {
        "total": 1,
        "surfaces": {
            "chat_coordinator": {
                "total": 1,
                "routes": {"lazy_getter": 1},
            },
        },
    }
    assert report["active_surfaces"] == [
        {"surface": "chat_coordinator.chat", "count": 2},
        {"surface": "chat_coordinator.stream_chat", "count": 1},
        {"surface": "orchestration_facade.chat_coordinator", "count": 1},
    ]
    assert has_deprecated_chat_shim_usage() is True


def test_chat_compat_telemetry_flat_snapshot_stays_stable():
    record_deprecated_chat_shim_access("agent_orchestrator", "_chat_coordinator_get", "compat")

    assert get_deprecated_chat_shim_telemetry() == {
        "agent_orchestrator._chat_coordinator_get.compat": 1,
        "total": 1,
    }
