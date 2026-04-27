# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.chat_compat_telemetry import (
    record_deprecated_chat_shim_access,
    reset_deprecated_chat_shim_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_chat_compat_telemetry():
    reset_deprecated_chat_shim_telemetry()
    yield
    reset_deprecated_chat_shim_telemetry()


def test_get_deprecated_chat_compat_report_returns_structured_telemetry():
    orchestrator = object.__new__(AgentOrchestrator)

    record_deprecated_chat_shim_access("chat_coordinator", "chat", "chat_service")
    record_deprecated_chat_shim_access(
        "orchestration_facade", "chat_coordinator", "lazy_getter"
    )

    report = orchestrator.get_deprecated_chat_compat_report()

    assert report["total"] == 2
    assert report["deprecated_surface_count"] == 2
    assert report["route_totals"] == {
        "chat_service": 1,
        "lazy_getter": 1,
    }
    assert report["active_components"] == [
        {"component": "chat_coordinator", "count": 1},
        {"component": "orchestration_facade", "count": 1},
    ]
    assert report["active_routes"] == [
        {"route": "chat_service", "count": 1},
        {"route": "lazy_getter", "count": 1},
    ]
    assert report["components"]["chat_coordinator"]["surfaces"]["chat"] == {
        "total": 1,
        "routes": {"chat_service": 1},
    }
    assert report["components"]["orchestration_facade"]["surfaces"]["chat_coordinator"] == {
        "total": 1,
        "routes": {"lazy_getter": 1},
    }
    assert report["removal_candidates"] == [
        {
            "surface": "chat_coordinator.chat",
            "count": 1,
            "routes": {"chat_service": 1},
        },
        {
            "surface": "orchestration_facade.chat_coordinator",
            "count": 1,
            "routes": {"lazy_getter": 1},
        },
    ]


def test_has_deprecated_chat_compat_usage_reflects_telemetry_state():
    orchestrator = object.__new__(AgentOrchestrator)

    assert orchestrator.has_deprecated_chat_compat_usage() is False

    record_deprecated_chat_shim_access("chat_coordinator", "chat", "chat_service")

    assert orchestrator.has_deprecated_chat_compat_usage() is True
