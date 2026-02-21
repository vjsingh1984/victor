# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Coordination runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy


@dataclass(frozen=True)
class CoordinationRuntimeComponents:
    """Coordination runtime handles exposed to the orchestrator facade."""

    recovery_coordinator: LazyRuntimeProxy[Any]
    chunk_generator: LazyRuntimeProxy[Any]
    tool_planner: LazyRuntimeProxy[Any]
    task_coordinator: LazyRuntimeProxy[Any]


def create_coordination_runtime_components(
    *,
    factory: Any,
) -> CoordinationRuntimeComponents:
    """Create lazy coordination components for orchestrator wiring."""

    return CoordinationRuntimeComponents(
        recovery_coordinator=LazyRuntimeProxy(
            factory=lambda: factory.create_recovery_coordinator(),
            name="recovery_coordinator",
        ),
        chunk_generator=LazyRuntimeProxy(
            factory=lambda: factory.create_chunk_generator(),
            name="chunk_generator",
        ),
        tool_planner=LazyRuntimeProxy(
            factory=lambda: factory.create_tool_planner(),
            name="tool_planner",
        ),
        task_coordinator=LazyRuntimeProxy(
            factory=lambda: factory.create_task_coordinator(),
            name="task_coordinator",
        ),
    )
