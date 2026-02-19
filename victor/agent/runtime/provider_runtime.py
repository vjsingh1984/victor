# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider runtime boundaries for AgentOrchestrator.

This module extracts provider runtime wiring from orchestrator construction and
adds lazy coordinator materialization to reduce startup overhead.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LazyRuntimeProxy(Generic[T]):
    """Thread-safe lazy proxy for runtime components."""

    def __init__(self, *, factory: Callable[[], T], name: str) -> None:
        self._factory = factory
        self._name = name
        self._instance: Optional[T] = None
        self._lock = threading.Lock()

    def get_instance(self) -> T:
        """Get the underlying instance, creating it on first access."""
        instance = self._instance
        if instance is not None:
            return instance

        with self._lock:
            if self._instance is None:
                self._instance = self._factory()
                logger.debug("Lazily initialized runtime component: %s", self._name)
            return self._instance

    @property
    def initialized(self) -> bool:
        """Whether the wrapped instance has been created."""
        return self._instance is not None

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.get_instance(), attr)


@dataclass(frozen=True)
class ProviderRuntimeComponents:
    """Provider runtime handles exposed to the orchestrator facade."""

    provider_coordinator: LazyRuntimeProxy[Any]
    provider_switch_coordinator: LazyRuntimeProxy[Any]


def create_provider_runtime_components(
    *,
    factory: Any,
    settings: Any,
    provider_manager: Any,
) -> ProviderRuntimeComponents:
    """Create lazy provider runtime components for orchestrator wiring."""

    def _build_provider_coordinator() -> Any:
        from victor.agent.provider_coordinator import (
            ProviderCoordinator,
            ProviderCoordinatorConfig,
        )

        return ProviderCoordinator(
            provider_manager=provider_manager,
            config=ProviderCoordinatorConfig(
                max_rate_limit_retries=getattr(settings, "max_rate_limit_retries", 3),
                enable_health_monitoring=getattr(settings, "provider_health_checks", True),
            ),
        )

    def _build_provider_switch_coordinator() -> Any:
        return factory.create_provider_switch_coordinator(
            provider_switcher=provider_manager._provider_switcher,
            health_monitor=provider_manager._health_monitor,
        )

    return ProviderRuntimeComponents(
        provider_coordinator=LazyRuntimeProxy(
            factory=_build_provider_coordinator,
            name="provider_coordinator",
        ),
        provider_switch_coordinator=LazyRuntimeProxy(
            factory=_build_provider_switch_coordinator,
            name="provider_switch_coordinator",
        ),
    )
