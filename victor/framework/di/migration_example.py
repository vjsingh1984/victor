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

"""
Migration Example: From Factory Pattern to DI Container

This example demonstrates how to migrate from the manual factory pattern
(victor.core.container) to the enhanced auto-resolution DI container
(victor.framework.di).

We'll migrate a simple Analytics service as a proof of concept.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pathlib import Path


# =============================================================================
# BEFORE: Manual Factory Pattern (victor.core.container)
# =============================================================================


class MetricsCollector:
    """Simple metrics collector - BEFORE version."""

    def __init__(self, log_file: Path, enabled: bool = True):
        self.log_file = log_file
        self.enabled = enabled
        self._metrics: Dict[str, float] = {}

    def record(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self._metrics[name] = value

    def get_metrics(self) -> Dict[str, float]:
        return self._metrics.copy()


class AnalyticsService:
    """Analytics service - BEFORE version with manual injection."""

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        log_file: Optional[Path] = None,
    ):
        # Manual dependency injection with defaults
        if metrics_collector is None:
            log_file = log_file or Path(".victor/logs/metrics.log")
            metrics_collector = MetricsCollector(log_file)

        self.metrics_collector = metrics_collector

    def track_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Track an analytics event."""
        self.metrics_collector.record(event_name, 1.0)


def create_analytics_service_legacy(settings: Any) -> AnalyticsService:
    """Factory function for AnalyticsService - BEFORE pattern.

    This is the current pattern used throughout victor:
    - Manual factory function
    - Explicit dependency passing
    - Lots of boilerplate
    """
    from victor.config.settings import get_project_paths

    paths = get_project_paths()
    log_file = paths.global_logs_dir / "analytics.jsonl"

    metrics = MetricsCollector(log_file, enabled=settings.enable_analytics)

    return AnalyticsService(
        metrics_collector=metrics,
        log_file=log_file,
    )


# =============================================================================
# AFTER: Auto-Resolution DI Container (victor.framework.di)
# =============================================================================


class MetricsCollectorV2:
    """Metrics collector - AFTER version with type hints.

    Key changes:
    - Clear type hints for auto-resolution
    - Dependencies declared in constructor
    - No default values for required deps
    """

    def __init__(self, log_file: Path, enabled: bool = True):
        self.log_file = log_file
        self.enabled = enabled
        self._metrics: Dict[str, float] = {}

    def record(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self._metrics[name] = value

    def get_metrics(self) -> Dict[str, float]:
        return self._metrics.copy()


class AnalyticsServiceV2:
    """Analytics service - AFTER version with auto-injection.

    Key changes:
    - Constructor dependencies are type-hinted
    - No manual factory needed
    - Dependencies auto-injected by DI container
    """

    def __init__(self, metrics_collector: MetricsCollectorV2):
        # Just declare the dependency, container handles the rest
        self.metrics_collector = metrics_collector

    def track_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Track an analytics event."""
        self.metrics_collector.record(event_name, 1.0)


def setup_analytics_with_di(settings: Any) -> AnalyticsServiceV2:
    """Setup AnalyticsService using DI container - AFTER pattern.

    Benefits:
    - Auto-resolution of constructor dependencies
    - Less boilerplate code
    - Type-safe dependency injection
    - Clearer dependency graph
    """
    from victor.framework.di import DIContainer, ServiceLifetime
    from victor.config.settings import get_project_paths

    paths = get_project_paths()
    log_file = paths.global_logs_dir / "analytics.jsonl"

    container = (
        DIContainer()
        # Register MetricsCollector as singleton
        .register_instance(
            MetricsCollectorV2,
            MetricsCollectorV2(log_file, enabled=settings.enable_analytics),
        )
        # Register AnalyticsService as transient
        # It will auto-inject MetricsCollector
        .register(AnalyticsServiceV2)
    )

    # Resolve AnalyticsService - dependencies auto-injected!
    analytics = container.get(AnalyticsServiceV2)

    return analytics


# =============================================================================
# COMPARISON: Usage Examples
# =============================================================================


def example_before_migration():
    """Example usage BEFORE migration."""
    from victor.config.settings import Settings

    # Load settings
    settings = Settings()

    # Create service with factory function
    analytics = create_analytics_service_legacy(settings)

    # Use service
    analytics.track_event("user_login", {"user_id": "123"})


def example_after_migration():
    """Example usage AFTER migration."""
    from victor.config.settings import Settings

    # Load settings
    settings = Settings()

    # Create service with DI container
    analytics = setup_analytics_with_di(settings)

    # Use service (same interface)
    analytics.track_event("user_login", {"user_id": "123"})


# =============================================================================
# MIGRATION STEPS
# =============================================================================


def migrate_step_1_add_type_hints():
    """
    STEP 1: Add type hints to constructor dependencies.

    BEFORE:
        class AnalyticsService:
            def __init__(self, metrics_collector=None):
                self.metrics_collector = metrics_collector

    AFTER:
        class AnalyticsService:
            def __init__(self, metrics_collector: MetricsCollector):
                self.metrics_collector = metrics_collector
    """
    pass


def migrate_step_2_remove_defaults():
    """
    STEP 2: Remove default values from required dependencies.

    BEFORE:
        def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
            if metrics_collector is None:
                metrics_collector = create_metrics_collector()

    AFTER:
        def __init__(self, metrics_collector: MetricsCollector):
            # Dependency is required, no defaults
            pass
    """
    pass


def migrate_step_3_register_in_container():
    """
    STEP 3: Register services in DI container.

    Add registration in bootstrap.py or service provider:

    container.register(MetricsCollector, lifetime=ServiceLifetime.SINGLETON)
    container.register(AnalyticsService)

    The container will auto-inject dependencies!
    """
    pass


def migrate_step_4_update_factory():
    """
    STEP 4: Update or remove factory functions.

    BEFORE:
        def create_analytics_service(settings):
            metrics = MetricsCollector(...)
            return AnalyticsService(metrics)

    AFTER:
        def create_analytics_service(settings):
            container = DIContainer()
            container.register_instance(
                MetricsCollector,
                MetricsCollector(...)
            )
            container.register(AnalyticsService)
            return container.get(AnalyticsService)
    """
    pass


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================


class AnalyticsServiceV3:
    """
    Final version with backward compatibility.

    Maintains compatibility with factory pattern while adding DI support.
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollectorV2] = None,
        # Allow manual injection for backward compatibility
        log_file: Optional[Path] = None,
    ):
        # Support both factory and DI patterns
        if metrics_collector is None:
            # Fallback to manual creation (factory pattern)
            log_file = log_file or Path(".victor/logs/metrics.log")
            metrics_collector = MetricsCollectorV2(log_file)

        self.metrics_collector = metrics_collector

    def track_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Track an analytics event."""
        self.metrics_collector.record(event_name, 1.0)


# =============================================================================
# SUMMARY OF BENEFITS
# =============================================================================


def migration_benefits():
    """
    Summary of benefits from migrating to DI container:

    1. Less Boilerplate
       - No manual factory functions needed
       - Dependencies auto-injected based on type hints

    2. Type Safety
       - Constructor dependencies are type-hinted
       - Compile-time type checking with mypy

    3. Testability
       - Easy to inject mock dependencies
       - Container supports test overrides

    4. Clearer Dependency Graph
       - All dependencies visible in constructor
       - Circular dependencies detected at runtime

    5. Lifecycle Management
       - Singleton, transient, scoped lifetimes
       - Automatic disposal of resources

    6. Backward Compatibility
       - Can coexist with existing factory pattern
       - Gradual migration possible
    """
    pass


if __name__ == "__main__":
    # Run comparison examples
    print("=== BEFORE Migration ===")
    example_before_migration()
    print("Analytics service created with factory pattern")

    print("\n=== AFTER Migration ===")
    example_after_migration()
    print("Analytics service created with DI container")

    print("\n=== Migration Benefits ===")
    migration_benefits()
    print("See migration_benefits() function for details")
