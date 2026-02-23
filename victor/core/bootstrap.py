# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bootstrap module for Victor dependency injection.

This module configures the DI container with default implementations
of all services. It's called once at application startup.

Design Principles:
- Explicit registration (no auto-wiring)
- Easy to substitute for testing
- Lazy initialization where appropriate
- Environment-aware configuration

Usage:
    from victor.core.bootstrap import bootstrap_container, get_container

    # Bootstrap with default settings
    container = bootstrap_container()

    # Or with custom settings
    container = bootstrap_container(settings=my_settings)

    # Resolve services
    metrics = container.get(MetricsServiceProtocol)
    logger = container.get(LoggerServiceProtocol)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from victor.config.settings import Settings, load_settings, get_project_paths
from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    get_container,
    set_container,
    MetricsServiceProtocol,
    LoggerServiceProtocol,
    CacheServiceProtocol,
    EmbeddingServiceProtocol,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Default Service Implementations
# =============================================================================


class NullMetricsService:
    """No-op metrics service for when metrics are disabled."""

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        pass

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        pass


class ConsoleLoggerService:
    """Simple console-based logger service."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._logger = logging.getLogger("victor.analytics")

    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        if not self._enabled:
            return

        log_level = getattr(logging, level.upper(), logging.INFO)
        extra = f" | {kwargs}" if kwargs else ""
        self._logger.log(log_level, f"{message}{extra}")

    @property
    def enabled(self) -> bool:
        return self._enabled


class InMemoryCacheService:
    """Simple in-memory cache service."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Simple LRU-style eviction if over limit
        if len(self._cache) >= self._max_size:
            # Remove oldest entry (first key)
            if self._cache:
                first_key = next(iter(self._cache))
                del self._cache[first_key]
        self._cache[key] = value

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


class LazyEmbeddingService:
    """Lazy-loading embedding service that initializes on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded embedding model: {self._model_name}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        if self._model is None:
            return [[0.0] * 384 for _ in texts]  # Return zero vectors
        return self._model.encode(texts, convert_to_numpy=False).tolist()

    def embed_single(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else [0.0] * 384


# =============================================================================
# Service Protocol Type Hints
# =============================================================================


class SignatureStoreProtocol:
    """Protocol for failed signature storage."""

    def is_known_failure(self, tool_name: str, args: Dict[str, Any]) -> bool: ...

    def record_failure(self, tool_name: str, args: Dict[str, Any], error_message: str) -> None: ...


class UsageLoggerProtocol:
    """Protocol for usage logging."""

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None: ...

    def is_enabled(self) -> bool: ...


# =============================================================================
# Bootstrap Function
# =============================================================================


def bootstrap_container(
    settings: Optional[Settings] = None,
    vertical: Optional[str] = None,
    override_services: Optional[Dict[Type, Any]] = None,
) -> ServiceContainer:
    """Bootstrap the DI container with default service implementations.

    This function configures the container with all required services.
    Call once at application startup.

    Args:
        settings: Optional Settings instance (loads from config if None)
        vertical: Optional vertical name to activate (e.g., "coding", "research")
                 If None, uses settings.default_vertical or "coding"
        override_services: Optional dict of service type -> instance for testing

    Returns:
        Configured ServiceContainer
    """
    if settings is None:
        settings = load_settings()

    container = ServiceContainer()

    # Register Settings as singleton
    container.register_instance(Settings, settings)

    # Register core services
    _register_core_services(container, settings)

    # Register event services
    _register_event_services(container, settings)

    # Register analytics services
    _register_analytics_services(container, settings)

    # Register embedding services
    _register_embedding_services(container, settings)

    # Register coding services (language plugins, indexing)
    _register_coding_services(container, settings)

    # Register signature store
    _register_signature_store(container, settings)

    # Register orchestrator services (Phase 10 DI Migration)
    _register_orchestrator_services(container, settings)

    # Register workflow services (SOLID Refactoring)
    _register_workflow_services(container, settings)

    # Register workflow compiler plugins (Plugin Architecture)
    _register_workflow_compiler_plugins(container, settings)

    # Apply runtime pressure/reporter configuration for extension loading.
    _configure_extension_loader_runtime(settings)

    # Register vertical services
    _register_vertical_services(container, settings, vertical)

    # Apply overrides for testing
    if override_services:
        for service_type, instance in override_services.items():
            container.register_or_replace(
                service_type,
                lambda c, inst=instance: inst,
                ServiceLifetime.SINGLETON,
            )

    # Set as global container
    set_container(container)

    logger.info("Bootstrapped service container")
    return container


def _register_core_services(container: ServiceContainer, settings: Settings) -> None:
    """Register core infrastructure services."""
    from victor.framework.capability_config_service import CapabilityConfigService
    from victor.framework.framework_integration_registry_service import (
        FrameworkIntegrationRegistryService,
    )

    # Cache service
    container.register(
        CacheServiceProtocol,
        lambda c: InMemoryCacheService(max_size=1000),
        ServiceLifetime.SINGLETON,
    )

    # Capability config service (DI-global store with per-session scope buckets)
    container.register(
        CapabilityConfigService,
        lambda c: CapabilityConfigService(),
        ServiceLifetime.SINGLETON,
    )

    # Framework integration registry facade service
    container.register(
        FrameworkIntegrationRegistryService,
        lambda c: FrameworkIntegrationRegistryService(),
        ServiceLifetime.SINGLETON,
    )


def _register_event_services(container: ServiceContainer, settings: Settings) -> None:
    """Register event backend and observability services.

    This registers the canonical event system (core/events) as the
    primary event backend, replacing the legacy observability/event_bus.py.
    """
    from victor.core.events import (
        IEventBackend,
        ObservabilityBus,
        AgentMessageBus,
        create_event_backend,
    )
    from victor.core.events.backends import build_backend_config_from_settings

    backend_config = build_backend_config_from_settings(settings)
    lazy_init = bool(getattr(settings, "event_backend_lazy_init", True))

    # Register IEventBackend as singleton
    container.register(
        IEventBackend,
        lambda c, cfg=backend_config, lazy=lazy_init: create_event_backend(
            config=cfg, lazy_init=lazy
        ),
        ServiceLifetime.SINGLETON,
    )

    # Register ObservabilityBus as singleton
    container.register(
        ObservabilityBus,
        lambda c: ObservabilityBus(backend=c.get(IEventBackend)),
        ServiceLifetime.SINGLETON,
    )

    # Register AgentMessageBus as singleton
    container.register(
        AgentMessageBus,
        lambda c: AgentMessageBus(backend=c.get(IEventBackend)),
        ServiceLifetime.SINGLETON,
    )

    logger.info(
        "Registered event services with backend: %s (overflow_policy=%s, queue_maxsize=%s)",
        backend_config.backend_type.value,
        backend_config.extra.get("queue_overflow_policy"),
        backend_config.extra.get("queue_maxsize"),
    )


def _configure_extension_loader_runtime(settings: Settings) -> None:
    """Apply settings-level extension-loader pressure/reporter configuration."""
    try:
        from victor.core.verticals.extension_loader import (
            VerticalExtensionLoader,
            start_extension_loader_metrics_reporter,
            stop_extension_loader_metrics_reporter,
        )
    except Exception as e:
        logger.debug("Failed to import extension loader runtime config hooks: %s", e)
        return

    def _setting_int(name: str, default: int) -> int:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return default
        return default

    def _setting_float(name: str, default: float) -> float:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return default
        return default

    def _setting_bool(name: str, default: bool) -> bool:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    def _setting_str(name: str, default: str) -> str:
        value = getattr(settings, name, default)
        return value if isinstance(value, str) and value.strip() else default

    VerticalExtensionLoader.configure_extension_loader_pressure(
        warn_queue_threshold=_setting_int("extension_loader_warn_queue_threshold", 24),
        error_queue_threshold=_setting_int("extension_loader_error_queue_threshold", 32),
        warn_in_flight_threshold=_setting_int("extension_loader_warn_in_flight_threshold", 6),
        error_in_flight_threshold=_setting_int("extension_loader_error_in_flight_threshold", 8),
        cooldown_seconds=_setting_float("extension_loader_pressure_cooldown_seconds", 5.0),
        emit_events=_setting_bool("extension_loader_emit_pressure_events", False),
    )

    if _setting_bool("extension_loader_metrics_reporter_enabled", False):
        start_extension_loader_metrics_reporter(
            interval_seconds=_setting_float(
                "extension_loader_metrics_reporter_interval_seconds", 60.0
            ),
            topic=_setting_str(
                "extension_loader_metrics_reporter_topic",
                "vertical.extensions.loader.metrics",
            ),
            source="BootstrapExtensionLoaderMetricsReporter",
            reset_after_emit=_setting_bool(
                "extension_loader_metrics_reporter_reset_after_emit", False
            ),
        )
    else:
        stop_extension_loader_metrics_reporter(timeout=2.0)


def _register_analytics_services(container: ServiceContainer, settings: Settings) -> None:
    """Register analytics and logging services."""

    # Metrics service
    container.register(
        MetricsServiceProtocol,
        lambda c: NullMetricsService(),  # No-op by default
        ServiceLifetime.SINGLETON,
    )

    # Logger service
    container.register(
        LoggerServiceProtocol,
        lambda c: ConsoleLoggerService(enabled=True),
        ServiceLifetime.SINGLETON,
    )

    # Enhanced usage logger
    paths = get_project_paths()
    container.register(
        UsageLoggerProtocol,
        lambda c: _create_usage_logger(paths.global_logs_dir / "usage.jsonl", settings),
        ServiceLifetime.SINGLETON,
    )


def _register_embedding_services(container: ServiceContainer, settings: Settings) -> None:
    """Register embedding/ML services."""

    container.register(
        EmbeddingServiceProtocol,
        lambda c: LazyEmbeddingService(settings.unified_embedding_model),
        ServiceLifetime.SINGLETON,
    )


def _register_coding_services(container: ServiceContainer, settings: Settings) -> None:
    """Register coding services (language plugins, indexing).

    This ensures language plugins are discovered at startup, not mid-conversation.
    """
    try:
        from victor_coding.languages.registry import get_language_registry

        registry = get_language_registry()
        # Only discover if not already discovered (check if plugins list is empty)
        if not registry._plugins:
            count = registry.discover_plugins()
            logger.info(f"Discovered {count} language plugins at startup")
        else:
            logger.debug(f"Language plugins already discovered: {len(registry._plugins)} plugins")
    except Exception as e:
        logger.warning(f"Failed to discover language plugins: {e}")


def _register_signature_store(container: ServiceContainer, settings: Settings) -> None:
    """Register failed signature store service."""

    container.register(
        SignatureStoreProtocol,
        lambda c: _create_signature_store(),
        ServiceLifetime.SINGLETON,
    )


def _create_usage_logger(log_file: Path, settings: Settings) -> Any:
    """Create usage logger with enhanced features if available."""
    try:
        from victor.analytics.enhanced_logger import EnhancedUsageLogger

        return EnhancedUsageLogger(
            log_file=log_file,
            enabled=True,
            scrub_pii=True,
            encrypt=False,  # Enable if needed
            max_log_size=10 * 1024 * 1024,  # 10MB
            backup_count=5,
            compress_rotated=True,
        )
    except Exception as e:
        logger.warning(f"Failed to create enhanced logger: {e}")
        # Fall back to basic logger
        from victor.analytics.logger import UsageLogger

        return UsageLogger(log_file, enabled=True)


def _create_signature_store() -> Any:
    """Create signature store for failed tool calls."""
    try:
        from victor.agent.signature_store import SignatureStore

        return SignatureStore()
    except Exception as e:
        logger.warning(f"Failed to create signature store: {e}")
        # Return a no-op store
        return NullSignatureStore()


class NullSignatureStore:
    """No-op signature store for when the real one fails to load."""

    def is_known_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        return False

    def record_failure(self, tool_name: str, args: Dict[str, Any], error_message: str) -> None:
        pass

    def clear_signature(self, tool_name: str, args: Dict[str, Any]) -> bool:
        return False


def _register_orchestrator_services(container: ServiceContainer, settings: Settings) -> None:
    """Register orchestrator-related services.

    Part of Phase 10 DI Migration - registers services used by AgentOrchestrator.
    This enables:
    - Type-safe dependency injection for orchestrator components
    - Easy testing via mock substitution
    - Proper lifecycle management (singleton vs scoped)

    Args:
        container: DI container to register services in
        settings: Application settings
    """
    try:
        from victor.agent.service_provider import configure_orchestrator_services

        configure_orchestrator_services(container, settings)
        logger.debug("Registered orchestrator services")
    except Exception as e:
        # Don't fail bootstrap if orchestrator services can't be registered
        # The orchestrator will fall back to direct instantiation
        logger.warning(f"Failed to register orchestrator services: {e}")


def _register_workflow_services(container: ServiceContainer, settings: Settings) -> None:
    """Register workflow-related services.

    Part of SOLID Refactoring - registers services used by the workflow system.
    This enables:
    - Type-safe dependency injection for workflow components
    - Protocol-based architecture (ISP, DIP compliance)
    - Proper lifecycle management (singleton vs scoped vs transient)

    Services registered:
    - Singleton: YAMLWorkflowLoader, WorkflowValidator, NodeExecutorFactory
    - Scoped: ExecutionContext, OrchestratorPool
    - Transient: WorkflowCompiler, WorkflowExecutor

    Args:
        container: DI container to register services in
        settings: Application settings
    """
    try:
        from victor.workflows.services.workflow_service_provider import configure_workflow_services

        configure_workflow_services(container, settings)
        logger.debug("Registered workflow services")
    except Exception as e:
        # Don't fail bootstrap if workflow services can't be registered
        # The workflow system will fall back to direct instantiation
        logger.warning(f"Failed to register workflow services: {e}")


def _register_workflow_compiler_plugins(
    container: ServiceContainer,
    settings: Settings,
) -> None:
    """Register workflow compiler plugins.

    Part of Plugin Architecture - registers built-in compiler plugins.
    This enables:
    - URI-based compiler creation (create_compiler("yaml://"))
    - Third-party plugin registration via entry points
    - Gradual migration from legacy to SOLID architecture

    Plugins registered:
    - YamlCompilerPlugin: YAML workflow compilation

    Args:
        container: DI container to register services in
        settings: Application settings
    """
    try:
        from victor.workflows.plugins import register_builtin_plugins

        register_builtin_plugins()
        logger.debug("Registered workflow compiler plugins")
    except Exception as e:
        # Don't fail bootstrap if plugins can't be registered
        # The plugin system will fall back to direct instantiation
        logger.warning(f"Failed to register workflow compiler plugins: {e}")


def _register_vertical_services(
    container: ServiceContainer,
    settings: Settings,
    vertical_name: Optional[str] = None,
) -> None:
    """Register vertical-specific services.

    Loads the specified vertical and registers its services with the container.
    This enables verticals to provide:
    - Custom middleware for tool execution
    - Safety patterns for dangerous operations
    - Prompt contributions for task hints
    - Mode configurations

    Args:
        container: DI container to register services in
        settings: Application settings
        vertical_name: Optional vertical name. If None, uses settings.default_vertical.
                       If settings.default_vertical is also not set, defaults to "coding".
    """
    # Determine which vertical to load
    # Priority: explicit parameter > settings.default_vertical > "coding"
    if vertical_name is None:
        default_vertical = getattr(settings, "default_vertical", None)
        if isinstance(default_vertical, str) and default_vertical.strip():
            vertical_name = default_vertical
        else:
            vertical_name = "coding"

    try:
        from victor.core.verticals.vertical_loader import activate_vertical_services
        from victor.core.verticals.protocols import VerticalExtensions

        activation = activate_vertical_services(container, settings, vertical_name)

        # Register the extensions as a service for framework access
        from victor.core.verticals.vertical_loader import get_vertical_loader

        loader = get_vertical_loader()
        extensions = loader.get_extensions()
        if extensions:
            if container.is_registered(VerticalExtensions):
                container.register_or_replace(
                    VerticalExtensions,
                    lambda c, ext=extensions: ext,
                    ServiceLifetime.SINGLETON,
                )
            else:
                container.register_instance(VerticalExtensions, extensions)

        logger.info(
            "Registered vertical services: %s (activated=%s, services_registered=%s)",
            vertical_name,
            activation.activated,
            activation.services_registered,
        )
    except ImportError as e:
        logger.debug(f"Vertical loading not available: {e}")
    except ValueError as e:
        logger.warning(f"Failed to load vertical '{vertical_name}': {e}")
    except Exception as e:
        # Don't fail bootstrap if vertical services can't be registered
        logger.warning(f"Failed to register vertical services: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================


def get_service(service_type: Type[T]) -> T:
    """Get a service from the global container.

    Args:
        service_type: Type of service to retrieve

    Returns:
        Service instance
    """
    return get_container().get(service_type)


def get_service_optional(service_type: Type[T]) -> Optional[T]:
    """Get a service from the global container, or None if not found.

    Args:
        service_type: Type of service to retrieve

    Returns:
        Service instance or None
    """
    return get_container().get_optional(service_type)


def ensure_bootstrapped(
    settings: Optional[Settings] = None,
    vertical: Optional[str] = None,
) -> ServiceContainer:
    """Ensure the container is bootstrapped with correct vertical.

    Semi-idempotent - only bootstraps if not already done, but will
    re-activate vertical if a different one is requested.

    Args:
        settings: Optional Settings instance
        vertical: Optional vertical name (e.g., "coding", "data_analysis")

    Returns:
        Service container
    """
    container = get_container()

    # Check if already bootstrapped (Settings registered)
    if container.is_registered(Settings):
        # Container already bootstrapped, but check if we need to switch verticals
        if vertical is not None:
            _ensure_vertical_activated(container, settings or container.get(Settings), vertical)
        return container

    return bootstrap_container(settings, vertical=vertical)


def _ensure_vertical_activated(
    container: ServiceContainer,
    settings: Settings,
    vertical_name: str,
) -> None:
    """Ensure the specified vertical is activated, switching if needed.

    This handles the case where the container was bootstrapped with one
    vertical but we need to use a different one (e.g., --vertical CLI flag).

    Args:
        container: DI container
        settings: Application settings
        vertical_name: Vertical name to ensure is active
    """
    try:
        from victor.core.verticals.vertical_loader import (
            activate_vertical_services,
            get_vertical_loader,
        )
        from victor.core.verticals.protocols import VerticalExtensions

        loader = get_vertical_loader()
        current_vertical = loader.active_vertical_name

        # If no vertical active or different vertical requested, (re)activate
        if current_vertical is None or current_vertical != vertical_name:
            logger.info(f"Switching vertical: {current_vertical or 'none'} -> {vertical_name}")
            activation = activate_vertical_services(container, settings, vertical_name)

            # Update the extensions in container
            extensions = loader.get_extensions()
            if extensions:
                container.register_or_replace(
                    VerticalExtensions,
                    lambda c, ext=extensions: ext,
                    ServiceLifetime.SINGLETON,
                )

            logger.info(
                "Vertical switched to: %s (activated=%s, services_registered=%s)",
                vertical_name,
                activation.activated,
                activation.services_registered,
            )
    except ImportError as e:
        logger.debug(f"Vertical loading not available: {e}")
    except ValueError as e:
        logger.warning(f"Failed to switch to vertical '{vertical_name}': {e}")
    except Exception as e:
        logger.warning(f"Failed to ensure vertical activation: {e}")
