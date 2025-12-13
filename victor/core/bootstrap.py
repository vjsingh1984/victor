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
    override_services: Optional[Dict[Type, Any]] = None,
) -> ServiceContainer:
    """Bootstrap the DI container with default service implementations.

    This function configures the container with all required services.
    Call once at application startup.

    Args:
        settings: Optional Settings instance (loads from config if None)
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

    # Register analytics services
    _register_analytics_services(container, settings)

    # Register embedding services
    _register_embedding_services(container, settings)

    # Register signature store
    _register_signature_store(container, settings)

    # Register orchestrator services (Phase 10 DI Migration)
    _register_orchestrator_services(container, settings)

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

    # Cache service
    container.register(
        CacheServiceProtocol,
        lambda c: InMemoryCacheService(max_size=1000),
        ServiceLifetime.SINGLETON,
    )


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


def ensure_bootstrapped(settings: Optional[Settings] = None) -> ServiceContainer:
    """Ensure the container is bootstrapped.

    Idempotent - only bootstraps if not already done.

    Args:
        settings: Optional Settings instance

    Returns:
        Service container
    """
    container = get_container()

    # Check if already bootstrapped (Settings registered)
    if container.is_registered(Settings):
        return container

    return bootstrap_container(settings)
