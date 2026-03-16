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

"""Vertical Extension Loading Capabilities.

This module provides a comprehensive extension loading system for Victor verticals,
enabling modular integration with framework components through lazy loading and
caching infrastructure.

Key Features:
- Fine-grained extension caching with composite keys to prevent collisions
- Lazy loading with auto-generated class names for reduced boilerplate
- Strict error handling with configurable extension requirements
- Support for 10+ extension types (middleware, safety, prompt, workflow, etc.)

Extension Types:
    - Middleware: Tool execution pipeline processing
    - Safety Extension: Dangerous operation pattern detection
    - Prompt Contributor: Task hints and prompt sections
    - Mode Config Provider: Operational mode configurations
    - Tool Dependency Provider: Tool execution patterns
    - Workflow Provider: Vertical-specific workflow definitions
    - Service Provider: DI container integration
    - RL Config Provider: Reinforcement learning configurations
    - Team Spec Provider: Multi-agent team formations
    - Enrichment Strategy: DSPy-like prompt optimization
    - Tiered Tool Config: Context-aware tool selection

Design Principles:
    - SRP Compliance: Extracted from VerticalBase for focused responsibility
    - Lazy Loading: Extensions loaded only when first accessed
    - Caching: Fine-grained cache with composite keys (ClassName:extension_key)
    - Error Tolerance: Graceful degradation with strict mode optional
    - Auto-Discovery: Auto-generated class names reduce boilerplate

Usage:
    from victor.core.verticals.extension_loader import VerticalExtensionLoader

    class MyVertical(VerticalExtensionLoader):
        @classmethod
        def get_safety_extension(cls):
            # Use generic factory (auto-generates "MyVerticalSafetyExtension")
            return cls._get_extension_factory(
                "safety_extension",
                "myvertical.safety",
            )

        @classmethod
        def get_prompt_contributor(cls):
            # Custom class name
            return cls._get_extension_factory(
                "prompt_contributor",
                "myvertical.prompts",
                "MyCustomPromptContributor",
            )

Error Handling:
    # Strict mode (any failure raises ExtensionLoadError)
    class StrictVertical(VerticalExtensionLoader):
        strict_extension_loading = True

    # Required extensions (must succeed even in non-strict mode)
    class CriticalVertical(VerticalExtensionLoader):
        required_extensions = {"safety", "middleware"}

Related Modules:
    - victor.core.verticals.base: VerticalBase (uses this loader)
    - victor.core.verticals.protocols: Extension protocol definitions
    - victor.core.verticals.metadata: Metadata provider
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import importlib.util
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Set, Type

from victor.core.verticals.extension_cache_manager import ExtensionCacheManager
from victor.core.verticals.extension_module_resolver import ExtensionModuleResolver
from victor.core.verticals.import_resolver import vertical_runtime_module_candidates

if TYPE_CHECKING:
    from victor.core.verticals.protocols import VerticalExtensions
    from victor.core.vertical_types import TieredToolConfig

logger = logging.getLogger(__name__)

_EXTENSION_LOADER_METRICS_TOPIC = "vertical.extensions.loader.metrics"
_EXTENSION_LOADER_PRESSURE_TOPIC = "vertical.extensions.loader.pressure"

_METRICS_REPORTER_LOCK = threading.Lock()
_METRICS_REPORTER_SINGLETON: Optional["ExtensionLoaderMetricsReporter"] = None


class ExtensionLoaderPressureMonitor:
    """Internal helper that owns all pressure monitoring and metrics state.

    Encapsulates the lightweight loader metrics counters, queue-pressure
    thresholds, cooldown logic, and event emission so that
    ``VerticalExtensionLoader`` can delegate to a single instance without
    spreading metrics bookkeeping across many class variables.

    Not part of the public API -- instantiated once as a class variable on
    ``VerticalExtensionLoader._pressure_monitor``.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "in_flight": 0,
            "max_in_flight": 0,
            "queued": 0,
            "max_queued": 0,
            "queue_waits": 0,
            "pressure_warnings": 0,
            "pressure_errors": 0,
        }
        self._metrics_lock = threading.RLock()

        # Queue pressure policy (P3 reliability).
        self.warn_queue_threshold: int = 24
        self.error_queue_threshold: int = 32
        self.warn_in_flight_threshold: int = 6
        self.error_in_flight_threshold: int = 8
        self.pressure_cooldown_seconds: float = 5.0
        self.emit_pressure_events: bool = False
        self.last_pressure_level: str = "ok"
        self.last_pressure_emit_ts: float = 0.0

        # Track optional modules already reported missing (avoid log spam).
        self._missing_extension_modules: Set[str] = set()
        self._missing_extension_modules_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def increment_metric(self, metric: str, delta: int = 1) -> None:
        """Increment extension loader metric counter."""
        with self._metrics_lock:
            self._metrics[metric] = self._metrics.get(metric, 0) + delta

    def update_peak_metric(self, metric: str, value: int) -> None:
        """Update extension loader metric peak value."""
        with self._metrics_lock:
            current = self._metrics.get(metric, 0)
            if value > current:
                self._metrics[metric] = value

    def get_metric(self, metric: str) -> int:
        """Return current value of a single metric under lock."""
        with self._metrics_lock:
            return self._metrics.get(metric, 0)

    def get_metrics_snapshot(
        self,
        *,
        max_workers: int = 0,
        queue_limit: int = 0,
    ) -> Dict[str, Any]:
        """Return snapshot of all loader metrics plus threshold config.

        Args:
            max_workers: Executor max_workers value to include in snapshot.
            queue_limit: Executor queue_limit value to include in snapshot.
        """
        with self._metrics_lock:
            snapshot: Dict[str, Any] = dict(self._metrics)
        snapshot["max_workers"] = max_workers
        snapshot["queue_limit"] = queue_limit
        snapshot["warn_queue_threshold"] = self.warn_queue_threshold
        snapshot["error_queue_threshold"] = self.error_queue_threshold
        snapshot["warn_in_flight_threshold"] = self.warn_in_flight_threshold
        snapshot["error_in_flight_threshold"] = self.error_in_flight_threshold
        snapshot["pressure_level"] = self.last_pressure_level
        return snapshot

    def reset_metrics(self) -> None:
        """Reset all metric counters and pressure state."""
        with self._metrics_lock:
            for key in (
                "submitted",
                "completed",
                "failed",
                "in_flight",
                "max_in_flight",
                "queued",
                "max_queued",
                "queue_waits",
                "pressure_warnings",
                "pressure_errors",
            ):
                self._metrics[key] = 0
            self.last_pressure_level = "ok"
            self.last_pressure_emit_ts = 0.0

    # ------------------------------------------------------------------
    # Pressure level evaluation
    # ------------------------------------------------------------------

    def pressure_level(self, *, queued: int, in_flight: int) -> str:
        """Return pressure level for current loader queue and in-flight counts."""
        if (
            queued >= self.error_queue_threshold
            or in_flight >= self.error_in_flight_threshold
        ):
            return "error"
        if (
            queued >= self.warn_queue_threshold
            or in_flight >= self.warn_in_flight_threshold
        ):
            return "warn"
        return "ok"

    # ------------------------------------------------------------------
    # Pressure event emission
    # ------------------------------------------------------------------

    def emit_pressure_event(
        self, level: str, snapshot: Dict[str, Any], reason: str
    ) -> None:
        """Emit queue-pressure signal for extension loader saturation."""
        try:
            from victor.core.events import get_observability_bus
            from victor.core.events.emit_helper import emit_event_sync

            bus = get_observability_bus()
            if bus is None:
                return
            emit_event_sync(
                bus,
                _EXTENSION_LOADER_PRESSURE_TOPIC,
                {
                    "level": level,
                    "reason": reason,
                    "metrics": snapshot,
                },
                source="VerticalExtensionLoader",
                use_background_loop=True,
                track_metrics=False,
            )
        except Exception as e:
            logger.debug("Failed emitting extension loader pressure event: %s", e)

    # ------------------------------------------------------------------
    # Combined pressure check (warn / error / emit)
    # ------------------------------------------------------------------

    def check_pressure(
        self,
        *,
        reason: str,
        max_workers: int = 0,
        queue_limit: int = 0,
    ) -> None:
        """Check queue pressure and optionally emit warning/error diagnostics."""
        snapshot = self.get_metrics_snapshot(
            max_workers=max_workers,
            queue_limit=queue_limit,
        )
        queued = int(snapshot.get("queued", 0))
        in_flight = int(snapshot.get("in_flight", 0))
        level = self.pressure_level(queued=queued, in_flight=in_flight)

        now = time.monotonic()
        should_emit = False
        with self._metrics_lock:
            previous_level = self.last_pressure_level
            cooldown = self.pressure_cooldown_seconds
            elapsed = now - self.last_pressure_emit_ts

            if level == "ok":
                self.last_pressure_level = "ok"
                return

            if level != previous_level or elapsed >= cooldown:
                self.last_pressure_level = level
                self.last_pressure_emit_ts = now
                should_emit = True
                if level == "error":
                    self._metrics["pressure_errors"] += 1
                else:
                    self._metrics["pressure_warnings"] += 1

        if not should_emit:
            return

        message = (
            "Extension loader pressure %s: queued=%s in_flight=%s "
            "(reason=%s, thresholds q:%s/%s in_f:%s/%s)"
        )
        args = (
            level.upper(),
            queued,
            in_flight,
            reason,
            self.warn_queue_threshold,
            self.error_queue_threshold,
            self.warn_in_flight_threshold,
            self.error_in_flight_threshold,
        )

        if level == "error":
            logger.error(message, *args)
        else:
            logger.warning(message, *args)

        if self.emit_pressure_events:
            self.emit_pressure_event(level, snapshot, reason)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        warn_queue_threshold: Optional[int] = None,
        error_queue_threshold: Optional[int] = None,
        warn_in_flight_threshold: Optional[int] = None,
        error_in_flight_threshold: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        emit_events: Optional[bool] = None,
    ) -> None:
        """Configure queue-pressure warning/error thresholds."""
        with self._metrics_lock:
            if warn_queue_threshold is not None:
                self.warn_queue_threshold = max(0, int(warn_queue_threshold))
            if error_queue_threshold is not None:
                self.error_queue_threshold = max(0, int(error_queue_threshold))
            if warn_in_flight_threshold is not None:
                self.warn_in_flight_threshold = max(0, int(warn_in_flight_threshold))
            if error_in_flight_threshold is not None:
                self.error_in_flight_threshold = max(0, int(error_in_flight_threshold))
            if cooldown_seconds is not None:
                self.pressure_cooldown_seconds = max(0.0, float(cooldown_seconds))
            if emit_events is not None:
                self.emit_pressure_events = bool(emit_events)

    # ------------------------------------------------------------------
    # Metrics event emission
    # ------------------------------------------------------------------

    def emit_metrics_event(
        self,
        *,
        max_workers: int = 0,
        queue_limit: int = 0,
        event_bus: Optional[Any] = None,
        topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
        source: str = "VerticalExtensionLoader",
        reset_after_emit: bool = False,
    ) -> Dict[str, Any]:
        """Emit current extension-loader metrics as an observability event."""
        metrics = self.get_metrics_snapshot(
            max_workers=max_workers,
            queue_limit=queue_limit,
        )

        bus = event_bus
        if bus is None:
            try:
                from victor.core.events import get_observability_bus

                bus = get_observability_bus()
            except Exception as e:
                logger.debug(
                    "Failed resolving observability bus for loader metrics: %s", e
                )
                bus = None

        if bus is not None:
            try:
                from victor.core.events.emit_helper import emit_event_sync

                emit_event_sync(
                    bus,
                    topic,
                    {"metrics": metrics},
                    source=source,
                    use_background_loop=True,
                    track_metrics=False,
                )
            except Exception as e:
                logger.debug("Failed emitting extension-loader metrics event: %s", e)

        if reset_after_emit:
            self.reset_metrics()
        return metrics

    # ------------------------------------------------------------------
    # Missing module tracking
    # ------------------------------------------------------------------

    def record_missing_module(self, cache_key: str) -> bool:
        """Record a missing module key. Return True if this is the first time."""
        with self._missing_extension_modules_lock:
            if cache_key not in self._missing_extension_modules:
                self._missing_extension_modules.add(cache_key)
                return True
        return False


class VerticalExtensionLoader(ABC):
    """Loader of vertical extensions.

    Handles loading and caching of vertical extensions including middleware,
    safety, prompt contributors, mode configs, and more.

    This is a mix-in class that provides extension loading capabilities
    to verticals while maintaining SRP compliance.

    Extension Loading Configuration:
        strict_extension_loading: When True, any extension loading failure
            raises ExtensionLoadError.
        required_extensions: Extensions that must load successfully even
            when strict_extension_loading=False.
    """

    # Extension loading configuration
    # When True, any extension loading failure raises ExtensionLoadError
    strict_extension_loading: ClassVar[bool] = False

    # Extensions that must load successfully even when strict_extension_loading=False
    # Valid values: "middleware", "safety", "prompt", "mode_config", "tool_deps",
    #               "workflow", "service", "rl_config", "team_spec", "enrichment",
    #               "tiered_tools"
    required_extensions: ClassVar[Set[str]] = set()

    # LSP capability for language intelligence
    _lsp_capability: ClassVar[Optional[Any]] = None

    # Extension cache (shared across all verticals)
    _cache_manager: ClassVar[ExtensionCacheManager] = ExtensionCacheManager()

    # Shared async extension loading infrastructure (P3)
    _extension_executor_max_workers: ClassVar[int] = 8
    _extension_executor_queue_limit: ClassVar[int] = 32
    _extension_executor: ClassVar[Optional[concurrent.futures.ThreadPoolExecutor]] = None
    _extension_executor_lock: ClassVar[threading.RLock] = threading.RLock()
    _extension_executor_semaphores: ClassVar[Dict[int, asyncio.Semaphore]] = {}

    # Pressure monitoring (metrics, thresholds, cooldown, missing-module tracking).
    _pressure_monitor: ClassVar[ExtensionLoaderPressureMonitor] = (
        ExtensionLoaderPressureMonitor()
    )

    # Module resolver (delegates candidate resolution, availability checks,
    # attribute loading, and class-name generation).
    _module_resolver: ClassVar[ExtensionModuleResolver] = ExtensionModuleResolver(
        _pressure_monitor
    )

    @classmethod
    def _cache_namespace(cls) -> str:
        """Return namespaced cache prefix for this vertical class."""
        return f"{cls.__name__}:{cls.__module__}:{cls.__qualname__}"

    @classmethod
    def get_lsp(cls) -> Optional[Any]:
        """Get the LSP capability for this vertical.

        Returns:
            LSPCapability instance or None
        """
        return cls._lsp_capability

    @classmethod
    def set_lsp(cls, lsp_capability: Any) -> None:
        """Set the LSP capability for this vertical.

        Enables language intelligence features for extension loading
        and code-related operations.

        Args:
            lsp_capability: LSPCapability instance
        """
        cls._lsp_capability = lsp_capability

    @classmethod
    def _get_shared_extension_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        """Get shared bounded executor for async extension loading."""
        base = VerticalExtensionLoader
        with base._extension_executor_lock:
            if base._extension_executor is None:
                base._extension_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=base._extension_executor_max_workers,
                    thread_name_prefix="victor-ext-loader",
                )
            return base._extension_executor

    @classmethod
    def _get_extension_load_semaphore(cls) -> asyncio.Semaphore:
        """Get per-event-loop semaphore for bounded async submission."""
        base = VerticalExtensionLoader
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        with base._extension_executor_lock:
            semaphore = base._extension_executor_semaphores.get(loop_id)
            if semaphore is None:
                semaphore = asyncio.Semaphore(base._extension_executor_queue_limit)
                base._extension_executor_semaphores[loop_id] = semaphore
            return semaphore

    @classmethod
    def _increment_loader_metric(cls, metric: str, delta: int = 1) -> None:
        """Increment extension loader metric counter."""
        VerticalExtensionLoader._pressure_monitor.increment_metric(metric, delta)

    @classmethod
    def _update_loader_peak_metric(cls, metric: str, value: int) -> None:
        """Update extension loader metric peak value."""
        VerticalExtensionLoader._pressure_monitor.update_peak_metric(metric, value)

    @classmethod
    def get_extension_loader_metrics(cls) -> Dict[str, Any]:
        """Return snapshot of shared async extension loader metrics."""
        base = VerticalExtensionLoader
        return base._pressure_monitor.get_metrics_snapshot(
            max_workers=base._extension_executor_max_workers,
            queue_limit=base._extension_executor_queue_limit,
        )

    @classmethod
    def reset_extension_loader_metrics(cls) -> None:
        """Reset async extension loader metrics counters."""
        VerticalExtensionLoader._pressure_monitor.reset_metrics()

    @classmethod
    def _extension_module_candidates(cls, module_suffix: str) -> List[str]:
        """Return possible module paths for an optional vertical extension."""
        vertical_name = getattr(cls, "name", None) or ""
        return cls._module_resolver.resolve_candidates(vertical_name, module_suffix)

    @classmethod
    def _extension_module_available(cls, module_path: str) -> bool:
        """Return True when the extension module can be imported."""
        return cls._module_resolver.is_available(
            module_path,
            vertical_display_name=getattr(cls, "name", cls.__name__),
            caller_class_name=cls.__name__,
        )

    @classmethod
    def _find_available_candidates(cls, suffix: str) -> List[str]:
        """Resolve and filter extension module candidates to available ones."""
        return [
            path
            for path in cls._extension_module_candidates(suffix)
            if cls._extension_module_available(path)
        ]

    @classmethod
    def _resolve_factory_extension(
        cls,
        extension_key: str,
        suffix: str,
        class_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Common pattern: find candidates → try _get_extension_factory on each.

        Used by get_safety_extension, get_prompt_contributor, get_team_spec_provider,
        get_enrichment_strategy, etc. to eliminate repeated candidate iteration.

        Args:
            extension_key: Cache key for the extension.
            suffix: Module suffix to resolve candidates for.
            class_name: Optional explicit class name (auto-generated if None).

        Returns:
            Extension instance or None if not found.
        """
        candidate_paths = cls._find_available_candidates(suffix)
        if not candidate_paths:
            return None

        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                return cls._get_extension_factory(
                    extension_key,
                    module_path,
                    class_name,
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error:
            raise last_error
        return None

    @classmethod
    def _resolve_class_or_factory_extension(
        cls,
        extension_key: str,
        suffix: str,
        class_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Common pattern: find candidates → try direct class import → fallback to factory.

        Used by get_mode_config_provider, get_rl_config_provider, get_rl_hooks,
        get_capability_provider, get_service_provider, etc.

        Args:
            extension_key: Cache key for the extension.
            suffix: Module suffix to resolve candidates for.
            class_name: Explicit class name to look for. If None, auto-generated.

        Returns:
            Extension instance or None if not found.
        """
        if class_name is None:
            class_name = cls._module_resolver.auto_generate_class_name(
                cls.__name__, extension_key
            )
        candidate_paths = cls._find_available_candidates(suffix)
        if not candidate_paths:
            return None

        last_error: Optional[Exception] = None
        # Try direct class import first (using __import__ to match original pattern)
        for module_path in candidate_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                provider_cls = getattr(module, class_name, None)
                if provider_cls is not None:
                    return provider_cls()
            except (ImportError, AttributeError) as exc:
                last_error = exc

        # Fallback to extension factory
        for module_path in candidate_paths:
            try:
                return cls._get_extension_factory(
                    extension_key, module_path, class_name,
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error:
            raise last_error
        return None

    @classmethod
    def configure_extension_loader_pressure(
        cls,
        *,
        warn_queue_threshold: Optional[int] = None,
        error_queue_threshold: Optional[int] = None,
        warn_in_flight_threshold: Optional[int] = None,
        error_in_flight_threshold: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        emit_events: Optional[bool] = None,
    ) -> None:
        """Configure queue-pressure warning/error thresholds for extension loading."""
        VerticalExtensionLoader._pressure_monitor.configure(
            warn_queue_threshold=warn_queue_threshold,
            error_queue_threshold=error_queue_threshold,
            warn_in_flight_threshold=warn_in_flight_threshold,
            error_in_flight_threshold=error_in_flight_threshold,
            cooldown_seconds=cooldown_seconds,
            emit_events=emit_events,
        )

    @classmethod
    def _pressure_level(cls, *, queued: int, in_flight: int) -> str:
        """Return pressure level for current loader queue and in-flight counts."""
        return VerticalExtensionLoader._pressure_monitor.pressure_level(
            queued=queued, in_flight=in_flight
        )

    @classmethod
    def _emit_pressure_event(cls, level: str, snapshot: Dict[str, Any], reason: str) -> None:
        """Emit queue-pressure signal for extension loader saturation."""
        VerticalExtensionLoader._pressure_monitor.emit_pressure_event(level, snapshot, reason)

    @classmethod
    def _check_pressure(cls, *, reason: str) -> None:
        """Check queue pressure and optionally emit warning/error diagnostics."""
        base = VerticalExtensionLoader
        base._pressure_monitor.check_pressure(
            reason=reason,
            max_workers=base._extension_executor_max_workers,
            queue_limit=base._extension_executor_queue_limit,
        )

    @classmethod
    def emit_extension_loader_metrics_event(
        cls,
        *,
        event_bus: Optional[Any] = None,
        topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
        source: str = "VerticalExtensionLoader",
        reset_after_emit: bool = False,
    ) -> Dict[str, Any]:
        """Emit current extension-loader metrics as an observability event."""
        base = VerticalExtensionLoader
        return base._pressure_monitor.emit_metrics_event(
            max_workers=base._extension_executor_max_workers,
            queue_limit=base._extension_executor_queue_limit,
            event_bus=event_bus,
            topic=topic,
            source=source,
            reset_after_emit=reset_after_emit,
        )

    # =========================================================================
    # Extension Caching Infrastructure
    # =========================================================================

    @classmethod
    def _get_cached_extension(cls, key: str, factory: callable) -> Any:
        """Get extension from cache or create and cache it.

        This helper enables fine-grained caching of individual extension
        instances, avoiding repeated object creation when extensions are
        accessed multiple times.

        The cache uses a composite key of (class_name, extension_key) to
        ensure proper isolation between different vertical subclasses.

        Args:
            key: Unique key for this extension type (e.g., "middleware",
                 "safety_extension", "workflow_provider")
            factory: Zero-argument callable that creates the extension instance.
                     Only called if the extension is not already cached.

        Returns:
            Cached or newly created extension instance.

        Example:
            @classmethod
            def get_middleware(cls) -> List[MiddlewareProtocol]:
                def _create():
                    from myvertical.middleware import MyMiddleware
                    return [MyMiddleware()]
                return cls._get_cached_extension("middleware", _create)
        """
        return cls._cache_manager.get_or_create(cls._cache_namespace(), key, factory)

    @classmethod
    def _get_cached_extension_value(cls, key: str) -> tuple[bool, Any]:
        """Return a cached extension value without caching misses."""
        return cls._cache_manager.get_if_cached(cls._cache_namespace(), key)

    @classmethod
    def _load_cached_optional_extension(
        cls,
        extension_key: str,
        loader: Callable[[], Optional[Any]],
    ) -> Optional[Any]:
        """Load an optional extension while caching hits but not misses."""
        return cls._cache_manager.load_optional(
            cls._cache_namespace(), extension_key, loader
        )

    @classmethod
    def _load_named_entry_point_extension(
        cls,
        extension_key: str,
        group: str,
    ) -> Optional[Any]:
        """Resolve an optional runtime extension from an explicit entry-point group."""
        try:
            from victor.framework.entry_point_loader import load_runtime_extension_from_entry_points
        except ImportError:
            return None

        return cls._load_cached_optional_extension(
            extension_key,
            lambda: load_runtime_extension_from_entry_points(cls.name, group),
        )

    @classmethod
    def _get_extension_factory(
        cls,
        extension_key: str,
        import_path: str,
        attribute_name: Optional[str] = None,
    ) -> Any:
        """Generic factory for lazy-loading and caching extensions.

                Eliminates boilerplate across all verticals by providing a single
                implementation of the lazy import + create + cache pattern.

                Args:
                    extension_key: Cache key (e.g., "safety_extension", "prompt_contributor")
                    import_path: Full Python import path (e.g., "victor.coding.safety")
                    attribute_name: Class name to import. If None, auto-generates from vertical name
                                  (e.g., "CodingSafetyExtension" for CodingAssistant)

                Returns:
                    Cached or newly created extension instance

                Example:
                    # Before (13 lines)
                    def get_safety_extension(cls):
                        def _create():
        try:
                                from victor_coding.safety import CodingSafetyExtension
        except ImportError:
            # External vertical package may not be installed
            pass
                            return CodingSafetyExtension()
                        return cls._get_cached_extension("safety_extension", _create)

                    # After (3 lines)
                    def get_safety_extension(cls):
                        return cls._get_extension_factory(
                            "safety_extension",
                            "victor.coding.safety",
                        )
        """

        resolver = cls._module_resolver

        def _create():
            # Determine the class name to import
            if attribute_name is None:
                class_name = resolver.auto_generate_class_name(
                    cls.__name__, extension_key
                )
            else:
                class_name = attribute_name

            # Lazy import + retrieve attribute, then instantiate
            return resolver.load_attribute(import_path, class_name)()

        # Use existing caching infrastructure
        return cls._get_cached_extension(extension_key, _create)

    # =========================================================================
    # Extension Protocol Methods (Optional)
    # =========================================================================
    # These methods enable verticals to provide framework extensions.
    # Override them to integrate with the framework's middleware, safety,
    # prompt, and configuration systems.

    @classmethod
    def get_middleware(cls) -> List[Any]:
        """Get middleware implementations for this vertical.

        Default implementation resolves a ``get_middleware()`` factory from the
        vertical's runtime middleware module when present.

        Returns:
            List of middleware implementations (MiddlewareProtocol)
        """
        candidate_paths = cls._find_available_candidates("middleware")
        if not candidate_paths:
            return []

        try:
            factory = cls._module_resolver.try_load_from_candidates(
                candidate_paths, "get_middleware"
            )
        except ImportError:
            return []

        if factory is None:
            return []

        return cls._get_cached_extension(
            "middleware",
            lambda factory=factory: list(factory() or []),
        )

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        """Get safety extension for this vertical.

        Returns:
            Safety extension (SafetyExtensionProtocol) or None
        """
        return cls._resolve_factory_extension("safety_extension", "safety")

    @classmethod
    def get_prompt_contributor(cls) -> Optional[Any]:
        """Get prompt contributor for this vertical.

        Returns:
            Prompt contributor (PromptContributorProtocol) or None
        """
        contributor = cls._load_named_entry_point_extension(
            "prompt_contributor",
            "victor.prompt_contributors",
        )
        if contributor is not None:
            return contributor

        return cls._resolve_factory_extension("prompt_contributor", "prompts")

    @classmethod
    def get_mode_config_provider(cls) -> Optional[Any]:
        """Get mode configuration provider for this vertical.

        Returns:
            Mode config provider (ModeConfigProviderProtocol) or None
        """
        provider = cls._load_named_entry_point_extension(
            "mode_config_provider", "victor.mode_configs",
        )
        if provider is not None:
            return provider

        vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
        return cls._resolve_class_or_factory_extension(
            "mode_config_provider", "mode_config",
            class_name=f"{vertical_name}ModeConfigProvider",
        )

    @classmethod
    def get_mode_config(cls) -> Dict[str, Any]:
        """Get mode configurations for this vertical.

        Returns operational modes like 'fast', 'thorough', 'explore' with
        their configurations (tool_budget, max_iterations, temperature).

        Default implementation provides standard modes. Override in subclasses
        for vertical-specific mode configurations.

        Returns:
            Dictionary mapping mode names to ModeConfig-like dicts.
        """
        return {
            "fast": {
                "name": "fast",
                "tool_budget": 10,
                "max_iterations": 20,
                "temperature": 0.7,
                "description": "Quick responses with limited tool usage",
            },
            "thorough": {
                "name": "thorough",
                "tool_budget": 50,
                "max_iterations": 50,
                "temperature": 0.7,
                "description": "Comprehensive analysis with extensive tool usage",
            },
            "explore": {
                "name": "explore",
                "tool_budget": 30,
                "max_iterations": 30,
                "temperature": 0.9,
                "description": "Exploratory mode with higher creativity",
            },
        }

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Get task-type-specific prompt hints.

        Returns hints for common task types (edit, search, explain, etc.)
        with tool priorities and budget recommendations.

        Default implementation provides standard hints. Override in subclasses
        for vertical-specific task type hints.

        Returns:
            Dictionary mapping task types to TaskTypeHint-like dicts.
        """
        return {
            "edit": {
                "task_type": "edit",
                "hint": "[EDIT MODE] Read target files first, then make focused modifications.",
                "tool_budget": 15,
                "priority_tools": ["read", "edit", "grep"],
            },
            "search": {
                "task_type": "search",
                "hint": "[SEARCH MODE] Use semantic search and grep for efficient discovery.",
                "tool_budget": 10,
                "priority_tools": ["grep", "code_search", "ls"],
            },
            "explain": {
                "task_type": "explain",
                "hint": "[EXPLAIN MODE] Read relevant code and provide clear explanations.",
                "tool_budget": 8,
                "priority_tools": ["read", "grep", "overview"],
            },
            "debug": {
                "task_type": "debug",
                "hint": "[DEBUG MODE] Investigate systematically, check logs and error messages.",
                "tool_budget": 20,
                "priority_tools": ["read", "grep", "shell", "run_tests"],
            },
            "implement": {
                "task_type": "implement",
                "hint": "[IMPLEMENT MODE] Plan first, implement incrementally, verify each step.",
                "tool_budget": 30,
                "priority_tools": ["read", "write", "edit", "shell"],
            },
        }

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[Any]:
        """Get tool dependency provider for this vertical.

        Default implementation uses the framework's create_vertical_tool_dependency_provider()
        factory function with the vertical's name. Override only if custom behavior needed.

        This eliminates ~25 LOC of duplicated wrapper code across verticals.

        Returns:
            Tool dependency provider (ToolDependencyProviderProtocol) or None
        """
        try:
            from victor.framework.entry_point_loader import (
                load_tool_dependency_provider_from_entry_points,
            )

            return cls._load_cached_optional_extension(
                "tool_dependency_provider",
                lambda: load_tool_dependency_provider_from_entry_points(cls.name),
            )
        except ImportError:
            try:
                from victor.core.tool_dependency_loader import (
                    create_vertical_tool_dependency_provider,
                )

                return cls._load_cached_optional_extension(
                    "tool_dependency_provider",
                    lambda: create_vertical_tool_dependency_provider(cls.name),
                )
            except (ImportError, ValueError):
                # If factory not available or vertical not recognized, return None
                return None

    @classmethod
    def get_tool_graph(cls) -> Optional[Any]:
        """Get tool execution graph for this vertical.

        Override to provide a custom ToolExecutionGraph that defines
        tool dependencies, transitions, and execution sequences.
        The graph is registered with the global ToolGraphRegistry.

        Returns:
            ToolExecutionGraph instance or None (no graph registered)
        """
        return None

    @classmethod
    def get_tiered_tool_config(cls) -> Optional["TieredToolConfig"]:
        """Get tiered tool configuration for this vertical.

        This is a placeholder for compatibility. The actual implementation
        is in VerticalMetadataProvider but duplicated here for standalone use.

        Returns:
            TieredToolConfig or None if vertical doesn't use tiered config.
        """
        return None

    # get_tiered_tools() removed (E5 M3) — use get_tiered_tool_config() instead

    @classmethod
    def get_rl_config_provider(cls) -> Optional[Any]:
        """Get RL configuration provider for this vertical.

        Returns:
            RL config provider (RLConfigProviderProtocol) or None
        """
        try:
            from victor.framework.entry_point_loader import (
                load_rl_config_provider_from_entry_points,
            )
        except ImportError:
            pass
        else:
            provider = cls._load_cached_optional_extension(
                "rl_config_provider",
                lambda: load_rl_config_provider_from_entry_points(cls.name),
            )
            if provider is not None:
                return provider

        vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
        return cls._resolve_class_or_factory_extension(
            "rl_config_provider", "rl", class_name=f"{vertical_name}RLConfig",
        )

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for outcome recording.

        Returns:
            RLHooks instance or None
        """
        vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
        return cls._resolve_class_or_factory_extension(
            "rl_hooks", "rl", class_name=f"{vertical_name}RLHooks",
        )

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Get team specification provider for this vertical.

        Default implementation tries direct import pattern first, then falls back
        to extension factory. Override only if custom behavior needed.

        This eliminates ~30 LOC of duplicated wrapper code across verticals.

        Returns:
            Team spec provider (TeamSpecProviderProtocol) or None
        """
        provider = cls._load_named_entry_point_extension(
            "team_spec_provider",
            "victor.team_spec_providers",
        )
        if provider is not None:
            return provider

        vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
        class_name = f"{vertical_name}TeamSpecProvider"
        candidate_paths = [
            path
            for path in cls._extension_module_candidates("teams")
            if cls._extension_module_available(path)
        ]
        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                provider_cls = getattr(module, class_name, None)
                if provider_cls is not None:
                    return provider_cls()
            except (ImportError, AttributeError) as exc:
                last_error = exc

        for module_path in candidate_paths:
            try:
                return cls._get_extension_factory(
                    "team_spec_provider",
                    module_path,
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        try:
            from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter

            provider = VerticalRuntimeAdapter.build_team_spec_provider(cls)
            if provider is not None:
                return provider
        except Exception as exc:
            if last_error is None:
                last_error = exc

        if last_error:
            raise last_error
        return None

    @classmethod
    def get_team_specs(cls) -> Dict[str, Any]:
        """Get team specifications for this vertical.

        Returns:
            Dict mapping team names to TeamSpec instances
        """

        provider = cls.get_team_spec_provider()
        if provider is None or not hasattr(provider, "get_team_specs"):
            return {}
        return provider.get_team_specs()

    @classmethod
    def get_capability_provider(cls) -> Optional[Any]:
        """Get capability provider for this vertical.

        Returns:
            Capability provider (BaseCapabilityProvider) or None
        """
        provider = cls._load_named_entry_point_extension(
            "capability_provider", "victor.capability_providers",
        )
        if provider is not None:
            return provider

        vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
        return cls._resolve_class_or_factory_extension(
            "capability_provider", "capabilities",
            class_name=f"{vertical_name}CapabilityProvider",
        )

    @classmethod
    def get_service_provider(cls) -> Optional[Any]:
        """Get service provider for this vertical.

        Returns:
            Service provider (ServiceProviderProtocol) or factory-created provider
        """
        provider = cls._load_named_entry_point_extension(
            "service_provider", "victor.service_providers",
        )
        if provider is not None:
            return provider

        vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
        class_name = f"{vertical_name}ServiceProvider"
        candidate_paths = cls._find_available_candidates("service_provider")
        last_error: Optional[Exception] = None

        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                last_error = exc
                continue

            provider_cls = getattr(module, class_name, None)
            if provider_cls is not None:
                return cls._get_cached_extension(
                    "service_provider",
                    lambda provider_cls=provider_cls: provider_cls(),
                )

            factory = getattr(module, "get_service_provider", None)
            if callable(factory):
                return cls._get_cached_extension("service_provider", factory)

        try:
            from victor.core.verticals.base_service_provider import VerticalServiceProviderFactory

            return VerticalServiceProviderFactory.create(cls)
        except ImportError:
            if last_error is not None:
                raise last_error
            return None

    @classmethod
    def get_composed_chains(cls) -> Dict[str, Any]:
        """Get composed tool chains for this vertical from runtime modules."""
        candidate_paths = cls._find_available_candidates("composed_chains")
        if not candidate_paths:
            return {}

        constant_name = f"{getattr(cls, 'name', cls.__name__).upper().replace('-', '_')}_CHAINS"
        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                last_error = exc
                continue

            factory = getattr(module, "get_composed_chains", None)
            if callable(factory):
                return cls._get_cached_extension(
                    "composed_chains",
                    lambda factory=factory: dict(factory() or {}),
                )

            chains = getattr(module, constant_name, None)
            if isinstance(chains, dict):
                return cls._get_cached_extension(
                    "composed_chains",
                    lambda chains=chains: dict(chains),
                )

        if last_error:
            raise last_error
        return {}

    @classmethod
    def get_personas(cls) -> Dict[str, Any]:
        """Get vertical personas from runtime team modules when available."""
        candidate_paths = cls._find_available_candidates("teams")
        if not candidate_paths:
            return {}

        constant_name = f"{getattr(cls, 'name', cls.__name__).upper().replace('-', '_')}_PERSONAS"
        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                last_error = exc
                continue

            factory = getattr(module, "get_personas", None)
            if callable(factory):
                return cls._get_cached_extension(
                    "personas",
                    lambda factory=factory: dict(factory() or {}),
                )

            personas = getattr(module, constant_name, None)
            if isinstance(personas, dict):
                return cls._get_cached_extension(
                    "personas",
                    lambda personas=personas: dict(personas),
                )

        if last_error:
            raise last_error
        return {}

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[Any]:
        """Get vertical-specific enrichment strategy.

        Override to provide vertical-specific prompt enrichment strategies
        for DSPy-like auto prompt optimization. Enrichments can include:
        - Knowledge graph symbols and related code snippets (coding)
        - Web search results and source citations (research)
        - Infrastructure context and command patterns (devops)
        - Schema context and query patterns (data analysis)

        Returns:
            EnrichmentStrategyProtocol implementation or None
        """
        return None

    @classmethod
    def get_extensions(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Get all extensions for this vertical with strict error handling.

        Aggregates all extension implementations for framework integration.
        Override for custom extension aggregation.

        LSP Compliance: This method ALWAYS returns a valid VerticalExtensions
        object, never None. Even on exceptions (in non-strict mode), it returns
        a VerticalExtensions with successfully loaded extensions.

        Error Handling Modes:
        - strict=True: Raises ExtensionLoadError on ANY extension failure
        - strict=False: Collects errors, logs warnings, returns partial extensions
        - strict=None: Uses class-level strict_extension_loading setting

        Required Extensions:
        Even when strict=False, extensions listed in required_extensions will
        raise ExtensionLoadError if they fail to load.

        Args:
            use_cache: If True (default), return cached extensions if available.
                       Set to False to force rebuild.
            strict: Override the class-level strict_extension_loading setting.
                    If None (default), uses cls.strict_extension_loading.

        Returns:
            VerticalExtensions containing all vertical extensions (never None)

        Raises:
            ExtensionLoadError: In strict mode or when a required extension fails
        """
        from victor.core.errors import ExtensionLoadError
        from victor.core.verticals.protocols import VerticalExtensions

        cache_key = cls._cache_namespace()

        # Return cached extensions if available and caching enabled
        if use_cache:
            with cls._cache_manager._lock:
                cached = cls._cache_manager._cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine strict mode
        is_strict = strict if strict is not None else cls.strict_extension_loading

        # Collect errors for reporting
        errors: List["ExtensionLoadError"] = []

        def _load_extension(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            """Load an extension with error handling.

            Args:
                extension_type: Type name for error reporting
                loader: Callable that loads the extension
                is_list: If True, the extension should be a list

            Returns:
                The loaded extension, or default value on error
            """
            try:
                result = loader()
                return result
            except Exception as e:
                is_required = extension_type in cls.required_extensions
                error = ExtensionLoadError(
                    message=f"Failed to load '{extension_type}' extension for vertical '{cls.name}': {e}",
                    extension_type=extension_type,
                    vertical_name=cls.name,
                    original_error=e,
                    is_required=is_required,
                )
                errors.append(error)

                # Log the error with appropriate severity
                if is_strict or is_required:
                    logger.error(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{cls.name}': {e}",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{cls.name}': {e}"
                    )

                # Return default value
                return [] if is_list else None

        # Load each extension with error handling
        middleware = _load_extension("middleware", cls.get_middleware, is_list=True)
        safety = _load_extension("safety", cls.get_safety_extension)
        prompt = _load_extension("prompt", cls.get_prompt_contributor)
        mode_config = _load_extension("mode_config", cls.get_mode_config_provider)
        tool_deps = _load_extension("tool_deps", cls.get_tool_dependency_provider)
        workflow = _load_extension("workflow", cls.get_workflow_provider)
        service = _load_extension("service", cls.get_service_provider)
        rl_config = _load_extension("rl_config", cls.get_rl_config_provider)
        team_spec = _load_extension("team_spec", cls.get_team_spec_provider)
        enrichment = _load_extension("enrichment", cls.get_enrichment_strategy)
        tiered_tools = _load_extension("tiered_tools", cls.get_tiered_tool_config)

        # Check for critical failures (strict mode or required extensions)
        critical_errors = [e for e in errors if is_strict or e.is_required]
        if critical_errors:
            # Raise the first critical error
            raise critical_errors[0]

        # Log summary if there were non-critical errors
        if errors:
            logger.warning(
                f"Vertical '{cls.name}' loaded with {len(errors)} extension error(s). "
                f"Affected extensions: {', '.join(e.extension_type for e in errors)}"
            )

        # Build extensions object
        extensions = VerticalExtensions(
            middleware=middleware if middleware else [],
            safety_extensions=[safety] if safety else [],
            prompt_contributors=[prompt] if prompt else [],
            mode_config_provider=mode_config,
            tool_dependency_provider=tool_deps,
            workflow_provider=workflow,
            service_provider=service,
            rl_config_provider=rl_config,
            team_spec_provider=team_spec,
            enrichment_strategy=enrichment,
            tiered_tool_config=tiered_tools,
        )

        # Cache the extensions
        with cls._cache_manager._lock:
            cls._cache_manager._cache[cache_key] = extensions
        return extensions

    @classmethod
    def _get_cached_extensions(cls, use_cache: bool) -> Optional["VerticalExtensions"]:
        """Return cached VerticalExtensions if available, else None."""
        if not use_cache:
            return None
        cache_key = cls._cache_namespace()
        with cls._cache_manager._lock:
            return cls._cache_manager._cache.get(cache_key)

    @classmethod
    def _resolve_strict_mode(cls, strict: Optional[bool]) -> bool:
        """Resolve effective strict mode from explicit arg or class default."""
        return strict if strict is not None else cls.strict_extension_loading

    @classmethod
    async def _submit_extension_tasks(
        cls,
        load_fn: callable,
    ) -> Dict[str, Any]:
        """Submit all extension loads in parallel with bounded concurrency.

        Args:
            load_fn: A callable(extension_type, loader, is_list) that loads
                     a single extension with error handling.

        Returns:
            Dict mapping extension type keys to loaded results.
        """
        loop = asyncio.get_running_loop()
        executor = cls._get_shared_extension_executor()
        semaphore = cls._get_extension_load_semaphore()

        async def _submit_with_limits(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            cls._increment_loader_metric("submitted")
            cls._increment_loader_metric("queued")
            queued_now = cls._pressure_monitor.get_metric("queued")
            cls._update_loader_peak_metric("max_queued", queued_now)
            cls._check_pressure(reason=f"{extension_type}.queued")

            if semaphore.locked():
                cls._increment_loader_metric("queue_waits")

            acquired = False
            try:
                await semaphore.acquire()
                acquired = True

                cls._increment_loader_metric("queued", -1)
                cls._increment_loader_metric("in_flight")
                in_flight_now = cls._pressure_monitor.get_metric("in_flight")
                cls._update_loader_peak_metric("max_in_flight", in_flight_now)
                cls._check_pressure(reason=f"{extension_type}.in_flight")

                return await loop.run_in_executor(
                    executor,
                    lambda: load_fn(extension_type, loader, is_list),
                )
            except Exception:
                cls._increment_loader_metric("failed")
                raise
            finally:
                if acquired:
                    cls._increment_loader_metric("in_flight", -1)
                    cls._increment_loader_metric("completed")
                    semaphore.release()
                else:
                    cls._increment_loader_metric("queued", -1)
                cls._check_pressure(reason=f"{extension_type}.post")

        futures = {
            "middleware": asyncio.create_task(
                _submit_with_limits("middleware", cls.get_middleware, True)
            ),
            "safety": asyncio.create_task(_submit_with_limits("safety", cls.get_safety_extension)),
            "prompt": asyncio.create_task(
                _submit_with_limits("prompt", cls.get_prompt_contributor)
            ),
            "mode_config": asyncio.create_task(
                _submit_with_limits("mode_config", cls.get_mode_config_provider)
            ),
            "tool_deps": asyncio.create_task(
                _submit_with_limits("tool_deps", cls.get_tool_dependency_provider)
            ),
            "workflow": asyncio.create_task(
                _submit_with_limits("workflow", cls.get_workflow_provider)
            ),
            "service": asyncio.create_task(
                _submit_with_limits("service", cls.get_service_provider)
            ),
            "rl_config": asyncio.create_task(
                _submit_with_limits("rl_config", cls.get_rl_config_provider)
            ),
            "team_spec": asyncio.create_task(
                _submit_with_limits("team_spec", cls.get_team_spec_provider)
            ),
            "enrichment": asyncio.create_task(
                _submit_with_limits("enrichment", cls.get_enrichment_strategy)
            ),
            "tiered_tools": asyncio.create_task(
                _submit_with_limits("tiered_tools", cls.get_tiered_tool_config)
            ),
        }

        results = {}
        for key, task in futures.items():
            results[key] = await task
        return results

    @classmethod
    def _validate_extension_errors(
        cls,
        errors: List[Any],
        is_strict: bool,
    ) -> None:
        """Check for critical extension errors and raise or log as appropriate."""
        critical_errors = [e for e in errors if is_strict or e.is_required]
        if critical_errors:
            raise critical_errors[0]

        if errors:
            logger.warning(
                f"Vertical '{cls.name}' loaded with {len(errors)} extension error(s). "
                f"Affected extensions: {', '.join(e.extension_type for e in errors)}"
            )

    @classmethod
    def _assemble_extensions(cls, results: Dict[str, Any]) -> "VerticalExtensions":
        """Build a VerticalExtensions from a results dict."""
        from victor.core.verticals.protocols import VerticalExtensions

        return VerticalExtensions(
            middleware=results["middleware"] if results["middleware"] else [],
            safety_extensions=[results["safety"]] if results["safety"] else [],
            prompt_contributors=[results["prompt"]] if results["prompt"] else [],
            mode_config_provider=results["mode_config"],
            tool_dependency_provider=results["tool_deps"],
            workflow_provider=results["workflow"],
            service_provider=results["service"],
            rl_config_provider=results["rl_config"],
            team_spec_provider=results["team_spec"],
            enrichment_strategy=results["enrichment"],
            tiered_tool_config=results["tiered_tools"],
        )

    @classmethod
    def _cache_extensions(cls, extensions: "VerticalExtensions") -> None:
        """Store extensions in the shared cache."""
        cache_key = cls._cache_namespace()
        with cls._cache_manager._lock:
            cls._cache_manager._cache[cache_key] = extensions

    @classmethod
    async def get_extensions_async(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Async version of get_extensions that loads extensions in parallel.

        Uses a thread pool executor to load all extensions concurrently,
        providing faster initialization when extensions involve I/O.

        Shares the same cache as the synchronous get_extensions() method.

        Args:
            use_cache: If True (default), return cached extensions if available.
            strict: Override the class-level strict_extension_loading setting.

        Returns:
            VerticalExtensions containing all vertical extensions (never None)

        Raises:
            ExtensionLoadError: In strict mode or when a required extension fails
        """
        from victor.core.errors import ExtensionLoadError

        cached = cls._get_cached_extensions(use_cache)
        if cached is not None:
            return cached

        is_strict = cls._resolve_strict_mode(strict)

        errors: List[ExtensionLoadError] = []
        errors_lock = threading.Lock()

        def _load_extension(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            """Load an extension with error handling (runs in thread pool)."""
            try:
                return loader()
            except Exception as e:
                cls._increment_loader_metric("failed")
                is_required = extension_type in cls.required_extensions
                error = ExtensionLoadError(
                    message=(
                        f"Failed to load '{extension_type}' extension "
                        f"for vertical '{cls.name}': {e}"
                    ),
                    extension_type=extension_type,
                    vertical_name=cls.name,
                    original_error=e,
                    is_required=is_required,
                )
                with errors_lock:
                    errors.append(error)

                if is_strict or is_required:
                    logger.error(
                        f"[{error.correlation_id}] {extension_type} extension failed "
                        f"for vertical '{cls.name}': {e}",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        f"[{error.correlation_id}] {extension_type} extension failed "
                        f"for vertical '{cls.name}': {e}"
                    )
                return [] if is_list else None

        results = await cls._submit_extension_tasks(_load_extension)
        cls._validate_extension_errors(errors, is_strict)
        extensions = cls._assemble_extensions(results)
        cls._cache_extensions(extensions)
        return extensions

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for this vertical.

        Default implementation tries direct import pattern, then falls back
        to extension factory. Override only if custom behavior needed.

        This eliminates ~6-8 LOC of duplicated wrapper code across verticals.

        Returns:
            Workflow provider instance (WorkflowProviderProtocol) or None
        """
        provider = cls._load_named_entry_point_extension(
            "workflow_provider",
            "victor.workflow_providers",
        )
        if provider is not None:
            return provider

        vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
        class_name = f"{vertical_name}WorkflowProvider"

        candidate_paths: List[str] = []
        for suffix in ("workflows", "workflows.provider"):
            for module_path in cls._extension_module_candidates(suffix):
                if cls._extension_module_available(module_path):
                    candidate_paths.append(module_path)

        if not candidate_paths:
            return None

        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                provider_class = getattr(module, class_name, None)
                if provider_class is not None:
                    return provider_class()
            except (ImportError, AttributeError) as exc:
                last_error = exc

        for module_path in candidate_paths:
            try:
                return cls._get_extension_factory(
                    "workflow_provider",
                    module_path,
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error:
            raise last_error
        return None

    @classmethod
    def clear_extension_cache(cls, *, clear_all: bool = False) -> None:
        """Clear the extension cache for this vertical.

        Args:
            clear_all: If True, clear cache for all verticals.
                       If False (default), clear only for this class.
        """
        if clear_all:
            with cls._cache_manager._lock:
                cls._cache_manager._cache.clear()
        else:
            namespaced_key = cls._cache_namespace()
            namespaced_prefix = f"{namespaced_key}:"
            legacy_key = cls.__name__
            legacy_prefix = f"{legacy_key}:"
            with cls._cache_manager._lock:
                # Clear namespaced cache entries
                cls._cache_manager._cache.pop(namespaced_key, None)
                namespaced_keys = [
                    k for k in cls._cache_manager._cache if k.startswith(namespaced_prefix)
                ]
                for key in namespaced_keys:
                    cls._cache_manager._cache.pop(key, None)

                # Backward compatibility: clear legacy class-name-only keys.
                cls._cache_manager._cache.pop(legacy_key, None)
                legacy_keys = [
                    k
                    for k in cls._cache_manager._cache
                    if k.startswith(legacy_prefix) and k.count(":") == 1
                ]
                for key in legacy_keys:
                    cls._cache_manager._cache.pop(key, None)


class ExtensionLoaderMetricsReporter:
    """Periodic publisher for extension-loader metrics snapshots."""

    def __init__(
        self,
        *,
        interval_seconds: float = 60.0,
        topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
        source: str = "ExtensionLoaderMetricsReporter",
        reset_after_emit: bool = False,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")

        self._interval_seconds = interval_seconds
        self._topic = topic
        self._source = source
        self._reset_after_emit = reset_after_emit
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        """Return True while periodic reporter thread is active."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start periodic metrics emission if not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="ext-loader-metrics-reporter",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop periodic metrics emission."""
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout=timeout)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run_loop(self) -> None:
        """Loop that emits extension-loader metrics at fixed intervals."""
        while not self._stop_event.wait(self._interval_seconds):
            try:
                VerticalExtensionLoader.emit_extension_loader_metrics_event(
                    topic=self._topic,
                    source=self._source,
                    reset_after_emit=self._reset_after_emit,
                )
            except Exception as e:
                logger.debug("ExtensionLoaderMetricsReporter loop error: %s", e)


def start_extension_loader_metrics_reporter(
    *,
    interval_seconds: float = 60.0,
    topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
    source: str = "ExtensionLoaderMetricsReporter",
    reset_after_emit: bool = False,
) -> ExtensionLoaderMetricsReporter:
    """Start singleton periodic reporter for extension-loader metrics."""
    global _METRICS_REPORTER_SINGLETON
    with _METRICS_REPORTER_LOCK:
        if _METRICS_REPORTER_SINGLETON is None:
            _METRICS_REPORTER_SINGLETON = ExtensionLoaderMetricsReporter(
                interval_seconds=interval_seconds,
                topic=topic,
                source=source,
                reset_after_emit=reset_after_emit,
            )
        _METRICS_REPORTER_SINGLETON.start()
        return _METRICS_REPORTER_SINGLETON


def stop_extension_loader_metrics_reporter(*, timeout: float = 2.0) -> None:
    """Stop and clear singleton extension-loader metrics reporter."""
    global _METRICS_REPORTER_SINGLETON
    with _METRICS_REPORTER_LOCK:
        reporter = _METRICS_REPORTER_SINGLETON
        _METRICS_REPORTER_SINGLETON = None
    if reporter is not None:
        reporter.stop(timeout=timeout)
