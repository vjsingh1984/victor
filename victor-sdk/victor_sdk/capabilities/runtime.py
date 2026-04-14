"""SDK-owned capability definitions and config helpers.

These are pure Python contracts that external vertical packages can import
without depending on the Victor framework runtime package layout.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import os
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T")


class CapabilityType(str, Enum):
    """Types of orchestrator capabilities."""

    TOOL = "tool"
    PROMPT = "prompt"
    MODE = "mode"
    SAFETY = "safety"
    RL = "rl"
    TEAM = "team"
    WORKFLOW = "workflow"
    VERTICAL = "vertical"


@dataclass
class OrchestratorCapability:
    """Explicit capability declaration for orchestrator features."""

    name: str
    capability_type: CapabilityType
    version: str = "1.0"
    setter: Optional[str] = None
    getter: Optional[str] = None
    attribute: Optional[str] = None
    description: str = ""
    required: bool = False
    deprecated: bool = False
    deprecated_message: str = ""

    def __post_init__(self) -> None:
        if not any([self.setter, self.getter, self.attribute]):
            raise ValueError(
                f"Capability '{self.name}' must specify at least one of: "
                "setter, getter, or attribute"
            )
        if not self._is_valid_version(self.version):
            raise ValueError(
                f"Capability '{self.name}' has invalid version '{self.version}'. "
                "Expected format: 'MAJOR.MINOR' (e.g., '1.0', '2.1')"
            )

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        try:
            parts = version.split(".")
            if len(parts) != 2:
                return False
            major, minor = int(parts[0]), int(parts[1])
            return major >= 0 and minor >= 0
        except (ValueError, AttributeError):
            return False


@dataclass
class CapabilityEntry:
    """Entry for a dynamically loaded capability."""

    capability: OrchestratorCapability
    handler: Optional[Callable[..., Any]] = None
    getter_handler: Optional[Callable[..., Any]] = None
    source_module: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.capability.name

    @property
    def version(self) -> str:
        return self.capability.version

    @property
    def capability_type(self) -> CapabilityType:
        return self.capability.capability_type


def capability(
    name: str,
    capability_type: CapabilityType = CapabilityType.TOOL,
    version: str = "1.0",
    description: Optional[str] = None,
    setter: Optional[str] = None,
    getter: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to mark a function as a capability handler."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._capability_meta = {  # type: ignore[attr-defined]
            "name": name,
            "capability_type": capability_type,
            "version": version,
            "description": description or func.__doc__ or "",
            "setter": setter or name,
            "getter": getter,
        }
        return func

    return decorator


@dataclass
class CapabilityMetadata:
    """Metadata for a registered capability."""

    name: str
    description: str
    version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class BaseCapabilityProvider(Generic[T]):
    """Base provider for named capability objects."""

    def get_capabilities(self) -> Dict[str, T]:
        raise NotImplementedError

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        raise NotImplementedError

    def get_capability(self, name: str) -> Optional[T]:
        return self.get_capabilities().get(name)

    def list_capabilities(self) -> List[str]:
        return list(self.get_capabilities().keys())

    def has_capability(self, name: str) -> bool:
        return name in self.get_capabilities()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "access_count": getattr(self, "_access_count", 0),
            "last_accessed": getattr(self, "_last_accessed", 0.0),
            "error_count": getattr(self, "_error_count", 0),
            "capability_count": len(self.get_capabilities()),
        }

    def record_access(self, capability_name: Optional[str] = None) -> None:
        self._access_count = getattr(self, "_access_count", 0) + 1
        self._last_accessed = time.time()

    def record_error(self, capability_name: Optional[str] = None) -> None:
        self._error_count = getattr(self, "_error_count", 0) + 1

    def get_observability_data(self) -> Dict[str, Any]:
        return {
            "capabilities": self.list_capabilities(),
            "metadata": {
                name: {
                    "description": metadata.description,
                    "version": metadata.version,
                    "dependencies": list(metadata.dependencies),
                    "tags": list(metadata.tags),
                }
                for name, metadata in self.get_capability_metadata().items()
            },
            "metrics": self.get_metrics(),
        }


DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY = "__global__"


class CapabilityConfigMergePolicy(str, Enum):
    """Merge behavior when writing capability configuration."""

    REPLACE = "replace"
    SHALLOW_MERGE = "shallow_merge"


class CapabilityConfigService:
    """Centralized runtime store for capability configuration."""

    def __init__(self) -> None:
        self._configs_by_scope: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def normalize_scope_key(scope_key: Optional[str]) -> str:
        if scope_key is None:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY
        normalized = str(scope_key).strip()
        return normalized or DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    def _get_scope_bucket(
        self,
        scope_key: Optional[str],
        *,
        create: bool = False,
    ) -> Dict[str, Any]:
        normalized_scope_key = self.normalize_scope_key(scope_key)
        if create:
            return self._configs_by_scope.setdefault(normalized_scope_key, {})
        return self._configs_by_scope.get(normalized_scope_key, {})

    def has_config(self, name: str, *, scope_key: Optional[str] = None) -> bool:
        return name in self._get_scope_bucket(scope_key)

    def get_config(self, name: str, default: Any = None, *, scope_key: Optional[str] = None) -> Any:
        return self._get_scope_bucket(scope_key).get(name, default)

    def set_config(
        self,
        name: str,
        config: Any,
        *,
        merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
        scope_key: Optional[str] = None,
    ) -> Any:
        bucket = self._get_scope_bucket(scope_key, create=True)
        if (
            merge_policy == CapabilityConfigMergePolicy.SHALLOW_MERGE
            and isinstance(bucket.get(name), dict)
            and isinstance(config, dict)
        ):
            merged = dict(bucket[name])
            merged.update(config)
            bucket[name] = merged
            return merged

        bucket[name] = config
        return config

    def apply_configs(
        self,
        configs: Dict[str, Any],
        *,
        merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
        scope_key: Optional[str] = None,
    ) -> None:
        for name, config in configs.items():
            self.set_config(name, config, merge_policy=merge_policy, scope_key=scope_key)

    def clear(
        self,
        name: Optional[str] = None,
        *,
        scope_key: Optional[str] = None,
        clear_all_scopes: bool = False,
    ) -> None:
        if clear_all_scopes:
            self._configs_by_scope.clear()
            return

        normalized_scope_key = self.normalize_scope_key(scope_key)
        if name is None:
            self._configs_by_scope.pop(normalized_scope_key, None)
            return
        bucket = self._configs_by_scope.get(normalized_scope_key)
        if bucket is None:
            return
        bucket.pop(name, None)
        if not bucket:
            self._configs_by_scope.pop(normalized_scope_key, None)


@runtime_checkable
class CapabilityConfigScopePortProtocol(Protocol):
    """Protocol for orchestrators exposing capability-config scope identity."""

    def get_capability_config_scope_key(self) -> str:
        """Return the scope key used for capability configuration isolation."""


@runtime_checkable
class CapabilityLoaderPortProtocol(Protocol):
    """Protocol for host capability loaders used by extracted verticals."""

    def _register_capability_internal(
        self,
        *,
        capability: Any,
        handler: Optional[Callable[..., Any]] = None,
        getter_handler: Optional[Callable[..., Any]] = None,
        source_module: Optional[str] = None,
    ) -> Any:
        """Register one capability entry with the host loader."""


def register_capability_entries(
    loader: CapabilityLoaderPortProtocol,
    entries: Iterable[CapabilityEntry],
    *,
    source_module: Optional[str] = None,
) -> CapabilityLoaderPortProtocol:
    """Populate a host capability loader from SDK capability entries."""

    for entry in entries:
        loader._register_capability_internal(
            capability=entry.capability,
            handler=entry.handler,
            getter_handler=entry.getter_handler,
            source_module=source_module or entry.source_module,
        )
    return loader


def build_capability_loader(
    entries: Iterable[CapabilityEntry],
    *,
    loader_factory: Callable[[], CapabilityLoaderPortProtocol],
    source_module: Optional[str] = None,
) -> CapabilityLoaderPortProtocol:
    """Create and populate a host capability loader via explicit dependency injection."""

    return register_capability_entries(
        loader_factory(),
        entries,
        source_module=source_module,
    )


def resolve_capability_config_service(orchestrator: Any) -> Optional[CapabilityConfigService]:
    """Resolve CapabilityConfigService via a structurally typed service container."""

    get_container = getattr(orchestrator, "get_service_container", None)
    container = (
        get_container() if callable(get_container) else getattr(orchestrator, "container", None)
    )
    if container is None or not hasattr(container, "get_optional"):
        return None

    try:
        service = container.get_optional(CapabilityConfigService)
    except Exception:
        return None

    return service if isinstance(service, CapabilityConfigService) else None


def resolve_capability_config_scope_key(orchestrator: Any) -> str:
    """Resolve capability config scope key using the explicit port first."""

    def _normalize(scope_key: Any) -> str:
        if scope_key is None:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY
        normalized = str(scope_key).strip()
        return normalized or DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    if isinstance(orchestrator, CapabilityConfigScopePortProtocol):
        try:
            return _normalize(orchestrator.get_capability_config_scope_key())
        except Exception:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    getter = getattr(orchestrator, "get_capability_config_scope_key", None)
    if callable(getter):
        try:
            return _normalize(getter())
        except Exception:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    for attr_name in ("capability_config_scope_key", "active_session_id", "session_id"):
        attr_value = getattr(orchestrator, attr_name, None)
        if attr_value:
            return _normalize(attr_value)

    return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY


def _ensure_not_private_fallback(attr_name: str, *, operation: str) -> None:
    if not attr_name.startswith("_"):
        return
    strict_mode = os.getenv("VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS", "").strip().lower()
    if strict_mode not in {"1", "true", "yes", "on"}:
        return
    raise RuntimeError(
        f"Private attribute fallback blocked for capability config {operation}: '{attr_name}'. "
        "Disable strict mode by unsetting VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS."
    )


def load_capability_config(
    orchestrator: Any,
    name: str,
    defaults: Dict[str, Any],
    *,
    fallback_attr: Optional[str] = None,
    legacy_service_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Load capability config from service-first storage with legacy fallback."""

    default_copy = deepcopy(defaults)
    service = resolve_capability_config_service(orchestrator)
    scope_key = resolve_capability_config_scope_key(orchestrator)
    if service is not None:
        if legacy_service_names:
            if service.has_config(name, scope_key=scope_key):
                return service.get_config(name, default_copy, scope_key=scope_key)
            for legacy_name in legacy_service_names:
                if service.has_config(legacy_name, scope_key=scope_key):
                    return service.get_config(legacy_name, default_copy, scope_key=scope_key)
            return default_copy

        return service.get_config(name, default_copy, scope_key=scope_key)

    target_attr = fallback_attr or name
    _ensure_not_private_fallback(target_attr, operation="read")
    return getattr(orchestrator, target_attr, default_copy)


def store_capability_config(
    orchestrator: Any,
    name: str,
    config: Dict[str, Any],
    *,
    fallback_attr: Optional[str] = None,
    require_existing_attr: bool = True,
    merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
) -> bool:
    """Store capability config in service-first storage with legacy fallback."""

    service = resolve_capability_config_service(orchestrator)
    scope_key = resolve_capability_config_scope_key(orchestrator)
    if service is not None:
        service.set_config(name, config, merge_policy=merge_policy, scope_key=scope_key)
        return True

    target_attr = fallback_attr or name
    _ensure_not_private_fallback(target_attr, operation="write")
    if not require_existing_attr or hasattr(orchestrator, target_attr):
        setattr(orchestrator, target_attr, config)
    return False


def update_capability_config_section(
    orchestrator: Any,
    *,
    root_name: str,
    section_name: str,
    section_config: Dict[str, Any],
    root_defaults: Dict[str, Any],
    fallback_attr: Optional[str] = None,
    require_existing_attr: bool = True,
) -> Dict[str, Any]:
    """Merge one section into a grouped capability config."""

    root_config = load_capability_config(
        orchestrator,
        root_name,
        root_defaults,
        fallback_attr=fallback_attr,
    )
    merged = dict(root_config) if isinstance(root_config, dict) else deepcopy(root_defaults)
    merged[section_name] = section_config
    store_capability_config(
        orchestrator,
        root_name,
        merged,
        fallback_attr=fallback_attr,
        require_existing_attr=require_existing_attr,
    )
    return merged


__all__ = [
    "BaseCapabilityProvider",
    "CapabilityConfigMergePolicy",
    "CapabilityConfigScopePortProtocol",
    "CapabilityConfigService",
    "CapabilityEntry",
    "CapabilityLoaderPortProtocol",
    "CapabilityMetadata",
    "CapabilityType",
    "DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY",
    "OrchestratorCapability",
    "build_capability_loader",
    "capability",
    "load_capability_config",
    "register_capability_entries",
    "resolve_capability_config_scope_key",
    "resolve_capability_config_service",
    "store_capability_config",
    "update_capability_config_section",
]
