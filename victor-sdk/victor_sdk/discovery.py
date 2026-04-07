"""Protocol discovery and registration.

This module provides utilities for discovering protocol implementations
via entry points and other mechanisms.

Entry Point Groups:
- victor.plugins: Main plugin registration (plugins register one or more verticals)
- victor.sdk.protocols: Protocol implementations (tool, safety, workflow, etc.)
- victor.sdk.capabilities: Capability providers (LSP, Git, etc.)
- victor.sdk.validators: Validator functions
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable

from victor_sdk.core.plugins import VictorPlugin
from victor_sdk.verticals.protocols.base import VerticalBase as SdkVerticalBase
from victor_sdk.verticals.protocols import (
    ToolProvider,
    SafetyProvider,
    WorkflowProvider,
    PromptProvider,
    TeamProvider,
    MiddlewareProvider,
    ModeConfigProvider,
    RLProvider,
    EnrichmentProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class ProtocolMetadata:
    """Metadata about a discovered protocol implementation.

    Attributes:
        name: Protocol/entry point name
        source_package: Package that registered this protocol
        version: Package version (if available)
        protocol_type: Type of protocol (e.g., "tool_provider")
        load_error: Error that occurred during loading (if any)
        is_loaded: Whether the protocol loaded successfully
    """

    name: str
    source_package: str
    protocol_type: str
    version: Optional[str] = None
    load_error: Optional[str] = None
    is_loaded: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryStats:
    """Statistics about protocol discovery.

    Attributes:
        total_verticals: Number of verticals discovered
        total_protocols: Number of protocols discovered
        total_capabilities: Number of capabilities discovered
        total_validators: Number of validators discovered
        failed_loads: Number of protocols that failed to load
    """

    total_verticals: int = 0
    total_protocols: int = 0
    total_capabilities: int = 0
    total_validators: int = 0
    failed_loads: int = 0

    def __str__(self) -> str:
        return (
            f"DiscoveryStats(verticals={self.total_verticals}, "
            f"protocols={self.total_protocols}, "
            f"capabilities={self.total_capabilities}, "
            f"validators={self.total_validators}, "
            f"failed={self.failed_loads})"
        )


class ProtocolRegistry:
    """Registry for discovering protocol implementations via entry points.

    This registry loads protocol implementations from installed packages
    that register them via entry points. It provides:

    1. Automatic discovery of verticals, protocols, capabilities, and validators
    2. Metadata tracking for all discovered implementations
    3. Error handling and reporting for failed loads
    4. Statistics about discovery results

    Example:
        registry = ProtocolRegistry()
        registry.load_from_entry_points()

        # Get discovered items
        verticals = registry.get_verticals()
        tools = registry.get_tool_providers()

        # Get metadata
        stats = registry.get_discovery_stats()
        metadata = registry.get_protocol_metadata()
    """

    # Entry point groups
    VERTICALS_GROUP = "victor.plugins"
    SDK_PROTOCOLS_GROUP = "victor.sdk.protocols"
    CAPABILITIES_GROUP = "victor.sdk.capabilities"
    VALIDATORS_GROUP = "victor.sdk.validators"

    def __init__(self, strict: bool = False) -> None:
        """Initialize the registry.

        Args:
            strict: If True, raise exceptions on load errors. If False, log errors
                   and continue loading.
        """
        self._strict = strict

        # Protocol storage
        self._tool_providers: List[ToolProvider] = []
        self._safety_providers: List[SafetyProvider] = []
        self._workflow_providers: List[WorkflowProvider] = []
        self._prompt_providers: List[PromptProvider] = []
        self._team_providers: List[TeamProvider] = []
        self._middleware_providers: List[MiddlewareProvider] = []
        self._mode_providers: List[ModeConfigProvider] = []
        self._rl_providers: List[RLProvider] = []
        self._enrichment_providers: List[EnrichmentProvider] = []

        self._capability_providers: Dict[str, Any] = {}
        self._validators: Dict[str, Callable] = {}
        self._verticals: Dict[str, Type[Any]] = {}

        # Metadata tracking
        self._protocol_metadata: Dict[str, ProtocolMetadata] = {}
        self._discovery_stats = DiscoveryStats()

    def load_from_entry_points(self, *, reload: bool = False) -> DiscoveryStats:
        """Load all protocol implementations from installed packages.

        Args:
            reload: If True, reload even if already loaded.

        Returns:
            DiscoveryStats with information about what was discovered.
        """
        if reload:
            self.clear()

        # Load verticals
        self._load_verticals()

        # Load protocol implementations
        self._load_protocols()

        # Load capability providers
        self._load_capabilities()

        # Load validators
        self._load_validators()

        return self._discovery_stats

    def _load_verticals(self) -> None:
        """Load verticals from victor.plugins entry point group."""
        try:
            for ep in importlib.metadata.entry_points(group=self.VERTICALS_GROUP):
                try:
                    candidate = ep.load()
                    for vertical_name, vertical_class in collect_verticals_from_candidate(
                        candidate
                    ).items():
                        self._verticals[vertical_name] = vertical_class
                        self._discovery_stats.total_verticals += 1
                        self._track_metadata(
                            name=vertical_name,
                            entry_point=ep,
                            protocol_type="vertical",
                        )
                except Exception as e:
                    self._handle_load_error(ep.name, "vertical", e)
        except Exception:
            logger.debug("No victor.plugins entry points found")

    def _load_protocols(self) -> None:
        """Load protocol implementations from victor.sdk.protocols group."""
        try:
            for ep in importlib.metadata.entry_points(group=self.SDK_PROTOCOLS_GROUP):
                try:
                    provider_class_or_instance = ep.load()

                    # Try to instantiate if it's a class
                    if isinstance(provider_class_or_instance, type):
                        try:
                            instance = provider_class_or_instance()
                        except Exception as e:
                            self._handle_load_error(ep.name, "protocol", e)
                            continue
                    else:
                        instance = provider_class_or_instance

                    # Register based on protocol type
                    registered = False
                    if isinstance(instance, ToolProvider):
                        self._tool_providers.append(instance)
                        registered = True
                    if isinstance(instance, SafetyProvider):
                        self._safety_providers.append(instance)
                        registered = True
                    if isinstance(instance, WorkflowProvider):
                        self._workflow_providers.append(instance)
                        registered = True
                    if isinstance(instance, PromptProvider):
                        self._prompt_providers.append(instance)
                        registered = True
                    if isinstance(instance, TeamProvider):
                        self._team_providers.append(instance)
                        registered = True
                    if isinstance(instance, MiddlewareProvider):
                        self._middleware_providers.append(instance)
                        registered = True
                    if isinstance(instance, ModeConfigProvider):
                        self._mode_providers.append(instance)
                        registered = True
                    if isinstance(instance, RLProvider):
                        self._rl_providers.append(instance)
                        registered = True
                    if isinstance(instance, EnrichmentProvider):
                        self._enrichment_providers.append(instance)
                        registered = True

                    if registered:
                        self._discovery_stats.total_protocols += 1
                        self._track_metadata(
                            name=ep.name,
                            entry_point=ep,
                            protocol_type="protocol",
                        )
                except Exception as e:
                    self._handle_load_error(ep.name, "protocol", e)
        except Exception:
            logger.debug("No victor.sdk.protocols entry points found")

    def _load_capabilities(self) -> None:
        """Load capability providers from victor.sdk.capabilities group."""
        try:
            for ep in importlib.metadata.entry_points(group=self.CAPABILITIES_GROUP):
                try:
                    capability_class_or_instance = ep.load()

                    # Try to instantiate if it's a class
                    if isinstance(capability_class_or_instance, type):
                        try:
                            instance = capability_class_or_instance()
                        except Exception as e:
                            self._handle_load_error(ep.name, "capability", e)
                            continue
                    else:
                        instance = capability_class_or_instance

                    self._capability_providers[ep.name] = instance
                    self._discovery_stats.total_capabilities += 1
                    self._track_metadata(
                        name=ep.name,
                        entry_point=ep,
                        protocol_type="capability",
                    )
                except Exception as e:
                    self._handle_load_error(ep.name, "capability", e)
        except Exception:
            logger.debug("No victor.sdk.capabilities entry points found")

    def _load_validators(self) -> None:
        """Load validators from victor.sdk.validators group."""
        try:
            for ep in importlib.metadata.entry_points(group=self.VALIDATORS_GROUP):
                try:
                    validator = ep.load()
                    self._validators[ep.name] = validator
                    self._discovery_stats.total_validators += 1
                    self._track_metadata(
                        name=ep.name,
                        entry_point=ep,
                        protocol_type="validator",
                    )
                except Exception as e:
                    self._handle_load_error(ep.name, "validator", e)
        except Exception:
            logger.debug("No victor.sdk.validators entry points found")

    def _track_metadata(self, name: str, entry_point: Any, protocol_type: str) -> None:
        """Track metadata for a discovered protocol.

        Args:
            name: Protocol/entry point name
            entry_point: EntryPoint object
            protocol_type: Type of protocol
        """
        try:
            # Try to get package version
            version = None
            if hasattr(entry_point, "dist") and entry_point.dist:
                version = getattr(entry_point.dist, "version", None)

            self._protocol_metadata[name] = ProtocolMetadata(
                name=name,
                source_package=getattr(entry_point, "value", name),
                protocol_type=protocol_type,
                version=version,
            )
        except Exception as e:
            logger.debug(f"Failed to track metadata for {name}: {e}")

    def _handle_load_error(self, name: str, protocol_type: str, error: Exception) -> None:
        """Handle a load error.

        Args:
            name: Protocol name
            protocol_type: Type of protocol
            error: Exception that occurred
        """
        self._discovery_stats.failed_loads += 1

        # Track error metadata
        self._protocol_metadata[name] = ProtocolMetadata(
            name=name,
            source_package="unknown",
            protocol_type=protocol_type,
            load_error=str(error),
            is_loaded=False,
        )

        if self._strict:
            raise RuntimeError(f"Failed to load {protocol_type} '{name}': {error}") from error
        else:
            logger.warning(f"Failed to load {protocol_type} '{name}': {error}")

    def clear(self) -> None:
        """Clear all discovered protocols and metadata."""
        self._tool_providers.clear()
        self._safety_providers.clear()
        self._workflow_providers.clear()
        self._prompt_providers.clear()
        self._team_providers.clear()
        self._middleware_providers.clear()
        self._mode_providers.clear()
        self._rl_providers.clear()
        self._enrichment_providers.clear()
        self._capability_providers.clear()
        self._validators.clear()
        self._verticals.clear()
        self._protocol_metadata.clear()
        self._discovery_stats = DiscoveryStats()

    # Getter methods
    def get_tool_providers(self) -> List[ToolProvider]:
        """Get all registered tool providers."""
        return self._tool_providers.copy()

    def get_safety_providers(self) -> List[SafetyProvider]:
        """Get all registered safety providers."""
        return self._safety_providers.copy()

    def get_workflow_providers(self) -> List[WorkflowProvider]:
        """Get all registered workflow providers."""
        return self._workflow_providers.copy()

    def get_prompt_providers(self) -> List[PromptProvider]:
        """Get all registered prompt providers."""
        return self._prompt_providers.copy()

    def get_team_providers(self) -> List[TeamProvider]:
        """Get all registered team providers."""
        return self._team_providers.copy()

    def get_middleware_providers(self) -> List[MiddlewareProvider]:
        """Get all registered middleware providers."""
        return self._middleware_providers.copy()

    def get_mode_providers(self) -> List[ModeConfigProvider]:
        """Get all registered mode providers."""
        return self._mode_providers.copy()

    def get_rl_providers(self) -> List[RLProvider]:
        """Get all registered RL providers."""
        return self._rl_providers.copy()

    def get_enrichment_providers(self) -> List[EnrichmentProvider]:
        """Get all registered enrichment providers."""
        return self._enrichment_providers.copy()

    def get_capability_provider(self, name: str) -> Optional[Any]:
        """Get a specific capability provider by name."""
        return self._capability_providers.get(name)

    def get_capability_providers(self) -> Dict[str, Any]:
        """Get all capability providers."""
        return self._capability_providers.copy()

    def get_validator(self, name: str) -> Optional[Callable]:
        """Get a specific validator by name."""
        return self._validators.get(name)

    def get_validators(self) -> Dict[str, Callable]:
        """Get all validators."""
        return self._validators.copy()

    def get_vertical(self, name: str) -> Optional[Type[Any]]:
        """Get a vertical class by name."""
        return self._verticals.get(name)

    def get_verticals(self) -> Dict[str, Type[Any]]:
        """Get all registered verticals."""
        return self._verticals.copy()

    def get_protocol_metadata(self, name: Optional[str] = None) -> Dict[str, ProtocolMetadata]:
        """Get metadata about discovered protocols.

        Args:
            name: If provided, get metadata for specific protocol.
                  If None, get metadata for all protocols.

        Returns:
            Dictionary of protocol metadata.
        """
        if name:
            return (
                {name: self._protocol_metadata.get(name)} if name in self._protocol_metadata else {}
            )
        return self._protocol_metadata.copy()

    def get_discovery_stats(self) -> DiscoveryStats:
        """Get statistics about protocol discovery.

        Returns:
            DiscoveryStats object with discovery information.
        """
        return self._discovery_stats

    def list_vertical_names(self) -> List[str]:
        """List all registered vertical names."""
        return list(self._verticals.keys())

    def list_capability_names(self) -> List[str]:
        """List all registered capability names."""
        return list(self._capability_providers.keys())

    def list_validator_names(self) -> List[str]:
        """List all registered validator names."""
        return list(self._validators.keys())

    def find_by_protocol_type(self, protocol_type: str) -> List[str]:
        """Find all protocols of a specific type.

        Args:
            protocol_type: Type to filter by (e.g., "tool_provider", "safety_provider")

        Returns:
            List of protocol names of that type.
        """
        return [
            name
            for name, meta in self._protocol_metadata.items()
            if meta.protocol_type == protocol_type and meta.is_loaded
        ]

    def get_failed_loads(self) -> List[str]:
        """Get list of protocols that failed to load.

        Returns:
            List of protocol names that had load errors.
        """
        return [name for name, meta in self._protocol_metadata.items() if not meta.is_loaded]


# Global registry instance
_global_registry: Optional[ProtocolRegistry] = None


def get_global_registry() -> ProtocolRegistry:
    """Get the global protocol registry instance.

    Returns:
        Global ProtocolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ProtocolRegistry()
        _global_registry.load_from_entry_points()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    _global_registry = None


def _looks_like_plugin(candidate: Any) -> bool:
    """Return True when *candidate* resembles the VictorPlugin protocol."""

    return all(hasattr(candidate, attr) for attr in ("name", "register", "get_cli_app"))


class _CollectingPluginContext:
    """Minimal PluginContext implementation for discovery-only plugin registration."""

    def __init__(self) -> None:
        self.verticals: list[type[Any]] = []

    def register_tool(self, tool_instance: Any) -> None:
        return None

    def register_vertical(self, vertical_class: type[Any]) -> None:
        self.verticals.append(vertical_class)

    def register_chunker(self, chunker_instance: Any) -> None:
        return None

    def register_command(self, name: str, app: Any) -> None:
        return None

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        return None

    def get_service(self, service_type: type[Any]) -> None:
        return None

    def get_settings(self) -> None:
        return None


def collect_verticals_from_candidate(candidate: Any) -> Dict[str, Type[SdkVerticalBase]]:
    """Collect SDK vertical classes from a direct vertical or VictorPlugin candidate.

    This helper is the definition-layer contract parser shared by SDK discovery,
    core runtime loading, and install/discovery smoke tests.
    """

    if isinstance(candidate, type) and issubclass(candidate, SdkVerticalBase):
        return {getattr(candidate, "name", candidate.__name__): candidate}

    plugin = candidate
    if isinstance(candidate, type):
        try:
            plugin = candidate()
        except Exception as exc:
            raise TypeError(
                "Entry point class must be an SDK VerticalBase subclass or an instantiable "
                "VictorPlugin"
            ) from exc

    if isinstance(plugin, VictorPlugin) or _looks_like_plugin(plugin):
        context = _CollectingPluginContext()
        plugin.register(context)
        return {
            getattr(vertical_class, "name", vertical_class.__name__): vertical_class
            for vertical_class in context.verticals
        }

    raise TypeError("Entry point must resolve to a VictorPlugin or an SDK VerticalBase subclass")


# Convenience functions for common operations
def discover_verticals() -> Dict[str, Type[Any]]:
    """Discover all verticals from entry points.

    Returns:
        Dictionary mapping vertical names to vertical classes.
    """
    registry = get_global_registry()
    return registry.get_verticals()


def discover_protocols(
    protocol_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover all protocols or protocols of a specific type.

    Args:
        protocol_type: Optional protocol type filter (e.g., "tool_provider")

    Returns:
        Dictionary mapping protocol names to protocol implementations.
    """
    registry = get_global_registry()

    if protocol_type == "tool_provider":
        return {f"tool_{i}": p for i, p in enumerate(registry.get_tool_providers())}
    elif protocol_type == "safety_provider":
        return {f"safety_{i}": p for i, p in enumerate(registry.get_safety_providers())}
    elif protocol_type == "workflow_provider":
        return {f"workflow_{i}": p for i, p in enumerate(registry.get_workflow_providers())}
    elif protocol_type == "prompt_provider":
        return {f"prompt_{i}": p for i, p in enumerate(registry.get_prompt_providers())}
    elif protocol_type == "team_provider":
        return {f"team_{i}": p for i, p in enumerate(registry.get_team_providers())}
    else:
        # Return all protocols as a flat dict
        all_protocols = {}
        for i, p in enumerate(registry.get_tool_providers()):
            all_protocols[f"tool_{i}"] = p
        for i, p in enumerate(registry.get_safety_providers()):
            all_protocols[f"safety_{i}"] = p
        for i, p in enumerate(registry.get_workflow_providers()):
            all_protocols[f"workflow_{i}"] = p
        for i, p in enumerate(registry.get_prompt_providers()):
            all_protocols[f"prompt_{i}"] = p
        for i, p in enumerate(registry.get_team_providers()):
            all_protocols[f"team_{i}"] = p
        return all_protocols


def get_discovery_summary() -> str:
    """Get a human-readable summary of discovered protocols.

    Returns:
        Formatted string with discovery information.
    """
    registry = get_global_registry()
    stats = registry.get_discovery_stats()
    metadata = registry.get_protocol_metadata()

    lines = [
        "=== Victor SDK Protocol Discovery Summary ===",
        f"",
        f"Statistics: {stats}",
        f"",
        f"Verticals ({stats.total_verticals}):",
    ]

    for name in sorted(registry.list_vertical_names()):
        meta = metadata.get(name)
        source = meta.source_package if meta else "unknown"
        lines.append(f"  - {name} (from {source})")

    lines.append(f"")
    lines.append(f"Protocols ({stats.total_protocols}):")

    # Count by type
    type_counts = {}
    for meta in metadata.values():
        if meta.protocol_type == "protocol":
            # This is a bit messy - let's just count the actual providers
            pass

    # Just list the provider counts
    lines.append(f"  - Tool providers: {len(registry.get_tool_providers())}")
    lines.append(f"  - Safety providers: {len(registry.get_safety_providers())}")
    lines.append(f"  - Workflow providers: {len(registry.get_workflow_providers())}")
    lines.append(f"  - Prompt providers: {len(registry.get_prompt_providers())}")
    lines.append(f"  - Team providers: {len(registry.get_team_providers())}")

    lines.append(f"")
    lines.append(f"Capabilities ({stats.total_capabilities}):")

    for name in sorted(registry.list_capability_names()):
        meta = metadata.get(name)
        source = meta.source_package if meta else "unknown"
        lines.append(f"  - {name} (from {source})")

    lines.append(f"")
    lines.append(f"Validators ({stats.total_validators}):")

    for name in sorted(registry.list_validator_names()):
        lines.append(f"  - {name}")

    if stats.failed_loads > 0:
        lines.append(f"")
        lines.append(f"Failed Loads ({stats.failed_loads}):")
        for name in registry.get_failed_loads():
            meta = metadata.get(name, {})
            lines.append(f"  - {name}: {meta.load_error if meta else 'unknown error'}")

    return "\n".join(lines)


def reload_discovery() -> DiscoveryStats:
    """Reload all protocols from entry points.

    This clears the existing registry and reloads everything.
    Useful for testing or when new packages are installed.

    Returns:
        DiscoveryStats with information about what was discovered.
    """
    reset_global_registry()
    registry = get_global_registry()
    return registry.load_from_entry_points(reload=True)
