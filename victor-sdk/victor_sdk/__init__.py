"""
Victor SDK - Protocol definitions for vertical development.

This package provides pure protocol/ABC definitions that external verticals
can depend on without pulling in the entire Victor framework.

Version: 1.0.0a1
"""

__version__ = "1.0.0"

# Core types
from victor_sdk.core.types import (
    VerticalConfig,
    StageDefinition,
    TieredToolConfig,
    ToolSet,
)

# Core exceptions
from victor_sdk.core.exceptions import (
    VerticalException,
    VerticalConfigurationError,
    VerticalProtocolError,
)

# Vertical protocols
from victor_sdk.verticals.protocols.base import VerticalBase

# Discovery and registration
from victor_sdk.discovery import (
    ProtocolRegistry,
    ProtocolMetadata,
    DiscoveryStats,
    get_global_registry,
    reset_global_registry,
    discover_verticals,
    discover_protocols,
    get_discovery_summary,
    reload_discovery,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "VerticalConfig",
    "StageDefinition",
    "TieredToolConfig",
    "ToolSet",
    # Exceptions
    "VerticalException",
    "VerticalConfigurationError",
    "VerticalProtocolError",
    # Base class
    "VerticalBase",
    # Discovery (Phase 4)
    "ProtocolRegistry",
    "ProtocolMetadata",
    "DiscoveryStats",
    "get_global_registry",
    "reset_global_registry",
    "discover_verticals",
    "discover_protocols",
    "get_discovery_summary",
    "reload_discovery",
]
