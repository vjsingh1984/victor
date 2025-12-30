"""
Victor AI Framework Core

Multi-provider LLM orchestration, tools, and protocols.
Install with: pip install victor-core

This package provides:
- Multi-provider LLM orchestration (25+ providers)
- Tool infrastructure and generic tools
- Protocol definitions (SOLID interfaces)
- Caching, MCP, and workflow support
- CLI/TUI interfaces

For coding-specific features (Tree-sitter, LSP, code analysis),
install victor-coding: pip install victor-coding
"""

__version__ = "0.3.0"

# Re-export core modules from the main victor package
# This enables gradual migration - packages can import from
# victor (victor-core) and later from victor_coding

try:
    # Import core components that should be available from victor-core
    from victor.config.settings import Settings, get_settings
    from victor.core.container import ServiceContainer
    from victor.verticals.base import VerticalBase, VerticalRegistry
    from victor.verticals.vertical_loader import (
        VerticalLoader,
        get_vertical_loader,
        load_vertical,
        discover_vertical_plugins,
        discover_tool_plugins,
    )

    __all__ = [
        "__version__",
        # Config
        "Settings",
        "get_settings",
        # DI
        "ServiceContainer",
        # Verticals
        "VerticalBase",
        "VerticalRegistry",
        "VerticalLoader",
        "get_vertical_loader",
        "load_vertical",
        "discover_vertical_plugins",
        "discover_tool_plugins",
    ]
except ImportError:
    # During initial setup, the main victor package may not be available
    # This allows the package to be installed standalone
    __all__ = ["__version__"]
