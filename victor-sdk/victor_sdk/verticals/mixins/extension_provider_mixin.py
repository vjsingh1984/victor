"""Extension provider mixin for VerticalBase."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from victor_sdk.core.types import (
    CapabilityRequirementLike,
    ToolRequirementLike,
)


class ExtensionProviderMixin:
    """Opt-in mixin providing extension-provider hooks.

    These methods return providers/strategies that the framework loads
    lazily at runtime. All default to None or empty collections.

    Methods:
        get_middleware, get_safety_extension, get_prompt_contributor,
        get_mode_config_provider, get_tool_dependency_provider,
        get_workflow_provider, get_service_provider,
        get_enrichment_strategy, get_tool_selection_strategy,
        get_tiered_tool_config, get_capability_provider,
        get_handlers, get_tool_graph, get_mode_config,
        get_tool_requirements, get_capability_requirements,
    """

    @classmethod
    def get_middleware(cls) -> List[Any]:
        """Return middleware implementations for this vertical."""
        return []

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        """Return the safety extension for this vertical, if any."""
        return None

    @classmethod
    def get_prompt_contributor(cls) -> Optional[Any]:
        """Return the prompt contributor for this vertical, if any."""
        return None

    @classmethod
    def get_mode_config_provider(cls) -> Optional[Any]:
        """Return the mode-config provider for this vertical, if any."""
        return None

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[Any]:
        """Return the tool-dependency provider for this vertical, if any."""
        return None

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Return the workflow provider for this vertical, if any."""
        return None

    @classmethod
    def get_service_provider(cls) -> Optional[Any]:
        """Return the DI service provider for this vertical, if any."""
        return None

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[Any]:
        """Return the enrichment strategy for this vertical, if any."""
        return None

    @classmethod
    def get_tool_selection_strategy(cls) -> Optional[Any]:
        """Return the tool-selection strategy for this vertical, if any."""
        return None

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[Any]:
        """Return the tiered tool config for this vertical, if any."""
        return None

    @classmethod
    def get_capability_provider(cls) -> Optional[Any]:
        """Return the capability provider for this vertical, if any."""
        return None

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Return workflow step handlers for this vertical."""
        return {}

    @classmethod
    def get_tool_graph(cls) -> Optional[Any]:
        """Return the tool dependency graph for this vertical, if any."""
        return None

    @classmethod
    def get_mode_config(cls) -> Dict[str, Any]:
        """Return mode configurations for this vertical."""
        return {}

    @classmethod
    def get_tool_requirements(cls) -> List[ToolRequirementLike]:
        """Return required tools for this vertical definition."""
        return cls.get_tools()  # type: ignore[attr-defined,no-any-return]

    @classmethod
    def get_capability_requirements(cls) -> List[CapabilityRequirementLike]:
        """Return required runtime capabilities for this vertical definition."""
        return []
