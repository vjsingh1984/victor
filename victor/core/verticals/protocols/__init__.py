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

"""Vertical Extension Protocols Package (ISP-Compliant).

This package provides segregated protocol interfaces for vertical-framework
integration, following the Interface Segregation Principle (ISP).

Each module contains protocols focused on a single responsibility:
- tool_provider: Tool selection and dependency protocols
- safety_provider: Safety pattern protocols
- team_provider: Multi-agent team protocols
- middleware: Tool execution middleware protocols
- prompt_provider: Prompt contribution protocols
- mode_provider: Mode configuration protocols
- workflow_provider: Workflow management protocols
- service_provider: DI service registration protocols
- rl_provider: Reinforcement learning protocols
- enrichment: Prompt enrichment protocols
- capability_provider: Capability, chain, and persona protocols

Usage:
    # Import from specific module (preferred for ISP)
    from victor.core.verticals.protocols.tool_provider import (
        ToolSelectionStrategyProtocol,
    )

    # Or import from package root (for convenience)
    from victor.core.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Tool Provider
from victor.core.verticals.protocols.tool_provider import (
    ToolSelectionContext,
    ToolSelectionResult,
    ToolSelectionStrategyProtocol,
    VerticalToolSelectionProviderProtocol,
    TieredToolConfigProviderProtocol,
    VerticalTieredToolProviderProtocol,
)

# Safety Provider
from victor.core.verticals.protocols.safety_provider import (
    SafetyExtensionProtocol,
)

# Import SafetyPattern at runtime for re-export
# (defined in victor.security.safety.types, re-exported from victor.security_analysis)
from victor.security.safety.types import SafetyPattern

# Team Provider
from victor.core.verticals.protocols.team_provider import (
    TeamSpecProviderProtocol,
    VerticalTeamProviderProtocol,
)

# Middleware
from victor.core.verticals.protocols.middleware import (
    MiddlewareProtocol,
    MiddlewarePriority,
    MiddlewareResult,
)

# Prompt Provider
from victor.core.verticals.protocols.prompt_provider import (
    PromptContributorProtocol,
    TaskTypeHint,
)

# Mode Provider
from victor.core.verticals.protocols.mode_provider import (
    ModeConfig,
    ModeConfigProviderProtocol,
)

# Workflow Provider
from victor.core.verticals.protocols.workflow_provider import (
    WorkflowProviderProtocol,
    VerticalWorkflowProviderProtocol,
)

# Service Provider
from victor.core.verticals.protocols.service_provider import (
    ServiceProviderProtocol,
)

# RL Provider
from victor.core.verticals.protocols.rl_provider import (
    RLConfigProviderProtocol,
    VerticalRLProviderProtocol,
)

# Enrichment
from victor.core.verticals.protocols.enrichment import (
    EnrichmentStrategyProtocol,
    VerticalEnrichmentProviderProtocol,
)

# Capability Provider
from victor.core.verticals.protocols.capability_provider import (
    CapabilityProviderProtocol,
    ChainProviderProtocol,
    PersonaProviderProtocol,
    VerticalPersonaProviderProtocol,
)

# Dynamic Extensions (OCP-Compliant)
from victor.core.verticals.protocols.extension import (
    StandardExtensionTypes,
    IExtension,
    IExtensionRegistry,
    ExtensionMetadata,
)

# ISP-Compliant Vertical Providers
from victor.core.verticals.protocols.providers import (
    MiddlewareProvider,
    SafetyProvider,
    WorkflowProvider,
    TeamProvider,
    RLProvider,
    EnrichmentProvider,
    ToolProvider,
    HandlerProvider,
    CapabilityProvider,
    ModeConfigProvider,
    PromptContributorProvider,
    ToolDependencyProvider,
    TieredToolConfigProvider,
    ServiceProvider,
)

# Re-export from core for backward compatibility
from victor.core.tool_types import ToolDependency, ToolDependencyProviderProtocol
from victor.core.vertical_types import StageDefinition, TieredToolConfig

# Protocol utilities (Phase 11.1)
from victor.core.verticals.protocols.utils import (
    check_protocol,
    check_protocol_optional,
    is_protocol_conformant,
    get_protocol_methods,
    protocol_error_message,
    require_protocol,
)

# =============================================================================
# Composite Vertical Extension
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerticalExtensions:
    """Container for all vertical extension implementations.

    Aggregates all extension protocols for a vertical, making it easy
    to pass vertical capabilities to framework components.

    Attributes:
        middleware: List of middleware implementations
        safety_extensions: List of safety extensions
        prompt_contributors: List of prompt contributors
        mode_config_provider: Mode configuration provider
        tool_dependency_provider: Tool dependency provider
        workflow_provider: Workflow provider
        service_provider: Service provider
        enrichment_strategy: Prompt enrichment strategy for DSPy-like optimization
        tool_selection_strategy: Strategy for vertical-specific tool selection
        tiered_tool_config: Tiered tool configuration for context-efficient selection
        _dynamic_extensions: Dynamic extensions from registry (OCP-compliant)
    """

    middleware: List[MiddlewareProtocol] = field(default_factory=list)
    safety_extensions: List[SafetyExtensionProtocol] = field(default_factory=list)
    prompt_contributors: List[PromptContributorProtocol] = field(default_factory=list)
    mode_config_provider: Optional[ModeConfigProviderProtocol] = None
    tool_dependency_provider: Optional[ToolDependencyProviderProtocol] = None
    workflow_provider: Optional[WorkflowProviderProtocol] = None
    service_provider: Optional[ServiceProviderProtocol] = None
    rl_config_provider: Optional[RLConfigProviderProtocol] = None
    team_spec_provider: Optional[TeamSpecProviderProtocol] = None
    enrichment_strategy: Optional[EnrichmentStrategyProtocol] = None
    tool_selection_strategy: Optional[ToolSelectionStrategyProtocol] = None
    tiered_tool_config: Optional[Any] = None  # TieredToolConfig
    _dynamic_extensions: Dict[str, List["IExtension"]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def get_all_task_hints(self) -> Dict[str, TaskTypeHint]:
        """Merge task hints from all contributors.

        Later contributors override earlier ones for same task type.

        Returns:
            Merged dict of task type hints
        """
        merged = {}
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            merged.update(contributor.get_task_type_hints())
        return merged

    def get_all_safety_patterns(self) -> List[SafetyPattern]:
        """Collect safety patterns from all extensions.

        Returns:
            Combined list of safety patterns
        """
        patterns = []
        for ext in self.safety_extensions:
            patterns.extend(ext.get_bash_patterns())
            patterns.extend(ext.get_file_patterns())
        return patterns

    def get_all_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configs from provider.

        Returns:
            Dict of mode configurations
        """
        if self.mode_config_provider:
            return self.mode_config_provider.get_mode_configs()
        return {}

    def get_extension(self, extension_type: str) -> List["IExtension"]:
        """Get dynamic extensions by type.

        Retrieves extensions from the dynamic extension registry,
        enabling OCP compliance by supporting unlimited extension types.

        Args:
            extension_type: Type of extension to retrieve (e.g., "analytics",
                          "tools", "middleware")

        Returns:
            List of extensions of the specified type (empty list if none)
        """
        return self._dynamic_extensions.get(extension_type, [])


__all__ = [
    # Tool Selection
    "ToolSelectionContext",
    "ToolSelectionResult",
    "ToolSelectionStrategyProtocol",
    "VerticalToolSelectionProviderProtocol",
    "TieredToolConfigProviderProtocol",
    "VerticalTieredToolProviderProtocol",
    # Safety
    "SafetyExtensionProtocol",
    "SafetyPattern",
    # Team
    "TeamSpecProviderProtocol",
    "VerticalTeamProviderProtocol",
    # Middleware
    "MiddlewareProtocol",
    "MiddlewarePriority",
    "MiddlewareResult",
    # Prompt
    "PromptContributorProtocol",
    "TaskTypeHint",
    # Mode
    "ModeConfig",
    "ModeConfigProviderProtocol",
    # Workflow
    "WorkflowProviderProtocol",
    "VerticalWorkflowProviderProtocol",
    # Service
    "ServiceProviderProtocol",
    # RL
    "RLConfigProviderProtocol",
    "VerticalRLProviderProtocol",
    # Enrichment
    "EnrichmentStrategyProtocol",
    "VerticalEnrichmentProviderProtocol",
    # Capability
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "VerticalPersonaProviderProtocol",
    # Dynamic Extensions (OCP-Compliant)
    "StandardExtensionTypes",
    "IExtension",
    "IExtensionRegistry",
    "ExtensionMetadata",
    # Re-exports from core
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    "StageDefinition",
    "TieredToolConfig",
    # Composite
    "VerticalExtensions",
    # ISP-Compliant Vertical Providers
    "MiddlewareProvider",
    "SafetyProvider",
    "WorkflowProvider",
    "TeamProvider",
    "RLProvider",
    "EnrichmentProvider",
    "ToolProvider",
    "HandlerProvider",
    "CapabilityProvider",
    "ModeConfigProvider",
    "PromptContributorProvider",
    "ToolDependencyProvider",
    "TieredToolConfigProvider",
    "ServiceProvider",
    # Protocol utilities (Phase 11.1)
    "check_protocol",
    "check_protocol_optional",
    "is_protocol_conformant",
    "get_protocol_methods",
    "protocol_error_message",
    "require_protocol",
]
