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

Protocol definitions are now canonically defined in victor-sdk to allow
external verticals to depend on the SDK only. This module re-exports them
for backward compatibility.

Preferred import path (external verticals):
    from victor_sdk.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
        ModeConfigProviderProtocol,
    )

Legacy import path (still works):
    from victor.core.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
        ModeConfigProviderProtocol,
    )
"""

# =============================================================================
# Re-export promoted protocols from SDK (canonical source)
# =============================================================================

try:
    from victor_sdk.verticals.protocols.promoted import (
        # Tool Selection
        ToolSelectionStrategyProtocol,
        VerticalToolSelectionProviderProtocol,
        TieredToolConfigProviderProtocol,
        VerticalTieredToolProviderProtocol,
        # Safety
        SafetyExtensionProtocol,
        # Team
        TeamSpecProviderProtocol,
        VerticalTeamProviderProtocol,
        # Middleware
        MiddlewareProtocol,
        # Prompt
        PromptContributorProtocol,
        # Mode
        ModeConfigProviderProtocol,
        # Workflow
        WorkflowProviderProtocol,
        VerticalWorkflowProviderProtocol,
        # Service
        ServiceProviderProtocol,
        # RL
        RLConfigProviderProtocol,
        VerticalRLProviderProtocol,
        # Enrichment
        EnrichmentStrategyProtocol,
        VerticalEnrichmentProviderProtocol,
        # Capability
        CapabilityProviderProtocol,
        ChainProviderProtocol,
        PersonaProviderProtocol,
        VerticalPersonaProviderProtocol,
        # Stage Contract
        StageContract,
        StageValidator,
        StageValidationResult,
        ValidationError,
        validate_stage_contract,
        StageContractMixin,
    )

    # Re-export promoted data types from SDK
    from victor_sdk.verticals.protocols.promoted_types import (
        MiddlewarePriority,
        MiddlewareResult,
        ModeConfig,
        ToolSelectionContext,
        ToolSelectionResult,
    )

    _SDK_AVAILABLE = True
except ImportError:
    # SDK not installed yet (e.g., during victor-ai installation)
    # These will be available after victor-sdk is installed
    _SDK_AVAILABLE = False

# Re-export SafetyPattern from its original location (dataclass, not Protocol)
from victor.security.safety.types import SafetyPattern

# Re-export TaskTypeHint from its original location (dataclass)
from victor.core.vertical_types import TaskTypeHint

# Backward compat alias for ToolSelectionContext
if _SDK_AVAILABLE:
    VerticalToolSelectionContext = ToolSelectionContext

# =============================================================================
# ISP-Compliant Vertical Providers (re-exported from SDK)
# =============================================================================

if _SDK_AVAILABLE:
    from victor_sdk.verticals.protocols import (
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
    )

# These provider protocols are defined locally (have methods not in SDK versions)
from victor.core.verticals.protocols.providers import (
    PromptContributorProvider,
    ToolDependencyProvider,
    TieredToolConfigProvider,
    ServiceProvider,
)

# Re-export from core for backward compatibility
from victor.core.tool_types import ToolDependency, ToolDependencyProviderProtocol
from victor.core.vertical_types import StageDefinition, TieredToolConfig

# =============================================================================
# Composite Vertical Extension
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    """

    middleware: List["MiddlewareProtocol"] = field(default_factory=list)
    safety_extensions: List["SafetyExtensionProtocol"] = field(default_factory=list)
    prompt_contributors: List["PromptContributorProtocol"] = field(default_factory=list)
    mode_config_provider: Optional["ModeConfigProviderProtocol"] = None
    tool_dependency_provider: Optional["ToolDependencyProviderProtocol"] = None
    workflow_provider: Optional["WorkflowProviderProtocol"] = None
    service_provider: Optional["ServiceProviderProtocol"] = None
    rl_config_provider: Optional["RLConfigProviderProtocol"] = None
    team_spec_provider: Optional["TeamSpecProviderProtocol"] = None
    enrichment_strategy: Optional["EnrichmentStrategyProtocol"] = None
    tool_selection_strategy: Optional["ToolSelectionStrategyProtocol"] = None
    tiered_tool_config: Optional[TieredToolConfig] = None

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

    def get_all_mode_configs(self) -> Dict[str, "ModeConfig"]:
        """Get mode configs from provider.

        Returns:
            Dict of mode configurations
        """
        if self.mode_config_provider:
            return self.mode_config_provider.get_mode_configs()
        return {}


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
    # Stage Contract (Phase 2: LSP Compliance)
    "StageContract",
    "StageValidator",
    "StageValidationResult",
    "ValidationError",
    "validate_stage_contract",
    "StageContractMixin",
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
]
