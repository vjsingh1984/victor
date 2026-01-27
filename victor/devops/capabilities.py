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

"""Dynamic capability definitions for the DevOps vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the DevOps vertical with custom functionality.

Refactored to use BaseVerticalCapabilityProvider, reducing from
696 lines to ~280 lines by eliminating duplicated patterns.

Example:
    # Use provider
    from victor.devops.capabilities import DevOpsCapabilityProvider

    provider = DevOpsCapabilityProvider()

    # Apply capabilities
    provider.apply_deployment_safety(orchestrator, require_approval_for_production=True)
    provider.apply_container_settings(orchestrator, runtime="podman")

    # Get configurations
    config = provider.get_capability_config(orchestrator, "deployment_safety")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, cast

from victor.framework.capabilities.base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
)
from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Handlers (configure_*, get_* functions)
# =============================================================================


def configure_deployment_safety(
    orchestrator: Any,
    *,
    require_approval_for_production: bool = True,
    require_backup_before_deploy: bool = True,
    enable_rollback: bool = True,
    protected_environments: Optional[List[str]] = None,
) -> None:
    """Configure deployment safety rules for the orchestrator.

    This capability configures the orchestrator's deployment safety
    checks to prevent dangerous operations.

    Args:
        orchestrator: Target orchestrator
        require_approval_for_production: Require approval for production deployments
        require_backup_before_deploy: Require backup before deployment
        enable_rollback: Enable automatic rollback on failure
        protected_environments: List of environments that require extra caution
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    deployment_safety_config = {
        "require_approval_for_production": require_approval_for_production,
        "require_backup_before_deploy": require_backup_before_deploy,
        "enable_rollback": enable_rollback,
        "protected_environments": protected_environments or ["production", "staging"],
    }
    context.set_capability_config("deployment_safety", deployment_safety_config)

    logger.info("Configured deployment safety rules")


def configure_container_settings(
    orchestrator: Any,
    *,
    runtime: str = "docker",
    default_registry: Optional[str] = None,
    security_scan_enabled: bool = True,
    max_image_size_mb: int = 2000,
) -> None:
    """Configure container settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        runtime: Container runtime to use (docker, podman)
        default_registry: Default container registry URL
        security_scan_enabled: Enable security scanning for images
        max_image_size_mb: Maximum allowed image size in MB
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    container_config = {
        "runtime": runtime,
        "default_registry": default_registry,
        "security_scan_enabled": security_scan_enabled,
        "max_image_size_mb": max_image_size_mb,
    }
    context.set_capability_config("container_settings", container_config)

    logger.info(f"Configured container settings: runtime={runtime}")


def get_container_settings(orchestrator: Any) -> Dict[str, Any]:
    """Get current container configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Container configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    config = context.get_capability_config(
        "container_settings",
        {
            "runtime": "docker",
            "default_registry": None,
            "security_scan_enabled": True,
            "max_image_size_mb": 2000,
        },
    )
    return cast(Dict[str, Any], config)


def configure_infrastructure_settings(
    orchestrator: Any,
    *,
    iac_tool: str = "terraform",
    auto_approve_non_destructive: bool = False,
    require_plan_before_apply: bool = True,
    state_backend: Optional[str] = None,
) -> None:
    """Configure Infrastructure as Code settings.

    Args:
        orchestrator: Target orchestrator
        iac_tool: IaC tool to use (terraform, opentofu, pulumi, cloudformation)
        auto_approve_non_destructive: Auto-approve non-destructive changes
        require_plan_before_apply: Require plan step before apply
        state_backend: Backend for storing IaC state
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    infra_config = {
        "iac_tool": iac_tool,
        "auto_approve_non_destructive": auto_approve_non_destructive,
        "require_plan_before_apply": require_plan_before_apply,
        "state_backend": state_backend,
    }
    context.set_capability_config("infrastructure_settings", infra_config)

    logger.info(f"Configured infrastructure settings: iac_tool={iac_tool}")


def configure_cicd_settings(
    orchestrator: Any,
    *,
    platform: str = "github_actions",
    run_tests_before_deploy: bool = True,
    require_passing_checks: bool = True,
    enable_security_scan: bool = True,
) -> None:
    """Configure CI/CD pipeline settings.

    Args:
        orchestrator: Target orchestrator
        platform: CI/CD platform (github_actions, gitlab_ci, jenkins)
        run_tests_before_deploy: Run tests before deployment
        require_passing_checks: Require all checks to pass
        enable_security_scan: Enable security scanning in pipeline
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    cicd_config = {
        "platform": platform,
        "run_tests_before_deploy": run_tests_before_deploy,
        "require_passing_checks": require_passing_checks,
        "enable_security_scan": enable_security_scan,
    }
    context.set_capability_config("cicd_settings", cicd_config)

    logger.info(f"Configured CI/CD settings: platform={platform}")


def configure_monitoring_settings(
    orchestrator: Any,
    *,
    metrics_backend: str = "prometheus",
    logging_backend: str = "loki",
    alerting_enabled: bool = True,
    dashboard_tool: str = "grafana",
) -> None:
    """Configure monitoring and observability settings.

    Args:
        orchestrator: Target orchestrator
        metrics_backend: Metrics collection backend
        logging_backend: Log aggregation backend
        alerting_enabled: Enable alerting
        dashboard_tool: Dashboard visualization tool
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    monitoring_config = {
        "metrics_backend": metrics_backend,
        "logging_backend": logging_backend,
        "alerting_enabled": alerting_enabled,
        "dashboard_tool": dashboard_tool,
    }
    context.set_capability_config("monitoring_settings", monitoring_config)

    logger.info(f"Configured monitoring settings: metrics={metrics_backend}")


# =============================================================================
# Capability Provider Class (Refactored to use BaseVerticalCapabilityProvider)
# =============================================================================


class DevOpsCapabilityProvider(BaseVerticalCapabilityProvider):
    """Provider for DevOps-specific capabilities.

    Refactored to inherit from BaseVerticalCapabilityProvider, eliminating
    ~420 lines of duplicated boilerplate code.

    Example:
        provider = DevOpsCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_deployment_safety(orchestrator, require_approval_for_production=True)
        provider.apply_container_settings(orchestrator, runtime="podman")

        # Get configurations
        config = provider.get_capability_config(orchestrator, "deployment_safety")
    """

    def __init__(self) -> None:
        """Initialize the DevOps capability provider."""
        super().__init__("devops")

    def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
        """Define DevOps capability definitions.

        Returns:
            Dictionary of DevOps capability definitions

        Note: Internal keys use _settings suffix for backward compatibility,
        but external names (generated in generate_capabilities_list) use
        shorter names without the suffix.
        """
        from victor.framework.protocols import CapabilityType

        return {
            "deployment_safety": CapabilityDefinition(
                name="deployment_safety",  # External name: devops_deployment_safety
                type=CapabilityType.SAFETY,
                description="Deployment safety rules for preventing dangerous operations",
                version="0.5.0",
                configure_fn="configure_deployment_safety",
                default_config={
                    "require_approval_for_production": True,
                    "require_backup_before_deploy": True,
                    "enable_rollback": True,
                    "protected_environments": ["production", "staging"],
                },
                tags=["safety", "deployment", "production"],
            ),
            "container_settings": CapabilityDefinition(
                name="container",  # External name: devops_container (without _settings)
                type=CapabilityType.TOOL,
                description="Container management and configuration",
                version="0.5.0",
                configure_fn="configure_container_settings",
                get_fn="get_container_settings",
                default_config={
                    "runtime": "docker",
                    "default_registry": None,
                    "security_scan_enabled": True,
                    "max_image_size_mb": 2000,
                },
                tags=["docker", "container", "podman"],
            ),
            "infrastructure_settings": CapabilityDefinition(
                name="infrastructure",  # External name: devops_infrastructure
                type=CapabilityType.TOOL,
                description="Infrastructure as Code configuration",
                version="0.5.0",
                configure_fn="configure_infrastructure_settings",
                default_config={
                    "iac_tool": "terraform",
                    "auto_approve_non_destructive": False,
                    "require_plan_before_apply": True,
                    "state_backend": None,
                },
                tags=["terraform", "iac", "infrastructure"],
            ),
            "cicd_settings": CapabilityDefinition(
                name="cicd",  # External name: devops_cicd
                type=CapabilityType.TOOL,
                description="CI/CD pipeline configuration",
                version="0.5.0",
                configure_fn="configure_cicd_settings",
                default_config={
                    "platform": "github_actions",
                    "run_tests_before_deploy": True,
                    "require_passing_checks": True,
                    "enable_security_scan": True,
                },
                tags=["cicd", "pipeline", "automation"],
            ),
            "monitoring_settings": CapabilityDefinition(
                name="monitoring",  # External name: devops_monitoring
                type=CapabilityType.TOOL,
                description="Monitoring and observability configuration",
                version="0.5.0",
                configure_fn="configure_monitoring_settings",
                default_config={
                    "metrics_backend": "prometheus",
                    "logging_backend": "loki",
                    "alerting_enabled": True,
                    "dashboard_tool": "grafana",
                },
                dependencies=["deployment_safety"],
                tags=["monitoring", "observability", "metrics", "logging"],
            ),
        }

    def generate_capabilities_list(self) -> List[CapabilityEntry]:
        """Generate CAPABILITIES list for CapabilityLoader discovery.

        Override base class to use definition.name instead of dict key for
        external capability names.

        Returns:
            List of CapabilityEntry instances
        """
        from victor.framework.protocols import OrchestratorCapability

        entries: List[CapabilityEntry] = []
        definitions = self._get_definitions()

        for key, definition in definitions.items():
            if not definition.configure_fn:
                continue

            # Use definition.name for external name (not the dict key)
            cap_metadata = {
                "name": f"{self._vertical_name}_{definition.name}",
                "version": definition.version,
                "setter": definition.configure_fn,
                "description": definition.description,
            }

            # Add getter if available
            if definition.get_fn:
                cap_metadata["getter"] = definition.get_fn

            capability = OrchestratorCapability(
                capability_type=definition.type,
                name=cap_metadata["name"],
                version=cap_metadata["version"],
                setter=cap_metadata["setter"],
                description=cap_metadata["description"],
                getter=cap_metadata.get("getter"),
            )

            # Get handler functions
            handler = getattr(self, definition.configure_fn, None)
            getter_handler = getattr(self, definition.get_fn, None) if definition.get_fn else None

            if not handler:
                logger.warning(
                    f"Handler function '{definition.configure_fn}' not found for capability '{key}'"
                )
                continue

            entry = CapabilityEntry(
                capability=capability,
                handler=handler,
                getter_handler=getter_handler,
            )
            entries.append(entry)

        return entries

    # Delegate to handler functions (required by BaseVerticalCapabilityProvider)
    def configure_deployment_safety(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure deployment safety capability."""
        configure_deployment_safety(orchestrator, **kwargs)

    def configure_container_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure container settings capability."""
        configure_container_settings(orchestrator, **kwargs)

    def get_container_settings(self, orchestrator: Any) -> Dict[str, Any]:
        """Get container settings configuration."""
        return get_container_settings(orchestrator)

    def configure_infrastructure_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure infrastructure settings capability."""
        configure_infrastructure_settings(orchestrator, **kwargs)

    def configure_cicd_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure CI/CD settings capability."""
        configure_cicd_settings(orchestrator, **kwargs)

    def configure_monitoring_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure monitoring settings capability."""
        configure_monitoring_settings(orchestrator, **kwargs)

    # Backward compatibility: provide apply_* methods
    def apply_deployment_safety(self, orchestrator: Any, **kwargs: Any) -> None:
        """Apply deployment safety capability (backward compatibility)."""
        self.apply_capability(orchestrator, "deployment_safety", **kwargs)

    def apply_container_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Apply container settings capability (backward compatibility)."""
        self.apply_capability(orchestrator, "container_settings", **kwargs)

    def apply_infrastructure_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Apply infrastructure settings capability (backward compatibility)."""
        self.apply_capability(orchestrator, "infrastructure_settings", **kwargs)

    def apply_cicd_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Apply CI/CD settings capability (backward compatibility)."""
        self.apply_capability(orchestrator, "cicd_settings", **kwargs)

    def apply_monitoring_settings(self, orchestrator: Any, **kwargs: Any) -> None:
        """Apply monitoring settings capability (backward compatibility)."""
        self.apply_capability(orchestrator, "monitoring_settings", **kwargs)


# =============================================================================
# Decorated Capability Functions (removed - now handled by BaseVerticalCapabilityProvider)
# =============================================================================


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


# Create singleton instance for generating CAPABILITIES list
_provider_instance: Optional[DevOpsCapabilityProvider] = None


def _get_provider() -> DevOpsCapabilityProvider:
    """Get or create provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = DevOpsCapabilityProvider()
    return _provider_instance


# Generate CAPABILITIES list from provider
CAPABILITIES: List[CapabilityEntry] = []


def _generate_capabilities_list() -> None:
    """Generate CAPABILITIES list from provider."""
    global CAPABILITIES
    if not CAPABILITIES:
        provider = _get_provider()
        CAPABILITIES.extend(provider.generate_capabilities_list())


_generate_capabilities_list()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_devops_capabilities() -> List[CapabilityEntry]:
    """Get all DevOps capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_devops_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for DevOps vertical.

    Returns:
        CapabilityLoader with DevOps capabilities registered
    """
    from victor.framework.capability_loader import CapabilityLoader

    provider = _get_provider()
    return provider.create_capability_loader()


def get_capability_configs() -> Dict[str, Any]:
    """Get DevOps capability configurations for centralized storage.

    Returns default DevOps configuration for VerticalContext storage.
    This replaces direct orchestrator deployment/container/cicd_config assignment.

    Returns:
        Dict with default DevOps capability configurations
    """
    # Preserve backward compatibility with old config key names
    return {
        "deployment_safety": {
            "require_approval_for_production": True,
            "require_backup_before_deploy": True,
            "enable_rollback": True,
            "protected_environments": ["production", "staging"],
        },
        "container_config": {
            "runtime": "docker",
            "default_registry": None,
            "security_scan_enabled": True,
            "max_image_size_mb": 2000,
        },
        "infrastructure_config": {
            "iac_tool": "terraform",
            "auto_approve_non_destructive": False,
            "require_plan_before_apply": True,
            "state_backend": None,
        },
        "cicd_config": {
            "platform": "github_actions",
            "run_tests_before_deploy": True,
            "require_passing_checks": True,
            "enable_security_scan": True,
        },
        "monitoring_config": {
            "metrics_backend": "prometheus",
            "logging_backend": "loki",
            "alerting_enabled": True,
            "dashboard_tool": "grafana",
        },
    }


__all__ = [
    # Handlers
    "configure_deployment_safety",
    "configure_container_settings",
    "configure_infrastructure_settings",
    "configure_cicd_settings",
    "configure_monitoring_settings",
    "get_container_settings",
    # Provider class
    "DevOpsCapabilityProvider",
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_devops_capabilities",
    "create_devops_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]
