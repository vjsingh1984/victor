"""DevOps Vertical Package - Complete implementation with extensions.

Competitive use case: Replaces/augments Docker Desktop AI, Terraform Assistant, Pulumi AI.

This vertical provides:
- Docker and container management
- CI/CD pipeline configuration
- Infrastructure as Code (IaC) generation
- Kubernetes manifest creation
- Monitoring and observability setup
"""

from victor.devops.assistant import DevOpsAssistant
from victor.devops.prompts import DevOpsPromptContributor
from victor.devops.mode_config import DevOpsModeConfigProvider
from victor.devops.safety import DevOpsSafetyExtension
from victor.devops.tool_dependencies import DevOpsToolDependencyProvider
from victor.devops.capabilities import DevOpsCapabilityProvider


# Auto-register escape hatches (OCP-compliant)
def _register_escape_hatches() -> None:
    """Register devops vertical's escape hatches with the global registry.

    This function runs on module import to automatically register escape hatches.
    Using the registry's discover_from_all_verticals() method is the OCP-compliant
    approach, as it doesn't require the framework to know about specific verticals.
    """
    try:
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry

        # Import escape hatches module to trigger side-effect registration
        from victor.devops import escape_hatches  # noqa: F401

        # Register with global registry
        registry = EscapeHatchRegistry.get_instance()
        registry.register_from_vertical(
            "devops",
            conditions=escape_hatches.CONDITIONS,
            transforms=escape_hatches.TRANSFORMS,
        )
    except Exception:
        # If registration fails, it's not critical
        pass


# Register on import
_register_escape_hatches()

__all__ = [
    "DevOpsAssistant",
    "DevOpsPromptContributor",
    "DevOpsModeConfigProvider",
    "DevOpsSafetyExtension",
    "DevOpsToolDependencyProvider",
    "DevOpsCapabilityProvider",
]
