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
from victor.devops.capabilities import DevOpsCapabilityProvider
from victor.devops.chat_workflow_provider import DevOpsChatWorkflowProvider
from victor.devops.mode_config import DevOpsModeConfigProvider
from victor.devops.prompts import DevOpsPromptContributor
from victor.devops.safety import DevOpsSafetyExtension
from victor.devops.tool_dependencies import DevOpsToolDependencyProvider

# Import lazy initializer for eliminating import side-effects
from victor.framework.lazy_initializer import get_initializer_for_vertical


# Auto-register escape hatches (OCP-compliant)
def _register_escape_hatches() -> None:
    """Register devops vertical's escape hatches with the global registry.

    Phase 5 Import Side-Effects Remediation:
    This function now uses lazy initialization via LazyInitializer to eliminate
    import-time side effects. Registration occurs on first use, not on import.
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


# Create lazy initializer (no import side-effect)
_lazy_init = get_initializer_for_vertical("devops", _register_escape_hatches)

__all__ = [
    "DevOpsAssistant",
    "DevOpsPromptContributor",
    "DevOpsModeConfigProvider",
    "DevOpsSafetyExtension",
    "DevOpsToolDependencyProvider",
    "DevOpsCapabilityProvider",
    "DevOpsChatWorkflowProvider",
]
