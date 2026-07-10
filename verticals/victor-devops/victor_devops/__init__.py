"""DevOps vertical package with lazy exports for contract-first installs.

This vertical provides:
- Docker and container management
- CI/CD pipeline configuration
- Infrastructure as Code (IaC) generation
- Kubernetes manifest creation
- Monitoring and observability setup

Enhanced Features:
- Enhanced safety with SafetyCoordinator (safety_enhanced.py)
- Enhanced conversation management with ConversationCoordinator (conversation_enhanced.py)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DevOpsAssistant",
    "DevOpsPlugin",
    "plugin",
    "DevOpsPromptContributor",
    "DevOpsModeConfigProvider",
    "DevOpsSafetyExtension",
    "EnhancedDevOpsSafetyExtension",
    "EnhancedDevOpsConversationManager",
    "DevOpsSafetyRules",
    "DevOpsContext",
    "DevOpsToolDependencyProvider",
    "DevOpsCapabilityProvider",
    "DevOpsSandboxProvider",
    "DevOpsPermissionProvider",
    "DevOpsHookProvider",
]

_EXPORTS = {
    "DevOpsAssistant": ("victor_devops.assistant", "DevOpsAssistant"),
    "DevOpsPlugin": ("victor_devops.plugin", "DevOpsPlugin"),
    "plugin": ("victor_devops.plugin", "plugin"),
    "DevOpsPromptContributor": ("victor_devops.prompts", "DevOpsPromptContributor"),
    "DevOpsModeConfigProvider": (
        "victor_devops.mode_config",
        "DevOpsModeConfigProvider",
    ),
    "DevOpsSafetyExtension": ("victor_devops.safety", "DevOpsSafetyExtension"),
    "EnhancedDevOpsSafetyExtension": (
        "victor_devops.safety_enhanced",
        "EnhancedDevOpsSafetyExtension",
    ),
    "EnhancedDevOpsConversationManager": (
        "victor_devops.conversation_enhanced",
        "EnhancedDevOpsConversationManager",
    ),
    "DevOpsSafetyRules": ("victor_devops.safety_enhanced", "DevOpsSafetyRules"),
    "DevOpsContext": ("victor_devops.conversation_enhanced", "DevOpsContext"),
    "DevOpsToolDependencyProvider": (
        "victor_devops.tool_dependencies",
        "DevOpsToolDependencyProvider",
    ),
    "DevOpsCapabilityProvider": (
        "victor_devops.capabilities",
        "DevOpsCapabilityProvider",
    ),
    "DevOpsSandboxProvider": ("victor_devops.protocols", "DevOpsSandboxProvider"),
    "DevOpsPermissionProvider": ("victor_devops.protocols", "DevOpsPermissionProvider"),
    "DevOpsHookProvider": ("victor_devops.protocols", "DevOpsHookProvider"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = import_module(module_name)  # nosemgrep
    return getattr(module, attribute_name)
