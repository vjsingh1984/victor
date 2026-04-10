"""Coding vertical package with lazy exports for SDK-first installs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CodingAssistant",
    "CodingMiddleware",
    "CodeCorrectionMiddleware",
    "CodingSafetyExtension",
    "EnhancedCodingSafetyExtension",
    "EnhancedCodingConversationManager",
    "CodingSafetyRules",
    "CodingContext",
    "CodingPromptContributor",
    "CodingModeConfigProvider",
    "CodingToolDependencyProvider",
    "CodingServiceProvider",
    "CodingCapabilityProvider",
    "get_coding_capabilities",
    "create_coding_capability_loader",
    "CodingPlugin",
    "plugin",
    "CodingSandboxProvider",
    "CodingPermissionProvider",
    "CodingHookProvider",
    "CodingCompactionProvider",
]

_EXPORTS = {
    "CodingAssistant": ("victor_coding.assistant", "CodingAssistant"),
    "CodingMiddleware": ("victor_coding.middleware", "CodingMiddleware"),
    "CodeCorrectionMiddleware": ("victor_coding.middleware", "CodeCorrectionMiddleware"),
    "CodingSafetyExtension": ("victor_coding.safety", "CodingSafetyExtension"),
    "EnhancedCodingSafetyExtension": (
        "victor_coding.safety_enhanced",
        "EnhancedCodingSafetyExtension",
    ),
    "EnhancedCodingConversationManager": (
        "victor_coding.conversation_enhanced",
        "EnhancedCodingConversationManager",
    ),
    "CodingSafetyRules": ("victor_coding.safety_enhanced", "CodingSafetyRules"),
    "CodingContext": ("victor_coding.conversation_enhanced", "CodingContext"),
    "CodingPromptContributor": ("victor_coding.prompts", "CodingPromptContributor"),
    "CodingModeConfigProvider": ("victor_coding.mode_config", "CodingModeConfigProvider"),
    "CodingServiceProvider": ("victor_coding.service_provider", "CodingServiceProvider"),
    "CodingCapabilityProvider": ("victor_coding.capabilities", "CodingCapabilityProvider"),
    "get_coding_capabilities": ("victor_coding.capabilities", "get_coding_capabilities"),
    "create_coding_capability_loader": (
        "victor_coding.capabilities",
        "create_coding_capability_loader",
    ),
    "CodingPlugin": ("victor_coding.plugin", "CodingPlugin"),
    "plugin": ("victor_coding.plugin", "plugin"),
    "CodingSandboxProvider": ("victor_coding.protocols", "CodingSandboxProvider"),
    "CodingPermissionProvider": ("victor_coding.protocols", "CodingPermissionProvider"),
    "CodingHookProvider": ("victor_coding.protocols", "CodingHookProvider"),
    "CodingCompactionProvider": ("victor_coding.protocols", "CodingCompactionProvider"),
}


def __getattr__(name: str) -> Any:
    if name == "CodingToolDependencyProvider":
        from victor.framework.extensions import create_vertical_tool_dependency_provider

        return create_vertical_tool_dependency_provider("coding")

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = import_module(module_name)
    return getattr(module, attribute_name)
