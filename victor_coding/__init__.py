"""Coding vertical package with lazy exports for SDK-first installs.

Victor's primary vertical for software development, providing:
- Code exploration and understanding
- Bug fixing and refactoring
- Feature implementation
- Testing and verification
- Git operations and version control

This package contains all coding-specific logic extracted from the framework,
enabling the framework to remain domain-agnostic while providing rich
coding assistant functionality.

Package Structure:
    assistant.py              - CodingAssistant vertical class
    plugin.py                 - CodingPlugin class (new SDK-first approach)
    middleware.py             - Code correction middleware
    safety.py                  - Coding-specific safety patterns (legacy)
    safety_enhanced.py        - Enhanced safety with SafetyCoordinator
    conversation_enhanced.py  - Enhanced conversation with ConversationCoordinator
    prompts.py                - Task type hints and prompt contributions
    mode_config.py            - Mode configurations and tool budgets
    tool_dependencies.py      - Code tool dependency graph
    service_provider.py       - DI service registration
    workflows/                - Coding-specific workflows

Usage:
    from victor_coding import CodingAssistant, CodingPlugin

    # Get vertical configuration
    config = CodingAssistant.get_config()

    # Get extensions for framework integration
    extensions = CodingAssistant.get_extensions()

    # Use enhanced features
    from victor_coding import EnhancedCodingSafetyExtension, EnhancedCodingConversationManager

    safety_ext = EnhancedCodingSafetyExtension()
    conv_mgr = EnhancedCodingConversationManager()
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CodingAssistant",
    "CodingPlugin",
    "plugin",
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
    "CodingSandboxProvider",
    "CodingPermissionProvider",
    "CodingHookProvider",
    "CodingCompactionProvider",
]

_EXPORTS = {
    "CodingAssistant": ("victor_coding.assistant", "CodingAssistant"),
    "CodingPlugin": ("victor_coding.plugin", "CodingPlugin"),
    "plugin": ("victor_coding.plugin", "plugin"),
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
    "CodingSandboxProvider": ("victor_coding.protocols", "CodingSandboxProvider"),
    "CodingPermissionProvider": ("victor_coding.protocols", "CodingPermissionProvider"),
    "CodingHookProvider": ("victor_coding.protocols", "CodingHookProvider"),
    "CodingCompactionProvider": ("victor_coding.protocols", "CodingCompactionProvider"),
}


def __getattr__(name: str) -> Any:
    if name == "CodingToolDependencyProvider":
        from victor_coding.tool_dependencies import get_provider

        return get_provider()

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = import_module(module_name)
    return getattr(module, attribute_name)
