"""Research vertical package with lazy exports for SDK-first installs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ResearchAssistant",
    "ResearchPromptContributor",
    "ResearchModeConfigProvider",
    "ResearchSafetyExtension",
    "ResearchCapabilityProvider",
    "ResearchToolDependencyProvider",
    "ResearchPlugin",
    "plugin",
    "ResearchSafetyRules",
    "EnhancedResearchSafetyExtension",
    "ResearchContext",
    "EnhancedResearchConversationManager",
]

_EXPORTS = {
    "ResearchAssistant": ("victor_research.assistant", "ResearchAssistant"),
    "ResearchPromptContributor": ("victor_research.prompts", "ResearchPromptContributor"),
    "ResearchModeConfigProvider": ("victor_research.mode_config", "ResearchModeConfigProvider"),
    "ResearchSafetyExtension": ("victor_research.safety", "ResearchSafetyExtension"),
    "ResearchCapabilityProvider": ("victor_research.capabilities", "ResearchCapabilityProvider"),
    "ResearchPlugin": ("victor_research.plugin", "ResearchPlugin"),
    "plugin": ("victor_research.plugin", "plugin"),
    "ResearchSafetyRules": ("victor_research.safety_enhanced", "ResearchSafetyRules"),
    "EnhancedResearchSafetyExtension": (
        "victor_research.safety_enhanced",
        "EnhancedResearchSafetyExtension",
    ),
    "ResearchContext": ("victor_research.conversation_enhanced", "ResearchContext"),
    "EnhancedResearchConversationManager": (
        "victor_research.conversation_enhanced",
        "EnhancedResearchConversationManager",
    ),
}


def __getattr__(name: str) -> Any:
    if name == "ResearchToolDependencyProvider":
        from victor_research.tool_dependencies import get_provider

        return get_provider()

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = import_module(module_name)
    return getattr(module, attribute_name)
