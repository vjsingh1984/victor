"""Framework-level capability method mappings.

Single source of truth for capability name → method name resolution.
This lives in the framework layer so that both framework/ and agent/
modules can import it without creating circular dependencies.

Usage:
    from victor.framework.capability_registry import (
        CAPABILITY_METHOD_MAPPINGS,
        get_method_for_capability,
    )

    method_name = get_method_for_capability("enabled_tools")
    # Returns "set_enabled_tools"
"""

from __future__ import annotations

from typing import Dict

# Maps capability names to their setter method names.
# This is the single source of truth for capability → method resolution.
CAPABILITY_METHOD_MAPPINGS: Dict[str, str] = {
    # Tool capabilities
    "enabled_tools": "set_enabled_tools",
    "tool_dependencies": "set_tool_dependencies",
    "tool_sequences": "set_tool_sequences",
    "tiered_tool_config": "set_tiered_tool_config",
    # Vertical capabilities
    "vertical_middleware": "apply_vertical_middleware",
    "vertical_safety_patterns": "apply_vertical_safety_patterns",
    "vertical_context": "set_vertical_context",
    # RL capabilities
    "rl_hooks": "set_rl_hooks",
    # Team capabilities
    "team_specs": "set_team_specs",
    # Mode capabilities
    "mode_configs": "set_mode_configs",
    "default_budget": "set_default_budget",
    # Prompt capabilities
    "custom_prompt": "set_custom_prompt",
    "prompt_section": "add_prompt_section",
    "task_type_hints": "set_task_type_hints",
    # Safety capabilities
    "safety_patterns": "add_safety_patterns",
    # Enrichment capabilities
    "enrichment_strategy": "set_enrichment_strategy",
    "enrichment_service": "enrichment_service",
}


def get_method_for_capability(capability_name: str) -> str:
    """Get the method name for a capability.

    This is the canonical way to resolve capability names to method names.
    Uses CAPABILITY_METHOD_MAPPINGS as the source of truth.

    Args:
        capability_name: Name of the capability

    Returns:
        Method name to call for this capability
    """
    return CAPABILITY_METHOD_MAPPINGS.get(capability_name, f"set_{capability_name}")
