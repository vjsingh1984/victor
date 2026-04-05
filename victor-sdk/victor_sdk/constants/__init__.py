"""Stable SDK constants and registries for vertical authors."""

from victor_sdk.constants.capability_ids import (
    CapabilityIds,
    get_all_capability_ids,
    is_known_capability_id,
)

from victor_sdk.constants.tool_names import (
    CANONICAL_TO_ALIASES,
    TOOL_ALIASES,
    ToolNameEntry,
    ToolNames,
    get_aliases,
    get_all_canonical_names,
    get_canonical_name,
    get_name_mapping,
    is_valid_tool_name,
)

__all__ = [
    "CapabilityIds",
    "get_all_capability_ids",
    "is_known_capability_id",
    "ToolNames",
    "ToolNameEntry",
    "TOOL_ALIASES",
    "CANONICAL_TO_ALIASES",
    "get_canonical_name",
    "get_aliases",
    "is_valid_tool_name",
    "get_all_canonical_names",
    "get_name_mapping",
]
