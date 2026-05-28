"""Compatibility tests for the SDK-backed tool naming registry."""

from victor_contracts import ToolNames as SdkToolNames
from victor_contracts.constants import get_canonical_name as sdk_get_canonical_name
from victor_contracts.constants import TOOL_ALIASES as sdk_tool_aliases

from victor.tools.tool_names import (
    TOOL_ALIASES as legacy_tool_aliases,
    ToolNames as LegacyToolNames,
    get_canonical_name as legacy_get_canonical_name,
)


def test_legacy_tool_names_module_reexports_sdk_registry() -> None:
    """Legacy imports continue to resolve to the SDK-owned registry."""

    assert LegacyToolNames is SdkToolNames
    assert legacy_get_canonical_name is sdk_get_canonical_name
    assert LegacyToolNames.SHELL == "shell"
    assert not hasattr(LegacyToolNames, "SHELL_READONLY")
    assert legacy_get_canonical_name("execute_bash") == SdkToolNames.SHELL
    assert legacy_get_canonical_name("shell_readonly") == "shell_readonly"
    assert LegacyToolNames.file_operations() == SdkToolNames.file_operations()


def test_legacy_tool_aliases_share_sdk_contract() -> None:
    """Core compatibility imports must not fork SDK tool-name semantics."""

    assert legacy_tool_aliases is sdk_tool_aliases
    assert legacy_tool_aliases["git_create_pr"] == SdkToolNames.PR
    assert legacy_get_canonical_name("git_create_pr") == sdk_get_canonical_name(
        "git_create_pr"
    )
