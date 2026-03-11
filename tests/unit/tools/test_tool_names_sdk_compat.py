"""Compatibility tests for the SDK-backed tool naming registry."""

from victor_sdk import ToolNames as SdkToolNames
from victor_sdk.constants import get_canonical_name as sdk_get_canonical_name

from victor.tools.tool_names import (
    ToolNames as LegacyToolNames,
    get_canonical_name as legacy_get_canonical_name,
)


def test_legacy_tool_names_module_reexports_sdk_registry() -> None:
    """Legacy imports continue to resolve to the SDK-owned registry."""

    assert LegacyToolNames is SdkToolNames
    assert legacy_get_canonical_name is sdk_get_canonical_name
    assert LegacyToolNames.SHELL == "shell"
    assert legacy_get_canonical_name("execute_bash") == SdkToolNames.SHELL
    assert LegacyToolNames.file_operations() == SdkToolNames.file_operations()
