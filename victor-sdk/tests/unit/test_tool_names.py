"""Unit tests for SDK-owned tool name constants."""

from victor_sdk import (
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


def test_tool_names_are_exported_from_sdk() -> None:
    """Top-level victor_sdk exports the canonical tool registry."""

    assert ToolNames.READ == "read"
    assert ToolNames.SHELL == "shell"
    assert ToolNames.GIT == "git"


def test_alias_resolution_matches_existing_runtime_contract() -> None:
    """Legacy aliases still resolve to canonical tool names."""

    assert get_canonical_name("execute_bash") == ToolNames.SHELL
    assert get_canonical_name("read_file") == ToolNames.READ
    assert get_aliases(ToolNames.GIT) >= {"git_status", "git_diff"}


def test_all_canonical_names_and_mapping_are_consistent() -> None:
    """Canonical-name helpers produce a complete self-mapping."""

    canonical_names = get_all_canonical_names()
    mapping = get_name_mapping()

    assert ToolNames.WRITE in canonical_names
    assert ToolNames.WEB_FETCH in canonical_names
    assert mapping[ToolNames.WRITE] == ToolNames.WRITE
    assert mapping["run_tests"] == ToolNames.TEST
    assert CANONICAL_TO_ALIASES[ToolNames.TEST] >= {"run_tests"}


def test_tool_name_entry_and_validation_helpers() -> None:
    """SDK helpers behave as a stable pure registry surface."""

    entry = ToolNameEntry(canonical=ToolNames.READ, aliases=frozenset({"read_file"}))

    assert entry.all_names() == {"read", "read_file"}
    assert is_valid_tool_name(ToolNames.READ) is True
    assert is_valid_tool_name("read_file") is True
    assert is_valid_tool_name("not_a_real_tool") is False
    assert TOOL_ALIASES["git_create_pr"] == ToolNames.PR


def test_common_file_operation_group_is_sdk_owned() -> None:
    """Shared tool-group helpers stay in the SDK contract layer."""

    assert ToolNames.file_operations() == (
        ToolNames.READ,
        ToolNames.WRITE,
        ToolNames.EDIT,
        ToolNames.GREP,
    )
