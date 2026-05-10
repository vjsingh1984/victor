from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.tool_access_policy import ToolAccessPolicy


def _make_policy(*, registrar=None, selector=None) -> ToolAccessPolicy:
    return ToolAccessPolicy(
        get_registrar=lambda: registrar,
        get_selector=lambda: selector,
        logger=MagicMock(),
    )


def test_access_policy_sets_enabled_tools_and_propagates_to_selector():
    selector = MagicMock()
    policy = _make_policy(selector=selector)

    policy.set_enabled_tools({"read", "shell"})

    assert policy.enabled_tools == {"read", "shell"}
    assert policy.get_enabled_tools() == {"read", "shell"}
    assert policy.is_tool_enabled("read") is True
    assert policy.is_tool_enabled("write") is False
    selector.set_enabled_tools.assert_called_once_with({"read", "shell"})


def test_access_policy_empty_enabled_tools_means_all_available():
    registrar = SimpleNamespace(get_tool_names=MagicMock(return_value=["read", "write"]))
    selector = MagicMock()
    policy = _make_policy(registrar=registrar, selector=selector)

    policy.set_enabled_tools(set())

    assert policy.enabled_tools is None
    assert policy.get_enabled_tools() == {"read", "write"}
    assert policy.is_tool_enabled("anything") is True
    selector.set_enabled_tools.assert_called_once_with(set())


def test_access_policy_get_available_tools_from_list_tools_objects():
    registrar = SimpleNamespace(
        list_tools=MagicMock(return_value=["read", SimpleNamespace(name="write")])
    )
    policy = _make_policy(registrar=registrar)

    assert policy.get_available_tools() == {"read", "write"}


def test_access_policy_get_available_tools_handles_registrar_error():
    registrar = SimpleNamespace(get_registered_tools=MagicMock(side_effect=RuntimeError("boom")))
    policy = _make_policy(registrar=registrar)

    assert policy.get_available_tools() == set()
    policy._logger.warning.assert_called_once()


def test_access_policy_resolves_shell_alias_when_shell_enabled():
    policy = _make_policy()
    policy.set_enabled_tools({"shell"})

    assert policy.resolve_tool_alias("bash") == "shell"
    assert policy.resolve_tool_alias("execute_bash") == "shell"


def test_access_policy_normalizes_camel_case_tool_names_before_lookup():
    policy = _make_policy()

    assert policy.resolve_tool_alias("setGlobalAxisManager") == "set_global_axis_manager"
    assert policy.resolve_tool_alias("executeBash") == "shell"


def test_access_policy_returns_canonical_alias_when_shell_disabled():
    policy = _make_policy()
    policy.set_enabled_tools({"read"})

    assert policy.resolve_tool_alias("bash") == "shell"
