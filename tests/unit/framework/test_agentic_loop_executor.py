"""Tests for the AgenticLoop executor enum + resolver (mirrors FEP-0012).

``use_stategraph_executor()`` is the single decision for whether the StateGraph
AgenticLoop executor runs. ``AUTO`` (default settings) defers to the legacy
``USE_STATEGRAPH_AGENTIC_LOOP`` flag (prior behavior); explicit ``STATEGRAPH``
forces it regardless of the flag. These tests lock in that AUTO preserves
existing behavior and that the enum is the single source of truth.
"""

import types

import pytest

from victor.core.feature_flags import FeatureFlag, is_feature_enabled
from victor.framework.agentic_loop_executor import (
    AgenticLoopExecutor,
    resolve_agentic_loop_executor,
    use_stategraph_executor,
)


def _set_executor(monkeypatch, value) -> None:
    """Override the settings model read by the resolver."""
    from victor.config import agentic_loop_settings as als

    monkeypatch.setattr(als, "AgenticLoopSettings", lambda: types.SimpleNamespace(executor=value))


def _flag(monkeypatch, on: bool) -> None:
    """Set USE_STATEGRAPH_AGENTIC_LOOP via env (auto-cleaned by monkeypatch)."""
    monkeypatch.setenv(
        FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP.get_env_var_name(),
        "true" if on else "false",
    )


def test_parse():
    assert AgenticLoopExecutor.parse("stategraph") is AgenticLoopExecutor.STATEGRAPH
    assert AgenticLoopExecutor.parse("auto") is AgenticLoopExecutor.AUTO
    assert (
        AgenticLoopExecutor.parse(AgenticLoopExecutor.STATEGRAPH) is AgenticLoopExecutor.STATEGRAPH
    )
    assert AgenticLoopExecutor.parse("garbage") is AgenticLoopExecutor.AUTO
    assert AgenticLoopExecutor.parse(None) is AgenticLoopExecutor.AUTO


def test_auto_flag_off_uses_legacy_loop(monkeypatch):
    _flag(monkeypatch, False)
    assert is_feature_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP) is False
    _set_executor(monkeypatch, AgenticLoopExecutor.AUTO)
    assert use_stategraph_executor() is False
    assert resolve_agentic_loop_executor() is AgenticLoopExecutor.AUTO


def test_auto_flag_on_uses_stategraph(monkeypatch):
    _flag(monkeypatch, True)
    assert is_feature_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP) is True
    _set_executor(monkeypatch, AgenticLoopExecutor.AUTO)
    assert use_stategraph_executor() is True
    assert resolve_agentic_loop_executor() is AgenticLoopExecutor.STATEGRAPH


def test_explicit_stategraph_overrides_flag_off(monkeypatch):
    _flag(monkeypatch, False)
    _set_executor(monkeypatch, AgenticLoopExecutor.STATEGRAPH)
    assert use_stategraph_executor() is True


def test_default_settings_executor_is_auto():
    from victor.config.agentic_loop_settings import AgenticLoopSettings

    assert AgenticLoopSettings().executor is AgenticLoopExecutor.AUTO
