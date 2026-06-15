# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for governance wiring in CoordinationBuildersMixin.

Covers the observability emitter and approval-handler resolution helpers,
plus end-to-end gating through the real MiddlewareChain.
"""

import asyncio

from victor.agent.factory.coordination_builders import CoordinationBuildersMixin
from victor.agent.middleware_chain import MiddlewareChain
from victor.config.settings import Settings
from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager


class _FakeContainer:
    def __init__(self, registry=None):
        self._registry = registry or {}

    def get_optional(self, key):
        return self._registry.get(key)


class _Host(CoordinationBuildersMixin):
    def __init__(self, settings, container=None):
        self.settings = settings
        self.container = container or _FakeContainer()
        self.model = "claude-opus-4-8"


def _governed_settings(**overrides):
    s = Settings()
    s.governance.enabled = True
    s.governance.max_tool_calls_per_session = 1
    for k, v in overrides.items():
        setattr(s.governance, k, v)
    return s


async def test_disabled_when_flag_off():
    mgr = get_feature_flag_manager()
    mgr.disable(FeatureFlag.USE_POLICY_ENGINE)
    chain = MiddlewareChain()
    _Host(_governed_settings())._maybe_add_policy_engine(chain)
    assert len(chain) == 0


async def test_enabled_gates_tool_calls():
    mgr = get_feature_flag_manager()
    mgr.enable(FeatureFlag.USE_POLICY_ENGINE)
    try:
        chain = MiddlewareChain()
        _Host(_governed_settings())._maybe_add_policy_engine(chain)
        assert len(chain) == 1
        first = await chain.process_before("read_file", {})
        second = await chain.process_before("read_file", {})
        assert first.proceed is True
        assert second.proceed is False
    finally:
        mgr.disable(FeatureFlag.USE_POLICY_ENGINE)


async def test_no_op_when_governance_disabled():
    mgr = get_feature_flag_manager()
    mgr.enable(FeatureFlag.USE_POLICY_ENGINE)
    try:
        s = Settings()
        s.governance.enabled = False
        s.governance.max_tool_calls_per_session = 1
        chain = MiddlewareChain()
        _Host(s)._maybe_add_policy_engine(chain)
        assert len(chain) == 0
    finally:
        mgr.disable(FeatureFlag.USE_POLICY_ENGINE)


async def test_deny_tools_blocks_via_builder():
    mgr = get_feature_flag_manager()
    mgr.enable(FeatureFlag.USE_POLICY_ENGINE)
    try:
        s = _governed_settings(max_tool_calls_per_session=0, deny_tools=["run_command"])
        chain = MiddlewareChain()
        _Host(s)._maybe_add_policy_engine(chain)
        assert len(chain) == 1
        denied = await chain.process_before("run_command", {})
        allowed = await chain.process_before("read_file", {})
        assert denied.proceed is False
        assert allowed.proceed is True
    finally:
        mgr.disable(FeatureFlag.USE_POLICY_ENGINE)


async def test_allow_tools_restricts_via_builder():
    mgr = get_feature_flag_manager()
    mgr.enable(FeatureFlag.USE_POLICY_ENGINE)
    try:
        s = _governed_settings(max_tool_calls_per_session=0, allow_tools=["read_file"])
        chain = MiddlewareChain()
        _Host(s)._maybe_add_policy_engine(chain)
        assert len(chain) == 1
        allowed = await chain.process_before("read_file", {})
        denied = await chain.process_before("run_command", {})
        assert allowed.proceed is True
        assert denied.proceed is False
    finally:
        mgr.disable(FeatureFlag.USE_POLICY_ENGINE)


def test_emitter_none_when_no_bus():
    host = _Host(_governed_settings())
    assert host._build_policy_emitter() is None


async def test_emitter_schedules_emit_on_bus():
    captured = []

    class _Bus:
        async def emit(self, topic, data, *, source="victor", correlation_id=None):
            captured.append((topic, data, source))
            return True

    from victor.core.events.backends import ObservabilityBus

    host = _Host(_governed_settings(), container=_FakeContainer({ObservabilityBus: _Bus()}))
    emitter = host._build_policy_emitter()
    assert emitter is not None

    emitter("policy.decision", {"decision": "deny"})
    # The emit is scheduled as a task on the running loop; let it run.
    await asyncio.sleep(0)
    assert captured
    topic, data, source = captured[0]
    assert topic == "policy.decision"
    assert source == "policy_engine"


def test_approval_handler_none_when_not_interactive():
    s = _governed_settings(interactive_approval=False)
    host = _Host(s)
    assert host._resolve_policy_approval_handler(s.governance) is None


def test_approval_handler_none_when_no_tty(monkeypatch):
    import sys

    s = _governed_settings(interactive_approval=True)
    host = _Host(s)
    # Force non-TTY.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert host._resolve_policy_approval_handler(s.governance) is None


def test_approval_handler_wired_when_interactive_tty(monkeypatch):
    import sys

    s = _governed_settings(interactive_approval=True)
    host = _Host(s)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    handler = host._resolve_policy_approval_handler(s.governance)
    assert handler is not None
    assert callable(handler)
