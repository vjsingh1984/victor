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

"""Tests for SessionFacade domain facade."""

import pytest
from unittest.mock import MagicMock

from victor.agent.facades.session_facade import SessionFacade
from victor.agent.facades.protocols import SessionFacadeProtocol


class TestSessionFacadeInit:
    """Tests for SessionFacade initialization."""

    def test_init_with_all_components(self):
        """SessionFacade initializes with all components provided."""
        state = MagicMock()
        accessor = MagicMock()
        ledger = MagicMock()
        lifecycle = MagicMock()

        facade = SessionFacade(
            session_state=state,
            session_accessor=accessor,
            session_ledger=ledger,
            lifecycle_manager=lifecycle,
            active_session_id="session-abc",
            memory_session_id="memory-123",
            profile_name="claude-sonnet",
            checkpoint_manager=MagicMock(),
        )

        assert facade.session_state is state
        assert facade.session_accessor is accessor
        assert facade.session_ledger is ledger
        assert facade.lifecycle_manager is lifecycle
        assert facade.active_session_id == "session-abc"
        assert facade.memory_session_id == "memory-123"
        assert facade.profile_name == "claude-sonnet"

    def test_init_with_minimal_components(self):
        """SessionFacade initializes with only required components."""
        state = MagicMock()
        accessor = MagicMock()
        ledger = MagicMock()

        facade = SessionFacade(
            session_state=state,
            session_accessor=accessor,
            session_ledger=ledger,
        )

        assert facade.session_state is state
        assert facade.session_accessor is accessor
        assert facade.session_ledger is ledger
        assert facade.lifecycle_manager is None
        assert facade.active_session_id is None
        assert facade.memory_session_id is None
        assert facade.profile_name is None
        assert facade.checkpoint_manager is None


class TestSessionFacadeProperties:
    """Tests for SessionFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create a SessionFacade with mock components."""
        return SessionFacade(
            session_state=MagicMock(name="state"),
            session_accessor=MagicMock(name="accessor"),
            session_ledger=MagicMock(name="ledger"),
            lifecycle_manager=MagicMock(name="lifecycle"),
            active_session_id="test-session",
            memory_session_id="test-memory",
            profile_name="test-profile",
            checkpoint_manager=MagicMock(name="checkpoint"),
        )

    def test_session_state_property(self, facade):
        """SessionState property returns the state manager."""
        assert facade.session_state._mock_name == "state"

    def test_session_accessor_property(self, facade):
        """SessionAccessor property returns the accessor."""
        assert facade.session_accessor._mock_name == "accessor"

    def test_session_ledger_property(self, facade):
        """SessionLedger property returns the ledger."""
        assert facade.session_ledger._mock_name == "ledger"

    def test_session_ledger_setter(self, facade):
        """SessionLedger setter updates the ledger."""
        new_ledger = MagicMock(name="new_ledger")
        facade.session_ledger = new_ledger
        assert facade.session_ledger is new_ledger

    def test_lifecycle_manager_property(self, facade):
        """LifecycleManager property returns the manager."""
        assert facade.lifecycle_manager._mock_name == "lifecycle"

    def test_active_session_id_property(self, facade):
        """ActiveSessionId property returns the session ID."""
        assert facade.active_session_id == "test-session"

    def test_active_session_id_setter(self, facade):
        """ActiveSessionId setter updates the ID."""
        facade.active_session_id = "new-session"
        assert facade.active_session_id == "new-session"

    def test_memory_session_id_property(self, facade):
        """MemorySessionId property returns the memory session ID."""
        assert facade.memory_session_id == "test-memory"

    def test_memory_session_id_setter(self, facade):
        """MemorySessionId setter updates the ID."""
        facade.memory_session_id = "new-memory"
        assert facade.memory_session_id == "new-memory"

    def test_checkpoint_manager_property(self, facade):
        """CheckpointManager property returns the manager."""
        assert facade.checkpoint_manager._mock_name == "checkpoint"


class TestSessionFacadeProtocolConformance:
    """Tests that SessionFacade satisfies SessionFacadeProtocol."""

    def test_satisfies_protocol(self):
        """SessionFacade structurally conforms to SessionFacadeProtocol."""
        facade = SessionFacade(
            session_state=MagicMock(),
            session_accessor=MagicMock(),
            session_ledger=MagicMock(),
        )
        assert isinstance(facade, SessionFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on SessionFacade."""
        required = [
            "session_state",
            "session_ledger",
            "lifecycle_manager",
        ]
        facade = SessionFacade(
            session_state=MagicMock(),
            session_accessor=MagicMock(),
            session_ledger=MagicMock(),
        )
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
