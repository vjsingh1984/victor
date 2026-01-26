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

"""Tests for protocol checking utilities (Phase 11.1).

Tests the standardized protocol checking functions that replace
inconsistent isinstance/hasattr patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pytest


# Test protocols
@runtime_checkable
class TestProtocol(Protocol):
    """A test protocol for checking."""

    def get_name(self) -> str: ...

    def get_value(self) -> int: ...


@runtime_checkable
class AnotherProtocol(Protocol):
    """Another test protocol."""

    def do_something(self) -> None: ...


class ConformingClass:
    """Class that conforms to TestProtocol."""

    def get_name(self) -> str:
        return "test"

    def get_value(self) -> int:
        return 42


class NonConformingClass:
    """Class that does not conform to TestProtocol."""

    def other_method(self) -> str:
        return "other"


class PartialConformingClass:
    """Class with only some methods."""

    def get_name(self) -> str:
        return "partial"


class TestCheckProtocol:
    """Tests for check_protocol function."""

    def test_returns_typed_object_when_conformant(self):
        """Test that conforming object is returned typed."""
        from victor.core.verticals.protocols.utils import check_protocol

        obj = ConformingClass()
        result = check_protocol(obj, TestProtocol)

        assert result is obj
        assert result.get_name() == "test"
        assert result.get_value() == 42

    def test_returns_none_when_not_conformant(self):
        """Test that non-conforming object returns None."""
        from victor.core.verticals.protocols.utils import check_protocol

        obj = NonConformingClass()
        result = check_protocol(obj, TestProtocol)

        assert result is None

    def test_no_hasattr_fallback(self):
        """Test that hasattr is not used as fallback."""
        from victor.core.verticals.protocols.utils import check_protocol

        # This class has hasattr("get_name") but doesn't fully conform
        obj = PartialConformingClass()
        result = check_protocol(obj, TestProtocol)

        # Should return None because isinstance check fails
        assert result is None

    def test_works_with_none_input(self):
        """Test that None input returns None."""
        from victor.core.verticals.protocols.utils import check_protocol

        result = check_protocol(None, TestProtocol)

        assert result is None

    def test_works_with_different_protocols(self):
        """Test checking different protocols."""
        from victor.core.verticals.protocols.utils import check_protocol

        obj = ConformingClass()

        # Does conform to TestProtocol
        result1 = check_protocol(obj, TestProtocol)
        assert result1 is obj

        # Does not conform to AnotherProtocol
        result2 = check_protocol(obj, AnotherProtocol)
        assert result2 is None


class TestCheckProtocolOptional:
    """Tests for check_protocol_optional function."""

    def test_returns_none_for_missing_attribute(self):
        """Test optional check when attribute is missing."""
        from victor.core.verticals.protocols.utils import check_protocol_optional

        class NoProvider:
            pass

        obj = NoProvider()
        result = check_protocol_optional(obj, "get_provider", TestProtocol)

        assert result is None

    def test_returns_typed_provider_when_available(self):
        """Test optional check when provider is available and conforming."""
        from victor.core.verticals.protocols.utils import check_protocol_optional

        class HasProvider:
            def get_provider(self):
                return ConformingClass()

        obj = HasProvider()
        result = check_protocol_optional(obj, "get_provider", TestProtocol)

        assert result is not None
        assert result.get_name() == "test"

    def test_returns_none_when_provider_not_conforming(self):
        """Test optional check when provider doesn't conform."""
        from victor.core.verticals.protocols.utils import check_protocol_optional

        class HasNonConformingProvider:
            def get_provider(self):
                return NonConformingClass()

        obj = HasNonConformingProvider()
        result = check_protocol_optional(obj, "get_provider", TestProtocol)

        assert result is None


class TestProtocolHelpers:
    """Tests for protocol helper functions."""

    def test_is_protocol_conformant(self):
        """Test is_protocol_conformant helper."""
        from victor.core.verticals.protocols.utils import is_protocol_conformant

        assert is_protocol_conformant(ConformingClass(), TestProtocol) is True
        assert is_protocol_conformant(NonConformingClass(), TestProtocol) is False
        assert is_protocol_conformant(None, TestProtocol) is False

    def test_get_protocol_methods(self):
        """Test getting methods from protocol."""
        from victor.core.verticals.protocols.utils import get_protocol_methods

        methods = get_protocol_methods(TestProtocol)

        assert "get_name" in methods
        assert "get_value" in methods

    def test_protocol_error_message(self):
        """Test protocol error message generation."""
        from victor.core.verticals.protocols.utils import protocol_error_message

        msg = protocol_error_message(NonConformingClass(), TestProtocol)

        assert "NonConformingClass" in msg
        assert "TestProtocol" in msg


class TestRealProtocols:
    """Tests using real Victor protocols."""

    def test_check_workflow_provider_protocol(self):
        """Test checking WorkflowProviderProtocol."""
        from victor.core.verticals.protocols import WorkflowProviderProtocol
        from victor.core.verticals.protocols.utils import check_protocol

        class FakeWorkflowProvider:
            def get_workflows(self) -> Dict[str, Any]:
                return {"workflow1": {}}

            def get_auto_workflows(self) -> List:
                return []

        provider = FakeWorkflowProvider()
        result = check_protocol(provider, WorkflowProviderProtocol)

        # Should conform since it has both methods
        assert result is not None

    def test_check_rl_config_provider_protocol(self):
        """Test checking RLConfigProviderProtocol."""
        from victor.core.verticals.protocols import RLConfigProviderProtocol
        from victor.core.verticals.protocols.utils import check_protocol

        class FakeRLConfigProvider:
            def get_rl_config(self) -> Dict[str, Any]:
                return {"active_learners": []}

            def get_rl_hooks(self) -> Optional[Any]:
                return None

        provider = FakeRLConfigProvider()
        result = check_protocol(provider, RLConfigProviderProtocol)

        # Should conform since it has all required methods
        assert result is not None

    def test_check_team_spec_provider_protocol(self):
        """Test checking TeamSpecProviderProtocol."""
        from victor.core.verticals.protocols import TeamSpecProviderProtocol
        from victor.core.verticals.protocols.utils import check_protocol

        class FakeTeamSpecProvider:
            def get_team_specs(self) -> Dict[str, Any]:
                return {"team1": {}}

            def get_default_team(self) -> Optional[str]:
                return None

        provider = FakeTeamSpecProvider()
        result = check_protocol(provider, TeamSpecProviderProtocol)

        # Should conform since it has all required methods
        assert result is not None

    def test_is_protocol_conformant_with_real_protocol(self):
        """Test is_protocol_conformant with real protocol and isinstance."""
        from victor.core.verticals.protocols import WorkflowProviderProtocol
        from victor.core.verticals.protocols.utils import is_protocol_conformant

        # Use isinstance directly to verify behavior
        class Provider:
            def get_workflows(self):
                return {}

            def get_auto_workflows(self):
                return []

        provider = Provider()

        # isinstance check (runtime_checkable)
        assert isinstance(provider, WorkflowProviderProtocol) == is_protocol_conformant(
            provider, WorkflowProviderProtocol
        )
