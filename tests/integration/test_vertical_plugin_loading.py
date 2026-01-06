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

"""Integration tests for vertical plugin loading system.

Tests the external vertical discovery via entry_points, validation,
name conflict handling, and idempotency of the discovery process.

Uses mocking to simulate entry_points discovery since we cannot install
actual external packages during tests.
"""

import logging
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from victor.core.verticals import VerticalBase, VerticalRegistry


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================


class ValidExternalVertical(VerticalBase):
    """A valid external vertical for testing."""

    name = "external_test"
    description = "A test external vertical"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "external_tool"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are an external test assistant."


class AnotherValidVertical(VerticalBase):
    """Another valid external vertical for testing."""

    name = "another_external"
    description = "Another test external vertical"
    version = "2.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "custom_tool"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are another external assistant."


class InvalidNoNameVertical(VerticalBase):
    """Invalid vertical with no name."""

    name = ""
    description = "Invalid vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Invalid"


class InvalidToolsReturnType(VerticalBase):
    """Invalid vertical with wrong get_tools return type."""

    name = "invalid_tools"
    description = "Invalid vertical"

    @classmethod
    def get_tools(cls) -> str:  # type: ignore
        return "not a list"  # Should be a list

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Invalid"


class InvalidPromptReturnType(VerticalBase):
    """Invalid vertical with wrong get_system_prompt return type."""

    name = "invalid_prompt"
    description = "Invalid vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> int:  # type: ignore
        return 123  # Should be a string


class ConflictingNameVertical(VerticalBase):
    """Vertical with name that conflicts with built-in coding."""

    name = "coding"  # Conflicts with built-in
    description = "Conflicting vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Conflicting"


def create_mock_entry_point(name: str, load_result):
    """Create a mock entry point object."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"test_package:{name}"
    ep.load.return_value = load_result
    return ep


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry to known state before and after each test."""
    # Store current state
    original_registry = dict(VerticalRegistry._registry)
    original_discovered = VerticalRegistry._external_discovered

    yield

    # Restore
    VerticalRegistry._registry = original_registry
    VerticalRegistry._external_discovered = original_discovered


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.integration
class TestExternalVerticalDiscovery:
    """Tests for discovering external verticals via entry_points."""

    def test_discover_valid_external_vertical(self):
        """External verticals can be discovered via entry_points."""
        # Create mock entry point
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        # Reset discovery flag
        VerticalRegistry.reset_discovery()

        # Mock entry_points at importlib.metadata level
        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            # Discover external verticals
            discovered = VerticalRegistry.discover_external_verticals()

            # Should have discovered our vertical
            assert "external_test" in discovered
            assert discovered["external_test"] is ValidExternalVertical

            # Should be registered
            assert VerticalRegistry.get("external_test") is ValidExternalVertical

    def test_discover_multiple_external_verticals(self):
        """Multiple external verticals can be discovered at once."""
        mock_ep1 = create_mock_entry_point("external_test", ValidExternalVertical)
        mock_ep2 = create_mock_entry_point("another_external", AnotherValidVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep1, mock_ep2]

            discovered = VerticalRegistry.discover_external_verticals()

            assert len(discovered) == 2
            assert "external_test" in discovered
            assert "another_external" in discovered

    def test_discover_with_no_entry_points(self):
        """Discovery handles case with no external entry points."""
        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = []

            discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}


@pytest.mark.integration
class TestInvalidVerticalRejection:
    """Tests for rejecting invalid external verticals with warnings."""

    def test_reject_non_class_entry_point(self, caplog):
        """Non-class entry points should be rejected with warning."""
        # Entry point returns a function instead of a class
        mock_ep = create_mock_entry_point("not_a_class", lambda: "function")

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}
            assert "not a class" in caplog.text.lower()

    def test_reject_non_verticalbase_class(self, caplog):
        """Classes not inheriting from VerticalBase should be rejected."""

        class NotAVertical:
            name = "fake"

        mock_ep = create_mock_entry_point("not_vertical", NotAVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}
            assert "does not inherit from VerticalBase" in caplog.text

    def test_reject_vertical_without_name(self, caplog):
        """Verticals without name attribute should be rejected."""
        mock_ep = create_mock_entry_point("no_name", InvalidNoNameVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}
            assert "no 'name' attribute" in caplog.text

    def test_reject_vertical_with_invalid_tools_return(self, caplog):
        """Verticals with invalid get_tools() return should be rejected."""
        mock_ep = create_mock_entry_point("invalid_tools", InvalidToolsReturnType)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}
            assert "must return a list" in caplog.text

    def test_reject_vertical_with_invalid_prompt_return(self, caplog):
        """Verticals with invalid get_system_prompt() return should be rejected."""
        mock_ep = create_mock_entry_point("invalid_prompt", InvalidPromptReturnType)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}
            assert "must return a string" in caplog.text

    def test_reject_vertical_with_load_error(self, caplog):
        """Entry points that fail to load should be handled gracefully."""
        mock_ep = MagicMock()
        mock_ep.name = "broken_ep"
        mock_ep.value = "broken_package:BrokenClass"
        mock_ep.load.side_effect = ImportError("Module not found")

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}
            assert "Failed to load external vertical" in caplog.text

    def test_valid_and_invalid_mixed(self, caplog):
        """Valid verticals should be registered even when some are invalid."""
        valid_ep = create_mock_entry_point("external_test", ValidExternalVertical)
        invalid_ep = create_mock_entry_point("no_name", InvalidNoNameVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [valid_ep, invalid_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            # Valid one should be discovered
            assert "external_test" in discovered
            assert len(discovered) == 1

            # Invalid one should be logged
            assert "no 'name' attribute" in caplog.text


@pytest.mark.integration
class TestNameConflictHandling:
    """Tests for handling name conflicts between external and built-in verticals."""

    def test_conflict_with_builtin_coding_rejected(self, caplog):
        """External vertical with name 'coding' should be rejected."""
        mock_ep = create_mock_entry_point("coding_conflict", ConflictingNameVertical)

        VerticalRegistry.reset_discovery()

        # Ensure coding is registered (it should be by default)
        assert VerticalRegistry.get("coding") is not None

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            # Should not be discovered due to conflict
            assert (
                "coding" not in discovered
                or discovered.get("coding") is not ConflictingNameVertical
            )

            # Original coding should still be there
            original = VerticalRegistry.get("coding")
            assert original is not None
            assert original is not ConflictingNameVertical

            # Warning should be logged
            assert "conflicts with existing vertical" in caplog.text

    def test_conflict_between_external_verticals(self, caplog):
        """First external vertical should win in case of duplicates."""

        class DuplicateVertical(VerticalBase):
            name = "external_test"  # Same name as ValidExternalVertical
            description = "Duplicate"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Duplicate"

        # First valid, second is duplicate
        valid_ep = create_mock_entry_point("external_test", ValidExternalVertical)
        duplicate_ep = create_mock_entry_point("duplicate", DuplicateVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [valid_ep, duplicate_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            # First one should be registered
            assert VerticalRegistry.get("external_test") is ValidExternalVertical
            # Conflict warning for second
            assert "conflicts with existing vertical" in caplog.text


@pytest.mark.integration
class TestDiscoveryIdempotency:
    """Tests for discovery process idempotency."""

    def test_discover_external_verticals_is_idempotent(self):
        """Calling discover_external_verticals multiple times should be safe."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            # First call discovers
            result1 = VerticalRegistry.discover_external_verticals()
            assert "external_test" in result1

            # Second call returns empty (already discovered)
            result2 = VerticalRegistry.discover_external_verticals()
            assert result2 == {}

            # Third call also returns empty
            result3 = VerticalRegistry.discover_external_verticals()
            assert result3 == {}

            # Entry points only loaded once
            assert mock_entry_points.call_count == 1

    def test_registry_state_consistent_after_multiple_discovers(self):
        """Registry should be consistent after multiple discovery calls."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            # Multiple discovery calls
            VerticalRegistry.discover_external_verticals()
            VerticalRegistry.discover_external_verticals()
            VerticalRegistry.discover_external_verticals()

            # Registry should have the vertical registered exactly once
            all_verticals = VerticalRegistry.list_all()
            external_count = sum(1 for name, _ in all_verticals if name == "external_test")
            assert external_count == 1

    def test_discovery_flag_prevents_reprocessing(self):
        """_external_discovered flag should prevent reprocessing."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()
        assert VerticalRegistry._external_discovered is False

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            VerticalRegistry.discover_external_verticals()
            assert VerticalRegistry._external_discovered is True

            # Even with different entry points, nothing new is processed
            mock_entry_points.return_value = [
                create_mock_entry_point("another", AnotherValidVertical)
            ]

            result = VerticalRegistry.discover_external_verticals()
            assert result == {}
            assert VerticalRegistry.get("another_external") is None


@pytest.mark.integration
class TestResetDiscovery:
    """Tests for VerticalRegistry.reset_discovery() functionality."""

    def test_reset_discovery_clears_flag(self):
        """reset_discovery should clear the _external_discovered flag."""
        VerticalRegistry._external_discovered = True

        VerticalRegistry.reset_discovery()

        assert VerticalRegistry._external_discovered is False

    def test_reset_discovery_allows_rediscovery(self):
        """reset_discovery should allow discover_external_verticals to run again."""
        mock_ep1 = create_mock_entry_point("external_test", ValidExternalVertical)
        mock_ep2 = create_mock_entry_point("another_external", AnotherValidVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            # First discovery with one vertical
            mock_entry_points.return_value = [mock_ep1]
            result1 = VerticalRegistry.discover_external_verticals()
            assert "external_test" in result1

            # Reset
            VerticalRegistry.reset_discovery()

            # Second discovery with different vertical
            mock_entry_points.return_value = [mock_ep2]
            result2 = VerticalRegistry.discover_external_verticals()
            assert "another_external" in result2

    def test_clear_also_resets_discovery(self):
        """VerticalRegistry.clear() should also reset the discovery flag."""
        VerticalRegistry._external_discovered = True

        VerticalRegistry.clear(reregister_builtins=False)

        assert VerticalRegistry._external_discovered is False
        assert VerticalRegistry._registry == {}

    def test_reset_discovery_preserves_registry(self):
        """reset_discovery should not clear registered verticals."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]
            VerticalRegistry.discover_external_verticals()

        # Vertical should be registered
        assert VerticalRegistry.get("external_test") is ValidExternalVertical

        # Reset discovery
        VerticalRegistry.reset_discovery()

        # Vertical should still be registered
        assert VerticalRegistry.get("external_test") is ValidExternalVertical


@pytest.mark.integration
class TestEntryPointGroupConstant:
    """Tests for entry point group configuration."""

    def test_entry_point_group_is_correct(self):
        """ENTRY_POINT_GROUP should be 'victor.verticals'."""
        assert VerticalRegistry.ENTRY_POINT_GROUP == "victor.verticals"

    def test_discover_uses_correct_entry_point_group(self):
        """discover_external_verticals should query the correct group."""
        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = []
            VerticalRegistry.discover_external_verticals()

            # Should have been called with the correct group
            mock_entry_points.assert_called_once_with(group="victor.verticals")


@pytest.mark.integration
class TestValidationHelperMethod:
    """Tests for _validate_external_vertical helper method."""

    def test_validate_rejects_none(self):
        """_validate_external_vertical should reject None."""
        result = VerticalRegistry._validate_external_vertical(None, "test_ep")
        assert result is False

    def test_validate_rejects_instance(self):
        """_validate_external_vertical should reject class instances."""
        instance = ValidExternalVertical()
        result = VerticalRegistry._validate_external_vertical(instance, "test_ep")
        assert result is False

    def test_validate_accepts_valid_class(self):
        """_validate_external_vertical should accept valid vertical classes."""
        result = VerticalRegistry._validate_external_vertical(ValidExternalVertical, "test_ep")
        assert result is True

    def test_validate_rejects_abstract_vertical_base(self):
        """_validate_external_vertical should reject VerticalBase itself (no name)."""
        # VerticalBase has empty name
        result = VerticalRegistry._validate_external_vertical(VerticalBase, "test_ep")
        # VerticalBase has name = "" which should fail validation
        assert result is False


@pytest.mark.integration
class TestLoggingBehavior:
    """Tests for proper logging during discovery."""

    def test_successful_discovery_logs_info(self, caplog):
        """Successful discovery should log info messages."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.INFO):
                VerticalRegistry.discover_external_verticals()

            assert "Discovered external vertical" in caplog.text
            assert "external_test" in caplog.text

    def test_discovery_summary_logged(self, caplog):
        """Discovery should log summary of discovered verticals."""
        mock_ep1 = create_mock_entry_point("external_test", ValidExternalVertical)
        mock_ep2 = create_mock_entry_point("another", AnotherValidVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep1, mock_ep2]

            with caplog.at_level(logging.INFO):
                VerticalRegistry.discover_external_verticals()

            # Summary should mention count
            assert "2 external vertical(s)" in caplog.text


@pytest.mark.integration
class TestEdgeCases:
    """Tests for edge cases in vertical plugin loading."""

    def test_entry_point_with_exception_in_abstract_method(self, caplog):
        """Verticals that raise exceptions in abstract methods should be rejected."""

        class ExceptionVertical(VerticalBase):
            name = "exception_vertical"
            description = "Raises exception"

            @classmethod
            def get_tools(cls) -> List[str]:
                raise RuntimeError("Intentional error")

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Exception"

        mock_ep = create_mock_entry_point("exception", ExceptionVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            with caplog.at_level(logging.WARNING):
                discovered = VerticalRegistry.discover_external_verticals()

            assert discovered == {}
            assert "failed validation" in caplog.text

    def test_python_310_entry_points_api(self):
        """Test compatibility with Python 3.10+ entry_points API."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        # Python 3.10+ uses group parameter
        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]
            VerticalRegistry.discover_external_verticals()

            mock_entry_points.assert_called_with(group="victor.verticals")

    def test_fallback_for_older_python(self):
        """Test fallback for older Python entry_points API."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            # Simulate older API that raises TypeError for group parameter
            call_count = [0]

            def old_api_simulation(**kwargs):
                call_count[0] += 1
                if "group" in kwargs:
                    raise TypeError("No group parameter in old API")
                return {"victor.verticals": [mock_ep]}

            mock_entry_points.side_effect = old_api_simulation

            # This should handle the TypeError gracefully
            discovered = VerticalRegistry.discover_external_verticals()

            # Should have discovered the vertical using fallback
            assert "external_test" in discovered
            # Should have called entry_points twice (once with group, once without)
            assert call_count[0] == 2


@pytest.mark.integration
class TestIntegrationWithBuiltinVerticals:
    """Tests for interaction between external and built-in verticals."""

    def test_builtin_verticals_not_affected_by_discovery(self):
        """Built-in verticals should not be affected by external discovery."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        # Get built-in verticals before discovery
        coding_before = VerticalRegistry.get("coding")
        research_before = VerticalRegistry.get("research")

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]
            VerticalRegistry.discover_external_verticals()

        # Built-in verticals should be unchanged
        assert VerticalRegistry.get("coding") is coding_before
        assert VerticalRegistry.get("research") is research_before

    def test_external_verticals_listed_with_builtins(self):
        """list_all should include both built-in and external verticals."""
        mock_ep = create_mock_entry_point("external_test", ValidExternalVertical)

        VerticalRegistry.reset_discovery()

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]
            VerticalRegistry.discover_external_verticals()

        names = VerticalRegistry.list_names()

        # Should include built-ins
        assert "coding" in names
        assert "research" in names

        # Should include external
        assert "external_test" in names
