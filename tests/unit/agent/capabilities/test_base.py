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

"""Tests for capability base classes."""

import pytest

from victor.agent.capabilities.base import CapabilityBase, CapabilitySpec


class TestCapabilitySpec:
    """Test CapabilitySpec dataclass."""

    def test_create_valid_spec(self):
        """Test creating a valid capability spec."""
        spec = CapabilitySpec(
            name="test_capability",
            method_name="set_test_capability",
            version="1.0",
            description="Test capability",
        )

        assert spec.name == "test_capability"
        assert spec.method_name == "set_test_capability"
        assert spec.version == "1.0"
        assert spec.description == "Test capability"

    def test_spec_defaults(self):
        """Test spec default values."""
        spec = CapabilitySpec(
            name="test",
            method_name="set_test",
        )

        assert spec.version == "1.0"
        assert spec.description == ""

    def test_spec_frozen(self):
        """Test that spec is immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        spec = CapabilitySpec(
            name="test",
            method_name="set_test",
        )

        with pytest.raises(FrozenInstanceError):
            spec.name = "new_name"

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            CapabilitySpec(
                name="",
                method_name="set_test",
            )

    def test_empty_method_name_raises_error(self):
        """Test that empty method_name raises ValueError."""
        with pytest.raises(ValueError, match="method_name cannot be empty"):
            CapabilitySpec(
                name="test",
                method_name="",
            )

    def test_invalid_version_format_raises_error(self):
        """Test that invalid version format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version"):
            CapabilitySpec(
                name="test",
                method_name="set_test",
                version="invalid",
            )

    def test_version_missing_minor_raises_error(self):
        """Test that version without minor number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version"):
            CapabilitySpec(
                name="test",
                method_name="set_test",
                version="1",
            )

    def test_version_negative_numbers_raises_error(self):
        """Test that negative version numbers raise ValueError."""
        with pytest.raises(ValueError, match="Invalid version"):
            CapabilitySpec(
                name="test",
                method_name="set_test",
                version="-1.0",
            )

    def test_valid_version_formats(self):
        """Test various valid version formats."""
        valid_versions = ["0.0", "0.1", "1.0", "10.20", "100.200"]

        for version in valid_versions:
            spec = CapabilitySpec(
                name="test",
                method_name="set_test",
                version=version,
            )
            assert spec.version == version


class TestCapabilityBase:
    """Test CapabilityBase abstract class."""

    def test_cannot_instantiate_base_class(self):
        """Test that CapabilityBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CapabilityBase()

    def test_subclass_must_implement_get_spec(self):
        """Test that subclass must implement get_spec."""

        class IncompleteCapability(CapabilityBase):
            pass

        with pytest.raises(NotImplementedError):
            IncompleteCapability.get_spec()

    def test_valid_subclass(self):
        """Test creating a valid capability subclass."""

        class ValidCapability(CapabilityBase):
            @classmethod
            def get_spec(cls) -> CapabilitySpec:
                return CapabilitySpec(
                    name="valid",
                    method_name="set_valid",
                    version="1.0",
                    description="Valid capability",
                )

        spec = ValidCapability.get_spec()
        assert spec.name == "valid"
        assert spec.method_name == "set_valid"
        assert spec.version == "1.0"
        assert spec.description == "Valid capability"

    def test_multiple_subclasses(self):
        """Test creating multiple capability subclasses."""

        class CapabilityA(CapabilityBase):
            @classmethod
            def get_spec(cls) -> CapabilitySpec:
                return CapabilitySpec(
                    name="capability_a",
                    method_name="set_capability_a",
                )

        class CapabilityB(CapabilityBase):
            @classmethod
            def get_spec(cls) -> CapabilitySpec:
                return CapabilitySpec(
                    name="capability_b",
                    method_name="set_capability_b",
                    version="2.0",
                )

        spec_a = CapabilityA.get_spec()
        spec_b = CapabilityB.get_spec()

        assert spec_a.name == "capability_a"
        assert spec_b.name == "capability_b"
        assert spec_b.version == "2.0"
