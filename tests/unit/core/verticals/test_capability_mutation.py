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

"""Tests for capability mutation tracking."""

import time

import pytest

from victor.core.verticals.capability_mutation import CapabilityMutation, CapabilityRollback


class TestCapabilityMutation:
    """Test suite for CapabilityMutation."""

    def test_creation(self):
        """Test basic mutation creation."""
        mutation = CapabilityMutation(
            capability="test_capability",
            args={"key": "value"},
            timestamp=time.time(),
        )
        assert mutation.capability == "test_capability"
        assert mutation.args == {"key": "value"}
        assert mutation.source == "vertical_integration"
        assert mutation.timestamp > 0

    def test_validation_empty_capability(self):
        """Test validation rejects empty capability name."""
        with pytest.raises(ValueError, match="capability cannot be empty"):
            CapabilityMutation(
                capability="",
                args={},
                timestamp=time.time(),
            )

    def test_validation_invalid_args_type(self):
        """Test validation rejects non-dict args."""
        with pytest.raises(TypeError, match="args must be a dict"):
            CapabilityMutation(
                capability="test",
                args="not_a_dict",
                timestamp=time.time(),
            )

    def test_validation_negative_timestamp(self):
        """Test validation rejects negative timestamp."""
        with pytest.raises(ValueError, match="timestamp must be non-negative"):
            CapabilityMutation(
                capability="test",
                args={},
                timestamp=-1.0,
            )

    def test_validation_invalid_source_type(self):
        """Test validation rejects non-string source."""
        with pytest.raises(TypeError, match="source must be a string"):
            CapabilityMutation(
                capability="test",
                args={},
                timestamp=time.time(),
                source=123,
            )

    def test_get_age(self):
        """Test age calculation."""
        ts = time.time() - 10
        mutation = CapabilityMutation(
            capability="test",
            args={},
            timestamp=ts,
        )
        age = mutation.get_age()
        assert age >= 10
        assert age < 11  # Should be close to 10 seconds

    def test_is_older_than_true(self):
        """Test is_older_than returns True when mutation is old."""
        ts = time.time() - 100
        mutation = CapabilityMutation(
            capability="test",
            args={},
            timestamp=ts,
        )
        assert mutation.is_older_than(50)
        assert mutation.is_older_than(99)

    def test_is_older_than_false(self):
        """Test is_older_than returns False when mutation is recent."""
        mutation = CapabilityMutation(
            capability="test",
            args={},
            timestamp=time.time(),
        )
        assert not mutation.is_older_than(100)
        assert not mutation.is_older_than(1000)


class TestCapabilityRollback:
    """Test suite for CapabilityRollback."""

    def test_creation(self):
        """Test basic rollback creation."""
        mutation = CapabilityMutation(
            capability="test",
            args={"new": "value"},
            timestamp=time.time(),
        )
        rollback = CapabilityRollback(
            mutation=mutation,
            previous_value={"old": "value"},
            rollback_timestamp=time.time(),
        )
        assert rollback.mutation == mutation
        assert rollback.previous_value == {"old": "value"}
        assert rollback.rollback_timestamp > 0

    def test_can_rollback_true(self):
        """Test can_rollback returns True when previous_value exists."""
        mutation = CapabilityMutation(
            capability="test",
            args={},
            timestamp=time.time(),
        )
        rollback = CapabilityRollback(
            mutation=mutation,
            previous_value={"old": "value"},
            rollback_timestamp=time.time(),
        )
        assert rollback.can_rollback()

    def test_can_rollback_false(self):
        """Test can_rollback returns False when previous_value is None."""
        mutation = CapabilityMutation(
            capability="test",
            args={},
            timestamp=time.time(),
        )
        rollback = CapabilityRollback(
            mutation=mutation,
            previous_value=None,
            rollback_timestamp=time.time(),
        )
        assert not rollback.can_rollback()

    def test_get_rollback_age(self):
        """Test rollback age calculation."""
        ts = time.time() - 5
        mutation = CapabilityMutation(
            capability="test",
            args={},
            timestamp=time.time(),
        )
        rollback = CapabilityRollback(
            mutation=mutation,
            previous_value={},
            rollback_timestamp=ts,
        )
        age = rollback.get_rollback_age()
        assert age >= 5
        assert age < 6  # Should be close to 5 seconds
