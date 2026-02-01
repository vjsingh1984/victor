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

"""Capability mutation tracking for vertical context.

This module provides data structures for tracking capability mutations
in a DIP-compliant manner, enabling observability and rollback of
vertical capability changes.

Design Philosophy:
- Track all capability mutations for observability
- Support rollback for error recovery
- Type-safe mutation records
- Minimal overhead for tracking

SOLID Compliance:
- SRP: Each class has single responsibility (mutation, rollback)
- OCP: Open for extension via inheritance, closed for modification
- LSP: Subtypes maintain behavior contracts
- ISP: Focused interfaces for specific use cases
- DIP: Depend on abstractions (CapabilityMutation protocol)

Usage:
    from victor.core.verticals.capability_mutation import CapabilityMutation

    mutation = CapabilityMutation(
        capability="allowed_tools",
        args={"tools": ["read", "write"]},
        timestamp=time.time(),
    )

    if mutation.is_older_than(3600):
        # Mutation is older than 1 hour
        pass
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CapabilityMutation:
    """Record of a capability application.

    Tracks when and how a capability was applied to a vertical context,
    enabling observability and rollback.

    Attributes:
        capability: Name of the capability that was applied
        args: Arguments passed to the capability
        timestamp: Unix timestamp when mutation was created
        source: Source of the mutation (e.g., "vertical_integration")

    Example:
        mutation = CapabilityMutation(
            capability="allowed_tools",
            args={"tools": ["read", "write"]},
            timestamp=time.time(),
        )
    """

    capability: str
    args: dict[str, Any]
    timestamp: float
    source: str = "vertical_integration"

    def __post_init__(self) -> None:
        """Validate mutation on creation.

        Raises:
            ValueError: If validation fails
            TypeError: If types are incorrect
        """
        if not self.capability:
            raise ValueError("capability cannot be empty")
        if not isinstance(self.args, dict):
            raise TypeError("args must be a dict")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")

    def get_age(self) -> float:
        """Get age of mutation in seconds.

        Returns:
            Age in seconds since creation
        """
        return time.time() - self.timestamp

    def is_older_than(self, seconds: float) -> bool:
        """Check if mutation is older than given seconds.

        Args:
            seconds: Threshold in seconds

        Returns:
            True if mutation is older than threshold
        """
        return self.get_age() > seconds


@dataclass
class CapabilityRollback:
    """Rollback information for capability mutation.

    Stores information needed to rollback a capability mutation
    to its previous state.

    Attributes:
        mutation: The mutation being rolled back
        previous_value: Previous value before mutation
        rollback_timestamp: When rollback was created

    Example:
        rollback = CapabilityRollback(
            mutation=mutation,
            previous_value={"tools": ["read"]},
            rollback_timestamp=time.time(),
        )

        if rollback.can_rollback():
            # Restore previous value
            apply_capability(rollback.previous_value)
    """

    mutation: CapabilityMutation
    previous_value: Any
    rollback_timestamp: float

    def can_rollback(self) -> bool:
        """Check if rollback is possible.

        Returns:
            True if previous_value is not None
        """
        return self.previous_value is not None

    def get_rollback_age(self) -> float:
        """Get age of rollback in seconds.

        Returns:
            Age in seconds since rollback was created
        """
        return time.time() - self.rollback_timestamp


__all__ = [
    "CapabilityMutation",
    "CapabilityRollback",
]
