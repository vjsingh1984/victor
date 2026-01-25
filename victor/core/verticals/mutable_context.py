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

"""Mutable vertical context with capability mutation tracking.

This module provides a DIP-compliant mutable context that tracks all
capability mutations, enabling observability and rollback without
direct orchestrator mutation.

Design Philosophy:
- Single point of entry for all capability mutations (DIP)
- Track mutations for observability and debugging
- Support rollback for error recovery
- Protocol-based access (no private attribute access)
- Backward compatible with VerticalContext

SOLID Compliance:
- SRP: Single responsibility - track and apply capability mutations
- OCP: Open for extension, closed for modification
- LSP: Substitutable for VerticalContext
- ISP: Focused interface for mutation operations
- DIP: Depend on VerticalContext abstraction, not orchestrator

Usage:
    from victor.core.verticals.mutable_context import MutableVerticalContext

    # Create mutable context
    context = MutableVerticalContext("coding", {})

    # Apply capabilities through context (DIP compliant)
    context.apply_capability("allowed_tools", tools=["read", "write"])
    context.apply_capability("system_prompt", prompt="You are a coding assistant")

    # Query capabilities
    if context.has_capability("allowed_tools"):
        tools = context.get_capability("allowed_tools")

    # Rollback if needed
    context.rollback_last()

    # Get mutation history for debugging
    history = context.get_mutation_history()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from victor.core.verticals.base import VerticalConfig
from victor.core.verticals.capability_mutation import CapabilityMutation, CapabilityRollback
from victor.core.verticals.context import VerticalContext

logger = __import__("logging").getLogger(__name__)


class MutableVerticalContext(VerticalContext):
    """Vertical context with mutation tracking (DIP compliant).

    Extends VerticalContext with capability mutation tracking, providing:
    - Single point of entry for all capability mutations (DIP)
    - Observable mutation history
    - Rollback support for error recovery
    - Type-safe capability application
    - Serialization support for debugging

    This class ensures dependency inversion by routing all capability
    mutations through the context rather than directly mutating the
    orchestrator.

    Attributes:
        name: Vertical name
        config: Vertical configuration
        _mutations: List of all capability mutations
        _capability_values: Current capability values
        _rollback_stack: Stack of rollback information

    Example:
        context = MutableVerticalContext("coding", {})

        # Apply capabilities (DIP compliant)
        context.apply_capability("allowed_tools", tools=["read", "write"])
        context.apply_capability("system_prompt", prompt="Custom prompt")

        # Query
        if context.has_capability("allowed_tools"):
            tools = context.get_capability("allowed_tools")["tools"]

        # Rollback
        context.rollback_last()

        # History
        for mutation in context.get_mutation_history():
            print(f"{mutation.capability}: {mutation.args}")
    """

    def __init__(self, name: str, config: Optional[Union["VerticalConfig", Dict[str, Any]]] = None):
        """Initialize mutable context.

        Args:
            name: Vertical name
            config: Vertical configuration dict or VerticalConfig object
        """
        # Handle dict config by converting to empty dict (VerticalConfig is complex)
        # For now, we pass None if config is a plain dict to avoid type errors
        actual_config: Optional["VerticalConfig"] = config if isinstance(config, dict) or config is None else None  # type: ignore[arg-type]
        super().__init__(name=name, config=actual_config)
        self._mutations: List[CapabilityMutation] = []
        self._capability_values: Dict[str, Any] = {}
        self._rollback_stack: List[CapabilityRollback] = []

    def apply_capability(self, capability_name: str, **kwargs: Any) -> None:
        """Apply a capability through context (DIP compliant).

        This is the single point of entry for all capability mutations,
        ensuring dependency inversion and observability. Instead of
        mutating orchestrator attributes directly, capabilities are
        applied through the context.

        Args:
            capability_name: Name of capability to apply
            **kwargs: Capability arguments

        Example:
            context.apply_capability("allowed_tools", tools=["read", "write"])
            context.apply_capability("system_prompt", prompt="Custom prompt")
        """
        # Store previous value for rollback
        previous_value = self._capability_values.get(capability_name)

        # Create mutation record
        mutation = CapabilityMutation(
            capability=capability_name,
            args=dict(kwargs),
            timestamp=time.time(),
            source="vertical_integration",
        )

        # Record for rollback
        if previous_value is not None:
            rollback = CapabilityRollback(
                mutation=mutation,
                previous_value=previous_value,
                rollback_timestamp=time.time(),
            )
            self._rollback_stack.append(rollback)

        # Apply mutation
        self._mutations.append(mutation)
        self._capability_values[capability_name] = kwargs

        # Update config (but don't mutate orchestrator directly)
        if self.config is not None:
            if "_applied_capabilities" not in self.config:
                (self.config)["_applied_capabilities"] = {}  # type: ignore[index]
            (self.config)["_applied_capabilities"][capability_name] = kwargs  # type: ignore[index]
        else:
            # Initialize config if it's None
            self.config = {"_applied_capabilities": {capability_name: kwargs}}  # type: ignore[assignment]

    def get_capability(self, capability_name: str) -> Optional[Any]:
        """Get applied capability value.

        Args:
            capability_name: Name of capability

        Returns:
            Capability value or None if not found
        """
        return self._capability_values.get(capability_name)

    def has_capability(self, capability_name: str) -> bool:
        """Check if capability has been applied.

        Args:
            capability_name: Name of capability

        Returns:
            True if capability has been applied
        """
        return capability_name in self._capability_values

    def get_mutation_history(self) -> List[CapabilityMutation]:
        """Get history of capability changes.

        Returns:
            List of all mutations in chronological order
        """
        return list(self._mutations)

    def get_mutation_count(self) -> int:
        """Get total number of mutations.

        Returns:
            Number of mutations applied
        """
        return len(self._mutations)

    def get_recent_mutations(self, seconds: float = 300.0) -> List[CapabilityMutation]:
        """Get mutations from last N seconds.

        Args:
            seconds: Time window in seconds (default: 300 = 5 minutes)

        Returns:
            List of recent mutations
        """
        return [m for m in self._mutations if not m.is_older_than(seconds)]

    def rollback_to(self, mutation_index: int) -> None:
        """Rollback to a specific mutation state.

        Removes all mutations after the specified index.

        Args:
            mutation_index: Index to rollback to (0-based)

        Raises:
            IndexError: If index is out of range

        Example:
            context.apply_capability("cap1", value=1)
            context.apply_capability("cap2", value=2)
            context.apply_capability("cap3", value=3)

            context.rollback_to(1)  # Keep cap1 and cap2, remove cap3
        """
        if mutation_index < 0 or mutation_index >= len(self._mutations):
            raise IndexError(f"Invalid mutation index: {mutation_index}")

        # Remove mutations after index
        mutations_to_remove = self._mutations[mutation_index + 1 :]

        for mutation in mutations_to_remove:
            # Remove from capability values
            self._capability_values.pop(mutation.capability, None)

            # Remove from config
            if self.config is not None and "_applied_capabilities" in self.config:
                self.config["_applied_capabilities"].pop(mutation.capability, None)

        # Truncate mutations list
        self._mutations = self._mutations[: mutation_index + 1]

    def rollback_last(self) -> bool:
        """Rollback the last mutation.

        Returns:
            True if rollback succeeded, False if no mutations to rollback

        Example:
            context.apply_capability("cap1", value=1)
            context.rollback_last()  # Removes cap1
        """
        if not self._mutations:
            return False

        last_mutation = self._mutations.pop()

        # Remove from capability values
        self._capability_values.pop(last_mutation.capability, None)

        # Remove from config
        if self.config is not None and "_applied_capabilities" in self.config:
            self.config["_applied_capabilities"].pop(last_mutation.capability, None)

        # Pop from rollback stack
        if self._rollback_stack:
            self._rollback_stack.pop()

        return True

    def clear_mutations(self) -> None:
        """Clear all mutation history.

        Resets the context to its initial state with no mutations.

        Example:
            context.clear_mutations()
            assert context.get_mutation_count() == 0
        """
        self._mutations.clear()
        self._capability_values.clear()
        self._rollback_stack.clear()
        if self.config is not None and "_applied_capabilities" in self.config:
            del self.config["_applied_capabilities"]  # type: ignore[attr-defined]

    def get_mutations_by_capability(self, capability_name: str) -> List[CapabilityMutation]:
        """Get all mutations for a specific capability.

        Args:
            capability_name: Name of capability

        Returns:
            List of mutations for the capability
        """
        return [m for m in self._mutations if m.capability == capability_name]

    def get_all_applied_capabilities(self) -> Dict[str, Any]:
        """Get all currently applied capabilities.

        Returns:
            Dict mapping capability names to their values
        """
        return dict(self._capability_values)

    def export_state(self) -> Dict[str, Any]:
        """Export current state for serialization.

        Useful for debugging and persistence.

        Returns:
            Dict representation of current state
        """
        return {
            "name": self.name,
            "config": self.config,
            "mutations": [
                {
                    "capability": m.capability,
                    "args": m.args,
                    "timestamp": m.timestamp,
                    "source": m.source,
                }
                for m in self._mutations
            ],
            "capability_values": self._capability_values,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from serialization.

        Reconstructs context from previously exported state.

        Args:
            state: State dict from export_state()

        Example:
            exported = context.export_state()
            new_context = MutableVerticalContext("test", {})
            new_context.import_state(exported)
        """
        self.name = state["name"]
        self.config = state["config"]

        # Reconstruct mutations
        self._mutations = []
        for m_data in state.get("mutations", []):
            mutation = CapabilityMutation(
                capability=m_data["capability"],
                args=m_data["args"],
                timestamp=m_data["timestamp"],
                source=m_data.get("source", "imported"),
            )
            self._mutations.append(mutation)

        self._capability_values = state.get("capability_values", {})

    # VerticalContext compatibility methods
    def apply_stages(self, stages: Dict[str, Any]) -> None:
        """Apply stage configuration (delegates to apply_capability)."""
        self.apply_capability("stages", stages=stages)
        super().apply_stages(stages)

    def apply_middleware(self, middleware: List[Any]) -> None:
        """Apply middleware (delegates to apply_capability)."""
        self.apply_capability("middleware", middleware=middleware)
        super().apply_middleware(middleware)

    def apply_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns (delegates to apply_capability)."""
        self.apply_capability("safety_patterns", patterns=patterns)
        super().apply_safety_patterns(patterns)

    def apply_system_prompt(self, prompt: str) -> None:
        """Apply system prompt (delegates to apply_capability)."""
        self.apply_capability("system_prompt", prompt=prompt)
        super().apply_system_prompt(prompt)


__all__ = [
    "MutableVerticalContext",
]
