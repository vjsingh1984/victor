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

"""Workflow State Manager - State Inspection, Visualization, and Rollback.

This module provides comprehensive state management capabilities:

Features:
- Capture and manage workflow state snapshots
- Visualize state changes and diffs
- Query state using JSONPath-like syntax
- Rollback state to previous snapshots
- Export state for analysis
- Monitor state mutations

Usage:
    manager = WorkflowStateManager()
    snapshot_id = manager.capture_state(current_state, node_id="node_1")

    # Visualize state
    print(manager.visualize_state(snapshot_id))

    # Query state
    value = manager.query_state(snapshot_id, "user.name")

    # Rollback
    manager.rollback_to(snapshot_id)
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StateSnapshot:
    """Snapshot of workflow state.

    Attributes:
        snapshot_id: Unique snapshot identifier
        timestamp: Snapshot timestamp
        node_id: Node ID (if captured during node execution)
        state: State data
        metadata: Additional metadata
        parent_id: Parent snapshot ID (for rollback chains)
    """
    snapshot_id: str
    timestamp: float
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: Optional[str] = None
    parent_id: Optional[str] = None


@dataclass
class StateDiff:
    """Difference between two states.

    Attributes:
        snapshot_id: Snapshot ID
        from_snapshot_id: Source snapshot ID
        to_snapshot_id: Target snapshot ID
        added_keys: Keys added
        removed_keys: Keys removed
        changed_keys: Keys changed
        unchanged_keys: Keys unchanged
    """
    snapshot_id: str
    from_snapshot_id: str
    to_snapshot_id: str
    added_keys: Dict[str, Any]
    removed_keys: Dict[str, Any]
    changed_keys: Dict[str, Dict[str, Any]]
    unchanged_keys: List[str]


@dataclass
class StateMutation:
    """Record of a state mutation.

    Attributes:
        mutation_id: Unique mutation identifier
        timestamp: Mutation timestamp
        node_id: Node ID that caused mutation
        key: Key that was mutated
        old_value: Old value
        new_value: New value
        mutation_type: Type of mutation (add, remove, change)
    """
    mutation_id: str
    timestamp: float
    node_id: Optional[str]
    key: str
    old_value: Any
    new_value: Any
    mutation_type: str  # "add", "remove", "change"


# =============================================================================
# Workflow State Manager
# =============================================================================


class WorkflowStateManager:
    """Workflow state manager.

    Manages workflow state with capabilities for capturing snapshots,
    visualizing changes, querying data, and rolling back.
    """

    def __init__(
        self,
        max_snapshots: int = 100,
        auto_capture: bool = False,
    ):
        """Initialize state manager.

        Args:
            max_snapshots: Maximum number of snapshots to keep
            auto_capture: Whether to auto-capture on node execution
        """
        self.max_snapshots = max_snapshots
        self.auto_capture = auto_capture

        # Snapshot storage
        self._snapshots: Dict[str, StateSnapshot] = {}
        self._snapshot_order: List[str] = []

        # Mutation log
        self._mutations: List[StateMutation] = []

        # Current state
        self._current_state: Dict[str, Any] = {}

    # =========================================================================
    # Snapshot Management
    # =========================================================================

    def capture_state(
        self,
        state: Dict[str, Any],
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Capture a state snapshot.

        Args:
            state: State to capture
            node_id: Optional node ID
            metadata: Optional metadata

        Returns:
            Snapshot ID
        """
        snapshot_id = str(uuid4())

        # Create deep copy to avoid mutation
        state_copy = copy.deepcopy(state)

        # Get parent snapshot
        parent_id = self._snapshot_order[-1] if self._snapshot_order else None

        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            state=state_copy,
            node_id=node_id,
            metadata=metadata or {},
            parent_id=parent_id,
        )

        # Add to storage
        self._snapshots[snapshot_id] = snapshot
        self._snapshot_order.append(snapshot_id)

        # Track mutations if we have a parent
        if parent_id:
            self._track_mutations(parent_id, snapshot_id, node_id)

        # Enforce max snapshots
        while len(self._snapshot_order) > self.max_snapshots:
            old_id = self._snapshot_order.pop(0)
            del self._snapshots[old_id]

        logger.debug(f"Captured state snapshot '{snapshot_id}' (node: {node_id})")

        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Get a snapshot by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            Snapshot or None
        """
        return self._snapshots.get(snapshot_id)

    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        """Get the most recent snapshot.

        Returns:
            Latest snapshot or None
        """
        if not self._snapshot_order:
            return None

        latest_id = self._snapshot_order[-1]
        return self._snapshots.get(latest_id)

    def list_snapshots(
        self,
        limit: Optional[int] = None,
    ) -> List[StateSnapshot]:
        """List snapshots in chronological order.

        Args:
            limit: Optional limit on number of snapshots

        Returns:
            List of snapshots
        """
        order = self._snapshot_order[-limit:] if limit else self._snapshot_order

        return [self._snapshots[sid] for sid in order]

    def remove_snapshot(self, snapshot_id: str) -> bool:
        """Remove a snapshot.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            True if removed, False if not found
        """
        if snapshot_id not in self._snapshots:
            return False

        del self._snapshots[snapshot_id]
        self._snapshot_order.remove(snapshot_id)

        logger.debug(f"Removed snapshot '{snapshot_id}'")
        return True

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        count = len(self._snapshot_order)
        self._snapshots.clear()
        self._snapshot_order.clear()
        logger.info(f"Cleared {count} snapshot(s)")

    # =========================================================================
    # State Visualization
    # =========================================================================

    def visualize_state(
        self,
        snapshot_id: Optional[str] = None,
        format: str = "text",
    ) -> str:
        """Visualize state.

        Args:
            snapshot_id: Snapshot ID (uses latest if None)
            format: Output format (text, json, yaml)

        Returns:
            Formatted state string
        """
        snapshot = (
            self.get_snapshot(snapshot_id) if snapshot_id
            else self.get_latest_snapshot()
        )

        if not snapshot:
            raise ValueError("No snapshot to visualize")

        state = snapshot.state

        if format == "text":
            return self._format_state_text(state)
        elif format == "json":
            return json.dumps(state, indent=2, default=str)
        elif format == "yaml":
            try:
                import yaml
                return yaml.dump(state, default_flow_style=False)
            except ImportError:
                raise ValueError("PyYAML not installed for YAML format")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_state_text(self, state: Dict[str, Any]) -> str:
        """Format state as text.

        Args:
            state: State dictionary

        Returns:
            Formatted text
        """
        lines = []
        lines.append("=" * 60)
        lines.append("WORKFLOW STATE")
        lines.append("=" * 60)

        # Filter internal keys
        display_state = {
            k: v for k, v in state.items() if not k.startswith("_")
        }

        # Sort keys for consistent output
        for key in sorted(display_state.keys()):
            value = display_state[key]

            # Truncate long values
            value_str = self._format_value(value)

            lines.append(f"{key}: {value_str}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_value(self, value: Any, max_length: int = 100) -> str:
        """Format a value for display.

        Args:
            value: Value to format
            max_length: Maximum string length

        Returns:
            Formatted string
        """
        value_str = str(value)

        if len(value_str) > max_length:
            return value_str[:max_length] + "..."

        return value_str

    def visualize_diff(
        self,
        from_snapshot_id: str,
        to_snapshot_id: Optional[str] = None,
    ) -> str:
        """Visualize difference between two snapshots.

        Args:
            from_snapshot_id: Source snapshot ID
            to_snapshot_id: Target snapshot ID (uses latest if None)

        Returns:
            Formatted diff string
        """
        from_snapshot = self.get_snapshot(from_snapshot_id)
        to_snapshot = (
            self.get_snapshot(to_snapshot_id) if to_snapshot_id
            else self.get_latest_snapshot()
        )

        if not from_snapshot or not to_snapshot:
            raise ValueError("One or both snapshots not found")

        diff = self.compute_diff(from_snapshot_id, to_snapshot_id)

        lines = []
        lines.append("=" * 60)
        lines.append(f"STATE DIFF: {from_snapshot_id} -> {to_snapshot_id}")
        lines.append("=" * 60)

        # Added keys
        if diff.added_keys:
            lines.append("\n[green]ADDED:[/]")
            for key, value in diff.added_keys.items():
                lines.append(f"  + {key}: {self._format_value(value)}")

        # Removed keys
        if diff.removed_keys:
            lines.append("\n[red]REMOVED:[/]")
            for key, value in diff.removed_keys.items():
                lines.append(f"  - {key}: {self._format_value(value)}")

        # Changed keys
        if diff.changed_keys:
            lines.append("\n[yellow]CHANGED:[/]")
            for key, change in diff.changed_keys.items():
                lines.append(f"  ~ {key}:")
                lines.append(f"      - {self._format_value(change['from'])}")
                lines.append(f"      + {self._format_value(change['to'])}")

        # Unchanged keys
        if diff.unchanged_keys:
            lines.append(f"\n[dim]UNCHANGED: {len(diff.unchanged_keys)} key(s)[/]")

        lines.append("=" * 60)

        return "\n".join(lines)

    # =========================================================================
    # State Querying
    # =========================================================================

    def query_state(
        self,
        snapshot_id: Optional[str],
        query: str,
    ) -> Any:
        """Query state using JSONPath-like syntax.

        Args:
            snapshot_id: Snapshot ID (uses latest if None)
            query: Query string (e.g., "user.name", "items[0].id")

        Returns:
            Query result or None
        """
        snapshot = (
            self.get_snapshot(snapshot_id) if snapshot_id
            else self.get_latest_snapshot()
        )

        if not snapshot:
            raise ValueError("No snapshot to query")

        state = snapshot.state
        return self._query_path(state, query)

    def _query_path(self, data: Any, path: str) -> Any:
        """Query data using path.

        Args:
            data: Data to query
            path: Path string

        Returns:
            Query result
        """
        if not path:
            return data

        keys = path.split(".")
        result = data

        for key in keys:
            if result is None:
                return None

            # Handle array indexing
            if "[" in key and key.endswith("]"):
                base_key = key.split("[")[0]
                index_str = key.split("[")[1].rstrip("]")

                try:
                    index = int(index_str)
                except ValueError:
                    return None

                if isinstance(result, dict):
                    result = result.get(base_key)
                    if isinstance(result, list) and index < len(result):
                        result = result[index]
                    else:
                        return None
                else:
                    return None
            else:
                if isinstance(result, dict):
                    result = result.get(key)
                else:
                    return None

        return result

    # =========================================================================
    # State Diffing
    # =========================================================================

    def compute_diff(
        self,
        from_snapshot_id: str,
        to_snapshot_id: str,
    ) -> StateDiff:
        """Compute difference between two snapshots.

        Args:
            from_snapshot_id: Source snapshot ID
            to_snapshot_id: Target snapshot ID

        Returns:
            State diff
        """
        from_snapshot = self.get_snapshot(from_snapshot_id)
        to_snapshot = self.get_snapshot(to_snapshot_id)

        if not from_snapshot or not to_snapshot:
            raise ValueError("One or both snapshots not found")

        from_state = from_snapshot.state
        to_state = to_snapshot.state

        all_keys = set(from_state.keys()) | set(to_state.keys())

        added_keys = {}
        removed_keys = {}
        changed_keys = {}
        unchanged_keys = []

        for key in all_keys:
            if key not in from_state:
                added_keys[key] = to_state[key]
            elif key not in to_state:
                removed_keys[key] = from_state[key]
            elif from_state[key] != to_state[key]:
                changed_keys[key] = {
                    "from": from_state[key],
                    "to": to_state[key],
                }
            else:
                unchanged_keys.append(key)

        return StateDiff(
            snapshot_id=str(uuid4()),
            from_snapshot_id=from_snapshot_id,
            to_snapshot_id=to_snapshot_id,
            added_keys=added_keys,
            removed_keys=removed_keys,
            changed_keys=changed_keys,
            unchanged_keys=unchanged_keys,
        )

    # =========================================================================
    # Rollback
    # =========================================================================

    def rollback_to(
        self,
        snapshot_id: str,
    ) -> Dict[str, Any]:
        """Rollback state to a snapshot.

        Args:
            snapshot_id: Snapshot ID to rollback to

        Returns:
            Rolled back state
        """
        snapshot = self.get_snapshot(snapshot_id)

        if not snapshot:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")

        # Create new snapshot with rolled back state
        new_state = copy.deepcopy(snapshot.state)
        new_snapshot_id = self.capture_state(
            new_state,
            node_id=None,
            metadata={"rollback_from": snapshot_id},
        )

        logger.info(f"Rolled back to snapshot '{snapshot_id}', created '{new_snapshot_id}'")

        return new_state

    def rollback_n(self, n: int) -> Optional[Dict[str, Any]]:
        """Rollback N snapshots.

        Args:
            n: Number of snapshots to rollback

        Returns:
            Rolled back state or None
        """
        if len(self._snapshot_order) <= n:
            raise ValueError(f"Cannot rollback {n} snapshots, only {len(self._snapshot_order)} available")

        target_index = len(self._snapshot_order) - n - 1
        if target_index < 0:
            raise ValueError("Cannot rollback before first snapshot")

        target_id = self._snapshot_order[target_index]
        return self.rollback_to(target_id)

    # =========================================================================
    # Mutation Tracking
    # =========================================================================

    def _track_mutations(
        self,
        parent_id: str,
        child_id: str,
        node_id: Optional[str],
    ) -> None:
        """Track mutations between snapshots.

        Args:
            parent_id: Parent snapshot ID
            child_id: Child snapshot ID
            node_id: Node ID that caused mutations
        """
        parent = self.get_snapshot(parent_id)
        child = self.get_snapshot(child_id)

        if not parent or not child:
            return

        diff = self.compute_diff(parent_id, child_id)

        # Track additions
        for key, value in diff.added_keys.items():
            mutation = StateMutation(
                mutation_id=str(uuid4()),
                timestamp=child.timestamp,
                node_id=node_id or child.node_id,
                key=key,
                old_value=None,
                new_value=value,
                mutation_type="add",
            )
            self._mutations.append(mutation)

        # Track removals
        for key, value in diff.removed_keys.items():
            mutation = StateMutation(
                mutation_id=str(uuid4()),
                timestamp=child.timestamp,
                node_id=node_id or child.node_id,
                key=key,
                old_value=value,
                new_value=None,
                mutation_type="remove",
            )
            self._mutations.append(mutation)

        # Track changes
        for key, change in diff.changed_keys.items():
            mutation = StateMutation(
                mutation_id=str(uuid4()),
                timestamp=child.timestamp,
                node_id=node_id or child.node_id,
                key=key,
                old_value=change["from"],
                new_value=change["to"],
                mutation_type="change",
            )
            self._mutations.append(mutation)

    def get_mutations(
        self,
        snapshot_id: Optional[str] = None,
        key: Optional[str] = None,
    ) -> List[StateMutation]:
        """Get mutations.

        Args:
            snapshot_id: Optional snapshot ID filter
            key: Optional key filter

        Returns:
            List of mutations
        """
        mutations = self._mutations

        if snapshot_id:
            snapshot = self.get_snapshot(snapshot_id)
            if snapshot:
                mutations = [
                    m for m in mutations
                    if m.timestamp <= snapshot.timestamp
                ]

        if key:
            mutations = [m for m in mutations if m.key == key]

        return mutations

    def get_mutation_history(
        self,
        key: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get mutation history for a key.

        Args:
            key: Key to track
            limit: Maximum number of mutations

        Returns:
            List of mutation records
        """
        mutations = [m for m in self._mutations if m.key == key]
        mutations = mutations[-limit:]

        return [
            {
                "timestamp": m.timestamp,
                "node_id": m.node_id,
                "type": m.mutation_type,
                "old_value": self._format_value(m.old_value),
                "new_value": self._format_value(m.new_value),
            }
            for m in mutations
        ]

    # =========================================================================
    # Export/Import
    # =========================================================================

    def export_state(
        self,
        snapshot_id: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """Export state as JSON.

        Args:
            snapshot_id: Snapshot ID (uses latest if None)
            output_path: Optional output file path

        Returns:
            JSON string
        """
        snapshot = (
            self.get_snapshot(snapshot_id) if snapshot_id
            else self.get_latest_snapshot()
        )

        if not snapshot:
            raise ValueError("No snapshot to export")

        data = {
            "snapshot_id": snapshot.snapshot_id,
            "timestamp": snapshot.timestamp,
            "node_id": snapshot.node_id,
            "metadata": snapshot.metadata,
            "state": snapshot.state,
        }

        json_str = json.dumps(data, indent=2, default=str)

        if output_path:
            output_path.write_text(json_str)
            logger.info(f"State exported to {output_path}")

        return json_str

    def import_state(
        self,
        data: Union[str, Dict[str, Any]],
    ) -> str:
        """Import state and create snapshot.

        Args:
            data: JSON string or dictionary

        Returns:
            New snapshot ID
        """
        if isinstance(data, str):
            data = json.loads(data)

        snapshot_id = self.capture_state(
            data["state"],
            node_id=data.get("node_id"),
            metadata=data.get("metadata", {}),
        )

        logger.info(f"State imported as snapshot '{snapshot_id}'")

        return snapshot_id


# =============================================================================
# Convenience Functions
# =============================================================================


def create_state_manager(
    max_snapshots: int = 100,
    auto_capture: bool = False,
) -> WorkflowStateManager:
    """Create a workflow state manager.

    Args:
        max_snapshots: Maximum snapshots to keep
        auto_capture: Auto-capture on node execution

    Returns:
        WorkflowStateManager instance
    """
    return WorkflowStateManager(
        max_snapshots=max_snapshots,
        auto_capture=auto_capture,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "StateSnapshot",
    "StateDiff",
    "StateMutation",
    # Main class
    "WorkflowStateManager",
    # Functions
    "create_state_manager",
]
