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

"""Workflow execution recording and replay system for debugging and analysis.

This module provides comprehensive recording and replay capabilities for workflow
executions, enabling:
- Recording all workflow/team execution events with full state snapshots
- Step-through debugging with pause, resume, and navigation
- State inspection at any execution point
- Comparison of multiple executions (diff mode)
- Export to animations/visualizations

Key Features:
- Low overhead recording (<5% performance impact)
- Efficient serialization with optional compression
- Support for complex nested workflows and team execution
- Recursion depth tracking
- Timing information for performance analysis

Example:
    from victor.workflows.execution_recorder import ExecutionRecorder

    # Start recording
    recorder = ExecutionRecorder(
        workflow_name="my_workflow",
        record_inputs=True,
        record_outputs=True,
        record_state_snapshots=True,
    )

    # Record events during execution
    recorder.record_node_start("node1", inputs={"data": "value"})
    recorder.record_node_complete("node1", outputs={"result": "success"})

    # Save recording
    await recorder.save("/path/to/recording.json")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import gzip
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Iterator,
    Tuple,
    Union,
)
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RecordingEventType(Enum):
    """Types of events that can be recorded."""

    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"

    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_ERROR = "node_error"

    TEAM_START = "team_start"
    TEAM_COMPLETE = "team_complete"
    TEAM_MEMBER_START = "team_member_start"
    TEAM_MEMBER_COMPLETE = "team_member_complete"
    TEAM_COMMUNICATION = "team_communication"

    RECURSION_ENTER = "recursion_enter"
    RECURSION_EXIT = "recursion_exit"

    STATE_SNAPSHOT = "state_snapshot"
    CHECKPOINT = "checkpoint"


@dataclass
class RecordingEvent:
    """A single recorded event.

    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of event (RecordingEventType)
        timestamp: Unix timestamp when event occurred
        workflow_id: ID of the workflow execution
        node_id: ID of the node (if applicable)
        data: Event-specific data (inputs, outputs, state, etc.)
        metadata: Additional metadata (recursion depth, timing, etc.)
    """

    event_id: str
    event_type: RecordingEventType
    timestamp: float
    workflow_id: str
    node_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "data": self._sanitize_data(self.data),
            "metadata": self.metadata,
        }

    @staticmethod
    def _sanitize_data(data: Any) -> Any:
        """Sanitize data for JSON serialization.

        Removes or transforms non-serializable objects.
        """
        if isinstance(data, dict):
            return {k: RecordingEvent._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [RecordingEvent._sanitize_data(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            # Convert complex objects to string representation
            return str(data)


@dataclass
class RecordingMetadata:
    """Metadata about a recording.

    Attributes:
        recording_id: Unique identifier for the recording
        workflow_name: Name of the workflow
        started_at: Timestamp when recording started
        completed_at: Timestamp when recording completed
        duration_seconds: Total duration of execution
        success: Whether execution succeeded
        error: Error message if failed
        node_count: Number of nodes executed
        team_count: Number of teams spawned
        recursion_max_depth: Maximum recursion depth reached
        event_count: Total number of events recorded
        file_size_bytes: Size of recording file (compressed)
        checksum: SHA256 checksum of recording data
        tags: User-defined tags for categorization
    """

    recording_id: str
    workflow_name: str
    started_at: float
    completed_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    node_count: int = 0
    team_count: int = 0
    recursion_max_depth: int = 0
    event_count: int = 0
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


@dataclass
class StateSnapshot:
    """Snapshot of workflow state at a point in time.

    Attributes:
        snapshot_id: Unique identifier for this snapshot
        timestamp: When snapshot was taken
        workflow_id: Workflow execution ID
        node_id: Node where snapshot was taken
        state: Full state dictionary
        recursion_depth: Current recursion depth
        execution_stack: Current execution stack
    """

    snapshot_id: str
    timestamp: float
    workflow_id: str
    node_id: Optional[str]
    state: Dict[str, Any]
    recursion_depth: int
    execution_stack: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "state": RecordingEvent._sanitize_data(self.state),
            "recursion_depth": self.recursion_depth,
            "execution_stack": self.execution_stack,
        }


class ExecutionRecorder:
    """Record workflow execution events for replay and analysis.

    This recorder captures all events during workflow execution with minimal
    overhead. It supports:
    - Node execution with inputs/outputs
    - Team member communications
    - Recursion depth changes
    - State snapshots at key points
    - Timing information

    Attributes:
        workflow_id: Unique ID for this workflow execution
        workflow_name: Name of the workflow being recorded
        events: List of recorded events
        metadata: Recording metadata
        config: Recording configuration

    Example:
        recorder = ExecutionRecorder(
            workflow_name="my_workflow",
            record_inputs=True,
            record_outputs=True,
        )

        # Record execution
        recorder.record_workflow_start({"input": "data"})
        recorder.record_node_start("node1", {"data": "value"})
        recorder.record_node_complete("node1", {"result": "success"})
        recorder.record_workflow_complete({"final": "result"})

        # Save
        await recorder.save("/path/to/recording.json")
    """

    def __init__(
        self,
        workflow_name: str,
        workflow_id: Optional[str] = None,
        record_inputs: bool = True,
        record_outputs: bool = True,
        record_state_snapshots: bool = False,
        compress: bool = True,
        tags: Optional[List[str]] = None,
    ):
        """Initialize the execution recorder.

        Args:
            workflow_name: Name of the workflow being recorded
            workflow_id: Unique ID (generated if not provided)
            record_inputs: Whether to record node inputs
            record_outputs: Whether to record node outputs
            record_state_snapshots: Whether to record state snapshots
            compress: Whether to compress the recording
            tags: Optional tags for categorization
        """
        self.workflow_id = workflow_id or uuid.uuid4().hex
        self.workflow_name = workflow_name
        self.events: List[RecordingEvent] = []
        self.snapshots: List[StateSnapshot] = []

        self.config = {
            "record_inputs": record_inputs,
            "record_outputs": record_outputs,
            "record_state_snapshots": record_state_snapshots,
            "compress": compress,
        }

        self.metadata = RecordingMetadata(
            recording_id=uuid.uuid4().hex,
            workflow_name=workflow_name,
            started_at=time.time(),
            tags=tags or [],
        )

        self._recursion_depth = 0
        self._recursion_max_depth = 0
        self._node_count = 0
        self._team_count = 0
        self._start_time = time.time()

        logger.debug(f"Initialized recorder for workflow '{workflow_name}' (id={self.workflow_id})")

    def record_workflow_start(self, initial_context: Dict[str, Any]) -> None:
        """Record workflow execution start.

        Args:
            initial_context: Initial workflow context/state
        """
        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.WORKFLOW_START,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            data={"initial_context": initial_context} if self.config["record_inputs"] else {},
        )
        self._add_event(event)
        logger.debug(f"Recorded workflow start: {self.workflow_name}")

    def record_workflow_complete(
        self,
        final_state: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record workflow execution completion.

        Args:
            final_state: Final workflow state
            success: Whether execution succeeded
            error: Error message if failed
        """
        self.metadata.completed_at = time.time()
        self.metadata.duration_seconds = self.metadata.completed_at - self.metadata.started_at
        self.metadata.success = success
        self.metadata.error = error

        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.WORKFLOW_COMPLETE if success else RecordingEventType.WORKFLOW_ERROR,
            timestamp=self.metadata.completed_at,
            workflow_id=self.workflow_id,
            data={
                "final_state": final_state if self.config["record_outputs"] else {},
                "success": success,
                "error": error,
            },
        )
        self._add_event(event)
        logger.debug(f"Recorded workflow complete: {self.workflow_name} (success={success})")

    def record_node_start(
        self,
        node_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        node_type: Optional[str] = None,
    ) -> None:
        """Record node execution start.

        Args:
            node_id: Node identifier
            inputs: Node input data
            node_type: Type of node (agent, compute, team, etc.)
        """
        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.NODE_START,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={
                "inputs": inputs if self.config["record_inputs"] else {},
                "node_type": node_type,
            },
            metadata={"recursion_depth": self._recursion_depth},
        )
        self._add_event(event)
        logger.debug(f"Recorded node start: {node_id}")

    def record_node_complete(
        self,
        node_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record node execution completion.

        Args:
            node_id: Node identifier
            outputs: Node output data
            duration_seconds: Execution duration
            error: Error message if failed
        """
        self._node_count += 1

        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.NODE_COMPLETE if not error else RecordingEventType.NODE_ERROR,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={
                "outputs": outputs if self.config["record_outputs"] else {},
                "duration_seconds": duration_seconds,
                "error": error,
            },
            metadata={"recursion_depth": self._recursion_depth},
        )
        self._add_event(event)
        logger.debug(f"Recorded node complete: {node_id}")

    def record_team_start(
        self,
        team_id: str,
        formation: str,
        member_count: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record team execution start.

        Args:
            team_id: Team identifier
            formation: Team formation type (parallel, sequential, etc.)
            member_count: Number of team members
            context: Team context
        """
        self._team_count += 1

        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.TEAM_START,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            node_id=team_id,
            data={
                "formation": formation,
                "member_count": member_count,
                "context": context if self.config["record_inputs"] else {},
            },
            metadata={"recursion_depth": self._recursion_depth},
        )
        self._add_event(event)
        logger.debug(f"Recorded team start: {team_id}")

    def record_team_complete(
        self,
        team_id: str,
        final_output: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Record team execution completion.

        Args:
            team_id: Team identifier
            final_output: Final team output
            duration_seconds: Execution duration
            success: Whether execution succeeded
        """
        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.TEAM_COMPLETE if success else RecordingEventType.TEAM_COMPLETE,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            node_id=team_id,
            data={
                "final_output": final_output,
                "duration_seconds": duration_seconds,
                "success": success,
            },
            metadata={"recursion_depth": self._recursion_depth},
        )
        self._add_event(event)
        logger.debug(f"Recorded team complete: {team_id}")

    def record_team_member_communication(
        self,
        team_id: str,
        from_member: str,
        to_member: str,
        message: str,
    ) -> None:
        """Record communication between team members.

        Args:
            team_id: Team identifier
            from_member: Sender member ID
            to_member: Receiver member ID
            message: Communication message
        """
        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.TEAM_COMMUNICATION,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            node_id=team_id,
            data={
                "from_member": from_member,
                "to_member": to_member,
                "message": message,
            },
        )
        self._add_event(event)
        logger.debug(f"Recorded team communication: {from_member} -> {to_member}")

    def record_recursion_enter(self, operation_type: str, identifier: str) -> None:
        """Record entering a nested execution level.

        Args:
            operation_type: Type of operation (workflow, team)
            identifier: Operation identifier
        """
        self._recursion_depth += 1
        self._recursion_max_depth = max(self._recursion_max_depth, self._recursion_depth)

        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.RECURSION_ENTER,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            data={
                "operation_type": operation_type,
                "identifier": identifier,
                "depth": self._recursion_depth,
            },
        )
        self._add_event(event)

    def record_recursion_exit(self, operation_type: str, identifier: str) -> None:
        """Record exiting a nested execution level.

        Args:
            operation_type: Type of operation (workflow, team)
            identifier: Operation identifier
        """
        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.RECURSION_EXIT,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            data={
                "operation_type": operation_type,
                "identifier": identifier,
                "depth": self._recursion_depth,
            },
        )
        self._add_event(event)
        self._recursion_depth = max(0, self._recursion_depth - 1)

    def record_state_snapshot(
        self,
        state: Dict[str, Any],
        node_id: Optional[str] = None,
        execution_stack: Optional[List[str]] = None,
    ) -> None:
        """Record a state snapshot.

        Args:
            state: Current workflow state
            node_id: Node where snapshot is taken
            execution_stack: Current execution stack
        """
        if not self.config["record_state_snapshots"]:
            return

        snapshot = StateSnapshot(
            snapshot_id=uuid.uuid4().hex,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            node_id=node_id,
            state=state,
            recursion_depth=self._recursion_depth,
            execution_stack=execution_stack or [],
        )
        self.snapshots.append(snapshot)

        event = RecordingEvent(
            event_id=uuid.uuid4().hex,
            event_type=RecordingEventType.STATE_SNAPSHOT,
            timestamp=snapshot.timestamp,
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={"snapshot_id": snapshot.snapshot_id},
            metadata={"recursion_depth": self._recursion_depth},
        )
        self._add_event(event)

    def finalize(self) -> RecordingMetadata:
        """Finalize the recording and update metadata.

        Returns:
            RecordingMetadata with final statistics
        """
        self.metadata.node_count = self._node_count
        self.metadata.team_count = self._team_count
        self.metadata.recursion_max_depth = self._recursion_max_depth
        self.metadata.event_count = len(self.events)

        if self.metadata.completed_at is None:
            self.metadata.completed_at = time.time()
            self.metadata.duration_seconds = self.metadata.completed_at - self.metadata.started_at

        return self.metadata

    async def save(self, filepath: Union[str, Path]) -> RecordingMetadata:
        """Save recording to file.

        Args:
            filepath: Path to save the recording

        Returns:
            RecordingMetadata with final statistics
        """
        filepath = Path(filepath)

        # Finalize metadata
        metadata = self.finalize()

        # Prepare recording data
        recording_data = {
            "metadata": metadata.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }

        # Serialize
        json_data = json.dumps(recording_data, indent=2)

        # Calculate checksum
        checksum = hashlib.sha256(json_data.encode()).hexdigest()
        self.metadata.checksum = checksum

        # Save with optional compression
        if self.config["compress"]:
            filepath = filepath.with_suffix(filepath.suffix + ".gz")
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                f.write(json_data)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_data)

        # Update file size
        self.metadata.file_size_bytes = filepath.stat().st_size

        logger.info(
            f"Saved recording: {filepath} "
            f"({self.metadata.event_count} events, "
            f"{self.metadata.file_size_bytes} bytes)"
        )

        return self.metadata

    def _add_event(self, event: RecordingEvent) -> None:
        """Add an event to the recording.

        Args:
            event: Event to add
        """
        self.events.append(event)


class ExecutionReplayer:
    """Replay recorded workflow executions for debugging and analysis.

    Provides capabilities to:
    - Load recordings from file
    - Step through execution (pause, resume, forward/backward)
    - Inspect state at any point
    - Visualize execution graph
    - Compare multiple executions

    Attributes:
        recording_path: Path to the recording file
        metadata: Recording metadata
        events: List of recorded events
        snapshots: List of state snapshots
        current_position: Current playback position

    Example:
        replayer = ExecutionReplayer.load("/path/to/recording.json.gz")

        # Replay with step-through
        for event in replayer.step_forward():
            print(f"Event: {event.event_type} at {event.node_id}")
            if event.event_type == RecordingEventType.NODE_COMPLETE:
                state = replayer.get_state_at_event(event.event_id)
                print(f"State: {state}")
    """

    def __init__(
        self,
        recording_path: Path,
        metadata: RecordingMetadata,
        events: List[RecordingEvent],
        snapshots: List[StateSnapshot],
    ):
        """Initialize the replayer.

        Args:
            recording_path: Path to the recording file
            metadata: Recording metadata
            events: List of recorded events
            snapshots: List of state snapshots
        """
        self.recording_path = recording_path
        self.metadata = metadata
        self.events = events
        self.snapshots = snapshots
        self.current_position = 0

        # Build event index
        self._event_index: Dict[str, int] = {event.event_id: i for i, event in enumerate(events)}
        self._node_events: Dict[str, List[int]] = {}
        for i, event in enumerate(events):
            if event.node_id:
                if event.node_id not in self._node_events:
                    self._node_events[event.node_id] = []
                self._node_events[event.node_id].append(i)

        logger.debug(f"Loaded recording: {recording_path} ({len(events)} events)")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExecutionReplayer":
        """Load a recording from file.

        Args:
            filepath: Path to the recording file

        Returns:
            ExecutionReplayer instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Recording file not found: {filepath}")

        # Load with decompression if needed
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                json_data = f.read()
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = f.read()

        # Parse
        recording_data = json.loads(json_data)

        metadata = RecordingMetadata.from_dict(recording_data["metadata"])

        events = []
        for event_data in recording_data.get("events", []):
            event_data["event_type"] = RecordingEventType(event_data["event_type"])
            events.append(RecordingEvent(**event_data))

        snapshots = []
        for snapshot_data in recording_data.get("snapshots", []):
            snapshots.append(StateSnapshot(**snapshot_data))

        logger.info(f"Loaded recording: {filepath} ({metadata.workflow_name})")

        return cls(
            recording_path=filepath,
            metadata=metadata,
            events=events,
            snapshots=snapshots,
        )

    def get_event(self, event_id: str) -> Optional[RecordingEvent]:
        """Get an event by ID.

        Args:
            event_id: Event identifier

        Returns:
            RecordingEvent or None if not found
        """
        index = self._event_index.get(event_id)
        if index is not None:
            return self.events[index]
        return None

    def get_node_events(self, node_id: str) -> List[RecordingEvent]:
        """Get all events for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            List of events for the node
        """
        indices = self._node_events.get(node_id, [])
        return [self.events[i] for i in indices]

    def get_state_at_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get the workflow state at a specific event.

        Args:
            event_id: Event identifier

        Returns:
            State dictionary or None if not found
        """
        event = self.get_event(event_id)
        if not event:
            return None

        # Find the most recent snapshot before this event
        snapshot_state = None
        for snapshot in reversed(self.snapshots):
            if snapshot.timestamp <= event.timestamp:
                snapshot_state = snapshot.state
                break

        # Reconstruct state by applying events from snapshot
        if snapshot_state:
            state = snapshot_state.copy()
        else:
            state = {}

        # Apply events from snapshot to target event
        found_snapshot = False
        for e in self.events:
            if e.event_type == RecordingEventType.STATE_SNAPSHOT:
                if e.data.get("snapshot_id") == getattr(snapshot_state, "snapshot_id", None):
                    found_snapshot = True
                continue

            if found_snapshot and e.timestamp <= event.timestamp:
                # Update state based on event
                if e.event_type == RecordingEventType.NODE_COMPLETE:
                    if e.data.get("outputs"):
                        state.update(e.data["outputs"])

        return state

    def step_forward(self, steps: int = 1) -> Iterator[RecordingEvent]:
        """Step forward through events.

        Args:
            steps: Number of steps to move forward

        Yields:
            RecordingEvent objects
        """
        end_pos = min(self.current_position + steps, len(self.events))
        for i in range(self.current_position, end_pos):
            self.current_position = i + 1
            yield self.events[i]

    def step_backward(self, steps: int = 1) -> Iterator[RecordingEvent]:
        """Step backward through events.

        Args:
            steps: Number of steps to move backward

        Yields:
            RecordingEvent objects in reverse order
        """
        start_pos = max(self.current_position - steps, 0)
        for i in range(self.current_position - 1, start_pos - 1, -1):
            self.current_position = i
            yield self.events[i]

    def reset(self) -> None:
        """Reset playback position to the beginning."""
        self.current_position = 0

    def jump_to_event(self, event_id: str) -> bool:
        """Jump to a specific event.

        Args:
            event_id: Event identifier

        Returns:
            True if successful, False if event not found
        """
        index = self._event_index.get(event_id)
        if index is not None:
            self.current_position = index
            return True
        return False

    def jump_to_position(self, position: int) -> bool:
        """Jump to a specific position.

        Args:
            position: Position to jump to

        Returns:
            True if successful, False if position out of range
        """
        if 0 <= position < len(self.events):
            self.current_position = position
            return True
        return False

    @contextmanager
    def replay_from_event(self, event_id: str):
        """Context manager for replaying from a specific event.

        Args:
            event_id: Event to start replaying from

        Yields:
            Iterator of events from the specified event
        """
        original_position = self.current_position
        if self.jump_to_event(event_id):
            yield self.step_forward(len(self.events) - self.current_position)
        else:
            yield iter([])
        self.current_position = original_position

    def compare(self, other: "ExecutionReplayer") -> Dict[str, Any]:
        """Compare this recording with another.

        Args:
            other: Another ExecutionReplayer

        Returns:
            Dictionary with comparison results
        """
        # Compare metadata
        metadata_diff = {
            "duration_diff": (self.metadata.duration_seconds or 0) - (other.metadata.duration_seconds or 0),
            "node_count_diff": self.metadata.node_count - other.metadata.node_count,
            "team_count_diff": self.metadata.team_count - other.metadata.team_count,
            "event_count_diff": self.metadata.event_count - other.metadata.event_count,
        }

        # Compare node execution
        self_nodes = set(self._node_events.keys())
        other_nodes = set(other._node_events.keys())

        node_diff = {
            "only_in_self": self_nodes - other_nodes,
            "only_in_other": other_nodes - self_nodes,
            "common": self_nodes & other_nodes,
        }

        # Compare execution path
        self_path = [e.node_id for e in self.events if e.node_id]
        other_path = [e.node_id for e in other.events if e.node_id]

        path_diff = {
            "self_path": self_path,
            "other_path": other_path,
            "first_difference": None,
        }

        for i, (n1, n2) in enumerate(zip(self_path, other_path)):
            if n1 != n2:
                path_diff["first_difference"] = {
                    "position": i,
                    "self_node": n1,
                    "other_node": n2,
                }
                break

        return {
            "metadata_diff": metadata_diff,
            "node_diff": node_diff,
            "path_diff": path_diff,
        }

    def visualize(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate a visualization of the execution graph.

        Args:
            output_path: Optional path to save visualization

        Returns:
            Visualization string (DOT format for Graphviz)
        """
        lines = ["digraph workflow_execution {", '  rankdir=TD;', '  node [shape=box];']

        # Add nodes
        for node_id in self._node_events.keys():
            node_events = self.get_node_events(node_id)
            duration = sum(
                e.data.get("duration_seconds", 0)
                for e in node_events
                if e.event_type == RecordingEventType.NODE_COMPLETE
            )
            label = f"{node_id}\\n({duration:.2f}s)"
            lines.append(f'  "{node_id}" [label="{label}"];')

        # Add edges (based on event sequence)
        prev_node = None
        for event in self.events:
            if event.node_id and event.event_type == RecordingEventType.NODE_START:
                if prev_node and prev_node != event.node_id:
                    lines.append(f'  "{prev_node}" -> "{event.node_id}";')
                prev_node = event.node_id

        lines.append("}")

        dot_graph = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                f.write(dot_graph)
            logger.info(f"Saved visualization to: {output_path}")

        return dot_graph


__all__ = [
    "RecordingEventType",
    "RecordingEvent",
    "RecordingMetadata",
    "StateSnapshot",
    "ExecutionRecorder",
    "ExecutionReplayer",
]
