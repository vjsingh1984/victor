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

"""Integration of execution recording into workflow execution.

This module provides utilities and wrappers to integrate ExecutionRecorder
into the workflow execution pipeline with minimal overhead.

Example:
    from victor.workflows.recording_integration import (
        enable_workflow_recording,
        save_workflow_recording,
    )

    # Enable recording for a workflow execution
    recorder = enable_workflow_recording(
        workflow_name="my_workflow",
        record_on_failure=True,
    )

    # Execute workflow (will be recorded)
    result = await executor.execute(workflow, context)

    # Save recording
    await save_workflow_recording(recorder, "/path/to/recording.json")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union, cast

from victor.workflows.execution_recorder import ExecutionRecorder, RecordingEventType
from victor.workflows.recording_storage import FileRecordingStorage

logger = logging.getLogger(__name__)

# Global recording context (using contextvars would be better for async)
_recorder_context: Optional[ExecutionRecorder] = None


def get_current_recorder() -> Optional[ExecutionRecorder]:
    """Get the current recorder for this execution context.

    Returns:
        ExecutionRecorder if recording is active, None otherwise
    """
    return _recorder_context


def enable_workflow_recording(
    workflow_name: str,
    record_inputs: bool = True,
    record_outputs: bool = True,
    record_state_snapshots: bool = False,
    compress: bool = True,
    tags: Optional[list[str]] = None,
) -> ExecutionRecorder:
    """Enable workflow recording for the current execution.

    Creates and registers an ExecutionRecorder that will capture events
    during workflow execution.

    Args:
        workflow_name: Name of the workflow being recorded
        record_inputs: Whether to record node inputs
        record_outputs: Whether to record node outputs
        record_state_snapshots: Whether to record state snapshots
        compress: Whether to compress recordings
        tags: Optional tags for categorization

    Returns:
        ExecutionRecorder instance

    Example:
        recorder = enable_workflow_recording("my_workflow")
        # ... execute workflow ...
        await recorder.save("/path/to/recording.json")
    """
    global _recorder_context

    _recorder_context = ExecutionRecorder(
        workflow_name=workflow_name,
        record_inputs=record_inputs,
        record_outputs=record_outputs,
        record_state_snapshots=record_state_snapshots,
        compress=compress,
        tags=tags,
    )

    logger.debug(f"Enabled workflow recording: {workflow_name}")

    return _recorder_context


def disable_workflow_recording() -> None:
    """Disable workflow recording for the current execution.

    Example:
        disable_workflow_recording()
    """
    global _recorder_context
    _recorder_context = None
    logger.debug("Disabled workflow recording")


@contextmanager
def record_workflow(
    workflow_name: str,
    record_inputs: bool = True,
    record_outputs: bool = True,
    record_state_snapshots: bool = False,
    compress: bool = True,
    tags: Optional[list[str]] = None,
):
    """Context manager for automatic workflow recording.

    Args:
        workflow_name: Name of the workflow
        record_inputs: Whether to record node inputs
        record_outputs: Whether to record node outputs
        record_state_snapshots: Whether to record state snapshots
        compress: Whether to compress recordings
        tags: Optional tags for categorization

    Yields:
        ExecutionRecorder instance

    Example:
        with record_workflow("my_workflow") as recorder:
            result = await executor.execute(workflow, context)
        # Recording automatically finalized
    """
    recorder = enable_workflow_recording(
        workflow_name=workflow_name,
        record_inputs=record_inputs,
        record_outputs=record_outputs,
        record_state_snapshots=record_state_snapshots,
        compress=compress,
        tags=tags,
    )

    try:
        yield recorder
    finally:
        # Finalize metadata before disabling
        recorder.finalize()
        disable_workflow_recording()


async def save_workflow_recording(
    recorder: ExecutionRecorder,
    filepath: Union[str, object],
    storage: Optional[FileRecordingStorage] = None,
) -> Dict[str, Any]:
    """Save a workflow recording to file or storage.

    Args:
        recorder: ExecutionRecorder instance
        filepath: Path to save (or None if using storage)
        storage: Optional FileRecordingStorage instance

    Returns:
        Recording metadata dictionary

    Example:
        # Save to specific file
        metadata = await save_workflow_recording(recorder, "/path/to/recording.json")

        # Save to storage backend
        storage = FileRecordingStorage(base_path="./recordings")
        recording_id = await storage.save(recorder)
    """
    if storage:
        recording_id = await storage.save(recorder)
        metadata = await storage.get_metadata(recording_id)
        return cast(Dict[str, Any], metadata)

    if filepath:
        metadata = await recorder.save(filepath)
        return metadata.to_dict()

    raise ValueError("Must provide either filepath or storage")


def record_node_execution_start(
    node_id: str,
    inputs: Optional[Dict[str, Any]] = None,
    node_type: Optional[str] = None,
) -> None:
    """Record node execution start (called by executor).

    Args:
        node_id: Node identifier
        inputs: Node input data
        node_type: Type of node
    """
    recorder = get_current_recorder()
    if recorder:
        recorder.record_node_start(node_id, inputs, node_type)


def record_node_execution_complete(
    node_id: str,
    outputs: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Record node execution completion (called by executor).

    Args:
        node_id: Node identifier
        outputs: Node output data
        duration_seconds: Execution duration
        error: Error message if failed
    """
    recorder = get_current_recorder()
    if recorder:
        recorder.record_node_complete(node_id, outputs, duration_seconds, error)


def record_recursion_enter(operation_type: str, identifier: str) -> None:
    """Record entering a nested execution level.

    Args:
        operation_type: Type of operation (workflow, team)
        identifier: Operation identifier
    """
    recorder = get_current_recorder()
    if recorder:
        recorder.record_recursion_enter(operation_type, identifier)


def record_recursion_exit(operation_type: str, identifier: str) -> None:
    """Record exiting a nested execution level.

    Args:
        operation_type: Type of operation (workflow, team)
        identifier: Operation identifier
    """
    recorder = get_current_recorder()
    if recorder:
        recorder.record_recursion_exit(operation_type, identifier)


def record_state_snapshot(
    state: Dict[str, Any],
    node_id: Optional[str] = None,
    execution_stack: Optional[list[str]] = None,
) -> None:
    """Record a state snapshot.

    Args:
        state: Current workflow state
        node_id: Node where snapshot is taken
        execution_stack: Current execution stack
    """
    recorder = get_current_recorder()
    if recorder:
        recorder.record_state_snapshot(state, node_id, execution_stack)


class RecordingConfig:
    """Configuration for automatic workflow recording.

    Attributes:
        enabled: Whether recording is enabled
        record_on_failure: Automatically record on workflow failure
        record_on_success: Automatically record on workflow success
        sampling_rate: Sampling rate (0.0 to 1.0) for probabilistic recording
        storage_path: Path to storage directory
        compress: Whether to compress recordings
        retention_policy: Retention policy for old recordings
    """

    def __init__(
        self,
        enabled: bool = False,
        record_on_failure: bool = True,
        record_on_success: bool = False,
        sampling_rate: float = 0.0,
        storage_path: str = "./recordings",
        compress: bool = True,
        retention_policy: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = enabled
        self.record_on_failure = record_on_failure
        self.record_on_success = record_on_success
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))
        self.storage_path = storage_path
        self.compress = compress
        self.retention_policy = retention_policy or {}


__all__ = [
    "get_current_recorder",
    "enable_workflow_recording",
    "disable_workflow_recording",
    "record_workflow",
    "save_workflow_recording",
    "record_node_execution_start",
    "record_node_execution_complete",
    "record_recursion_enter",
    "record_recursion_exit",
    "record_state_snapshot",
    "RecordingConfig",
]
