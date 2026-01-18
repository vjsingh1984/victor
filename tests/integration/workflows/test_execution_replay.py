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

"""Integration tests for workflow execution replay system."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from victor.workflows.execution_recorder import (
    ExecutionRecorder,
    ExecutionReplayer,
    RecordingEventType,
    RecordingMetadata,
    StateSnapshot,
)
from victor.workflows.recording_storage import (
    FileRecordingStorage,
    InMemoryRecordingStorage,
    RecordingQuery,
    RetentionPolicy,
)
from victor.workflows.recording_integration import (
    record_workflow,
    enable_workflow_recording,
    save_workflow_recording,
)


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "recordings"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def sample_recording_data():
    """Sample recording data for testing."""
    return {
        "workflow_name": "test_workflow",
        "initial_context": {"input": "test_data"},
        "nodes": [
            {
                "id": "node1",
                "type": "agent",
                "inputs": {"data": "value"},
                "outputs": {"result": "success"},
                "duration": 1.5,
            },
            {
                "id": "node2",
                "type": "compute",
                "inputs": {"previous": "success"},
                "outputs": {"final": "done"},
                "duration": 0.5,
            },
        ],
        "success": True,
    }


class TestExecutionRecorder:
    """Tests for ExecutionRecorder."""

    def test_recorder_initialization(self):
        """Test recorder initialization."""
        recorder = ExecutionRecorder(
            workflow_name="test_workflow",
            record_inputs=True,
            record_outputs=True,
        )

        assert recorder.workflow_name == "test_workflow"
        assert len(recorder.events) == 0
        assert recorder.config["record_inputs"] is True
        assert recorder.config["record_outputs"] is True

    def test_record_workflow_lifecycle(self):
        """Test recording workflow lifecycle."""
        recorder = ExecutionRecorder(workflow_name="test_workflow")

        # Record start
        recorder.record_workflow_start({"input": "data"})

        # Record nodes
        recorder.record_node_start("node1", {"data": "value"})
        recorder.record_node_complete("node1", {"result": "success"}, 1.5)

        recorder.record_node_start("node2", {"previous": "success"})
        recorder.record_node_complete("node2", {"final": "done"}, 0.5)

        # Record completion
        recorder.record_workflow_complete({"output": "final_result"}, success=True)

        # Verify events
        assert len(recorder.events) == 6  # start, 2x(node_start, node_complete), complete

        # Verify metadata
        metadata = recorder.finalize()
        assert metadata.workflow_name == "test_workflow"
        assert metadata.success is True
        assert metadata.node_count == 2
        assert metadata.event_count == 6

    def test_record_recursion(self):
        """Test recording recursion depth changes."""
        recorder = ExecutionRecorder(workflow_name="test_workflow")

        recorder.record_recursion_enter("workflow", "outer")
        recorder.record_recursion_enter("team", "inner")
        recorder.record_recursion_exit("team", "inner")
        recorder.record_recursion_exit("workflow", "outer")

        assert recorder._recursion_depth == 0
        assert recorder._recursion_max_depth == 2

        # Verify events
        recursion_events = [e for e in recorder.events if "recursion" in e.event_type.value]
        assert len(recursion_events) == 4

    def test_record_team_execution(self):
        """Test recording team execution."""
        recorder = ExecutionRecorder(workflow_name="test_workflow")

        recorder.record_team_start(
            team_id="review_team",
            formation="parallel",
            member_count=3,
            context={"task": "review code"},
        )

        recorder.record_team_member_communication(
            team_id="review_team",
            from_member="member1",
            to_member="member2",
            message="Here's my analysis",
        )

        recorder.record_team_complete(
            team_id="review_team",
            final_output="Consensus reached",
            duration_seconds=5.0,
            success=True,
        )

        # Verify team events
        team_events = [e for e in recorder.events if "team" in e.event_type.value]
        assert len(team_events) == 3

    def test_record_state_snapshots(self):
        """Test recording state snapshots."""
        recorder = ExecutionRecorder(
            workflow_name="test_workflow",
            record_state_snapshots=True,
        )

        recorder.record_workflow_start({"input": "data"})

        recorder.record_state_snapshot(
            state={"key": "value1"},
            node_id="node1",
            execution_stack=["workflow:test", "node:node1"],
        )

        recorder.record_state_snapshot(
            state={"key": "value2"},
            node_id="node2",
            execution_stack=["workflow:test", "node:node2"],
        )

        assert len(recorder.snapshots) == 2
        assert recorder.snapshots[0].state == {"key": "value1"}
        assert recorder.snapshots[1].state == {"key": "value2"}

    @pytest.mark.asyncio
    async def test_save_and_load_recording(self, tmp_path):
        """Test saving and loading recording."""
        recorder = ExecutionRecorder(workflow_name="test_workflow", compress=False)

        recorder.record_workflow_start({"input": "data"})
        recorder.record_node_start("node1", {"data": "value"})
        recorder.record_node_complete("node1", {"result": "success"}, 1.0)
        recorder.record_workflow_complete({"output": "final"}, success=True)

        # Save to file
        filepath = tmp_path / "recording.json"
        metadata = await recorder.save(filepath)

        assert metadata.file_size_bytes is not None
        assert metadata.checksum is not None
        assert filepath.exists()

        # Load recording
        replayer = ExecutionReplayer.load(filepath)

        assert replayer.metadata.workflow_name == "test_workflow"
        assert len(replayer.events) == 4
        assert replayer.metadata.success is True

    @pytest.mark.asyncio
    async def test_compressed_recording(self, tmp_path):
        """Test saving compressed recording."""
        recorder = ExecutionRecorder(workflow_name="test_workflow", compress=True)

        recorder.record_workflow_start({"input": "data"})
        recorder.record_node_start("node1", {"data": "value"})
        recorder.record_node_complete("node1", {"result": "success"}, 1.0)
        recorder.record_workflow_complete({"output": "final"}, success=True)

        # Save compressed - the save() method will add .gz
        base_filepath = tmp_path / "recording.json"
        actual_filepath = tmp_path / "recording.json.gz"  # This is what will be created
        await recorder.save(base_filepath)

        # Should create .gz file
        assert actual_filepath.exists()

        # Load compressed
        replayer = ExecutionReplayer.load(actual_filepath)
        assert len(replayer.events) == 4


class TestExecutionReplayer:
    """Tests for ExecutionReplayer."""

    @pytest.fixture
    async def sample_recording(self, tmp_path):
        """Create a sample recording for testing."""
        recorder = ExecutionRecorder(
            workflow_name="sample_workflow",
            record_inputs=True,
            record_outputs=True,
            record_state_snapshots=True,
            compress=False,
        )

        recorder.record_workflow_start({"input": "test"})
        recorder.record_node_start("node1", {"data": "value"}, node_type="agent")
        recorder.record_node_complete("node1", {"result": "success"}, 1.0)
        recorder.record_state_snapshot({"result": "success"}, node_id="node1", execution_stack=["workflow:sample"])
        recorder.record_node_start("node2", {"previous": "success"}, node_type="compute")
        recorder.record_node_complete("node2", {"final": "done"}, 0.5)
        recorder.record_workflow_complete({"final": "done"}, success=True)

        filepath = tmp_path / "sample.json"
        await recorder.save(filepath)

        return filepath

    def test_load_recording(self, sample_recording):
        """Test loading a recording."""
        replayer = ExecutionReplayer.load(sample_recording)

        assert replayer.metadata.workflow_name == "sample_workflow"
        assert len(replayer.events) == 7  # workflow_start, node_start, node_complete, state_snapshot, node_start, node_complete, workflow_complete
        assert len(replayer.snapshots) == 1

    def test_step_forward(self, sample_recording):
        """Test stepping forward through events."""
        replayer = ExecutionReplayer.load(sample_recording)

        events = list(replayer.step_forward(steps=2))
        assert len(events) == 2
        assert events[0].event_type == RecordingEventType.WORKFLOW_START
        assert events[1].event_type == RecordingEventType.NODE_START

        # Position should be at 2
        assert replayer.current_position == 2

    def test_step_backward(self, sample_recording):
        """Test stepping backward through events."""
        replayer = ExecutionReplayer.load(sample_recording)
        replayer.current_position = 4

        events = list(replayer.step_backward(steps=2))
        assert len(events) == 2

        # Position should be at 2
        assert replayer.current_position == 2

    def test_get_event(self, sample_recording):
        """Test getting a specific event."""
        replayer = ExecutionReplayer.load(sample_recording)

        # Get first event
        first_event_id = replayer.events[0].event_id
        event = replayer.get_event(first_event_id)

        assert event is not None
        assert event.event_type == RecordingEventType.WORKFLOW_START

    def test_get_node_events(self, sample_recording):
        """Test getting events for a specific node."""
        replayer = ExecutionReplayer.load(sample_recording)

        node1_events = replayer.get_node_events("node1")

        # node1 has NODE_START, NODE_COMPLETE, and STATE_SNAPSHOT events
        assert len(node1_events) == 3
        assert node1_events[0].event_type == RecordingEventType.NODE_START
        assert node1_events[1].event_type == RecordingEventType.NODE_COMPLETE
        assert node1_events[2].event_type == RecordingEventType.STATE_SNAPSHOT

    def test_jump_to_event(self, sample_recording):
        """Test jumping to a specific event."""
        replayer = ExecutionReplayer.load(sample_recording)

        # Jump to node2 completion (node2 has NODE_START and NODE_COMPLETE events)
        node2_events = replayer.get_node_events("node2")
        node2_complete = [e for e in node2_events if e.event_type == RecordingEventType.NODE_COMPLETE][0]
        success = replayer.jump_to_event(node2_complete.event_id)

        assert success is True
        # Position should be at the index of node2 completion event
        assert replayer.current_position == replayer.events.index(node2_complete)

    @pytest.mark.asyncio
    async def test_compare_recordings(self, tmp_path, sample_recording):
        """Test comparing two recordings."""
        # Create second recording with different path
        recorder2 = ExecutionRecorder(workflow_name="sample_workflow", compress=False)
        recorder2.record_workflow_start({"input": "test"})
        recorder2.record_node_start("node1", {"data": "value"}, node_type="agent")
        recorder2.record_node_complete("node1", {"result": "success"}, 1.0)
        # Skip node2
        recorder2.record_workflow_complete({"result": "success"}, success=True)

        filepath2 = tmp_path / "sample2.json"
        await recorder2.save(filepath2)

        # Compare
        replayer1 = ExecutionReplayer.load(sample_recording)
        replayer2 = ExecutionReplayer.load(filepath2)

        diff = replayer1.compare(replayer2)

        assert diff["node_diff"]["only_in_self"] == {"node2"}
        assert diff["node_diff"]["only_in_other"] == set()
        assert len(diff["node_diff"]["common"]) == 1  # node1

    def test_visualize(self, sample_recording, tmp_path):
        """Test generating visualization."""
        replayer = ExecutionReplayer.load(sample_recording)

        output_path = tmp_path / "workflow.dot"
        dot_graph = replayer.visualize(output_path)

        assert "digraph workflow_execution" in dot_graph
        assert output_path.exists()


class TestFileRecordingStorage:
    """Tests for FileRecordingStorage."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, temp_storage_dir):
        """Test saving and loading recordings."""
        storage = FileRecordingStorage(base_path=temp_storage_dir)

        recorder = ExecutionRecorder(workflow_name="test_workflow")
        recorder.record_workflow_start({"input": "data"})
        recorder.record_node_start("node1", {"data": "value"})
        recorder.record_node_complete("node1", {"result": "success"}, 1.0)
        recorder.record_workflow_complete({"output": "final"}, success=True)

        # Save
        recording_id = await storage.save(recorder)
        assert recording_id is not None

        # Load
        replayer = await storage.load(recording_id)
        assert replayer.metadata.workflow_name == "test_workflow"

    @pytest.mark.asyncio
    async def test_list_recordings(self, temp_storage_dir):
        """Test listing recordings."""
        storage = FileRecordingStorage(base_path=temp_storage_dir)

        # Save multiple recordings
        for i in range(3):
            recorder = ExecutionRecorder(workflow_name=f"workflow_{i}")
            recorder.record_workflow_start({"index": i})
            recorder.record_workflow_complete({}, success=True)
            await storage.save(recorder)

        # List all
        recordings = await storage.list()
        assert len(recordings) == 3

        # Filter by workflow name
        query = RecordingQuery(workflow_name="workflow_1")
        filtered = await storage.list(query)
        assert len(filtered) == 1
        assert filtered[0]["workflow_name"] == "workflow_1"

    @pytest.mark.asyncio
    async def test_delete_recording(self, temp_storage_dir):
        """Test deleting a recording."""
        storage = FileRecordingStorage(base_path=temp_storage_dir)

        recorder = ExecutionRecorder(workflow_name="test_workflow")
        recorder.record_workflow_start({"input": "data"})
        recorder.record_workflow_complete({}, success=True)

        recording_id = await storage.save(recorder)

        # Delete
        success = await storage.delete(recording_id)
        assert success is True

        # Verify deleted
        metadata = await storage.get_metadata(recording_id)
        assert metadata is None

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, temp_storage_dir):
        """Test getting storage statistics."""
        storage = FileRecordingStorage(base_path=temp_storage_dir)

        # Save some recordings
        for i in range(3):
            recorder = ExecutionRecorder(workflow_name="test_workflow")
            recorder.record_workflow_start({"index": i})
            recorder.record_workflow_complete({}, success=(i % 2 == 0))
            await storage.save(recorder)

        stats = await storage.get_storage_stats()

        assert stats["total_recordings"] == 3
        assert stats["success_count"] == 2
        assert stats["failed_count"] == 1
        assert "test_workflow" in stats["workflow_counts"]

    @pytest.mark.asyncio
    async def test_retention_policy(self, temp_storage_dir):
        """Test applying retention policy."""
        from datetime import datetime, timedelta

        storage = FileRecordingStorage(base_path=temp_storage_dir)

        # Save some recordings with different timestamps
        for i in range(5):
            recorder = ExecutionRecorder(workflow_name="test_workflow")
            recorder.record_workflow_start({"index": i})
            recorder.record_workflow_complete({}, success=True)

            # Manually adjust timestamp
            recorder.metadata.started_at = (datetime.now() - timedelta(days=i)).timestamp()

            await storage.save(recorder)

        # Apply retention policy (keep only last 2 days)
        policy = RetentionPolicy(max_age_days=2)
        result = await storage.apply_retention_policy(policy, dry_run=False)

        assert result["to_delete"] > 0
        assert result["total_recordings"] == 5


class TestInMemoryRecordingStorage:
    """Tests for InMemoryRecordingStorage."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading in memory."""
        storage = InMemoryRecordingStorage()

        recorder = ExecutionRecorder(workflow_name="test_workflow")
        recorder.record_workflow_start({"input": "data"})
        recorder.record_node_start("node1", {"data": "value"})
        recorder.record_node_complete("node1", {"result": "success"}, 1.0)
        recorder.record_workflow_complete({"output": "final"}, success=True)

        # Save
        recording_id = await storage.save(recorder)
        assert recording_id is not None

        # Load
        replayer = await storage.load(recording_id)
        assert replayer.metadata.workflow_name == "test_workflow"

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all recordings."""
        storage = InMemoryRecordingStorage()

        recorder = ExecutionRecorder(workflow_name="test_workflow")
        recorder.record_workflow_start({"input": "data"})
        recorder.record_workflow_complete({}, success=True)
        await storage.save(recorder)

        assert len(storage._recordings) == 1

        storage.clear()
        assert len(storage._recordings) == 0


class TestRecordingIntegration:
    """Tests for recording integration utilities."""

    def test_record_workflow_context_manager(self):
        """Test using record_workflow context manager."""
        with record_workflow("test_workflow") as recorder:
            recorder.record_workflow_start({"input": "data"})
            recorder.record_node_start("node1", {"data": "value"})
            recorder.record_node_complete("node1", {"result": "success"}, 1.0)
            recorder.record_workflow_complete({}, success=True)

        assert recorder.metadata.success is True
        assert recorder.metadata.node_count == 1

    def test_enable_and_disable_recording(self):
        """Test enabling and disabling recording."""
        from victor.workflows.recording_integration import (
            enable_workflow_recording,
            disable_workflow_recording,
            get_current_recorder,
        )

        # Enable
        recorder = enable_workflow_recording("test_workflow")
        assert get_current_recorder() is not None
        assert get_current_recorder() == recorder

        # Disable
        disable_workflow_recording()
        assert get_current_recorder() is None


@pytest.mark.integration
class TestEndToEndRecording:
    """End-to-end integration tests for recording system."""

    @pytest.mark.asyncio
    async def test_record_simple_workflow_execution(self, temp_storage_dir):
        """Test recording a simple workflow execution."""
        from victor.workflows.definition import WorkflowBuilder

        # Create simple workflow
        builder = WorkflowBuilder("simple_test")
        builder.add_agent("node1", "assistant", "Process data")
        builder.add_agent("node2", "assistant", "Finalize")
        workflow = builder.build()

        # Record execution
        with record_workflow("simple_test") as recorder:
            recorder.record_workflow_start({"data": "input"})

            # Simulate node execution
            recorder.record_node_start("node1", {"data": "input"}, node_type="agent")
            recorder.record_node_complete("node1", {"output": "processed"}, 1.0)

            recorder.record_node_start("node2", {"previous": "processed"}, node_type="agent")
            recorder.record_node_complete("node2", {"final": "done"}, 0.5)

            recorder.record_workflow_complete({"result": "done"}, success=True)

        # Save to storage
        storage = FileRecordingStorage(base_path=temp_storage_dir)
        recording_id = await storage.save(recorder)

        # Verify
        replayer = await storage.load(recording_id)
        assert replayer.metadata.workflow_name == "simple_test"
        assert replayer.metadata.success is True
        assert replayer.metadata.node_count == 2

    @pytest.mark.asyncio
    async def test_replay_with_state_inspection(self, temp_storage_dir):
        """Test replaying with state inspection."""
        # Create recording with state snapshots
        recorder = ExecutionRecorder(
            workflow_name="state_test",
            record_state_snapshots=True,
            compress=False,  # Disable compression for simpler file handling
        )

        recorder.record_workflow_start({"input": "data"})

        recorder.record_node_start("node1", {"data": "value"}, node_type="agent")
        recorder.record_state_snapshot(
            state={"processed": True},
            node_id="node1",
            execution_stack=["workflow:state_test"],
        )
        recorder.record_node_complete("node1", {"result": "success"}, 1.0)

        recorder.record_workflow_complete({"final": "done"}, success=True)

        # Save
        filepath = temp_storage_dir / "state_test.json"
        await recorder.save(filepath)

        # Replay and inspect
        replayer = ExecutionReplayer.load(filepath)

        # Get state at node1 completion
        node1_events = replayer.get_node_events("node1")
        complete_event = [e for e in node1_events if e.event_type == RecordingEventType.NODE_COMPLETE][0]

        state = replayer.get_state_at_event(complete_event.event_id)
        assert state is not None
