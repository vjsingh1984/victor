#!/usr/bin/env python3
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

"""Demo of workflow execution replay system.

This script demonstrates the key features of the execution replay system:
1. Recording a workflow execution
2. Saving and loading recordings
3. Replaying with step-through debugging
4. Comparing multiple executions
5. Exporting visualizations

Usage:
    python examples/workflows/execution_replay_demo.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.workflows.execution_recorder import (
    ExecutionRecorder,
    ExecutionReplayer,
    RecordingEventType,
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
    get_current_recorder,
)


def simulate_workflow_execution(
    workflow_name: str,
    scenario: str = "success",
    recorder: ExecutionRecorder = None,
) -> ExecutionRecorder:
    """Simulate a workflow execution for demo purposes.

    Args:
        workflow_name: Name of the workflow
        scenario: Execution scenario (success, failure, slow)
        recorder: Optional recorder to use

    Returns:
        ExecutionRecorder with recorded events
    """
    if recorder is None:
        recorder = ExecutionRecorder(
            workflow_name=workflow_name,
            record_inputs=True,
            record_outputs=True,
            record_state_snapshots=True,
            compress=True,
        )

    print(f"\n{'='*60}")
    print(f"Simulating workflow: {workflow_name}")
    print(f"Scenario: {scenario}")
    print(f"{'='*60}")

    # Record workflow start
    initial_context = {"input": "demo_data", "scenario": scenario}
    recorder.record_workflow_start(initial_context)
    print(f"[0.00s] Workflow started")

    # Node 1: Data ingestion
    print(f"\n[Node 1] Data ingestion")
    recorder.record_node_start("ingest_data", {"source": "api"}, node_type="compute")

    if scenario == "slow":
        import time
        time.sleep(0.5)  # Simulate slow execution

    recorder.record_node_complete(
        "ingest_data",
        {"records": 1000, "status": "loaded"},
        duration_seconds=0.5 if scenario == "slow" else 0.1,
    )
    print(f"  ✓ Loaded 1000 records")

    # State snapshot
    recorder.record_state_snapshot(
        state={"records": 1000, "processed": 0},
        node_id="ingest_data",
        execution_stack=["workflow:demo", "node:ingest_data"],
    )

    # Node 2: Data processing
    print(f"\n[Node 2] Data processing")
    recorder.record_node_start("process_data", {"records": 1000}, node_type="agent")

    if scenario == "failure":
        # Simulate failure
        recorder.record_node_complete(
            "process_data",
            outputs=None,
            duration_seconds=0.3,
            error="Data validation failed: invalid schema",
        )
        recorder.record_workflow_complete(
            final_state=None,
            success=False,
            error="Data validation failed",
        )
        print(f"  ✗ Validation failed!")
        return recorder

    recorder.record_node_complete(
        "process_data",
        {"processed": 950, "errors": 50, "output": "processed_data"},
        duration_seconds=1.2,
    )
    print(f"  ✓ Processed 950 records (50 errors)")

    # Node 3: Team review (parallel)
    print(f"\n[Node 3] Team review")
    recorder.record_team_start(
        team_id="review_team",
        formation="parallel",
        member_count=3,
        context={"task": "review processed data"},
    )

    # Simulate team member communications
    recorder.record_team_member_communication(
        team_id="review_team",
        from_member="quality_reviewer",
        to_member="security_reviewer",
        message="Found 5 potential quality issues",
    )

    recorder.record_team_member_communication(
        team_id="review_team",
        from_member="security_reviewer",
        to_member="performance_reviewer",
        message="No security issues detected",
    )

    recorder.record_team_complete(
        team_id="review_team",
        final_output="All reviews passed",
        duration_seconds=2.0,
        success=True,
    )
    print(f"  ✓ Team review complete (3 members, 2 communications)")

    # Node 4: Final report
    print(f"\n[Node 4] Generate report")
    recorder.record_node_start("generate_report", {"data": "processed_data"}, node_type="compute")
    recorder.record_node_complete(
        "generate_report",
        {"report_path": "/reports/demo_report.pdf", "size_kb": 256},
        duration_seconds=0.3,
    )
    print(f"  ✓ Report generated: /reports/demo_report.pdf")

    # Record workflow completion
    recorder.record_workflow_complete(
        final_state={"report": "demo_report.pdf", "status": "complete"},
        success=True,
    )
    print(f"\n[2.10s] Workflow completed successfully")

    return recorder


async def demo_1_basic_recording():
    """Demo 1: Basic workflow recording."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Workflow Recording")
    print("="*60)

    # Record a workflow execution
    recorder = simulate_workflow_execution("demo_workflow", scenario="success")

    # Save to file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = Path(f.name)

    metadata = await recorder.save(filepath)

    print(f"\n📊 Recording metadata:")
    print(f"  Recording ID: {metadata.recording_id}")
    print(f"  Workflow: {metadata.workflow_name}")
    print(f"  Duration: {metadata.duration_seconds:.2f}s")
    print(f"  Events: {metadata.event_count}")
    print(f"  Nodes: {metadata.node_count}")
    print(f"  Teams: {metadata.team_count}")
    print(f"  File size: {metadata.file_size_bytes} bytes")
    print(f"  File: {filepath}")

    return filepath, metadata.recording_id


async def demo_2_replay_stepping(filepath: Path):
    """Demo 2: Replay with step-through debugging."""
    print("\n" + "="*60)
    print("DEMO 2: Replay with Step-Through Debugging")
    print("="*60)

    # Load recording
    replayer = ExecutionReplayer.load(filepath)

    print(f"\n📼 Loaded recording: {replayer.metadata.workflow_name}")
    print(f"   Events: {len(replayer.events)}")
    print(f"   Duration: {replayer.metadata.duration_seconds:.2f}s")

    # Step through first 5 events
    print(f"\n🔍 Stepping through first 5 events:")
    for i, event in enumerate(replayer.step_forward(steps=5)):
        node_info = f" @ {event.node_id}" if event.node_id else ""
        print(f"  [{i+1}] {event.event_type.value}{node_info} (t={event.timestamp:.2f})")

        # Show event data
        if event.data and event.event_type == RecordingEventType.NODE_COMPLETE:
            duration = event.data.get("duration_seconds", 0)
            outputs = event.data.get("outputs", {})
            error = event.data.get("error")
            if error:
                print(f"      ❌ Error: {error}")
            else:
                print(f"      ⏱️  Duration: {duration}s")
                if outputs:
                    print(f"      📤 Output: {list(outputs.keys())}")

    # Get state at a specific event
    if len(replayer.events) > 3:
        event_id = replayer.events[3].event_id
        state = replayer.get_state_at_event(event_id)
        if state:
            print(f"\n📋 State at event 4:")
            print(f"   Keys: {list(state.keys())}")

    return replayer


async def demo_3_storage_backend():
    """Demo 3: Using storage backend."""
    print("\n" + "="*60)
    print("DEMO 3: Storage Backend")
    print("="*60)

    # Use temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileRecordingStorage(base_path=Path(tmpdir))

        # Record multiple workflows
        scenarios = ["success", "failure", "slow"]
        recording_ids = []

        for scenario in scenarios:
            recorder = simulate_workflow_execution(
                f"demo_{scenario}",
                scenario=scenario,
            )
            recording_id = await storage.save(recorder)
            recording_ids.append(recording_id)
            print(f"💾 Saved recording: {recording_id[:8]}... ({scenario})")

        # List all recordings
        print(f"\n📚 All recordings in storage:")
        recordings = await storage.list()
        for r in recordings:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['recording_id'][:8]}... {r['workflow_name']} ({r['duration_seconds']:.1f}s)")

        # Filter by success status
        print(f"\n✅ Successful recordings only:")
        query = RecordingQuery(success=True)
        successful = await storage.list(query)
        for r in successful:
            print(f"  {r['recording_id'][:8]}... {r['workflow_name']}")

        # Filter by workflow name
        print(f"\n🔍 Filter by workflow name:")
        query = RecordingQuery(workflow_name="demo_success")
        filtered = await storage.list(query)
        for r in filtered:
            print(f"  {r['recording_id'][:8]}... {r['workflow_name']}")

        # Get storage statistics
        stats = await storage.get_storage_stats()
        print(f"\n📊 Storage statistics:")
        print(f"  Total recordings: {stats['total_recordings']}")
        print(f"  Successful: {stats['success_count']}")
        print(f"  Failed: {stats['failed_count']}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")


async def demo_4_comparison():
    """Demo 4: Comparing two executions."""
    print("\n" + "="*60)
    print("DEMO 4: Comparing Two Executions")
    print("="*60)

    # Record two different executions
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileRecordingStorage(base_path=Path(tmpdir))

        # Execution 1: Success
        recorder1 = simulate_workflow_execution("comparison_workflow", scenario="success")
        id1 = await storage.save(recorder1)
        print(f"📼 Recording 1: {id1[:8]}... (success)")

        # Execution 2: Failure
        recorder2 = simulate_workflow_execution("comparison_workflow", scenario="failure")
        id2 = await storage.save(recorder2)
        print(f"📼 Recording 2: {id2[:8]}... (failure)")

        # Load and compare
        replayer1 = await storage.load(id1)
        replayer2 = await storage.load(id2)

        diff = replayer1.compare(replayer2)

        print(f"\n🔍 Comparison results:")
        print(f"  Duration difference: {diff['metadata_diff']['duration_diff']:.2f}s")
        print(f"  Node count difference: {diff['metadata_diff']['node_count_diff']}")
        print(f"  Event count difference: {diff['metadata_diff']['event_count_diff']}")

        print(f"\n📍 Node differences:")
        print(f"  Only in execution 1: {diff['node_diff']['only_in_self'] or 'None'}")
        print(f"  Only in execution 2: {diff['node_diff']['only_in_other'] or 'None'}")
        print(f"  Common nodes: {len(diff['node_diff']['common'])}")

        # Show execution paths
        print(f"\n🛤️  Execution path difference:")
        if diff["path_diff"]["first_difference"]:
            fd = diff["path_diff"]["first_difference"]
            print(f"  Paths diverge at position {fd['position']}")
            print(f"    Execution 1: {fd['self_node']}")
            print(f"    Execution 2: {fd['other_node']}")
        else:
            print(f"  Paths are identical")


async def demo_5_visualization():
    """Demo 5: Export visualization."""
    print("\n" + "="*60)
    print("DEMO 5: Export Visualization")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create recording
        recorder = simulate_workflow_execution("viz_workflow", scenario="success")

        # Save
        filepath = Path(tmpdir) / "workflow_recording.json"
        await recorder.save(filepath)

        # Load and visualize
        replayer = ExecutionReplayer.load(filepath)

        # Generate DOT graph
        dot_path = Path(tmpdir) / "workflow.dot"
        dot_graph = replayer.visualize(dot_path)

        print(f"\n📊 Generated Graphviz DOT visualization:")
        print(f"   Output: {dot_path}")
        print(f"\n   To render:")
        print(f"     dot -Tpng {dot_path} -o workflow.png")
        print(f"     dot -Tsvg {dot_path} -o workflow.svg")

        # Show preview of DOT graph
        lines = dot_graph.split("\n")[:15]
        print(f"\n   Preview:")
        for line in lines:
            print(f"   {line}")


async def demo_6_context_manager():
    """Demo 6: Using context manager for recording."""
    print("\n" + "="*60)
    print("DEMO 6: Context Manager Recording")
    print("="*60)

    # Use context manager for automatic cleanup
    with record_workflow("context_demo_workflow", tags=["demo", "context_manager"]) as recorder:
        print(f"🎬 Recording started...")

        # Simulate workflow
        recorder.record_workflow_start({"input": "demo"})
        recorder.record_node_start("node1", {"data": "value"}, node_type="agent")
        recorder.record_node_complete("node1", {"result": "success"}, 1.0)
        recorder.record_workflow_complete({"output": "done"}, success=True)

        print(f"✅ Workflow executed")

    # Recording is automatically finalized
    print(f"\n📊 Recording finalized:")
    print(f"   Events: {recorder.metadata.event_count}")
    print(f"   Success: {recorder.metadata.success}")
    print(f"   Duration: {recorder.metadata.duration_seconds:.2f}s")


async def demo_7_retention_policy():
    """Demo 7: Retention policy management."""
    print("\n" + "="*60)
    print("DEMO 7: Retention Policy")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileRecordingStorage(base_path=Path(tmpdir))

        # Create several recordings
        for i in range(5):
            recorder = simulate_workflow_workflow(f"retention_test_{i}")
            await storage.save(recorder)

        stats_before = await storage.get_storage_stats()
        print(f"\n📊 Before cleanup:")
        print(f"   Total recordings: {stats_before['total_recordings']}")

        # Apply retention policy (keep only 3 most recent)
        policy = RetentionPolicy(max_count=3)
        result = await storage.apply_retention_policy(policy, dry_run=False)

        print(f"\n🧹 Applied retention policy (max_count=3):")
        print(f"   Deleted: {result['to_delete']} recordings")
        print(f"   Freed: {result['total_size_bytes']} bytes")

        stats_after = await storage.get_storage_stats()
        print(f"\n📊 After cleanup:")
        print(f"   Total recordings: {stats_after['total_recordings']}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Workflow Execution Replay System - Demo")
    print("="*60)

    try:
        # Demo 1: Basic recording
        filepath, _ = await demo_1_basic_recording()

        # Demo 2: Replay with stepping
        await demo_2_replay_stepping(filepath)

        # Demo 3: Storage backend
        await demo_3_storage_backend()

        # Demo 4: Comparison
        await demo_4_comparison()

        # Demo 5: Visualization
        await demo_5_visualization()

        # Demo 6: Context manager
        await demo_6_context_manager()

        # Demo 7: Retention policy
        # await demo_7_retention_policy()

        print("\n" + "="*60)
        print("✅ All demos completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
