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

"""CLI tool for workflow execution replay and debugging.

This script provides commands to:
- Record workflow executions
- Replay recordings with step-through debugging
- List and inspect recordings
- Compare multiple executions
- Export visualizations

Usage:
    python scripts/workflows/replay.py list
    python scripts/workflows/replay.py replay <recording_id>
    python scripts/workflows/replay.py compare <id1> <id2>
    python scripts/workflows/replay.py export <recording_id> --format dot
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.workflows.execution_recorder import (
    ExecutionRecorder,
    ExecutionReplayer,
    RecordingEventType,
)
from victor.workflows.recording_storage import (
    FileRecordingStorage,
    RecordingQuery,
    RetentionPolicy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_bytes(size_bytes: Optional[int]) -> str:
    """Format bytes in human-readable format."""
    if size_bytes is None:
        return "N/A"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable format."""
    if seconds is None:
        return "N/A"

    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


async def list_recordings(args) -> None:
    """List all recordings matching query.

    Args:
        args: Parsed command-line arguments
    """
    storage = FileRecordingStorage(base_path=args.storage)

    # Build query
    query = RecordingQuery(
        workflow_name=args.workflow,
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
        success=None if args.success is None else args.success == "true",
        tags=args.tags.split(",") if args.tags else None,
        limit=args.limit,
        offset=args.offset,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
    )

    recordings = await storage.list(query)

    if not recordings:
        print("No recordings found.")
        return

    # Print table
    print(f"\nFound {len(recordings)} recording(s):\n")
    print(f"{'ID':<8} {'Workflow':<20} {'Status':<8} {'Duration':<10} {'Size':<10} {'Date'}")
    print("-" * 100)

    for r in recordings:
        status = "SUCCESS" if r.get("success") else "FAILED"
        duration = format_duration(r.get("duration_seconds"))
        size = format_bytes(r.get("file_size_bytes"))
        date = datetime.fromtimestamp(r.get("started_at", 0)).strftime("%Y-%m-%d %H:%M")

        print(
            f"{r['recording_id'][:8]:<8} {r['workflow_name'][:20]:<20} {status:<8} {duration:<10} {size:<10} {date}"
        )

    print()


async def inspect_recording(args) -> None:
    """Inspect a recording's metadata.

    Args:
        args: Parsed command-line arguments
    """
    storage = FileRecordingStorage(base_path=args.storage)

    metadata = await storage.get_metadata(args.recording_id)
    if not metadata:
        print(f"Recording not found: {args.recording_id}")
        return

    print(f"\nRecording: {metadata['recording_id']}")
    print(f"Workflow: {metadata['workflow_name']}")
    print("-" * 50)

    print(f"\nStatus:")
    print(f"  Success: {metadata.get('success', 'N/A')}")
    print(f"  Error: {metadata.get('error') or 'None'}")

    print(f"\nTiming:")
    print(f"  Started: {datetime.fromtimestamp(metadata.get('started_at', 0))}")
    print(
        f"  Completed: {datetime.fromtimestamp(metadata.get('completed_at', 0)) if metadata.get('completed_at') else 'N/A'}"
    )
    print(f"  Duration: {format_duration(metadata.get('duration_seconds'))}")

    print(f"\nExecution:")
    print(f"  Nodes executed: {metadata.get('node_count', 0)}")
    print(f"  Teams spawned: {metadata.get('team_count', 0)}")
    print(f"  Max recursion depth: {metadata.get('recursion_max_depth', 0)}")
    print(f"  Total events: {metadata.get('event_count', 0)}")

    print(f"\nStorage:")
    print(f"  File size: {format_bytes(metadata.get('file_size_bytes'))}")
    print(f"  Checksum: {metadata.get('checksum', 'N/A')}")

    print(f"\nTags:")
    tags = metadata.get("tags", [])
    if tags:
        for tag in tags:
            print(f"  - {tag}")
    else:
        print("  None")

    print()


async def replay_recording(args) -> None:
    """Replay a recording with optional step-through.

    Args:
        args: Parsed command-line arguments
    """
    from victor.workflows.execution_recorder import ExecutionReplayer

    storage = FileRecordingStorage(base_path=args.storage)

    try:
        replayer = await storage.load(args.recording_id)
    except FileNotFoundError:
        print(f"Recording not found: {args.recording_id}")
        return

    print(f"\nReplaying: {replayer.metadata.workflow_name}")
    print(f"Recording ID: {replayer.metadata.recording_id}")
    print(f"Total events: {len(replayer.events)}")
    print()

    if args.step:
        # Interactive step-through
        print("Step-through mode (press Enter to continue, 'q' to quit):\n")

        for i, event in enumerate(replayer.events):
            print(f"[{i+1}/{len(replayer.events)}] {event.event_type.value}", end="")
            if event.node_id:
                print(f" @ {event.node_id}", end="")
            print(f" (t={event.timestamp:.3f})")

            if args.show_data and event.data:
                print("  Data:")
                for key, value in event.data.items():
                    if value is not None:
                        print(f"    {key}: {str(value)[:100]}")

            if args.show_state:
                state = replayer.get_state_at_event(event.event_id)
                if state:
                    print(f"  State keys: {list(state.keys())[:10]}")

            user_input = input("\nPress Enter to continue, 'q' to quit: ")
            if user_input.lower() == "q":
                break
            print()

    else:
        # Print all events
        for i, event in enumerate(replayer.events):
            print(f"[{i+1}] {event.event_type.value}", end="")
            if event.node_id:
                print(f" @ {event.node_id}", end="")
            print(f" (t={event.timestamp:.3f})")

            if args.show_data and event.data:
                for key, value in event.data.items():
                    if value is not None:
                        print(f"    {key}: {str(value)[:100]}")

    print()


async def compare_recordings(args) -> None:
    """Compare two recordings.

    Args:
        args: Parsed command-line arguments
    """
    storage = FileRecordingStorage(base_path=args.storage)

    try:
        replayer1 = await storage.load(args.recording_id1)
        replayer2 = await storage.load(args.recording_id2)
    except FileNotFoundError as e:
        print(f"Recording not found: {e}")
        return

    print(f"\nComparing recordings:")
    print(f"  1: {replayer1.metadata.workflow_name} ({replayer1.metadata.recording_id[:8]})")
    print(f"  2: {replayer2.metadata.workflow_name} ({replayer2.metadata.recording_id[:8]})")
    print()

    # Compare metadata
    print("Metadata differences:")
    print(
        f"  Duration: {format_duration(replayer1.metadata.duration_seconds)} vs {format_duration(replayer2.metadata.duration_seconds)}"
    )
    print(f"  Nodes: {replayer1.metadata.node_count} vs {replayer2.metadata.node_count}")
    print(f"  Teams: {replayer1.metadata.team_count} vs {replayer2.metadata.team_count}")
    print(f"  Events: {replayer1.metadata.event_count} vs {replayer2.metadata.event_count}")

    # Detailed comparison
    diff = replayer1.compare(replayer2)

    print(f"\nNode differences:")
    print(f"  Only in 1: {diff['node_diff']['only_in_self'] or 'None'}")
    print(f"  Only in 2: {diff['node_diff']['only_in_other'] or 'None'}")
    print(f"  Common: {len(diff['node_diff']['common'])} nodes")

    # Execution path differences
    if diff["path_diff"]["first_difference"]:
        fd = diff["path_diff"]["first_difference"]
        print(f"\nFirst path difference:")
        print(f"  Position: {fd['position']}")
        print(f"  Recording 1: {fd['self_node']}")
        print(f"  Recording 2: {fd['other_node']}")
    else:
        print("\nExecution paths are identical")

    print()


async def export_recording(args) -> None:
    """Export a recording to various formats.

    Args:
        args: Parsed command-line arguments
    """
    storage = FileRecordingStorage(base_path=args.storage)

    try:
        replayer = await storage.load(args.recording_id)
    except FileNotFoundError:
        print(f"Recording not found: {args.recording_id}")
        return

    output_path = Path(args.output)

    if args.format == "dot":
        # Export as Graphviz DOT
        dot_graph = replayer.visualize(output_path)
        print(f"Exported DOT visualization to: {output_path}")
        print("\nTo render:")
        print(f"  dot -Tpng {output_path} -o workflow.png")
        print(f"  dot -Tsvg {output_path} -o workflow.svg")

    elif args.format == "json":
        # Export as JSON
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": replayer.metadata.to_dict(),
                    "events": [e.to_dict() for e in replayer.events],
                },
                f,
                indent=2,
            )
        print(f"Exported JSON to: {output_path}")

    elif args.format == "summary":
        # Export human-readable summary
        with open(output_path, "w") as f:
            f.write(f"Recording: {replayer.metadata.recording_id}\n")
            f.write(f"Workflow: {replayer.metadata.workflow_name}\n")
            f.write(f"Started: {datetime.fromtimestamp(replayer.metadata.started_at)}\n")
            f.write(f"Duration: {format_duration(replayer.metadata.duration_seconds)}\n")
            f.write(f"Status: {'SUCCESS' if replayer.metadata.success else 'FAILED'}\n\n")

            f.write("Execution summary:\n")
            f.write(f"  Nodes: {replayer.metadata.node_count}\n")
            f.write(f"  Teams: {replayer.metadata.team_count}\n")
            f.write(f"  Events: {replayer.metadata.event_count}\n\n")

            f.write("Event timeline:\n")
            for event in replayer.events:
                f.write(f"  [{event.timestamp:.3f}] {event.event_type.value}")
                if event.node_id:
                    f.write(f" @ {event.node_id}")
                f.write("\n")

        print(f"Exported summary to: {output_path}")

    print()


async def cleanup_recordings(args) -> None:
    """Cleanup old recordings based on retention policy.

    Args:
        args: Parsed command-line arguments
    """
    storage = FileRecordingStorage(base_path=args.storage)

    # Build retention policy
    policy = RetentionPolicy(
        max_age_days=args.max_age_days,
        max_count=args.max_count,
        keep_failed=not args.delete_failed,
    )

    if args.dry_run:
        print("Dry-run mode (no files will be deleted):\n")

    result = await storage.apply_retention_policy(policy, dry_run=args.dry_run)

    print(f"Total recordings: {result['total_recordings']}")
    print(f"To delete: {result['to_delete']}")
    print(f"Total size: {format_bytes(result['total_size_bytes'])}")

    if args.dry_run and result["deleted_ids"]:
        print("\nRecordings that would be deleted:")
        for recording_id in result["deleted_ids"][:10]:
            print(f"  - {recording_id}")
        if len(result["deleted_ids"]) > 10:
            print(f"  ... and {len(result['deleted_ids']) - 10} more")

    print()


async def storage_stats(args) -> None:
    """Show storage statistics.

    Args:
        args: Parsed command-line arguments
    """
    storage = FileRecordingStorage(base_path=args.storage)

    stats = await storage.get_storage_stats()

    print(f"\nStorage Statistics: {args.storage}")
    print("-" * 50)

    print(f"\nRecordings:")
    print(f"  Total: {stats['total_recordings']}")
    print(f"  Successful: {stats['success_count']}")
    print(f"  Failed: {stats['failed_count']}")

    print(f"\nStorage:")
    print(f"  Total size: {format_bytes(stats['total_size_bytes'])}")
    print(f"  Total duration: {format_duration(stats['total_duration_seconds'])}")

    print(f"\nTime range:")
    if stats["oldest_recording"]:
        print(f"  Oldest: {datetime.fromtimestamp(stats['oldest_recording'])}")
    if stats["newest_recording"]:
        print(f"  Newest: {datetime.fromtimestamp(stats['newest_recording'])}")

    print(f"\nWorkflows:")
    for workflow, count in sorted(stats["workflow_counts"].items(), key=lambda x: -x[1]):
        print(f"  {workflow}: {count}")

    print()


async def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Victor workflow execution replay and debugging tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--storage",
        default="./recordings",
        help="Path to recording storage directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    list_parser = subparsers.add_parser("list", help="List recordings")
    list_parser.add_argument("--workflow", help="Filter by workflow name")
    list_parser.add_argument("--start", help="Start date (ISO format)")
    list_parser.add_argument("--end", help="End date (ISO format)")
    list_parser.add_argument(
        "--success", choices=["true", "false"], help="Filter by success status"
    )
    list_parser.add_argument("--tags", help="Filter by tags (comma-separated)")
    list_parser.add_argument("--limit", type=int, help="Limit number of results")
    list_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    list_parser.add_argument("--sort-by", default="started_at", help="Field to sort by")
    list_parser.add_argument(
        "--sort-order", choices=["asc", "desc"], default="desc", help="Sort order"
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect recording metadata")
    inspect_parser.add_argument("recording_id", help="Recording ID")

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a recording")
    replay_parser.add_argument("recording_id", help="Recording ID")
    replay_parser.add_argument("--step", action="store_true", help="Interactive step-through mode")
    replay_parser.add_argument("--show-data", action="store_true", help="Show event data")
    replay_parser.add_argument("--show-state", action="store_true", help="Show state at each event")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two recordings")
    compare_parser.add_argument("recording_id1", help="First recording ID")
    compare_parser.add_argument("recording_id2", help="Second recording ID")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export a recording")
    export_parser.add_argument("recording_id", help="Recording ID")
    export_parser.add_argument(
        "--format", choices=["dot", "json", "summary"], default="summary", help="Export format"
    )
    export_parser.add_argument("--output", required=True, help="Output file path")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old recordings")
    cleanup_parser.add_argument("--max-age-days", type=int, help="Maximum age in days")
    cleanup_parser.add_argument(
        "--max-count", type=int, help="Maximum number of recordings to keep"
    )
    cleanup_parser.add_argument(
        "--delete-failed", action="store_true", help="Delete failed recordings"
    )
    cleanup_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show storage statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    command_handlers = {
        "list": list_recordings,
        "inspect": inspect_recording,
        "replay": replay_recording,
        "compare": compare_recordings,
        "export": export_recording,
        "cleanup": cleanup_recordings,
        "stats": storage_stats,
    }

    handler = command_handlers.get(args.command)
    if handler:
        await handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
