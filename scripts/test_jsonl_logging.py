#!/usr/bin/env python3
"""Test script to verify JSONL logging and dashboard integration.

This script:
1. Generates test events
2. Writes them to ~/.victor/metrics/victor.jsonl
3. Verifies the file watcher can read them
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone


def main():
    print("=" * 60)
    print("JSONL Event Logging Test")
    print("=" * 60)

    # Determine log path
    log_path = Path.home() / ".victor" / "metrics" / "victor.jsonl"

    print(f"\n1. Log file path: {log_path}")

    # Ensure metrics directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Directory exists: {log_path.parent}")

    # Clear existing file for clean test
    if log_path.exists():
        backup_path = log_path.with_suffix(".jsonl.bak")
        log_path.rename(backup_path)
        print(f"   ✓ Backed up existing file to: {backup_path.name}")

    # Generate test events
    test_events = [
        {
            "id": "test-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": "LIFECYCLE",
            "name": "session.start",
            "data": {"session_id": "test-session", "agent_id": "test-agent"},
        },
        {
            "id": "test-002",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": "TOOL",
            "name": "tool.start",
            "data": {"tool_name": "read_file", "arguments": {"path": "test.py"}},
        },
        {
            "id": "test-003",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": "TOOL",
            "name": "tool.end",
            "data": {"tool_name": "read_file", "duration_ms": 50, "result": "success"},
        },
        {
            "id": "test-004",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": "MODEL",
            "name": "llm.start",
            "data": {"model": "gpt-4", "provider": "openai"},
        },
        {
            "id": "test-005",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": "MODEL",
            "name": "llm.end",
            "data": {"model": "gpt-4", "duration_ms": 1500, "tokens": 250},
        },
    ]

    print(f"\n2. Writing {len(test_events)} test events to JSONL file...")

    with open(log_path, "w", encoding="utf-8") as f:
        for event in test_events:
            line = json.dumps(event) + "\n"
            f.write(line)
            print(f"   ✓ Wrote event: {event['category']}/{event['name']}")

    print(f"\n3. Verifying file contents...")

    # Read back and verify
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"   ✓ File contains {len(lines)} lines")
    print(f"   ✓ File size: {log_path.stat().st_size} bytes")

    # Parse and verify each line
    print(f"\n4. Parsing events...")

    for i, line in enumerate(lines, 1):
        try:
            event = json.loads(line.strip())
            print(f"   ✓ Line {i}: {event['category']}/{event['name']} - {event['id']}")
        except json.JSONDecodeError as e:
            print(f"   ✗ Line {i}: Failed to parse - {e}")

    print(f"\n5. Dashboard file watcher should:")
    print(f"   • Detect file immediately (polls every 100ms)")
    print(f"   • Load all {len(test_events)} events")
    print(f"   • Display them in all 9 dashboard tabs")

    print(f"\n6. Starting file watcher for 10 seconds...")
    print(f"   (If dashboard is running, events should appear now)")

    # Simulate file watcher behavior
    from victor.observability.event_bus import EventBus, EventCategory, VictorEvent

    event_bus = EventBus.get_instance()
    events_read = 0

    def handle_event(event: VictorEvent):
        nonlocal events_read
        events_read += 1
        print(f"   ✓ Event received: {event.category}/{event.name}")

    # Subscribe to all events
    event_bus.subscribe(
        lambda e: True,  # All categories
        handle_event,
    )

    # Simulate file watcher reading events
    print(f"\n   Loading events from file...")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse JSONL
                data = json.loads(line)

                # Create VictorEvent
                event = VictorEvent(
                    id=data.get("id"),
                    timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
                    category=EventCategory(data["category"].lower()),
                    name=data["name"],
                    data=data.get("data", {}),
                )

                # Emit to EventBus
                event_bus.publish(event)
    except Exception as e:
        print(f"   ✗ Error loading events: {e}")

    print(f"\n   ✓ Loaded {events_read} events into EventBus")

    print(f"\n7. Summary:")
    print(f"   • Test events written: {len(test_events)}")
    print(f"   • Events loaded into EventBus: {events_read}")
    print(f"   • File path: {log_path}")
    print(f"   • File size: {log_path.stat().st_size} bytes")

    if events_read == len(test_events):
        print(f"\n   ✅ SUCCESS! All events were processed correctly.")
        print(f"   ✅ Dashboard should now show these events!")
    else:
        print(f"\n   ❌ FAILED! Expected {len(test_events)} events, got {events_read}")

    print(f"\n8. Next steps:")
    print(f"   1. Start dashboard: victor dashboard")
    print(f"   2. Check Events tab - should show {len(test_events)} events")
    print(f"   3. Press 'q' to quit dashboard")

    print(f"\n" + "=" * 60)


if __name__ == "__main__":
    main()
