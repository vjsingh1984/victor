#!/usr/bin/env python3
"""Simple test to emit events while dashboard is running.

Usage:
    # Terminal 1: Start dashboard
    victor dashboard

    # Terminal 2: Run this test
    python scripts/test_dashboard_live.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from victor.observability.bridge import ObservabilityBridge
from victor.observability.event_bus import EventBus


async def main():
    print("Starting event emission test...")
    print("Make sure the dashboard is running (victor dashboard)")
    print("\nPress Ctrl+C to stop\n")

    # Get the observability bridge (should be initialized by orchestrator)
    bridge = ObservabilityBridge.get_instance()

    # Counter for events
    event_count = 0

    try:
        while True:
            event_count += 1
            print(f"\n[Emitting event {event_count}]")

            # Emit a tool event
            tool_name = f"test_tool_{event_count % 3}"
            bridge.tool_start(
                tool_name,
                {"test_arg": f"value_{event_count}"},
                test_run=True,
            )

            print(f"  ✓ Tool start: {tool_name}")

            await asyncio.sleep(0.5)

            # End the tool
            bridge.tool_end(
                tool_name,
                100.0 + (event_count * 10),
                result=f"Test result {event_count}",
                test_run=True,
            )

            print(f"  ✓ Tool end: {tool_name}")

            # Every 3 events, emit a different type
            if event_count % 3 == 0:
                bridge.state_transition(
                    "state_a",
                    "state_b",
                    0.9,
                    test_run=True,
                )
                print(f"  ✓ State transition")

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print(f"\n\nStopped after {event_count} event sets")
        print("Check the dashboard - events should be visible in all tabs")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
