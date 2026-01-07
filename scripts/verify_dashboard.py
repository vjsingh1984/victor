#!/usr/bin/env python3
"""Minimal test to verify dashboard event reception.

This script emits a few events and exits, making it easy to verify
that the dashboard receives and displays them correctly.

Usage:
    # Terminal 1: Start dashboard
    victor dashboard

    # Terminal 2: Run this test
    python scripts/verify_dashboard.py

You should see events appear in the dashboard immediately.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from victor.observability.bridge import ObservabilityBridge
from victor.observability.event_bus import EventBus
from victor.observability.ollama_helper import get_default_ollama_model, is_ollama_available


def main():
    print("=" * 60)
    print("  Victor Dashboard - Event Verification Test")
    print("=" * 60)
    print("\nThis will emit 10 test events to verify the dashboard works.")
    print("Make sure the dashboard is running (victor dashboard)\n")

    input("Press Enter when dashboard is ready...")

    # Detect ollama model
    if is_ollama_available():
        model = get_default_ollama_model()
        print(f"\n✓ Using detected ollama model: {model}\n")
    else:
        model = "gpt-oss"  # Fallback
        print(f"\n⚠️  Ollama not available, using fallback: {model}\n")

    # Initialize observability
    bridge = ObservabilityBridge(event_bus=EventBus.get_instance())

    print("\n📡 Emitting events...\n")

    # Emit session lifecycle events
    print("1. Session start")
    bridge.session_start("test-session-123", test_agent="agent-1")
    time.sleep(0.5)

    # Emit tool events
    tools = ["read_file", "write_file", "search"]
    for i, tool in enumerate(tools, 2):
        print(f"{i}. Tool: {tool}")
        bridge.tool_start(tool, {"arg": "value"}, test_run=True)
        time.sleep(0.3)
        bridge.tool_end(tool, 100.0, result="Success", test_run=True)
        time.sleep(0.3)

    # Emit state event
    print("6. State transition")
    bridge.state_transition("thinking", "acting", 0.85, test_run=True)
    time.sleep(0.5)

    # Emit model event (using dynamically detected ollama model)
    print("7. Model request")
    bridge.model_request("ollama", model, 1000, test_run=True)
    time.sleep(0.3)

    print("8. Model response")
    bridge.model_response(
        "ollama",
        model,
        1000,
        500,
        1500.0,
        test_run=True,
    )
    time.sleep(0.5)

    # Emit error event
    print("9. Error (recoverable)")
    try:
        raise ValueError("Test error")
    except Exception as e:
        bridge.error(e, recoverable=True, test_run=True)
    time.sleep(0.3)

    # End session
    print("10. Session end")
    bridge.session_end("test-session-123")

    print("\n" + "=" * 60)
    print("✅ Test complete! Check the dashboard:")
    print("   • Events tab - Should show all 10 events")
    print("   • Table tab - Should show categorized events")
    print("   • Tools tab - Should show 3 tools with stats")
    print("   • Tool Calls tab - Should show 3 tool calls")
    print("   • State tab - Should show 1 transition")
    print("=" * 60)
    print("\nPress 'q' in the dashboard to exit\n")


if __name__ == "__main__":
    main()
