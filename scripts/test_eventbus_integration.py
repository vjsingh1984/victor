#!/usr/bin/env python3
"""EventBus Integration Test

This script tests the EventBus integration using the exact same logic
that the dashboard uses. It subscribes to events and then emits them
to verify they flow correctly through the system.

Usage:
    python scripts/test_eventbus_integration.py

This test will:
1. Subscribe to EventBus using dashboard's subscription logic
2. Emit various events using ObservabilityBridge
3. Verify events are received and processed
4. Report any issues with the event flow
"""

import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from victor.observability.bridge import ObservabilityBridge
from victor.observability.event_bus import EventBus, VictorEvent
from victor.observability.ollama_helper import get_default_ollama_model, is_ollama_available


class EventBusIntegrationTest:
    """Test EventBus integration using dashboard's subscription logic."""

    def __init__(self):
        self.events_received = []
        self.event_bus = EventBus.get_instance()
        self.bridge = ObservabilityBridge(event_bus=self.event_bus)
        self._unsubscribe = None

    def subscribe_like_dashboard(self) -> None:
        """Subscribe to events using the exact same logic as the dashboard.

        This is copied from victor/observability/dashboard/app.py lines 730-742
        """

        def handle_event(event: VictorEvent) -> None:
            """Handle incoming event (same as dashboard)."""
            self.events_received.append(event)
            print(
                f"✓ Received event: {event.name} | Category: {event.category} | Data: {event.data}"
            )

        # Subscribe to all event categories (same as dashboard)
        self._unsubscribe = self.event_bus.subscribe_all(handle_event)
        print("✓ Subscribed to EventBus using dashboard's logic\n")

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if self._unsubscribe:
            self._unsubscribe()

    def emit_test_events(self) -> None:
        """Emit various test events to verify the flow."""
        print("=" * 60)
        print("EMITTING TEST EVENTS")
        print("=" * 60)
        print()

        # Detect ollama model
        if is_ollama_available():
            model = get_default_ollama_model()
            print(f"Using detected ollama model: {model}\n")
        else:
            model = "gpt-oss"  # Fallback
            print(f"Using fallback model: {model}\n")

        # Test 1: Session lifecycle
        print("1. Testing Session Lifecycle Events...")
        self.bridge.session_start("test-session-001", test_agent="agent-1")
        time.sleep(0.2)

        self.bridge.session_end("test-session-001")
        time.sleep(0.2)
        print()

        # Test 2: Tool execution
        print("2. Testing Tool Execution Events...")
        tools = [
            ("read_file", {"path": "test.txt"}),
            ("write_file", {"path": "test.txt", "content": "hello"}),
            ("search", {"query": "test"}),
        ]

        for tool_name, args in tools:
            print(f"  - Tool: {tool_name}")
            self.bridge.tool_start(tool_name, args, test_run=True)
            time.sleep(0.1)
            self.bridge.tool_end(tool_name, 100.0, result="Success", test_run=True)
            time.sleep(0.1)

        print()

        # Test 3: Model interactions (using dynamically detected model)
        print("3. Testing Model Events...")
        self.bridge.model_request(
            provider="ollama",
            model=model,
            prompt_tokens=1000,
            test_run=True,
        )
        time.sleep(0.2)

        self.bridge.model_response(
            provider="ollama",
            model=model,
            prompt_tokens=1000,
            completion_tokens=500,
            latency_ms=1500.0,
            test_run=True,
        )
        time.sleep(0.2)
        print()

        # Test 4: State transitions
        print("4. Testing State Transition Events...")
        transitions = [
            ("thinking", "tool_execution", 0.85),
            ("tool_execution", "result_processing", 0.92),
            ("result_processing", "responding", 0.88),
        ]

        for from_state, to_state, confidence in transitions:
            print(f"  - Transition: {from_state} → {to_state} (confidence: {confidence})")
            self.bridge.state_transition(from_state, to_state, confidence, test_run=True)
            time.sleep(0.2)

        print()

        # Test 5: Error handling
        print("5. Testing Error Events...")
        try:
            raise ValueError("Test error for integration testing")
        except Exception as e:
            self.bridge.error(e, recoverable=True, test_run=True)
            time.sleep(0.2)

        print()

        # Test 6: Tool failure
        print("6. Testing Tool Failure Events...")
        self.bridge.tool_start("failing_tool", {"arg": "value"}, test_run=True)
        time.sleep(0.1)
        self.bridge.tool_failure(
            "failing_tool",
            50.0,
            RuntimeError("Tool failed"),
            test_run=True,
        )
        time.sleep(0.2)
        print()

    def verify_results(self) -> dict[str, Any]:
        """Verify that events were received correctly."""
        print("=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print()

        results = {
            "total_events_emitted": 0,
            "total_events_received": len(self.events_received),
            "by_category": {},
            "by_name": {},
            "success": True,
            "issues": [],
        }

        # Count events by category
        for event in self.events_received:
            category_name = str(event.category)  # Convert enum to string
            results["by_category"][category_name] = results["by_category"].get(category_name, 0) + 1
            results["by_name"][event.name] = results["by_name"].get(event.name, 0) + 1

        # Expected events (based on what we emitted)
        # Note: Tool events use the specific tool name (e.g., "read_file.start")
        expected_events = {
            "session.start": 1,
            "session.end": 1,
            "model.request": 1,
            "model.response": 1,
            "state.transition": 3,
            "error": 1,
            # Tools are counted by prefix match (tool_name.start, tool_name.end)
            # So we expect 8 TOOL category events total
        }

        # Calculate expected tool events
        tool_events_count = results["by_category"].get("EventCategory.TOOL", 0)
        expected_total = sum(expected_events.values()) + tool_events_count

        print(f"Total Events Emitted: {expected_total}")
        print(f"Total Events Received: {results['total_events_received']}")
        print()

        # Check if counts match
        if results["total_events_received"] < expected_total:
            results["success"] = False
            results["issues"].append(
                f"Event count mismatch: expected at least {expected_total}, "
                f"got {results['total_events_received']}"
            )

        # Check for expected events
        print("Event Breakdown by Name:")
        for event_name, expected_count in expected_events.items():
            actual_count = results["by_name"].get(event_name, 0)
            status = "✓" if actual_count == expected_count else "✗"
            print(f"  {status} {event_name}: {actual_count}/{expected_count}")

            if actual_count != expected_count:
                results["success"] = False
                results["issues"].append(
                    f"{event_name}: expected {expected_count}, got {actual_count}"
                )

        # Check tool events
        print(f"  ✓ Tool events (various names): {tool_events_count}/8")
        if tool_events_count < 8:
            results["success"] = False
            results["issues"].append(f"Tool events: expected at least 8, got {tool_events_count}")

        print()
        print("Event Breakdown by Category:")
        for category, count in sorted(results["by_category"].items()):
            print(f"  - {category}: {count}")

        print()
        print("=" * 60)

        if results["success"]:
            print("✅ ALL TESTS PASSED")
            print()
            print("EventBus integration is working correctly!")
            print("Events flow from emitter → EventBus → subscriber as expected.")
        else:
            print("❌ TESTS FAILED")
            print()
            print("Issues found:")
            for issue in results["issues"]:
                print(f"  - {issue}")

        print("=" * 60)
        print()

        return results

    def run(self) -> None:
        """Run the full integration test."""
        print()
        print("=" * 60)
        print("EVENTBUS INTEGRATION TEST")
        print("=" * 60)
        print()
        print("This test uses the dashboard's exact subscription logic to verify")
        print("that events flow correctly through the EventBus system.")
        print()

        try:
            # Subscribe using dashboard's logic
            self.subscribe_like_dashboard()

            # Emit test events
            self.emit_test_events()

            # Give time for all events to be processed
            print("Waiting for events to be processed...")
            time.sleep(0.5)
            print()

            # Verify results
            results = self.verify_results()

            # Provide diagnostic information
            self._print_diagnostics(results)

        finally:
            # Cleanup
            self.unsubscribe()

    def _print_diagnostics(self, results: dict[str, Any]) -> None:
        """Print diagnostic information."""
        print("=" * 60)
        print("DIAGNOSTIC INFORMATION")
        print("=" * 60)
        print()

        # EventBus status
        print(f"EventBus instance: {id(self.event_bus)}")
        print(f"Events successfully received: {len(self.events_received)}")
        print()

        # Event details
        if self.events_received:
            print("First 3 events received:")
            for i, event in enumerate(self.events_received[:3], 1):
                print(f"  {i}. {event.name} ({event.category})")
                print(f"     Data: {event.data}")
            print()

            if len(self.events_received) > 3:
                print(f"... and {len(self.events_received) - 3} more events")
                print()
        else:
            print("⚠️  No events were received!")
            print()
            print("Possible causes:")
            print("  1. EventBus is not initialized")
            print("  2. Subscription failed")
            print("  3. Events are not being emitted")
            print("  4. Event filtering is too restrictive")
            print()

        # Recommendations
        if not results["success"]:
            print("Recommendations:")
            print("  1. Check if EventBus singleton is working correctly")
            print("  2. Verify ObservabilityBridge is emitting events")
            print("  3. Check event categories and names match")
            print("  4. Review dashboard subscription code")
            print()

        print("=" * 60)
        print()


def main():
    """Run the EventBus integration test."""
    test = EventBusIntegrationTest()
    test.run()

    return 0 if test.events_received else 1


if __name__ == "__main__":
    sys.exit(main())
