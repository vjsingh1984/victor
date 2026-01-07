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

"""Demo script for Victor Observability System.

This script demonstrates the complete observability system by emitting
real events through all emitters and displaying them in the dashboard.

Usage:
    # Terminal 1: Start the dashboard
    python -m victor.observability.dashboard.app

    # Terminal 2: Run this demo
    python scripts/demo_observability.py

The dashboard will show events in real-time as they are emitted.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from victor.observability.bridge import ObservabilityBridge
from victor.observability.event_bus import EventBus
from victor.observability.ollama_helper import get_default_ollama_model, is_ollama_available


def print_banner():
    """Print demo banner."""
    print("\n" + "=" * 70)
    print("  Victor Observability System - Live Demo")
    print("=" * 70)
    print("\nThis demo emits events through the observability system.")
    print("Make sure the dashboard is running in another terminal:")
    print("  python -m victor.observability.dashboard.app")
    print("\n" + "=" * 70 + "\n")


async def demo_session_lifecycle(bridge: ObservabilityBridge, model: str = "gpt-oss"):
    """Demo session lifecycle events."""
    print("📍 [1/5] Demo: Session Lifecycle Events")

    session_id = "demo-session-001"
    print(f"  → Starting session: {session_id}")

    bridge.session_start(
        session_id,
        agent_id="agent-orchestrator",
        model=model,
        provider="ollama",
    )

    await asyncio.sleep(0.5)
    print(f"  ✓ Session started\n")

    return session_id


async def demo_tool_execution(bridge: ObservabilityBridge, session_id: str):
    """Demo tool execution events."""
    print("📍 [2/5] Demo: Tool Execution Events")

    # Simulate tool execution
    tools = [
        ("read_file", {"path": "src/main.py"}),
        ("search", {"query": "def process_*"}),
        ("analyze", {"target": "code"}),
    ]

    for tool_name, arguments in tools:
        print(f"  → Executing: {tool_name}")
        bridge.tool_start(tool_name, arguments, session_id=session_id)

        # Simulate execution time
        await asyncio.sleep(0.2)

        # Complete tool
        duration = 200.0
        result = f"Success: {tool_name} completed"
        bridge.tool_end(tool_name, duration, result=result, session_id=session_id)
        print(f"  ✓ {tool_name} completed in {duration}ms")

    print()


async def demo_model_calls(bridge: ObservabilityBridge, session_id: str, model: str = "gpt-oss"):
    """Demo model interaction events."""
    print("📍 [3/5] Demo: Model Interaction Events")

    # Simulate model request/response (using ollama - free, local model)
    print("  → Making model request...")
    bridge.model_request(
        provider="ollama",
        model=model,
        prompt_tokens=1000,
        session_id=session_id,
    )

    await asyncio.sleep(0.3)

    print("  → Receiving model response...")
    bridge.model_response(
        provider="ollama",
        model=model,
        prompt_tokens=1000,
        completion_tokens=500,
        latency_ms=1500.0,
        session_id=session_id,
    )

    print("  ✓ Model call completed (1500ms, 1500 tokens)\n")


async def demo_state_transitions(bridge: ObservabilityBridge, session_id: str):
    """Demo state transition events."""
    print("📍 [4/5] Demo: State Transition Events")

    transitions = [
        ("thinking", "tool_execution", 0.85),
        ("tool_execution", "result_synthesis", 0.92),
        ("result_synthesis", "response_generation", 0.95),
    ]

    for old_stage, new_stage, confidence in transitions:
        print(f"  → Transition: {old_stage} → {new_stage} (confidence: {confidence})")
        bridge.state_transition(old_stage, new_stage, confidence, session_id=session_id)
        await asyncio.sleep(0.2)

    print("  ✓ All transitions completed\n")


async def demo_error_tracking(bridge: ObservabilityBridge, session_id: str):
    """Demo error tracking events."""
    print("📍 [5/5] Demo: Error Tracking Events")

    # Simulate recoverable error
    print("  → Simulating recoverable error...")
    try:
        raise ValueError("Simulated error for demo")
    except Exception as e:
        bridge.error(
            e,
            recoverable=True,
            context={"component": "tool_executor", "tool": "read_file"},
            session_id=session_id,
        )
        print("  ✓ Error tracked (recoverable)")

    await asyncio.sleep(0.3)

    # Simulate successful recovery
    print("  → Error recovered, continuing execution...")
    bridge.tool_start("read_file", {"path": "src/recovered.py"}, session_id=session_id)
    await asyncio.sleep(0.2)
    bridge.tool_end("read_file", 150.0, result="Recovered successfully", session_id=session_id)
    print("  ✓ Recovery successful\n")


async def demo_aggregation_metrics(bridge: ObservabilityBridge, session_id: str):
    """Demo that generates metrics for dashboard aggregation."""
    print("📍 [BONUS] Demo: Generating Metrics Data")

    print("  → Generating multiple tool calls for metrics...")

    # Generate various tool executions with different timings
    tools_and_times = [
        ("read_file", 100),
        ("write_file", 150),
        ("search", 200),
        ("analyze", 300),
        ("read_file", 120),
        ("search", 180),
    ]

    for tool_name, duration in tools_and_times:
        bridge.tool_start(tool_name, {"arg": "value"}, session_id=session_id)
        await asyncio.sleep(0.05)  # Small delay
        bridge.tool_end(tool_name, float(duration), session_id=session_id)

    print(f"  ✓ Generated {len(tools_and_times)} tool executions")

    # Calculate expected metrics
    read_times = [100, 120]
    write_times = [150]
    search_times = [200, 180]
    analyze_times = [300]

    avg_read = sum(read_times) / len(read_times)
    avg_search = sum(search_times) / len(search_times)

    print(f"\n  📊 Expected Dashboard Metrics:")
    print(f"     - read_file: {len(read_times)} calls, {avg_read:.0f}ms avg")
    print(f"     - search: {len(search_times)} calls, {avg_search:.0f}ms avg")
    print(f"     - write_file: {len(write_times)} calls")
    print(f"     - analyze: {len(analyze_times)} calls")
    print()


async def run_demo():
    """Run the complete observability demo."""
    print_banner()

    # Detect ollama model
    print("Detecting ollama models...")
    if is_ollama_available():
        model = get_default_ollama_model()
        print(f"✓ Using ollama model: {model}\n")
    else:
        model = "gpt-oss"  # Fallback
        print(f"⚠️  Ollama not available, using fallback: {model}")
        print("   Install ollama: brew install ollama && ollama pull gpt-oss\n")

    # Initialize observability system
    print("Initializing observability system...")
    event_bus = EventBus.get_instance()
    bridge = ObservabilityBridge(event_bus=event_bus)
    print("✓ Observability system initialized\n")

    # Wait a moment for user to see the message
    await asyncio.sleep(1.0)

    # Pass model to demos
    await _run_demo_with_model(bridge, model)


async def _run_demo_with_model(bridge: ObservabilityBridge, model: str):
    """Run all demos with the detected model."""
    try:
        # Run all demos
        session_id = await demo_session_lifecycle(bridge, model=model)
        await demo_tool_execution(bridge, session_id)
        await demo_model_calls(bridge, session_id, model=model)
        await demo_state_transitions(bridge, session_id)
        await demo_error_tracking(bridge, session_id)
        await demo_aggregation_metrics(bridge, session_id)

        # End session
        print("=" * 70)
        print("  Demo Complete!")
        print("=" * 70)
        print("\n📊 Check the dashboard to see:")
        print("   • Events tab - Real-time event log")
        print("   • Table tab - Categorized events")
        print("   • Tools tab - Aggregated tool stats")
        print("   • Tool Calls tab - Detailed call history")
        print("   • State tab - State transitions")
        print("   • Metrics tab - Performance metrics")
        print("\n💡 Tip: Press 'q' in the dashboard to exit\n")

        # End the session
        bridge.session_end(session_id)
        print(f"✓ Session {session_id} ended")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped. Goodbye!")
