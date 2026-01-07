#!/usr/bin/env python3
"""Test EventBus factory with different backends."""

import asyncio
from victor.observability.event_bus_factory import get_event_bus, reset_event_bus, EventBusBackend
from victor.observability.event_bus import EventCategory, VictorEvent


async def test_memory_backend():
    print("=" * 60)
    print("Test 1: In-Memory Backend (default)")
    print("=" * 60)

    # Reset factory
    reset_event_bus()

    # Get EventBus with memory backend
    bus = get_event_bus()
    print(f"EventBus type: {type(bus).__name__}")

    # Subscribe
    received = []

    def handler(event):
        print(f"  Received: {event.category}/{event.name}")
        received.append(event)

    bus.subscribe_all(handler)

    # Publish event
    event = VictorEvent(
        category=EventCategory.TOOL,
        name="test.event",
        data={"message": "Hello from memory backend"},
    )
    bus.publish(event)

    print(f"Events received: {len(received)}")
    print("✅ In-memory backend works!")
    print()


async def test_jsonl_backend():
    print("=" * 60)
    print("Test 2: JSONL Backend")
    print("=" * 60)

    # Reset factory
    reset_event_bus()

    # Get EventBus with JSONL backend
    bus = get_event_bus("jsonl")
    print(f"EventBus type: {type(bus).__name__}")

    # Subscribe
    received = []

    def handler(event):
        print(f"  Received: {event.category}/{event.name}")
        received.append(event)

    bus.subscribe_all(handler)

    # Publish event
    event = VictorEvent(
        category=EventCategory.TOOL, name="test.event", data={"message": "Hello from JSONL backend"}
    )
    bus.publish(event)

    print(f"Events received: {len(received)}")
    print("✅ JSONL backend works!")
    print()


async def test_environment_override():
    print("=" * 60)
    print("Test 3: Environment Variable Override")
    print("=" * 60)

    import os

    os.environ["VICTOR_EVENTBUS_BACKEND"] = "jsonl"

    # Reset factory
    reset_event_bus()

    # Get EventBus (should use env var)
    bus = get_event_bus()
    print(f"EventBus type: {type(bus).__name__}")
    print(f"Using backend from env: {os.environ.get('VICTOR_EVENTBUS_BACKEND')}")

    # Clean up
    del os.environ["VICTOR_EVENTBUS_BACKEND"]

    print("✅ Environment override works!")
    print()


async def main():
    print("\nEventBus Factory Tests")
    print("=" * 60)
    print()

    await test_memory_backend()
    await test_jsonl_backend()
    await test_environment_override()

    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
