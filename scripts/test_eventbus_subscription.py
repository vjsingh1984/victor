#!/usr/bin/env python3
"""Test EventBus subscription in isolation."""

import asyncio
from victor.observability.event_bus import EventBus, EventCategory, VictorEvent
from victor.observability.dashboard.app import ObservabilityDashboard


async def main():
    # Create EventBus
    bus = EventBus.get_instance()
    print(f"EventBus instance: {bus}")

    # Create a test event
    event = VictorEvent(category=EventCategory.LIFECYCLE, name="test.event", data={"test": "data"})
    print(f"\nCreated test event: {event.category}/{event.name}")

    # Subscribe a simple handler
    received = []

    def handler(evt):
        print(f"  Handler called: {evt.category}/{evt.name}")
        received.append(evt)

    print("\nSubscribing handler...")
    unsubscribe = bus.subscribe_all(handler)
    print("Subscribed!")

    # Publish event
    print("\nPublishing event...")
    bus.publish(event)
    print("Published!")

    # Check if handler was called
    print(f"\nHandler received {len(received)} events")
    if received:
        print("✅ EventBus subscription works!")
    else:
        print("❌ EventBus subscription FAILED!")

    # Clean up
    unsubscribe()


if __name__ == "__main__":
    asyncio.run(main())
