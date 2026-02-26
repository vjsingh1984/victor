"""
Victor Agent Streaming Example

This example shows how to stream agent responses in real-time.
"""

import asyncio
from victor import Agent


async def main():
    """Stream agent responses as they arrive."""
    # Create an agent
    agent = Agent.create()

    print("Agent is thinking...")

    # Stream the response
    async for event in agent.stream("Tell me a short story about a robot"):
        if event.type == "content":
            # Print content as it arrives
            print(event.content, end="", flush=True)
        elif event.type == "thinking":
            print("\n[Thinking...]", flush=True)
        elif event.type == "error":
            print(f"\nError: {event.content}")

    print()  # New line after completion


if __name__ == "__main__":
    asyncio.run(main())
