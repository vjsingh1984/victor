"""
Victor Temperature Example

Shows how temperature affects agent responses.
"""

import asyncio
from victor import Agent


async def low_temperature():
    """Focused, deterministic responses."""
    agent = Agent.create(temperature=0.0)
    result = await agent.run("Write a haiku about Python")
    print(f"Temperature 0.0 (Focused):\n{result.content}\n")


async def medium_temperature():
    """Balanced creativity and focus."""
    agent = Agent.create(temperature=0.5)
    result = await agent.run("Write a haiku about Python")
    print(f"Temperature 0.5 (Balanced):\n{result.content}\n")


async def high_temperature():
    """Creative, varied responses."""
    agent = Agent.create(temperature=1.0)
    result = await agent.run("Write a haiku about Python")
    print(f"Temperature 1.0 (Creative):\n{result.content}\n")


async def main():
    """Compare different temperatures."""
    print("=== Temperature Comparison ===\n")
    print("Writing a haiku about Python with different temperatures:\n")

    await low_temperature()
    await medium_temperature()
    await high_temperature()


if __name__ == "__main__":
    asyncio.run(main())
