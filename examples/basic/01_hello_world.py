"""Basic Example 01: Hello World

A simple introduction to creating and running a Victor agent.
"""

import asyncio
from victor import Agent


async def main():
    """Create a simple agent and say hello."""

    # Create an agent with default settings
    agent = Agent.create()

    # Run the agent with a simple prompt
    result = await agent.run("Hello! Please introduce yourself.")

    # Print the response
    print(f"Agent: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
