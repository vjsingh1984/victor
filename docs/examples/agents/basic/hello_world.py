"""
Victor Hello World Example

The simplest possible Victor agent example.
"""

import asyncio
from victor import Agent


async def main():
    """Create an agent and say hello."""
    # Create agent with default settings
    agent = Agent.create()

    # Ask a simple question
    result = await agent.run("Hello! What is Victor?")

    # Print the response
    print(f"Agent says: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
