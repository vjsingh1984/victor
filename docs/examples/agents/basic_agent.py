"""
Basic Victor Agent Example

This example shows the simplest way to create and use a Victor agent.
"""

import asyncio
from victor import Agent


async def main():
    """Create a basic agent and ask it a question."""
    # Create an agent with default settings
    agent = Agent.create()

    # Ask a simple question
    result = await agent.run("What is the capital of France?")

    # Print the response
    print(f"Response: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
