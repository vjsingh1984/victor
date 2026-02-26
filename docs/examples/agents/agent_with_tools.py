"""
Victor Agent with Tools Example

This example shows how to create an agent with specific tools.
"""

import asyncio
from victor import Agent


async def main():
    """Create an agent with filesystem tools."""
    # Create an agent with filesystem tools
    agent = Agent.create(
        tools=["read", "write", "ls", "grep"]
    )

    # Ask the agent to analyze the current directory
    result = await agent.run(
        "List all Python files in the current directory and count total lines of code"
    )

    print(f"Analysis:\n{result.content}")


if __name__ == "__main__":
    asyncio.run(main())
