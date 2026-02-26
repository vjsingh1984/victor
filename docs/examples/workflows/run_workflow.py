"""
Run a Victor Workflow Example

This example shows how to run a workflow defined in YAML.
"""

import asyncio
from victor import Agent


async def main():
    """Run a simple workflow."""
    # Create an agent
    agent = Agent.create()

    # Run the workflow with input
    result = await agent.run_workflow(
        "simple_workflow.yaml",
        input={"user_input": "I need help with Python programming"}
    )

    print(f"Workflow result:\n{result.content}")


if __name__ == "__main__":
    asyncio.run(main())
