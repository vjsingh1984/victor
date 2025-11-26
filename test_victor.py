#!/usr/bin/env python3
"""Test Victor one-shot command."""

import asyncio
import sys
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings


async def test_victor():
    """Test Victor with a simple code creation task."""
    print("Loading settings...")
    settings = load_settings()

    print("Creating agent...")
    agent = await AgentOrchestrator.from_settings(settings, "default")

    # Change to victor_test directory
    import os
    os.chdir("/Users/vijaysingh/code/codingagent/victor_test")

    message = """Create a Python file called calculator.py in the current directory with a Calculator class.

The class should have these methods:
- add(a, b): returns a + b
- subtract(a, b): returns a - b
- multiply(a, b): returns a * b
- divide(a, b): returns a / b (handle division by zero)

Include proper docstrings and type hints. Use the write_file tool to create the file in the current working directory."""

    print(f"Sending message: {message}")
    print("-" * 70)

    try:
        response = await agent.chat(message)
        print("Response:")
        print(response.content)
        print("-" * 70)

        if response.tool_calls:
            print(f"\nTool calls made: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls, 1):
                tool_name = tool_call.get('name') if isinstance(tool_call, dict) else tool_call.name
                print(f"{i}. {tool_name}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.provider.close()


if __name__ == "__main__":
    asyncio.run(test_victor())
