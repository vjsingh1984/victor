"""
Victor Provider Selection Example

Shows how to use different LLM providers with Victor.
"""

import asyncio
from victor import Agent


async def openai_example():
    """Use OpenAI's GPT-4."""
    agent = Agent.create(
        provider="openai",
        model="gpt-4"
    )
    result = await agent.run("Explain quantum computing in one sentence")
    print(f"OpenAI: {result.content}")


async def anthropic_example():
    """Use Anthropic's Claude."""
    agent = Agent.create(
        provider="anthropic",
        model="claude-3-opus-20240229"
    )
    result = await agent.run("Explain quantum computing in one sentence")
    print(f"Anthropic: {result.content}")


async def ollama_example():
    """Use local Ollama model (no API key needed)."""
    agent = Agent.create(
        provider="ollama",
        model="llama2"
    )
    result = await agent.run("Explain quantum computing in one sentence")
    print(f"Ollama: {result.content}")


async def main():
    """Run all provider examples."""
    print("=== Provider Examples ===\n")

    # OpenAI
    print("Using OpenAI GPT-4:")
    await openai_example()
    print()

    # Anthropic
    print("Using Anthropic Claude:")
    await anthropic_example()
    print()

    # Ollama (if available)
    try:
        print("Using Ollama (local):")
        await ollama_example()
    except Exception as e:
        print(f"Ollama not available: {e}")


if __name__ == "__main__":
    asyncio.run(main())
