"""Example using xAI Grok provider."""

import asyncio
import os

from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.xai_provider import XAIProvider


async def main():
    """Run examples with Grok."""
    # Check for API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY environment variable not set")
        print("Get your API key from: https://console.x.ai/")
        return

    print("🤖 xAI Grok Example\n")
    print("=" * 60)

    # Create Grok provider
    provider = XAIProvider(api_key=api_key)

    # Create agent
    agent = AgentOrchestrator(
        provider=provider,
        model="grok-beta",
        temperature=0.8,
    )

    # Example 1: General question
    print("\n📝 Example 1: General Question")
    print("-" * 60)
    response = await agent.chat("What makes Grok unique compared to other AI models?")
    print(f"Grok: {response.content}")

    # Example 2: Technical explanation
    print("\n\n💻 Example 2: Technical Explanation")
    print("-" * 60)
    agent.reset_conversation()

    response = await agent.chat(
        "Explain the differences between REST and GraphQL APIs. "
        "Give a code example for each."
    )
    print(f"Grok: {response.content}")

    # Example 3: Streaming response
    print("\n\n🌊 Example 3: Streaming Response")
    print("-" * 60)
    agent.reset_conversation()

    print("Grok: ", end="", flush=True)
    async for chunk in agent.stream_chat(
        "Write a Python decorator that measures function execution time."
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    # Example 4: Creative + Technical
    print("\n\n🎨 Example 4: Creative + Technical")
    print("-" * 60)
    agent.reset_conversation()

    response = await agent.chat(
        "Design a whimsical API for a virtual pet simulator. "
        "Include 5 endpoints with fun names and descriptions."
    )
    print(f"Grok: {response.content}")

    # Example 5: Multi-turn debugging
    print("\n\n🐛 Example 5: Multi-turn Debugging")
    print("-" * 60)
    agent.reset_conversation()

    response1 = await agent.chat(
        "I'm getting a 'list index out of range' error in Python. What could cause this?"
    )
    print(f"User: I'm getting a 'list index out of range' error in Python. What could cause this?")
    print(f"Grok: {response1.content[:200]}...")

    response2 = await agent.chat(
        "The error happens when I do: result = my_list[len(my_list)]"
    )
    print(f"\nUser: The error happens when I do: result = my_list[len(my_list)]")
    print(f"Grok: {response2.content}")

    # Clean up
    await provider.close()

    print("\n\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
