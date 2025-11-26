# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example using Anthropic Claude provider."""

import asyncio
import os

from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.anthropic_provider import AnthropicProvider


async def main():
    """Run examples with Claude."""
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Get your API key from: https://console.anthropic.com/")
        return

    print("🤖 Anthropic Claude Example\n")
    print("=" * 60)

    # Create Claude provider
    provider = AnthropicProvider(api_key=api_key)

    # Create agent
    agent = AgentOrchestrator(
        provider=provider,
        model="claude-sonnet-4-5",  # or claude-3-opus, claude-3-sonnet
        temperature=1.0,
    )

    # Example 1: Simple question
    print("\n📝 Example 1: Simple Question")
    print("-" * 60)
    response = await agent.chat("Explain async/await in Python in 2 sentences.")
    print(f"Claude: {response.content}")
    print(f"\nTokens used: {response.usage}")

    # Example 2: Code generation
    print("\n\n💻 Example 2: Code Generation")
    print("-" * 60)
    agent.reset_conversation()  # Start fresh
    response = await agent.chat(
        "Write a Python function to find the longest palindrome substring. "
        "Include docstring and example."
    )
    print(f"Claude:\n{response.content}")

    # Example 3: Streaming
    print("\n\n🌊 Example 3: Streaming Response")
    print("-" * 60)
    agent.reset_conversation()
    print("Claude: ", end="", flush=True)

    async for chunk in agent.stream_chat(
        "Write a haiku about programming."
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    # Example 4: Multi-turn conversation
    print("\n\n💬 Example 4: Multi-turn Conversation")
    print("-" * 60)
    agent.reset_conversation()

    response1 = await agent.chat("I'm building a REST API. Should I use FastAPI or Flask?")
    print(f"User: I'm building a REST API. Should I use FastAPI or Flask?")
    print(f"Claude: {response1.content[:200]}...")

    response2 = await agent.chat("Why is async important for this choice?")
    print(f"\nUser: Why is async important for this choice?")
    print(f"Claude: {response2.content[:200]}...")

    # Clean up
    await provider.close()
    print("\n\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
