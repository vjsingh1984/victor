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

"""Example using OpenAI GPT provider."""

import asyncio
import os

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.openai_provider import OpenAIProvider


async def main():
    """Run examples with GPT."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Get your API key from: https://platform.openai.com/api-keys")
        return

    print("🤖 OpenAI GPT Example\n")
    print("=" * 60)

    # Create OpenAI provider
    provider = OpenAIProvider(api_key=api_key)

    # Example with GPT-4o
    print("\n💎 Using GPT-4o")
    print("-" * 60)

    settings = Settings()
    agent = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="gpt-4o",
        temperature=0.7,
    )

    # Example 1: Code review
    print("\n📝 Example 1: Code Review")
    print("-" * 60)

    code_to_review = """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price'] * item['qty']
    return total
"""

    response = await agent.chat(
        f"Review this Python code and suggest improvements:\n\n{code_to_review}"
    )
    print(f"GPT-4o: {response.content}")

    # Example 2: Creative writing
    print("\n\n✍️ Example 2: Creative Writing")
    print("-" * 60)
    agent.reset_conversation()

    response = await agent.chat(
        "Write a creative product name and tagline for an AI coding assistant that works with any LLM."
    )
    print(f"GPT-4o: {response.content}")

    # Example with GPT-4o mini (faster, cheaper)
    print("\n\n⚡ Using GPT-4o mini (faster)")
    print("-" * 60)

    provider2 = OpenAIProvider(api_key=api_key)
    agent2 = AgentOrchestrator(
        settings=settings,
        provider=provider2,
        model="gpt-4o-mini",
        temperature=0.5,
    )

    print("\nGPT-4o mini: ", end="", flush=True)
    async for chunk in agent2.stream_chat("List 5 Python best practices in one sentence each."):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    # Example 3: Problem solving
    print("\n\n🧩 Example 3: Problem Solving")
    print("-" * 60)
    agent.reset_conversation()

    response = await agent.chat(
        "I have a list of 1 million integers. I need to find the top 10 largest numbers efficiently. "
        "What's the best approach and why?"
    )
    print(f"GPT-4o: {response.content}")

    # Clean up
    await provider.close()
    await provider2.close()

    print("\n\n✅ Examples completed!")
    print("\n💡 Tip: Use GPT-4o mini for quick tasks, GPT-4o for complex reasoning")


if __name__ == "__main__":
    asyncio.run(main())
