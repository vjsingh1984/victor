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

"""Example workflow using multiple providers strategically."""

import asyncio
import os

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.openai_provider import OpenAIProvider


async def main():
    """Demonstrate strategic use of multiple providers."""
    print("🎯 Multi-Provider Workflow Example\n")
    print("=" * 60)
    print("Strategy: Use the right model for each task\n")

    # Task: Build a complete feature with tests

    # Step 1: Brainstorm with local Ollama (FREE!)
    print("\n📝 Step 1: Brainstorming (Using Ollama - FREE)")
    print("-" * 60)

    settings = Settings()
    ollama = OllamaProvider()
    try:
        models = await ollama.list_models()
        if not models:
            print("⚠️  No Ollama models available. Skipping Ollama step.")
            ollama_available = False
        else:
            ollama_available = True
            model_name = models[0]["name"]

            agent_ollama = AgentOrchestrator(
                settings=settings,
                provider=ollama,
                model=model_name,
                temperature=0.8,
            )

            response = await agent_ollama.chat(
                "Brainstorm 3 creative names for a Python function that validates email addresses. "
                "Just list the names, one per line."
            )
            print(f"Ollama: {response.content}")
            chosen_name = "validate_email_format"  # Pick one
            print(f"\n✅ Chose: {chosen_name}")
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        ollama_available = False
        chosen_name = "validate_email_format"

    # Step 2: Implement with GPT-4o mini (CHEAP & FAST)
    print("\n\n💻 Step 2: Implementation (Using GPT-4o mini - Fast & Cheap)")
    print("-" * 60)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        gpt4o_mini = OpenAIProvider(api_key=openai_key)
        agent_gpt4o_mini = AgentOrchestrator(
            settings=settings,
            provider=gpt4o_mini,
            model="gpt-4o-mini",
            temperature=0.5,
        )

        response = await agent_gpt4o_mini.chat(
            f"Write a Python function called {chosen_name} that validates email addresses. "
            "Use regex, include docstring, handle edge cases."
        )
        print(f"GPT-4o mini: {response.content[:500]}...")
        implementation = response.content
        await gpt4o_mini.close()
    else:
        print("⚠️  OPENAI_API_KEY not set. Skipping GPT-4o mini step.")
        implementation = "# Implementation would go here"

    # Step 3: Review with Claude (BEST QUALITY)
    print("\n\n🔍 Step 3: Code Review (Using Claude - Best Quality)")
    print("-" * 60)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        claude = AnthropicProvider(api_key=anthropic_key)
        agent_claude = AgentOrchestrator(
            settings=settings,
            provider=claude,
            model="claude-sonnet-4-5",
            temperature=0.7,
        )

        response = await agent_claude.chat(
            f"Review this code for security, edge cases, and best practices:\n\n{implementation}\n\n"
            "Provide specific suggestions for improvement."
        )
        print(f"Claude: {response.content}")
        await claude.close()
    else:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping Claude review.")

    # Step 4: Generate tests with Ollama again (FREE!)
    print("\n\n🧪 Step 4: Generate Tests (Using Ollama - FREE)")
    print("-" * 60)

    if ollama_available:
        response = await agent_ollama.chat(
            f"Write pytest unit tests for this function:\n\n{implementation}\n\n"
            "Include tests for: valid emails, invalid emails, edge cases."
        )
        print(f"Ollama: {response.content[:500]}...")
    else:
        print("⚠️  Ollama not available. Skipping test generation.")

    # Summary
    print("\n\n📊 Workflow Summary")
    print("-" * 60)
    print("✅ Step 1: Brainstorm with Ollama (FREE)")
    print("✅ Step 2: Implement with GPT-4o mini ($0.15 per 1M input tokens)")
    print("✅ Step 3: Review with Claude ($0.003 per 1K tokens)")
    print("✅ Step 4: Generate tests with Ollama (FREE)")
    print("\n💰 Total cost: ~$0.01 vs. using GPT-4 for everything: ~$0.10")
    print("📈 Cost savings: 90% while maintaining quality!")

    print("\n\n💡 Key Takeaways:")
    print("- Use Ollama for brainstorming and simple tasks (FREE)")
    print("- Use GPT-4o mini for quick implementations (CHEAP)")
    print("- Use Claude for critical reviews (QUALITY)")
    print("- Mix and match for optimal cost/quality balance")

    # Clean up
    if ollama_available:
        await ollama.close()

    print("\n✅ Multi-provider workflow completed!")


if __name__ == "__main__":
    asyncio.run(main())
