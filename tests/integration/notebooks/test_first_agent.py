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

"""Integration tests for 01_first_agent notebook using Ollama.

Tests the first agent tutorial with real LLM calls using Ollama running on localhost.
Requires Ollama to be running with a model pulled.

To setup Ollama:
    brew install ollama  # macOS
    ollama serve &
    ollama pull qwen2.5-coder:7b  # or another model
"""

import os
import subprocess
import pytest

from victor.framework import Agent


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OLLAMA_HOST") and not os.path.exists("/tmp/ollama.pid"),
    reason="Requires Ollama running on localhost"
)
class TestFirstAgentNotebook:
    """Integration tests for the first agent notebook.

    Based on: docs/tutorials/notebooks/01_first_agent.ipynb
    """

    @pytest.mark.asyncio
    async def test_basic_agent_with_default_settings(self):
        """Test creating agent with default settings and running a query."""
        # Use default provider (should be configured or fallback)
        agent = await Agent.create()

        # Run a simple query
        result = await agent.run("What is 2 + 2? Reply with just the number.")

        assert result.success is True
        assert result.content
        assert "4" in result.content.lower()
        print(f"Agent response: {result.content}")

    @pytest.mark.asyncio
    async def test_agent_with_ollama_provider(self):
        """Test agent with explicit Ollama provider."""
        # Check if Ollama is available
        try:
            # Check if ollama is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5,
            )
            ollama_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            ollama_available = False

        pytest.skipif(not ollama_available, "Ollama not available")

        # Create agent with Ollama
        agent = await Agent.create(
            provider="ollama",
            model="qwen2.5-coder:7b",  # Or any available model
        )

        # Run a query
        result = await agent.run("What is Python? Answer in one sentence.")
        assert result.success is True
        assert result.content

    @pytest.mark.asyncio
    async def test_streaming_responses(self):
        """Test streaming responses for real-time feedback."""
        agent = await Agent.create()

        # Stream response and collect events
        events_received = []
        content_parts = []

        async for event in agent.stream("Count to 3. Reply with just the numbers."):
            events_received.append(event.type)

            if event.type == "content":
                content_parts.append(event.content)
            elif event.type == "tool_call":
                print(f"[Tool: {event.tool_name}]")
            elif event.type == "thinking":
                print("[Thinking...]")

        assert len(events_received) > 0
        assert "content" in events_received
        assert "".join(content_parts)

    @pytest.mark.asyncio
    async def test_multi_turn_conversations(self):
        """Test maintaining context across multiple messages."""
        agent = await Agent.create()

        # First message
        response1 = await agent.chat("My favorite color is blue.")
        assert response1.success is True
        assert response1.content

        # Agent remembers context
        response2 = await agent.chat("What is my favorite color?")
        assert response2.success is True

        # Should remember "blue" from previous turn
        content_lower = response2.content.lower()
        assert "blue" in content_lower

    @pytest.mark.asyncio
    async def test_agent_with_explicit_provider_config(self):
        """Test agent creation with explicit provider configuration."""
        # Test with common providers
        providers_to_test = []

        # Check which providers are configured
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_to_test.append(("anthropic", "claude-sonnet-4-20250514"))

        if os.getenv("OPENAI_API_KEY"):
            providers_to_test.append(("openai", "gpt-4o"))

        # Check for Ollama
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                providers_to_test.append(("ollama", "qwen2.5-coder:7b"))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if not providers_to_test:
            pytest.skip("No LLM provider configured (need ANTHROPIC_API_KEY, OPENAI_API_KEY, or Ollama)")

        for provider, model in providers_to_test:
            print(f"\nTesting with provider={provider}, model={model}")

            agent = await Agent.create(provider=provider, model=model)
            result = await agent.run("Say 'test'")

            assert result.success is True
            assert "test" in result.content.lower()

    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test agent handles errors gracefully."""
        agent = await Agent.create()

        # Test with empty query
        result = await agent.run("")
        assert result.success is True  # Should handle empty input

        # Test with very long query
        long_query = "Explain the entire history of the universe in 10 words. " * 10
        result = await agent.run(long_query)
        assert result.success is True


if __name__ == "__main__":
    # Run tests directly
    import sys

    pytest.main([__file__, "-v", "-s"])
