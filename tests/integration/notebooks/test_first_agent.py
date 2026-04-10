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

These tests run against local Ollama models only.
"""

import json
import os
import socket
from urllib.parse import urlparse
from urllib.request import urlopen

import pytest

from victor.framework import Agent
from victor.framework.events import EventType

PREFERRED_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _ollama_base_url() -> str:
    """Return normalized Ollama base URL from environment."""
    raw = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _ollama_host_port() -> tuple[str, int]:
    """Resolve Ollama host and port from OLLAMA_HOST."""
    parsed = urlparse(_ollama_base_url())
    return parsed.hostname or "localhost", parsed.port or 11434


def _check_ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    host, port = _ollama_host_port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _list_ollama_models() -> list[str]:
    """Return available local Ollama model names."""
    try:
        with urlopen(f"{_ollama_base_url()}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    names: list[str] = []
    for model in payload.get("models", []):
        name = model.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _select_model(models: list[str]) -> str:
    """Select preferred model with sensible fallback."""
    if PREFERRED_OLLAMA_MODEL in models:
        return PREFERRED_OLLAMA_MODEL
    if "llama3.1:8b" in models:
        return "llama3.1:8b"
    return models[0]


@pytest.fixture(scope="module")
def ollama_model() -> str:
    """Pick a local Ollama model for integration tests."""
    models = _list_ollama_models()
    if not models:
        pytest.skip("No Ollama models available")
    return _select_model(models)


@pytest.fixture
async def ollama_agent(ollama_model: str):
    """Create and clean up an Ollama-backed agent."""
    agent = await Agent.create(provider="ollama", model=ollama_model)
    try:
        yield agent
    finally:
        await agent.close()


@pytest.mark.integration
@pytest.mark.skipif(not _check_ollama_available(), reason="Ollama server not available")
class TestFirstAgentNotebook:
    """Integration tests for the first agent notebook.

    Based on: docs/tutorials/notebooks/01_first_agent.ipynb
    """

    @pytest.mark.asyncio
    async def test_basic_agent_with_default_settings(self, ollama_agent):
        """Test creating agent with default settings and running a query."""
        result = await ollama_agent.run("What is 2 + 2? Reply with just the number 4.")

        assert result.success is True
        assert result.content
        assert "4" in result.content.lower()
        print(f"Agent response: {result.content}")

    @pytest.mark.asyncio
    async def test_agent_with_ollama_provider(self, ollama_model):
        """Test agent with explicit Ollama provider."""
        async with await Agent.create(provider="ollama", model=ollama_model) as agent:
            result = await agent.run("What is Python? Answer in one sentence.")
            assert result.success is True
            assert result.content

    @pytest.mark.asyncio
    async def test_streaming_responses(self, ollama_agent):
        """Test streaming responses for real-time feedback."""
        events_received = []
        content_parts = []

        async for event in ollama_agent.stream("Count to 3. Reply with just the numbers."):
            events_received.append(event.type)

            if event.type == EventType.CONTENT and event.content:
                content_parts.append(event.content)
            elif event.type == EventType.TOOL_CALL:
                print(f"[Tool: {event.tool_name}]")
            elif event.type == EventType.THINKING:
                print("[Thinking...]")

        assert len(events_received) > 0
        assert EventType.CONTENT in events_received
        assert "".join(content_parts).strip()

    @pytest.mark.asyncio
    async def test_multi_turn_conversations(self, ollama_agent):
        """Test maintaining context across multiple messages."""
        session = ollama_agent.chat("My favorite color is blue. Remember this.")
        response2 = await session.send("What is my favorite color? Reply with one word.")
        assert response2.success is True
        assert response2.content

        content_lower = response2.content.lower()
        assert "blue" in content_lower

    @pytest.mark.asyncio
    async def test_agent_with_explicit_provider_config(self, ollama_model):
        """Test agent creation with explicit provider configuration."""
        async with await Agent.create(provider="ollama", model=ollama_model) as agent:
            result = await agent.run("Say 'test' and nothing else.")
            assert result.success is True
            assert "test" in result.content.lower()

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, ollama_agent):
        """Test agent handles errors gracefully."""
        result = await ollama_agent.run("")
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error

        long_query = "Explain the entire history of the universe in 10 words. " * 10
        result = await ollama_agent.run(long_query)
        assert isinstance(result.success, bool)


if __name__ == "__main__":
    # Run tests directly
    import sys

    pytest.main([__file__, "-v", "-s"])
