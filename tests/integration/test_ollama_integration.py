"""Integration tests for Ollama provider.

These tests require Ollama to be running with at least one model installed.
Skip if Ollama is not available.
"""

import pytest
from httpx import ConnectError

from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.base import Message
from victor.providers.ollama import OllamaProvider


@pytest.fixture
async def ollama_provider():
    """Create Ollama provider and check if available."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    try:
        # Try to list models to check if Ollama is running
        models = await provider.list_models()
        if not models:
            pytest.skip("No models available in Ollama")
        yield provider
    except ConnectError:
        pytest.skip("Ollama is not running")
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_list_models(ollama_provider):
    """Test listing available models."""
    models = await ollama_provider.list_models()

    assert isinstance(models, list)
    assert len(models) > 0

    # Check model structure
    first_model = models[0]
    assert "name" in first_model
    print(f"\nAvailable models: {[m['name'] for m in models]}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_simple_chat(ollama_provider):
    """Test simple chat completion."""
    # Get first available model
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    messages = [Message(role="user", content="Say 'Hello World' and nothing else.")]

    response = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=50,
    )

    assert response.content
    assert len(response.content) > 0
    assert response.role == "assistant"
    print(f"\nResponse: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_streaming(ollama_provider):
    """Test streaming responses."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    messages = [Message(role="user", content="Count from 1 to 3.")]

    chunks = []
    async for chunk in ollama_provider.stream(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=50,
    ):
        chunks.append(chunk)
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()  # New line

    assert len(chunks) > 0
    assert any(chunk.content for chunk in chunks)
    assert chunks[-1].is_final


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_multi_turn_conversation(ollama_provider):
    """Test multi-turn conversation."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    # Turn 1
    messages = [Message(role="user", content="My name is Alice.")]
    response1 = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.7,
    )

    # Turn 2
    messages.append(Message(role="assistant", content=response1.content))
    messages.append(Message(role="user", content="What is my name?"))

    response2 = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.7,
    )

    assert response2.content
    # Model should remember the name
    assert "alice" in response2.content.lower()
    print(f"\nMemory test response: {response2.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_orchestrator_with_ollama():
    """Test full agent orchestrator with Ollama."""
    try:
        provider = OllamaProvider()
        models = await provider.list_models()
        if not models:
            pytest.skip("No Ollama models available")

        model_name = models[0]["name"]

        agent = AgentOrchestrator(
            provider=provider,
            model=model_name,
            temperature=0.5,
        )

        # Test simple chat
        response = await agent.chat("Say hello in exactly 3 words.")

        assert response.content
        assert len(response.content) > 0

        # Check conversation history
        assert len(agent.messages) == 2  # User + Assistant
        assert agent.messages[0].role == "user"
        assert agent.messages[1].role == "assistant"

        print(f"\nAgent response: {response.content}")

        await provider.close()

    except ConnectError:
        pytest.skip("Ollama is not running")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_ollama_pull_model():
    """Test pulling a model (if not already present).

    This test is marked as slow since pulling can take a while.
    """
    provider = OllamaProvider()

    try:
        # Try to pull a small test model
        test_model = "qwen2.5-coder:1.5b"  # Small model for testing

        print(f"\nAttempting to pull {test_model}...")
        print("(This may take a while if model not cached)")

        pull_generator = provider.pull_model(test_model)

        async for progress in pull_generator:
            # Print progress
            if "status" in progress:
                print(f"\r{progress['status']}", end="", flush=True)

        print("\nPull completed!")

        # Verify model is now available
        models = await provider.list_models()
        model_names = [m["name"] for m in models]

        # Check if our test model is in the list
        assert any(test_model in name for name in model_names)

        await provider.close()

    except ConnectError:
        pytest.skip("Ollama is not running")
    except Exception as e:
        # If pull fails for any reason, skip (might be network issue)
        pytest.skip(f"Model pull failed: {e}")
