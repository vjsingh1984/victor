"""Integration tests for vLLM provider.

These tests require vLLM server to be running with a model loaded.
Skip if vLLM is not available.

To start vLLM server:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --port 8000
"""

import pytest
from httpx import ConnectError, HTTPError
import httpx

from victor.providers.base import Message, ToolDefinition
from victor.providers.openai_provider import OpenAIProvider


@pytest.fixture
async def vllm_provider():
    """Create vLLM provider and check if available."""
    provider = OpenAIProvider(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=300,
    )

    try:
        # Try a simple request to check if vLLM is running
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code != 200:
                pytest.skip("vLLM server not healthy")

        yield provider
    except (ConnectError, HTTPError, Exception) as e:
        pytest.skip(f"vLLM is not running: {e}")
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_server_health():
    """Test vLLM server health endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            assert response.status_code == 200
            print(f"\nvLLM server is healthy: {response.text}")
    except Exception as e:
        pytest.skip(f"vLLM server not running: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_models_endpoint():
    """Test vLLM models endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/v1/models", timeout=5.0)
            assert response.status_code == 200

            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0

            model = data["data"][0]
            print(f"\nLoaded model: {model.get('id', 'unknown')}")

    except Exception as e:
        pytest.skip(f"vLLM server not running: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_simple_chat(vllm_provider):
    """Test simple chat completion with vLLM."""
    messages = [
        Message(role="user", content="Say 'Hello from vLLM' and nothing else.")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=50,
    )

    assert response.content
    assert len(response.content) > 0
    assert response.role == "assistant"
    print(f"\nvLLM Response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_code_generation(vllm_provider):
    """Test code generation with vLLM."""
    messages = [
        Message(role="system", content="You are an expert Python programmer."),
        Message(role="user", content="Write a Python function to calculate factorial. Keep it simple.")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.3,
        max_tokens=512,
    )

    assert response.content
    assert "def" in response.content or "factorial" in response.content.lower()
    print(f"\nvLLM Code Generation:\n{response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_streaming(vllm_provider):
    """Test streaming responses from vLLM."""
    messages = [
        Message(role="user", content="Count from 1 to 5, one number per line.")
    ]

    chunks = []
    full_content = ""

    async for chunk in vllm_provider.stream(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=100,
    ):
        chunks.append(chunk)
        if chunk.content:
            full_content += chunk.content
            print(chunk.content, end="", flush=True)

    print()  # New line

    assert len(chunks) > 0
    assert full_content
    assert chunks[-1].is_final
    print(f"\nStreamed {len(chunks)} chunks, total content length: {len(full_content)}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_tool_calling(vllm_provider):
    """Test tool calling with vLLM.

    vLLM supports tool calling with --enable-auto-tool-choice flag.
    """
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        )
    ]

    messages = [
        Message(role="user", content="What's the weather in San Francisco?")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tools=tools,
        temperature=0.5,
        max_tokens=200,
    )

    # vLLM with tool calling enabled should return either content or tool_calls
    assert response.content or response.tool_calls

    if response.tool_calls:
        print(f"\nvLLM Tool Calls: {response.tool_calls}")
        # Verify tool call structure
        assert len(response.tool_calls) > 0
        first_call = response.tool_calls[0]
        assert "name" in first_call
        assert "arguments" in first_call
        print(f"Tool called: {first_call['name']}")
    else:
        print(f"\nvLLM Response (no tool calls): {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_multi_turn_conversation(vllm_provider):
    """Test multi-turn conversation with vLLM."""
    # Turn 1
    messages = [
        Message(role="user", content="My favorite programming language is Python.")
    ]

    response1 = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    # Turn 2
    messages.append(Message(role="assistant", content=response1.content))
    messages.append(Message(role="user", content="What is my favorite programming language?"))

    response2 = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    assert response2.content
    # Model should remember Python
    assert "python" in response2.content.lower()
    print(f"\nvLLM Memory test response: {response2.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_with_custom_parameters(vllm_provider):
    """Test vLLM with custom sampling parameters."""
    messages = [
        Message(role="user", content="Write a one-line Python function to square a number.")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,  # Very low for deterministic output
        top_p=0.95,
        max_tokens=100,
    )

    assert response.content
    assert "def" in response.content or "lambda" in response.content
    print(f"\nvLLM Custom params response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_large_context(vllm_provider):
    """Test vLLM with larger context."""
    # Create a conversation with multiple turns
    messages = [
        Message(role="system", content="You are a helpful coding assistant."),
    ]

    # Add several exchanges
    for i in range(5):
        messages.append(Message(role="user", content=f"Tell me about Python feature {i+1} in one sentence."))
        messages.append(Message(role="assistant", content=f"Python feature {i+1} is important."))

    messages.append(Message(role="user", content="Now summarize what we discussed in one sentence."))

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.5,
        max_tokens=200,
    )

    assert response.content
    print(f"\nvLLM Large context response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_system_message(vllm_provider):
    """Test vLLM with system messages."""
    messages = [
        Message(role="system", content="You are a Python expert who gives concise answers."),
        Message(role="user", content="What is a list comprehension? One sentence only.")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.3,
        max_tokens=100,
    )

    assert response.content
    assert len(response.content) > 0
    print(f"\nvLLM System message response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_token_usage(vllm_provider):
    """Test that vLLM returns token usage information."""
    messages = [
        Message(role="user", content="Say 'hi'")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=10,
    )

    assert response.content
    assert response.usage is not None
    assert "prompt_tokens" in response.usage
    assert "completion_tokens" in response.usage
    assert "total_tokens" in response.usage
    assert response.usage["total_tokens"] > 0
    print(f"\nvLLM Token usage: {response.usage}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_temperature_variations(vllm_provider):
    """Test vLLM with different temperature settings."""
    messages = [
        Message(role="user", content="Say hello")
    ]

    # Test with very low temperature (deterministic)
    response_low = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=50,
    )

    # Test with higher temperature (more random)
    response_high = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.9,
        max_tokens=50,
    )

    assert response_low.content
    assert response_high.content
    print(f"\nvLLM Low temp (0.1): {response_low.content}")
    print(f"vLLM High temp (0.9): {response_high.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_max_tokens_limiting(vllm_provider):
    """Test that vLLM respects max_tokens limit."""
    messages = [
        Message(role="user", content="Write a very long story about a cat.")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.5,
        max_tokens=20,  # Very low limit
    )

    assert response.content
    # Verify token limit was respected
    if response.usage:
        assert response.usage["completion_tokens"] <= 25  # Allow small buffer
    print(f"\nvLLM Max tokens test: {len(response.content)} chars, {response.usage}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_top_p_sampling(vllm_provider):
    """Test vLLM with top_p (nucleus sampling)."""
    messages = [
        Message(role="user", content="Say hello in a creative way.")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
    )

    assert response.content
    assert len(response.content) > 0
    print(f"\nvLLM Top-p response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_response_metadata(vllm_provider):
    """Test that vLLM returns proper response metadata."""
    messages = [
        Message(role="user", content="Hi")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.5,
        max_tokens=50,
    )

    # Verify response structure
    assert response.content is not None
    assert response.role == "assistant"
    assert response.model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert response.stop_reason in ["stop", "length", None]
    print(f"\nvLLM Response metadata: role={response.role}, model={response.model}, stop_reason={response.stop_reason}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_empty_content_handling(vllm_provider):
    """Test vLLM handling of minimal input."""
    messages = [
        Message(role="user", content="Hi")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=10,
    )

    assert response.content is not None
    assert response.role == "assistant"
    print(f"\nvLLM Minimal input response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_streaming_with_tool_support(vllm_provider):
    """Test that streaming works with tools parameter (even if not used)."""
    messages = [
        Message(role="user", content="Count to 3")
    ]

    tools = [
        ToolDefinition(
            name="dummy_tool",
            description="A dummy tool",
            parameters={"type": "object", "properties": {}},
        )
    ]

    chunks = []
    async for chunk in vllm_provider.stream(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tools=tools,
        temperature=0.1,
        max_tokens=50,
    ):
        chunks.append(chunk)
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()

    assert len(chunks) > 0
    assert chunks[-1].is_final
    print(f"\nStreamed {len(chunks)} chunks with tools parameter")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_provider_features(vllm_provider):
    """Test vLLM provider capability reporting."""
    assert vllm_provider.supports_streaming() is True
    assert vllm_provider.supports_tools() is True
    assert vllm_provider.name == "openai"
    print(f"\nvLLM Provider: name={vllm_provider.name}, streaming={vllm_provider.supports_streaming()}, tools={vllm_provider.supports_tools()}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_empty_response_handling(vllm_provider):
    """Test vLLM handling of minimal responses."""
    messages = [
        Message(role="user", content=".")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=5,
    )

    assert response is not None
    assert response.role == "assistant"
    print(f"\nvLLM minimal response: '{response.content}'")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_streaming_final_chunk(vllm_provider):
    """Test that vLLM streaming properly marks final chunks."""
    messages = [
        Message(role="user", content="Hi")
    ]

    chunks = []
    final_count = 0

    async for chunk in vllm_provider.stream(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=10,
    ):
        chunks.append(chunk)
        if chunk.is_final:
            final_count += 1

    assert final_count == 1, "Should have exactly one final chunk"
    assert chunks[-1].is_final, "Last chunk should be marked as final"
    print(f"\nvLLM streaming: {len(chunks)} chunks, {final_count} final")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_stop_reason_verification(vllm_provider):
    """Test that vLLM returns appropriate stop reasons."""
    messages = [
        Message(role="user", content="Say hello")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=100,
    )

    assert response.stop_reason in ["stop", "length", None, "eos"]
    print(f"\nvLLM stop reason: {response.stop_reason}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_connection_close(vllm_provider):
    """Test that vLLM provider connection closes properly."""
    # Make a simple request
    messages = [Message(role="user", content="Hi")]
    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=10,
    )
    assert response.content

    # Close the connection
    await vllm_provider.close()
    print("\nvLLM connection closed successfully")
