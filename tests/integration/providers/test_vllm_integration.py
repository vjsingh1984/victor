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

"""Integration tests for vLLM provider.

These tests automatically launch a vLLM server before running tests
and shut it down after completion.

Uses a small model (Qwen/Qwen2.5-Coder-1.5B-Instruct) for faster testing.
"""

import pytest
from httpx import ConnectError, HTTPError
import httpx
import subprocess
import time
import os
import signal
import sys
import platform


# Early skip for ARM CPUs (vLLM has known compatibility issues)
_is_arm = platform.processor().startswith(("arm", "aarch64")) or platform.machine().startswith(
    ("arm", "aarch64")
)
if _is_arm:
    pytestmark = pytest.mark.skipif(
        True,
        reason="Skipping vLLM tests on ARM CPU (known compatibility issues during model warm-up)",
    )
else:
    pytestmark = []

# These imports are intentionally after checking availability
from victor.providers.base import Message, ToolDefinition  # noqa: E402
from victor.providers.openai_provider import OpenAIProvider  # noqa: E402


@pytest.fixture(scope="session")
def vllm_server():
    """Launch vLLM server for testing and shut it down after tests.

    This fixture:
    1. Checks if running in CI environment (GitHub Actions, etc.) - skips if true
    2. Launches vLLM server with a small model for local testing
    3. Waits for server to be ready (up to 120 seconds)
    4. Yields control to tests
    5. Shuts down server after all tests complete

    In CI environments, these tests are skipped because vLLM requires
    significant resources and model downloads that may not be allowed.

    Note: ARM CPU detection happens at module level for faster skipping.
    """
    # Detect CI environment
    is_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"

    if is_ci:
        pytest.skip("Skipping vLLM tests in CI environment (requires local model download)")

    # Check if vllm is installed
    try:
        import vllm

        vllm_available = True
    except ImportError:
        vllm_available = False
        pytest.skip("vllm package not installed. Install with: pip install vllm")

    if not vllm_available:
        return

    # Model and server configuration - using a very small model for faster testing
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    port = 8000
    host = "localhost"

    # Launch vLLM server as subprocess
    print(f"\n🚀 Launching vLLM server with model {model_name}...")

    vllm_process = None
    server_ready = False
    startup_attempt = 0
    max_startup_attempts = 1

    while startup_attempt < max_startup_attempts:
        startup_attempt += 1
        try:
            # Start vLLM server
            vllm_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    model_name,
                    "--port",
                    str(port),
                    "--host",
                    host,
                    "--max-model-len",
                    "2048",  # Reduce memory usage for ARM CPUs
                    "--dtype",
                    "float16",  # Use float16 instead of bfloat16 for better compatibility
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for server to be ready
            max_wait = 120  # seconds
            start_time = time.time()
            server_ready = False

            while time.time() - start_time < max_wait:
                try:
                    response = httpx.get(f"http://{host}:{port}/health", timeout=2.0)
                    if response.status_code == 200:
                        server_ready = True
                        print(f"✅ vLLM server is ready at http://{host}:{port}")
                        break
                except Exception:
                    time.sleep(2)

                    # Check if process is still running
                    poll_result = vllm_process.poll()
                    if poll_result is not None:
                        stdout, stderr = vllm_process.communicate()
                        print(f"❌ vLLM server exited unexpectedly with code {poll_result}")
                        print(f"STDOUT: {stdout}")
                        print(f"STDERR: {stderr}")

                        if startup_attempt < max_startup_attempts:
                            print(
                                f"🔄 Retrying startup (attempt {startup_attempt + 1}/{max_startup_attempts})..."
                            )
                            time.sleep(2)
                            break
                        else:
                            pytest.skip(
                                f"vLLM server failed to start after {max_startup_attempts} attempts (exit code: {poll_result})"
                            )

                time.sleep(2)

            if server_ready:
                break

        except Exception as e:
            print(f"❌ Error starting vLLM server: {e}")
            if startup_attempt < max_startup_attempts:
                print(
                    f"🔄 Retrying startup (attempt {startup_attempt + 1}/{max_startup_attempts})..."
                )
                time.sleep(2)
            else:
                pytest.skip(
                    f"vLLM server failed to start after {max_startup_attempts} attempts: {e}"
                )

        if not server_ready and startup_attempt >= max_startup_attempts:
            pytest.skip(f"vLLM server did not start after {max_startup_attempts} attempts")

    # Yield control to tests
    try:
        yield vllm_process
    finally:
        # Clean up: shut down vLLM server
        if vllm_process:
            print("\n🛑 Shutting down vLLM server...")
            try:
                vllm_process.terminate()
                vllm_process.wait(timeout=10)
                print("✅ vLLM server shut down successfully")
            except subprocess.TimeoutExpired:
                print("⚠️  vLLM server did not shut down gracefully, forcing...")
                vllm_process.kill()
                vllm_process.wait()
                print("✅ vLLM server forcefully shut down")
            except Exception as e:
                print(f"❌ Error shutting down vLLM server: {e}")


@pytest.fixture
async def vllm_provider(vllm_server):
    """Create vLLM provider and check if available.

    This fixture depends on vllm_server fixture, so it will only
    run if the vLLM server was successfully launched.
    """
    provider = OpenAIProvider(
        api_key="vllm",  # Placeholder
        base_url="http://localhost:8000/v1",
        timeout=300,
    )

    try:
        # Try to list models to check if vLLM is running
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/v1/models", timeout=5.0)
            if response.status_code != 200:
                pytest.skip("vLLM server not responding")

            data = response.json()
            if not data.get("data"):
                pytest.skip("No models loaded in vLLM")

        yield provider
    except (ConnectError, HTTPError, Exception) as e:
        pytest.skip(f"vLLM is not running: {e}")
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_server_health():
    """Test vLLM server availability."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health", timeout=5.0)
        assert response.status_code == 200
        print("\n✅ vLLM server is healthy")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_models_endpoint():
    """Test listing models from vLLM."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/v1/models", timeout=5.0)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

        models = [m["id"] for m in data["data"]]
        print(f"\n✅ Loaded models in vLLM: {models}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_simple_chat(vllm_provider):
    """Test simple chat completion with vLLM."""
    messages = [Message(role="user", content="Say 'Hello from vLLM' and nothing else.")]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,
        max_tokens=50,
    )

    assert response.content
    assert len(response.content) > 0
    assert response.role == "assistant"
    print(f"\n✅ vLLM Response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_code_generation(vllm_provider):
    """Test code generation with vLLM."""
    messages = [
        Message(role="system", content="You are an expert Python programmer."),
        Message(
            role="user",
            content="Write a simple Python function to add two numbers. Just the function, no explanation.",
        ),
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.2,
        max_tokens=512,
    )

    assert response.content
    assert "def" in response.content or "add" in response.content.lower()
    print(f"\n✅ vLLM Code Generation:\n{response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_streaming(vllm_provider):
    """Test streaming responses from vLLM."""
    messages = [Message(role="user", content="Count from 1 to 5, one number per line.")]

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
    print(f"\n✅ Streamed {len(chunks)} chunks, total content length: {len(full_content)}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_tool_calling(vllm_provider):
    """Test tool calling with vLLM (if model supports it)."""
    tools = [
        ToolDefinition(
            name="get_current_time",
            description="Get the current time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone name"},
                },
                "required": ["timezone"],
            },
        )
    ]

    messages = [Message(role="user", content="What time is it in New York?")]

    try:
        response = await vllm_provider.chat(
            messages=messages,
            model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            tools=tools,
            temperature=0.5,
            max_tokens=200,
        )

        # vLLM might not support tool calling for all models
        # Just verify we get a response
        assert response.content or response.tool_calls

        if response.tool_calls:
            print(f"\n✅ vLLM Tool Calls: {response.tool_calls}")
        else:
            print(f"\n✅ vLLM Response (no tool calls): {response.content}")

    except Exception as e:
        # Some models might not support tool calling
        pytest.skip(f"Model might not support tool calling: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_multi_turn_conversation(vllm_provider):
    """Test multi-turn conversation with vLLM."""
    # Turn 1
    messages = [Message(role="user", content="My name is Bob.")]

    response1 = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    # Turn 2
    messages.append(Message(role="assistant", content=response1.content))
    messages.append(Message(role="user", content="What is my name?"))

    response2 = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    assert response2.content
    # Model should remember Bob
    assert "bob" in response2.content.lower()
    print(f"\n✅ vLLM Memory test response: {response2.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vllm_with_custom_parameters(vllm_provider):
    """Test vLLM with custom sampling parameters."""
    messages = [
        Message(role="user", content="Write a one-line function to double a number in Python.")
    ]

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.1,  # Very low for deterministic output
        top_p=0.95,
        max_tokens=100,
    )

    assert response.content
    assert (
        "def" in response.content
        or "lambda" in response.content
        or "double" in response.content.lower()
    )
    print(f"\n✅ vLLM Custom params response: {response.content}")


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
        messages.append(
            Message(
                role="user",
                content=f"What is important about code quality aspect {i+1}? One sentence.",
            )
        )
        messages.append(Message(role="assistant", content=f"Code quality aspect {i+1} matters."))

    messages.append(Message(role="user", content="Summarize our discussion in one sentence."))

    response = await vllm_provider.chat(
        messages=messages,
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.5,
        max_tokens=200,
    )

    assert response.content
    print(f"\n✅ vLLM Large context response: {response.content}")
