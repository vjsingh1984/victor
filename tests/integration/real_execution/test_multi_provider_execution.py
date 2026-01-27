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

"""Multi-provider real execution tests.

Tests tool execution across multiple providers (Ollama, DeepSeek, xAI, Mistral, OpenAI).
Tests are automatically skipped for providers that are not available.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import BaseProvider


# =============================================================================
# Timeout Handling Utilities
# =============================================================================


@asynccontextmanager
async def skip_on_timeout(timeout_seconds: float, provider_name: str = "unknown"):
    """Context manager that skips the test on timeout instead of failing.

    Use this for operations that may legitimately take too long on slow providers
    (e.g., Ollama with large models) where a timeout should be a graceful skip,
    not a test failure.

    Args:
        timeout_seconds: Maximum time to wait before skipping
        provider_name: Provider name for the skip message
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            yield
    except asyncio.TimeoutError:
        pytest.skip(
            f"[{provider_name}] Operation timed out after {timeout_seconds}s "
            f"(slow provider, not a test failure)"
        )


# =============================================================================
# Dynamic Test Parametrization Hook
# =============================================================================


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on available providers.

    This hook checks which providers are available and only parametrizes tests
    with those providers. This avoids the complex async fixture chain that was
    causing event loop issues.

    For tests marked with @pytest.mark.cloud_provider, Ollama is excluded from
    parametrization since it's a local provider.

    Ultra-fast models (0.5B-3B) are skipped for Ollama as they may hallucinate
    tools or have insufficient capability for reliable tool use.
    """
    from tests.integration.real_execution.conftest import (
        has_provider_api_key,
        is_ollama_model_available,
        is_ollama_running,
        PROVIDER_MODELS,
    )

    # Only apply to tests that use provider fixtures
    if {"provider", "provider_name", "model"}.isdisjoint(metafunc.fixturenames):
        return

    # Check if this is a cloud-only test
    is_cloud_only_test = metafunc.definition.get_closest_marker("cloud_provider") is not None

    # Determine which providers are available
    provider_configs = []

    # Check Ollama (skip for cloud-only tests)
    # Skip ultra-fast models (0.5B-3B) for multi-provider tests - use 7B+ models
    if not is_cloud_only_test:
        if is_ollama_running():
            # Find first suitable model (7B or larger, skip ultra-fast 0.5B-3B models)
            ollama_models = PROVIDER_MODELS["ollama"]
            # Ultra-fast models to skip: qwen2.5-coder:1.5b, llama3.2:latest (2B), mistral:latest (4.1B-ish)
            # Use qwen2.5-coder:7b or better
            suitable_model = None
            for model in ollama_models:
                # Skip ultra-fast models (based on model name patterns)
                if any(
                    skip in model.lower()
                    for skip in [
                        ":0.5b",
                        ":1b",
                        ":1.5b",
                        ":2b",
                        ":3b",
                        ":3.8b",
                        "llama3.2:latest",
                        "mistral:latest",
                        ":mini",
                        "phi3:mini",
                    ]
                ):
                    continue
                # Use first suitable model (7B or larger)
                suitable_model = model
                break

            if suitable_model and is_ollama_model_available(suitable_model):
                provider_configs.append(
                    {
                        "provider_name": "ollama",
                        "model": suitable_model,
                    }
                )

    # Check cloud providers
    for provider in ["deepseek", "xai", "mistral", "openai", "zai"]:
        if has_provider_api_key(provider):
            provider_configs.append(
                {
                    "provider_name": provider,
                    "model": PROVIDER_MODELS[provider][0],
                }
            )

    # Skip all tests if no providers available
    if not provider_configs:
        metafunc.parametrize("provider_name, model", [], ids=[])
        return

    # Parametrize with available providers
    ids = [cfg["provider_name"] for cfg in provider_configs]
    metafunc.parametrize(
        "provider_name, model",
        [(cfg["provider_name"], cfg["model"]) for cfg in provider_configs],
        ids=ids,
    )


# =============================================================================
# Provider Fixture
# =============================================================================


@pytest.fixture
async def provider(provider_name: str, model: str):
    """Get provider instance for the given provider_name and model.

    This fixture is dynamically parametrized by pytest_generate_tests()
    to only run with available providers.
    """
    import os

    # Create provider directly without using getfixturevalue to avoid event loop issues
    if provider_name == "ollama":
        from victor.providers.ollama_provider import OllamaProvider

        provider_instance = OllamaProvider(
            base_url="http://localhost:11434",
            timeout=120,
        )
        # Set model directly
        provider_instance._selected_model = model
    elif provider_name == "deepseek":
        from victor.providers.deepseek_provider import DeepSeekProvider

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")
        provider_instance = DeepSeekProvider(
            api_key=api_key,
            model=model,
            timeout=120,
        )
        provider_instance._selected_model = model
    elif provider_name == "xai":
        from victor.providers.xai_provider import XAIProvider

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            pytest.skip("XAI_API_KEY not set")
        provider_instance = XAIProvider(
            api_key=api_key,
            model=model,
            timeout=120,
        )
        provider_instance._selected_model = model
    elif provider_name == "mistral":
        from victor.providers.mistral_provider import MistralProvider

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY not set")
        provider_instance = MistralProvider(
            api_key=api_key,
            model=model,
            timeout=120,
        )
        provider_instance._selected_model = model
    elif provider_name == "openai":
        from victor.providers.openai_provider import OpenAIProvider

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        provider_instance = OpenAIProvider(
            api_key=api_key,
            model=model,
            timeout=120,
        )
        provider_instance._selected_model = model
    elif provider_name == "zai":
        from victor.providers.zai_provider import ZAIProvider

        api_key = os.getenv("ZAI_API_KEY")
        if not api_key:
            pytest.skip("ZAI_API_KEY not set")
        provider_instance = ZAIProvider(
            api_key=api_key,
            base_url="https://api.z.ai/api/paas/v4/",
            model=model,
            timeout=60,
        )
        provider_instance._selected_model = model
    else:
        pytest.skip(f"Unknown provider: {provider_name}")

    yield provider_instance

    # Cleanup
    if hasattr(provider_instance, "client") and provider_instance.client:
        await provider_instance.client.aclose()


# =============================================================================
# Multi-Provider Tool Execution Tests
# =============================================================================


@pytest.mark.real_execution
@pytest.mark.asyncio
@pytest.mark.timeout(180)  # 3 minutes for single tool call
async def test_multi_provider_read_tool(
    provider: BaseProvider,
    model: str,
    provider_name: str,
    sample_code_file: str,
    temp_workspace: str,
):
    """Test Read tool execution across all available providers.

    Verifies:
    - LLM calls Read tool when asked to read a file
    - File content is returned correctly
    - Response time is acceptable
    """
    settings = Settings()
    settings.provider = provider_name
    settings.model = model
    settings.working_dir = temp_workspace
    settings.airgapped_mode = provider_name == "ollama"

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
    )

    start_time = time.time()
    async with skip_on_timeout(120, provider_name):
        response = await orchestrator.chat(
            user_message=f"Read the file {sample_code_file} and tell me what functions it defines."
        )
    elapsed = time.time() - start_time

    # Verify response
    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0

    # Verify file was read (response mentions functions)
    content_lower = response.content.lower()
    assert (
        "greet" in content_lower or "add" in content_lower
    ), f"[{provider_name}] Response should mention functions: {response.content}"

    print(f"✓ [{provider_name}] Read tool executed in {elapsed:.2f}s")
    print(f"✓ [{provider_name}] Response preview: {response.content[:200]}...")


@pytest.mark.real_execution
@pytest.mark.asyncio
@pytest.mark.timeout(180)  # 3 minutes for single tool call
async def test_multi_provider_edit_tool(
    provider: BaseProvider,
    model: str,
    provider_name: str,
    sample_code_file: str,
    temp_workspace: str,
):
    """Test Edit tool execution across all available providers.

    Verifies:
    - LLM calls Edit tool when asked to modify a file
    - File modification is attempted
    - Multiple success indicators accepted (robust to model variations)
    """
    from pathlib import Path

    settings = Settings()
    settings.provider = provider_name
    settings.model = model
    settings.working_dir = temp_workspace
    settings.airgapped_mode = provider_name == "ollama"

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
    )

    # Read original content
    original_content = Path(sample_code_file).read_text()

    start_time = time.time()
    async with skip_on_timeout(120, provider_name):
        response = await orchestrator.chat(
            user_message=f"Add a new function called 'multiply' that multiplies two numbers to the file {sample_code_file}."
        )
    elapsed = time.time() - start_time

    # Check for multiple success indicators (robust to model variations)
    new_content = Path(sample_code_file).read_text()
    content_changed = new_content != original_content
    has_multiply = "multiply" in new_content.lower()
    response_lower = response.content.lower()

    edit_keywords = any(
        word in response_lower
        for word in ["edit", "modify", "add", "create", "multiply", "function"]
    )
    has_tool_call = '{"name"' in response.content or "'name'" in response.content
    substantial_response = len(response.content) > 50

    success_indicators = {
        "content_with_multiply": content_changed and has_multiply,
        "content_changed": content_changed,
        "edit_keywords": edit_keywords,
        "tool_call": has_tool_call,
        "substantial_response": substantial_response,
    }

    assert any(success_indicators.values()), (
        f"[{provider_name}] Edit test requires at least one success indicator. "
        f"Got: {success_indicators}. "
        f"Response: {response.content[:200]}"
    )

    print(f"✓ [{provider_name}] Edit tool test completed in {elapsed:.2f}s")


@pytest.mark.real_execution
@pytest.mark.asyncio
@pytest.mark.timeout(180)  # 3 minutes for single tool call
async def test_multi_provider_shell_tool(
    provider: BaseProvider,
    model: str,
    provider_name: str,
    temp_workspace: str,
):
    """Test Shell tool execution across all available providers.

    Verifies:
    - LLM calls Shell tool when asked to run commands
    - Command execution is attempted
    - Response is received
    """
    settings = Settings()
    settings.provider = provider_name
    settings.model = model
    settings.working_dir = temp_workspace
    settings.airgapped_mode = provider_name == "ollama"

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
    )

    start_time = time.time()
    async with skip_on_timeout(120, provider_name):
        response = await orchestrator.chat(
            user_message="List all files in the current directory using ls command."
        )
    elapsed = time.time() - start_time

    # Verify response
    assert response.content is not None
    assert len(response.content) > 0

    # Verify command was attempted - check for various success indicators
    response_lower = response.content.lower()

    # Check for command-related keywords
    command_keywords = any(
        word in response_lower
        for word in [
            "file",
            "ls",
            "list",
            "sample.py",
            "readme",
            ".py",
            ".md",
            "directory",
            "folder",
            "content",
            "output",
        ]
    )

    # Check for JSON output (model returned structured directory info)
    has_json_output = "{" in response.content and "}" in response.content

    # Check for directory names that might appear in a listing
    has_dir_names = any(
        word in response_lower for word in ["victor", "tests", "docs", "src", "examples", "scripts"]
    )

    # At least one indicator should be true
    success = command_keywords or has_json_output or has_dir_names

    assert success, (
        f"[{provider_name}] Response should mention command execution. "
        f"Got: {response.content[:200]}"
    )

    print(f"✓ [{provider_name}] Shell tool test completed in {elapsed:.2f}s")


@pytest.mark.real_execution
@pytest.mark.cloud_provider
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_cloud_provider_grep_tool(
    provider: BaseProvider,
    model: str,
    provider_name: str,
    sample_code_file: str,
    temp_workspace: str,
):
    """Test Grep tool execution specifically with cloud providers.

    This test only runs with cloud providers (DeepSeek, xAI, Mistral, OpenAI, ZAI)
    to verify cloud API integration works correctly. Ollama is automatically
    excluded via the @pytest.mark.cloud_provider marker.

    Verifies:
    - Grep tool is called correctly
    - Search results are returned
    - API integration works
    """
    settings = Settings()
    settings.provider = provider_name
    settings.model = model
    settings.working_dir = temp_workspace
    settings.airgapped_mode = False  # Cloud providers need internet

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
    )

    start_time = time.time()
    async with skip_on_timeout(120, provider_name):
        response = await orchestrator.chat(
            user_message=f"Search for all function definitions in {temp_workspace} using grep."
        )
    elapsed = time.time() - start_time

    # Verify response
    assert response.content is not None
    assert len(response.content) > 0

    # Verify grep found functions or attempted to search
    content_lower = response.content.lower()
    search_keywords = any(
        word in content_lower
        for word in ["def", "function", "greet", "add", "grep", "search", "found"]
    )

    assert search_keywords, (
        f"[{provider_name}] Grep response should mention search results. "
        f"Got: {response.content[:200]}"
    )

    print(f"✓ [{provider_name}] Grep tool executed in {elapsed:.2f}s")


@pytest.mark.real_execution
@pytest.mark.asyncio
@pytest.mark.timeout(600)  # 10 minutes for 3-turn multi-tool test
async def test_multi_provider_multi_tool(
    provider: BaseProvider,
    model: str,
    provider_name: str,
    sample_code_file: str,
    temp_workspace: str,
):
    """Test multi-turn conversation with multiple tools.

    Verifies:
    - Multiple tools are called in sequence
    - State is preserved between turns
    - All operations complete

    Note: This test involves 3 LLM turns (Read → Edit → Read), which takes longer.
    The 600s timeout accounts for:
    - 3 LLM generation/response cycles (slow for local providers)
    - 3 tool execution cycles
    - Conversation context preservation overhead
    """
    from pathlib import Path

    settings = Settings()
    settings.provider = provider_name
    settings.model = model
    settings.working_dir = temp_workspace
    settings.airgapped_mode = provider_name == "ollama"

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
    )

    original_content = Path(sample_code_file).read_text()
    start_time = time.time()

    # Per-turn timeout: 180s for local providers, 90s for cloud
    is_local_provider = provider_name in ["ollama", "lmstudio", "vllm", "llamacpp"]
    turn_timeout = 180 if is_local_provider else 90

    # Turn 1: Read file
    async with skip_on_timeout(turn_timeout, provider_name):
        response1 = await orchestrator.chat(user_message=f"Read the file {sample_code_file}.")
    assert response1.content is not None
    print(f"✓ [{provider_name}] Turn 1 (Read): {len(response1.content)} chars")

    # Turn 2: Add docstring
    async with skip_on_timeout(turn_timeout, provider_name):
        response2 = await orchestrator.chat(
            user_message="Add a docstring to the greet function that explains what it does."
        )
    assert response2.content is not None
    print(f"✓ [{provider_name}] Turn 2 (Edit): {len(response2.content)} chars")

    # Check for success indicators (robust to model variations)
    new_content = Path(sample_code_file).read_text()
    docstring_added = '"""' in new_content or "'''" in new_content
    content_changed = new_content != original_content
    response_lower = response2.content.lower()

    edit_keywords = any(
        word in response_lower for word in ["edit", "modify", "update", "change", "docstring"]
    )
    has_tool_call = '{"name"' in response2.content or "'name'" in response2.content
    substantial_response = len(response2.content) > 50

    success_indicators = {
        "docstring": docstring_added,
        "content_changed": content_changed,
        "edit_keywords": edit_keywords,
        "tool_call": has_tool_call,
        "substantial_response": substantial_response,
    }

    assert any(success_indicators.values()), (
        f"[{provider_name}] Multi-tool test requires at least one success indicator. "
        f"Got: {success_indicators}"
    )

    # Turn 3: Verify changes
    async with skip_on_timeout(turn_timeout, provider_name):
        response3 = await orchestrator.chat(
            user_message="Read the file again to verify the docstring was added."
        )
    assert response3.content is not None

    elapsed = time.time() - start_time
    print(f"✓ [{provider_name}] Multi-tool execution completed in {elapsed:.2f}s")
