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

"""Fixtures for real execution integration tests with multi-provider support.

Supports both local providers (Ollama) and cloud providers (DeepSeek, xAI, Mistral, OpenAI).
Tests are automatically skipped in CI/CD when API keys are not available.
"""

import os
import socket
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import pytest
import httpx

from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.mistral_provider import MistralProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.xai_provider import XAIProvider
from victor.providers.zai_provider import ZAIProvider


# =============================================================================
# Local Provider Checks (Ollama)
# =============================================================================


def is_ollama_running() -> bool:
    """Check if Ollama server is running at localhost:11434."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        return result == 0
    except Exception:
        return False


def is_ollama_model_available(model: str = "qwen2.5-coder:14b") -> bool:
    """Check if Ollama model is available."""
    if not is_ollama_running():
        return False

    try:
        response = httpx.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(model in m for m in model_names)
        return False
    except Exception:
        return False


# =============================================================================
# Cloud Provider API Key Checks
# =============================================================================


def has_provider_api_key(provider: str) -> bool:
    """Check if provider API key is configured.

    Args:
        provider: Provider name (deepseek, xai, mistral, openai, zai)

    Returns:
        True if API key is available in environment
    """
    env_vars = {
        "deepseek": "DEEPSEEK_API_KEY",
        "xai": "XAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "openai": "OPENAI_API_KEY",
        "zai": "ZAI_API_KEY",
    }
    env_var = env_vars.get(provider.lower())
    return bool(os.getenv(env_var))


# =============================================================================
# Provider Model Configurations
# =============================================================================


# Cheapest/fastest models for each provider (for cost-effective testing)
PROVIDER_MODELS: Dict[str, List[str]] = {
    "ollama": [
        "qwen2.5-coder:14b",  # Fast and capable
        "gpt-oss-tools:20b-64K",  # Alternative with good tool support
        "qwen2.5-coder:7b",  # Fastest fallback
    ],
    "deepseek": [
        "deepseek-chat",  # Cheapest, good for general use
        "deepseek-coder",  # For coding tasks
    ],
    "xai": [
        "grok-beta",  # Cheapest Grok model
        "grok-2-1212",  # More capable
    ],
    "mistral": [
        "mistral-small-latest",  # Cheapest, fast
        "codestral-latest",  # For coding tasks
        "mistral-large-latest",  # Most capable
    ],
    "openai": [
        "gpt-4o-mini",  # Cheapest, fast
        "gpt-4o",  # More capable
    ],
    "zai": [
        "glm-4-flash",  # Cheapest
        "glm-4-plus",  # More capable
    ],
}


def get_provider_model(provider: str) -> Optional[str]:
    """Get the best available model for a provider.

    Args:
        provider: Provider name

    Returns:
        Model name or None if provider not supported
    """
    models = PROVIDER_MODELS.get(provider.lower())
    return models[0] if models else None


# =============================================================================
# Provider Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Check if Ollama is available for testing."""
    return is_ollama_running()


@pytest.fixture(scope="session")
def ollama_model_available() -> bool:
    """Check if Ollama model is available for testing."""
    return any(is_ollama_model_available(model) for model in PROVIDER_MODELS["ollama"])


@pytest.fixture(scope="session")
def deepseek_available() -> bool:
    """Check if DeepSeek API key is available."""
    return has_provider_api_key("deepseek")


@pytest.fixture(scope="session")
def xai_available() -> bool:
    """Check if xAI API key is available."""
    return has_provider_api_key("xai")


@pytest.fixture(scope="session")
def mistral_available() -> bool:
    """Check if Mistral API key is available."""
    return has_provider_api_key("mistral")


@pytest.fixture(scope="session")
def openai_available() -> bool:
    """Check if OpenAI API key is available."""
    return has_provider_api_key("openai")


@pytest.fixture(scope="session")
def zai_available() -> bool:
    """Check if ZAI API key is available."""
    return has_provider_api_key("zai")


# =============================================================================
# Async Provider Factories
# =============================================================================


@pytest.fixture
async def ollama_provider() -> AsyncGenerator[OllamaProvider, None]:
    """Create Ollama provider for testing.

    Uses smaller models for faster execution:
    - qwen2.5-coder:14b (preferred, fast and capable)
    - gpt-oss-tools:20b-64K (alternative, good tool support)
    - qwen2.5-coder:7b (fallback, fastest)
    """
    if not is_ollama_running():
        pytest.skip("Ollama not available at localhost:11434")

    # Try to find an available model
    model = None
    for candidate_model in PROVIDER_MODELS["ollama"]:
        if is_ollama_model_available(candidate_model):
            model = candidate_model
            break

    if not model:
        pytest.skip("No suitable Ollama model found. " "Run: ollama pull qwen2.5-coder:14b")

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        timeout=120,
    )
    provider._selected_model = model

    yield provider

    if hasattr(provider, "client"):
        await provider.client.aclose()


@pytest.fixture
async def deepseek_provider() -> AsyncGenerator[DeepSeekProvider, None]:
    """Create DeepSeek provider for testing.

    Uses deepseek-chat (cheapest model) for cost-effective testing.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")

    model = get_provider_model("deepseek")
    provider = DeepSeekProvider(
        api_key=api_key,
        model=model,
        timeout=120,
    )
    provider._selected_model = model

    yield provider

    if hasattr(provider, "client"):
        await provider.client.aclose()


@pytest.fixture
async def xai_provider() -> AsyncGenerator[XAIProvider, None]:
    """Create xAI (Grok) provider for testing.

    Uses grok-beta (cheapest model) for cost-effective testing.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        pytest.skip("XAI_API_KEY not set")

    model = get_provider_model("xai")
    provider = XAIProvider(
        api_key=api_key,
        model=model,
        timeout=120,
    )
    provider._selected_model = model

    yield provider

    if hasattr(provider, "client"):
        await provider.client.aclose()


@pytest.fixture
async def mistral_provider() -> AsyncGenerator[MistralProvider, None]:
    """Create Mistral provider for testing.

    Uses mistral-small-latest (cheapest, fast) for testing.
    Note: Mistral has a free tier (500K tokens/min).
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")

    model = get_provider_model("mistral")
    provider = MistralProvider(
        api_key=api_key,
        model=model,
        timeout=120,
    )
    provider._selected_model = model

    yield provider

    if hasattr(provider, "client"):
        await provider.client.aclose()


@pytest.fixture
async def openai_provider() -> AsyncGenerator[OpenAIProvider, None]:
    """Create OpenAI provider for testing.

    Uses gpt-4o-mini (cheapest model) for cost-effective testing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    model = get_provider_model("openai")
    provider = OpenAIProvider(
        api_key=api_key,
        model=model,
        timeout=120,
    )
    provider._selected_model = model

    yield provider

    if hasattr(provider, "client"):
        await provider.client.aclose()


@pytest.fixture
async def zai_provider() -> AsyncGenerator[ZAIProvider, None]:
    """Create ZAI provider for testing."""
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        pytest.skip("ZAI_API_KEY not set")

    model = get_provider_model("zai")
    provider = ZAIProvider(
        api_key=api_key,
        base_url="https://api.z.ai/api/paas/v4/",
        model=model,
        timeout=60,
    )
    provider._selected_model = model

    yield provider

    if hasattr(provider, "client"):
        await provider.client.aclose()


# =============================================================================
# Model Name Fixtures
# =============================================================================


@pytest.fixture
def ollama_model_name(ollama_provider: OllamaProvider) -> str:
    """Get the selected Ollama model name."""
    return getattr(ollama_provider, "_selected_model", "qwen2.5-coder:14b")


@pytest.fixture
def deepseek_model_name(deepseek_provider: DeepSeekProvider) -> str:
    """Get the selected DeepSeek model name."""
    return getattr(deepseek_provider, "_selected_model", "deepseek-chat")


@pytest.fixture
def xai_model_name(xai_provider: XAIProvider) -> str:
    """Get the selected xAI model name."""
    return getattr(xai_provider, "_selected_model", "grok-beta")


@pytest.fixture
def mistral_model_name(mistral_provider: MistralProvider) -> str:
    """Get the selected Mistral model name."""
    return getattr(mistral_provider, "_selected_model", "mistral-small-latest")


@pytest.fixture
def openai_model_name(openai_provider: OpenAIProvider) -> str:
    """Get the selected OpenAI model name."""
    return getattr(openai_provider, "_selected_model", "gpt-4o-mini")


@pytest.fixture
def zai_model_name(zai_provider: ZAIProvider) -> str:
    """Get the selected ZAI model name."""
    return getattr(zai_provider, "_selected_model", "glm-4-flash")


# =============================================================================
# Test Workspace Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace(tmp_path: Path) -> str:
    """Create temporary workspace for file operations."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return str(workspace)


@pytest.fixture
def sample_code_file(temp_workspace: str) -> str:
    """Create a sample Python file for testing."""
    file_path = Path(temp_workspace) / "sample.py"
    file_path.write_text(
        """
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    print(greet("World"))
    print(add(1, 2))
"""
    )
    return str(file_path)


@pytest.fixture
def sample_readme_file(temp_workspace: str) -> str:
    """Create a sample README file for testing."""
    file_path = Path(temp_workspace) / "README.md"
    file_path.write_text(
        """
# Sample Project

This is a sample project for testing.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

```bash
python sample.py
```
"""
    )
    return str(file_path)


# =============================================================================
# Timeout Configurations
# =============================================================================

TIMEOUT_SHORT = 30  # Simple queries, no tools
TIMEOUT_MEDIUM = 60  # Single tool calls
TIMEOUT_LONG = 120  # Multi-turn, multiple tools
TIMEOUT_XLONG = 180  # Complex workflows, file operations


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "real_execution: Mark test as real execution (no mocks)")
    config.addinivalue_line("markers", "cloud_provider: Mark test as cloud provider test")
    config.addinivalue_line("markers", "benchmark: Mark test as performance benchmark")
    config.addinivalue_line(
        "markers", "requires_provider(provider): Mark test as requiring specific provider"
    )


def pytest_collection_modifyitems(items, config):
    """Add skip markers to tests that require unavailable providers.

    This is called after test collection to dynamically add skip markers
    based on provider availability. This allows CI/CD to skip tests gracefully
    when providers or API keys are not available.
    """
    for item in items:
        # Skip real_execution tests if no provider is available
        if item.get_closest_marker("real_execution"):
            # Check if any provider is available
            has_provider = (
                is_ollama_running()
                or has_provider_api_key("deepseek")
                or has_provider_api_key("xai")
                or has_provider_api_key("mistral")
                or has_provider_api_key("openai")
                or has_provider_api_key("zai")
            )

            if not has_provider:
                item.add_marker(
                    pytest.mark.skip(
                        reason="No provider available. "
                        "Either run Ollama locally or set API keys for cloud providers: "
                        "DEEPSEEK_API_KEY, XAI_API_KEY, MISTRAL_API_KEY, OPENAI_API_KEY, ZAI_API_KEY"
                    )
                )

        # Check for provider-specific requirements
        requires_provider = item.get_closest_marker("requires_provider")
        if requires_provider:
            provider = requires_provider.args[0] if requires_provider.args else None
            if provider:
                provider_lower = provider.lower()
                if provider_lower == "ollama":
                    if not is_ollama_running():
                        item.add_marker(
                            pytest.mark.skip(
                                reason=f"Ollama not available. "
                                f"Install: brew install ollama && ollama serve"
                            )
                        )
                elif provider_lower in ["deepseek", "xai", "mistral", "openai", "zai"]:
                    if not has_provider_api_key(provider_lower):
                        env_var = provider_lower.upper() + "_API_KEY"
                        item.add_marker(
                            pytest.mark.skip(
                                reason=f"{env_var} not set. "
                                f"Set environment variable to run {provider} tests."
                            )
                        )

        # Skip cloud_provider tests if no cloud provider is available
        if item.get_closest_marker("cloud_provider"):
            has_cloud_provider = (
                has_provider_api_key("deepseek")
                or has_provider_api_key("xai")
                or has_provider_api_key("mistral")
                or has_provider_api_key("openai")
                or has_provider_api_key("zai")
            )

            if not has_cloud_provider:
                item.add_marker(
                    pytest.mark.skip(
                        reason="No cloud provider API keys available. "
                        "Set one of: DEEPSEEK_API_KEY, XAI_API_KEY, MISTRAL_API_KEY, "
                        "OPENAI_API_KEY, ZAI_API_KEY"
                    )
                )
