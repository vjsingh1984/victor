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
#
# Ollama models organized by speed/capability:
# - Ultra-fast (0.5B-3B): qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:3b, phi3:mini, gemma2:2b
# - Fast (7B-8B): qwen2.5-coder:7b, deepseek-coder:7b, phi3:3.8b, llama3.2:3b
# - Balanced (14B-20B): qwen2.5-coder:14b, deepseek-v3:32b, gpt-oss-tools:20b-64K
# - Capable (30B+): qwen3:32b, qwen3:72b, deepseek-v3:70b, glm-4:9b
#
# LM Studio models (user downloads, typical fast options):
# - Phi-3 models (3.8B): Very fast, good for simple tasks
# - Qwen2.5/Qwen3 smaller models (3B, 7B): Fast for coding
# - DeepSeek-Coder models (7B, 32B): Excellent for coding
# - Llama 3.2 models (1B, 3B): Ultra-fast
# - GLM-4 models (9B, 34B): Capable for complex tasks
#
# Complex 30B+ Category Models (Qwen3, DeepSeek, GLM):
# - qwen3:32b - Strong reasoning, tool use (Qwen3 > Qwen2.5)
# - qwen3:72b - Best quality for complex tasks
# - deepseek-v3:32b - Excellent coding capabilities
# - deepseek-r1:32b - Reasoning-optimized
# - glm-4:9b - Fast and capable (Zhipu AI)
# - glm-4:34b - Complex task specialist

PROVIDER_MODELS: Dict[str, List[str]] = {
    "ollama": [
        # === ULTRA-FAST MODELS (0.5B-3B) ===
        # Use for: Simple queries, error detection, basic file reads
        # Installed models from 'ollama list':
        "qwen2.5-coder:1.5b",  # 986 MB - FASTEST, great for simple tasks
        "llama3.2:latest",  # 2.0 GB - Fast, general purpose
        "mistral:latest",  # 4.1 GB - Fast, good quality
        # === FAST MODELS (7B-8B) ===
        # Use for: Coding tasks, syntax analysis, single tool calls
        # Installed models from 'ollama list':
        "qwen2.5-coder:7b",  # 4.7 GB - Excellent for coding
        "llama3.1:8b",  # 4.9 GB - Fast, capable
        "mistral:7b-instruct",  # 4.4 GB - Fast, good for tools
        "gemma3:12b",  # 8.1 GB - Fast coding model
        # === BALANCED MODELS (14B-20B) ===
        # Use for: Multi-step tasks, tool orchestration, complex reasoning
        # Installed models from 'ollama list':
        "deepseek-coder-v2:16b",  # 8.9 GB - Fast for coding + tools
        "qwen25-coder-tools:14b-64K",  # 9.0 GB - Tool-optimized 14B
        "deepseek-r1:14b",  # 9.0 GB - Reasoning-optimized 14B
        "qwen2.5-coder:14b",  # 9.0 GB - Balanced capability
        "gpt-oss-tools:20b-64K",  # 13 GB - Good tool support
        "phi4-reasoning:plus",  # 11 GB - Reasoning specialist
        # === CAPABLE MODELS (30B+) for Complex Tasks ===
        # Use for: Heavy tool orchestration, complex reasoning, multi-file operations
        # Installed models from 'ollama list':
        # Qwen3 series (newest, best for complex tasks)
        "qwen3-coder-tools:30b-64K",  # 18 GB - EXCELLENT for coding + tools
        "qwen3-coder-tools:30b",  # 18 GB - Qwen3 Coder 30B
        "qwen3-coder-tools:30b-262K",  # 18 GB - Qwen3 with 262K context
        "qwen3-coder-tools:30b-128k",  # 18 GB - Qwen3 with 128K context
        "qwen3:32b",  # 20 GB - Qwen3 32B general
        "qwen3:30b",  # 18 GB - Qwen3 30B general
        # DeepSeek series (excellent for coding and reasoning)
        "deepseek-coder:33b",  # 18 GB - DeepSeek Coder 33B
        "deepseek-coder-tools:33b",  # 18 GB - With tool optimizations
        "deepseek-coder-tools:33b-128K",  # 18 GB - 128K context
        "deepseek-coder-tools:33b-262K",  # 18 GB - 262K context
        "deepseek-coder-tools:33b-instruct",  # 18 GB - Instruction-tuned
        "deepseek-r1:32b",  # 19 GB - DeepSeek R1 reasoning
        "deepseek-r1-tools:32b",  # 19 GB - R1 with tool support
        "deepseek-r1-tools:32b-262K",  # 19 GB - R1 with long context
        # Other capable 30B+ models
        "gemma3:27b",  # 17 GB - Gemma3 27B
        "gemma3-tools:27b",  # 17 GB - Gemma3 with tools
        "gemma3-tools:27b-128K",  # 17 GB - Gemma3 128K context
        "mixtral:8x7b",  # 26 GB - Mixtral 8x7b Mixture of Experts
        # === LARGE MODELS (70B+) for Maximum Capability ===
        # Use only when: Maximum reasoning capability is required
        # Installed models from 'ollama list':
        "deepseek-r1:70b",  # 42 GB - Maximum reasoning
        "deepseek-r1-tools:70b-64K",  # 42 GB - With tools
        "deepseek-r1-tools:70b-96K",  # 42 GB - 96K context
        "deepseek-r1-tools:70b-128K",  # 42 GB - 128K context
        "llama3.1:70b",  # 42 GB - Llama 3.1 70B
        "llama3.3:70b",  # 42 GB - Llama 3.3 70B
        "llama3.1-tools:70b-64K",  # 42 GB - With tools
        "llama3.1-tools:70b-96K",  # 42 GB - 96K context
        "llama3.1-tools:70b-128K",  # 42 GB - 128K context
        "llama3.1-tools:70b-262K",  # 42 GB - 262K context
        "llama3.3-tools:70b-64K",  # 42 GB - Llama 3.3 with tools
        "llama3.3-tools:70b-96K",  # 42 GB - 96K context
    ],
    "lmstudio": [
        # Models available from LM Studio API at localhost:1234
        # Organized by speed/capability (fastest first)
        # === ULTRA-FAST MODELS (1B-3B) ===
        # Use for: Simple queries, error detection, basic file reads
        "qwen2.5-coder-1.5b",  # FASTEST, great for simple tasks
        "llama3.2",  # Fast, general purpose
        "mistral",  # Fast, good quality
        # === FAST MODELS (7B-12B) ===
        # Use for: Coding tasks, syntax analysis, single tool calls
        "qwen2.5-coder-7b",  # Excellent for coding
        "llama3.1-8b",  # Fast, capable
        "llama3.1-8b-instruct",  # Instruction-tuned 8B
        "mistral-7b-instruct",  # Fast, good for tools
        "gemma3-12b",  # Fast coding model
        "mistral-tools-7b-instruct",  # With tool optimizations
        # === BALANCED MODELS (14B-20B) ===
        # Use for: Multi-step tasks, tool orchestration, complex reasoning
        "deepseek-coder-v2-tools-16b",  # Fast for coding + tools
        "deepseek-coder-v2-tools-16b-64k",  # With 64K context
        "deepseek-coder-v2-16b",  # Capable 16B model
        "qwen25-coder-tools-14b-64k",  # Tool-optimized 14B
        "deepseek-r1-tools-14b-64k",  # Reasoning-optimized 14B
        "deepseek-r1-14b",  # DeepSeek R1 14B
        "qwen2.5-coder-14b",  # Balanced capability
        "gpt-oss-tools-20b-64k",  # Good tool support
        "gpt-oss-20b",  # GPT-OSS 20B
        "phi4-reasoning-tools-plus",  # Reasoning specialist
        "phi-4-16k-custom-tools",  # Phi-4 with tools
        "devstral-tools",  # Devstral with tools
        "mixtral-tools-8x7b",  # Mixtral MoE
        # === CAPABLE MODELS (30B+) for Complex Tasks ===
        # Use for: Heavy tool orchestration, complex reasoning, multi-file operations
        # Qwen3 series (newest, best for complex tasks)
        "qwen3-coder-tools-30b-64K",  # EXCELLENT for coding + tools
        "qwen3-coder-tools-30b-128k",  # With 128K context
        "qwen3-coder-tools-30b",  # Qwen3 Coder 30B
        "qwen3-coder-tools-30b-262k",  # With 262K context
        "qwen3-30b",  # Qwen3 30B general
        "qwen3-30b-a3b",  # A3B variant
        "qwen3-30b-40k-financial",  # Financial-specialized
        "qwen3-32b",  # Qwen3 32B general
        # DeepSeek series (excellent for coding and reasoning)
        "deepseek-coder-tools-33b",  # DeepSeek Coder 33B with tools
        "deepseek-coder-tools-33b-128k",  # With 128K context
        "deepseek-coder-tools-33b-262k",  # With 262K context
        "deepseek-coder-tools-33b-instruct",  # Instruction-tuned
        "deepseek-coder-33b",  # DeepSeek Coder 33B
        "deepseek-coder-33b-instruct",  # Instruction-tuned 33B
        "deepseek-r1-tools-32b",  # R1 with tool support
        "deepseek-r1-tools-32b-262k",  # R1 with long context
        "deepseek-r1-32b",  # DeepSeek R1 reasoning
        # Other capable 30B+ models
        "gemma3-tools-27b",  # Gemma3 27B with tools
        "gemma3-tools-27b-128k",  # With 128K context
        "gemma3-27b",  # Gemma3 27B
        "mixtral-tools-8x7b-65k",  # Mixtral with 65K context
        "mixtral-8x7b-32k",  # Mixtral with 32K context
        "mixtral-8x7b",  # Mixtral MoE general
        "qwen2.5-coder-tools-32b-262k",  # Qwen2.5 Coder with 262K context
        "qwen2.5-coder-32b",  # Qwen2.5 Coder 32B
        "qwen2.5-32b-instruct",  # Qwen2.5 32B instruction-tuned
        "codellama-34b-python",  # CodeLlama 34B Python
        "phi4-reasoning-plus",  # Phi-4 reasoning specialist
        # === LARGE MODELS (70B+) for Maximum Capability ===
        # Use only when: Maximum reasoning capability is required
        "deepseek-r1-70b",  # Maximum reasoning
        "deepseek-r1-tools-70b-64k",  # With tools
        "deepseek-r1-tools-70b-96k",  # With 96K context
        "deepseek-r1-tools-70b-128k",  # With 128K context
        "llama3.1-70b",  # Llama 3.1 70B
        "llama3.3-70b",  # Llama 3.3 70B
        "llama-3.3-70b-instruct-128k-custom",  # Custom 128K variant
        "llama3.1-tools-70b-64k",  # With tools
        "llama3.1-tools-70b-96k",  # With 96K context
        "llama3.1-tools-70b-128k",  # With 128K context
        "llama3.1-tools-70b-262k",  # With 262K context
        "llama3.3-tools-70b-64k",  # Llama 3.3 with tools
        "llama3.3-tools-70b-96k",  # With 96K context
        "llama3.3-tools-70b-128k",  # With 128K context
    ],
    "deepseek": [
        "deepseek-chat",  # Cheapest, good for general use (~$0.14/$0.28 per 1M)
        "deepseek-coder",  # For coding tasks (optimized for code)
        # DeepSeek V3 (newer, more capable)
        "deepseek-v3",  # Latest model, excellent for complex tasks
        "deepseek-r1",  # Reasoning-optimized model
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
        "glm-4-0520",  # Latest GLM-4 model
        "glm-4-long",  # Long context version (128K+)
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

    Uses fast available models for reliability (skips ultra-fast models < 7B):
    - Fast (7B-8B): qwen2.5-coder:7b, phi3:3.8b, llama3.1:8b - RECOMMENDED
    - Balanced (14B): qwen2.5-coder:14b, deepseek-coder-v2:16b - BETTER
    - Capable (20B+): gpt-oss-tools:20b-64K - BEST

    NOTE: Ultra-fast models (0.5B-3B) are skipped because they:
    - Hallucinate tools (confuse test names with tool names)
    - Have poor tool understanding
    - Produce unreliable results

    To install recommended models:
      ollama pull qwen2.5-coder:7b   # Fast for coding, ~4.7GB (RECOMMENDED)
      ollama pull qwen2.5-coder:14b  # Balanced capability, ~9GB
    """
    if not is_ollama_running():
        pytest.skip("Ollama not available at localhost:11434")

    # Try to find an available model (skip ultra-fast models < 7B for reliability)
    model = None
    for candidate_model in PROVIDER_MODELS["ollama"]:
        # Skip ultra-fast models (< 7B) - they hallucinate tools and are unreliable
        if any(
            skip in candidate_model.lower()
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
                "gemma2:2b",
            ]
        ):
            continue

        if is_ollama_model_available(candidate_model):
            model = candidate_model
            break

    if not model:
        pytest.skip(
            "No suitable Ollama model found (7B+ required). "
            "Install: ollama pull qwen2.5-coder:7b (4.7GB) "
            "or ollama pull qwen2.5-coder:14b (9GB)"
        )

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
                                reason="Ollama not available. "
                                "Install: brew install ollama && ollama serve"
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
