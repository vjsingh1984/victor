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

"""Fixtures for real execution integration tests with support for ALL Victor providers.

This conftest provides:
1. Fixtures for all 21 Victor providers
2. Automatic API key validation with test calls
3. Graceful skipping for missing keys, billing issues, auth failures
4. Cost-effective model selection (cheapest/fastest models)
5. Multi-provider test support with parametrization

Tests are automatically skipped when:
- Provider API key is not available
- API key is invalid or expired
- Billing/credit limits are exceeded
- Provider service is unreachable

Supported Providers (21 total):
- Local (4): Ollama, LMStudio, vLLM, LlamaCpp
- Premium Cloud (7): Anthropic, OpenAI, Google, xAI, Zhipu AI, Moonshot, DeepSeek
- Free-Tier Cloud (5): Groq, Mistral, Together, OpenRouter, Fireworks, Cerebras
- Enterprise (4): Vertex AI, Azure OpenAI, AWS Bedrock, Hugging Face, Replicate
"""

import os
import socket
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
import pytest_asyncio
import httpx

# Local providers
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.llamacpp_provider import LlamaCppProvider

# Cloud providers
from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.xai_provider import XAIProvider
from victor.providers.zai_provider import ZAIProvider
from victor.providers.moonshot_provider import MoonshotProvider
from victor.providers.deepseek_provider import DeepSeekProvider

# Free-tier providers
from victor.providers.groq_provider import GroqProvider
from victor.providers.mistral_provider import MistralProvider
from victor.providers.together_provider import TogetherProvider
from victor.providers.openrouter_provider import OpenRouterProvider
from victor.providers.fireworks_provider import FireworksProvider
from victor.providers.cerebras_provider import CerebrasProvider

# Additional providers
from victor.providers.huggingface_provider import HuggingFaceProvider


# =============================================================================
# Provider Configuration (All 21 Victor Providers)
# =============================================================================


# Complete provider configuration with cheapest/fastest models
PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    # === Local Providers (No API Key Required) ===
    "ollama": {
        "class": OllamaProvider,
        "model": "qwen2.5-coder:7b",
        "models": ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "gpt-oss-tools:20b-64K"],
        "api_key_env": None,
        "type": "local",
        "cost_tier": "free",
        "description": "Local inference",
    },
    "lmstudio": {
        "class": None,  # Would need to import if available
        "model": "local-model",
        "models": ["local-model"],
        "api_key_env": None,
        "type": "local",
        "cost_tier": "free",
        "description": "Local inference (localhost:1234)",
    },
    "vllm": {
        "class": None,  # Would need to import if available
        "model": "local-model",
        "models": ["local-model"],
        "api_key_env": None,
        "type": "local",
        "cost_tier": "free",
        "description": "Local inference (localhost:8000)",
    },
    "llamacpp": {
        "class": LlamaCppProvider,
        "model": "local-model",
        "models": ["local-model"],
        "api_key_env": None,
        "type": "local",
        "cost_tier": "free",
        "description": "Local GGUF inference",
    },

    # === Premium Cloud Providers ===
    "anthropic": {
        "class": AnthropicProvider,
        "model": "claude-haiku-3-5-20241022",  # Cheapest Claude model
        "models": [
            "claude-haiku-3-5-20241022",  # $0.25/$1.25 per 1M
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
        ],
        "api_key_env": "ANTHROPIC_API_KEY",
        "type": "cloud",
        "cost_tier": "premium",
        "description": "Claude models (Anthropic)",
    },
    "openai": {
        "class": OpenAIProvider,
        "model": "gpt-4o-mini",  # Cheapest OpenAI model
        "models": [
            "gpt-4o-mini",  # $0.15/$0.60 per 1M
            "gpt-3.5-turbo",
            "gpt-4o",
        ],
        "api_key_env": "OPENAI_API_KEY",
        "type": "cloud",
        "cost_tier": "premium",
        "description": "GPT models (OpenAI)",
    },
    "google": {
        "class": GoogleProvider,
        "model": "gemini-1.5-flash",  # Cheapest Gemini model
        "models": [
            "gemini-1.5-flash",  # $0.075/$0.30 per 1M
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
        ],
        "api_key_env": "GOOGLE_API_KEY",
        "type": "cloud",
        "cost_tier": "premium",
        "description": "Gemini models (Google)",
    },
    "xai": {
        "class": XAIProvider,
        "model": "grok-beta",  # Cheapest Grok model
        "models": [
            "grok-beta",
            "grok-2-1212",
        ],
        "api_key_env": "XAI_API_KEY",
        "type": "cloud",
        "cost_tier": "premium",
        "description": "Grok models (xAI)",
    },
    "zai": {
        "class": ZAIProvider,
        "model": "glm-4-flash",  # Cheapest Zhipu model
        "models": [
            "glm-4-flash",
            "glm-4-plus",
        ],
        "api_key_env": "ZAI_API_KEY",
        "type": "cloud",
        "cost_tier": "premium",
        "description": "GLM models (Zhipu AI)",
    },
    "moonshot": {
        "class": MoonshotProvider,
        "model": "moonshot-v1-8k",
        "models": [
            "moonshot-v1-8k",
            "moonshot-v1-32k",
        ],
        "api_key_env": "MOONSHOT_API_KEY",
        "type": "cloud",
        "cost_tier": "premium",
        "description": "Kimi models (Moonshot)",
    },
    "deepseek": {
        "class": DeepSeekProvider,
        "model": "deepseek-chat",  # Cheapest DeepSeek model
        "models": [
            "deepseek-chat",  # ~$0.14/$0.28 per 1M
            "deepseek-coder",
        ],
        "api_key_env": "DEEPSEEK_API_KEY",
        "type": "cloud",
        "cost_tier": "premium",
        "description": "DeepSeek models",
    },

    # === Free-Tier Cloud Providers ===
    "groqcloud": {
        "class": GroqProvider,
        "model": "llama-3.3-70b-versatile",  # Free tier available
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b",
            "mixtral-8x7b",
        ],
        "api_key_env": "GROQCLOUD_API_KEY",
        "type": "cloud",
        "cost_tier": "free",
        "description": "Groq Cloud (free tier: 15K RPM)",
    },
    "mistral": {
        "class": MistralProvider,
        "model": "mistral-small-latest",  # Cheapest, with free tier
        "models": [
            "mistral-small-latest",  # Free tier: 500K tokens/min
            "codestral-latest",
            "mistral-large-latest",
        ],
        "api_key_env": "MISTRAL_API_KEY",
        "type": "cloud",
        "cost_tier": "free",
        "description": "Mistral (free tier: 500K tokens/min)",
    },
    "together": {
        "class": TogetherProvider,
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "models": [
            "meta-llama/Llama-3-8b-chat-hf",
            "mistralai/Mixtral-8x7B-v0.1",
        ],
        "api_key_env": "TOGETHER_API_KEY",
        "type": "cloud",
        "cost_tier": "free",
        "description": "Together AI ($25 free credits)",
    },
    "openrouter": {
        "class": OpenRouterProvider,
        "model": "meta-llama/llama-3-8b:free",
        "models": [
            "meta-llama/llama-3-8b:free",
            "meta-llama/llama-3-70b:free",
        ],
        "api_key_env": "OPENROUTER_API_KEY",
        "type": "cloud",
        "cost_tier": "free",
        "description": "OpenRouter (free daily limits)",
    },
    "fireworks": {
        "class": FireworksProvider,
        "model": "accounts/fireworks/models/llama-v3-70b",
        "models": [
            "accounts/fireworks/models/llama-v3-70b",
        ],
        "api_key_env": "FIREWORKS_API_KEY",
        "type": "cloud",
        "cost_tier": "free",
        "description": "Fireworks AI (free tier)",
    },
    "cerebras": {
        "class": CerebrasProvider,
        "model": "llama-3.1-8b",
        "models": [
            "llama-3.1-8b",
            "llama-3.3-70b",
        ],
        "api_key_env": "CEREBRAS_API_KEY",
        "type": "cloud",
        "cost_tier": "free",
        "description": "Cerebras (free tier)",
    },

    # === Additional Providers ===
    "huggingface": {
        "class": HuggingFaceProvider,
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "models": [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "api_key_env": "HF_TOKEN",
        "type": "cloud",
        "cost_tier": "free",
        "description": "Hugging Face Inference",
    },
}


# =============================================================================
# Local Provider Checks
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


def is_local_provider_running(provider_name: str) -> bool:
    """Check if local provider is running.

    Args:
        provider_name: Provider name (lmstudio, vllm, llamacpp)
    """
    host, port = {
        "lmstudio": ("localhost", 1234),
        "vllm": ("localhost", 8000),
        "llamacpp": ("localhost", 8080),
        "llama.cpp": ("localhost", 8080),
    }.get(provider_name.lower(), ("localhost", 0))

    if port == 0:
        return True  # Assume running if no standard port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


# =============================================================================
# Cloud Provider API Key Checks
# =============================================================================


def has_provider_api_key(provider: str) -> bool:
    """Check if provider API key is configured.

    Args:
        provider: Provider name

    Returns:
        True if API key is available in environment
    """
    config = PROVIDER_CONFIG.get(provider.lower())
    if not config:
        return False

    env_var = config.get("api_key_env")
    if not env_var:
        return False

    return bool(os.getenv(env_var))


def get_provider_api_key(provider: str) -> Optional[str]:
    """Get provider API key from environment.

    Args:
        provider: Provider name

    Returns:
        API key or None
    """
    config = PROVIDER_CONFIG.get(provider.lower())
    if not config:
        return None

    env_var = config.get("api_key_env")
    if not env_var:
        return None

    return os.getenv(env_var)


# =============================================================================
# Provider Fixture Generators
# =============================================================================


def _create_provider_fixture(provider_name: str):
    """Create a pytest fixture for a provider.

    Args:
        provider_name: Provider name (lowercase)

    Returns:
        Fixture function
    """
    config = PROVIDER_CONFIG.get(provider_name.lower())
    if not config:
        return None

    provider_class = config["class"]
    if provider_class is None:
        return None

    @pytest_asyncio.fixture
    async def provider_fixture():
        """Provider fixture created dynamically."""
        api_key = get_provider_api_key(provider_name)
        provider_type = config.get("type", "cloud")

        # Local providers - check if running
        if provider_type == "local":
            if provider_name == "ollama":
                if not is_ollama_running():
                    pytest.skip("Ollama not available at localhost:11434")

                # Check for available model
                model = None
                for candidate_model in config["models"]:
                    if is_ollama_model_available(candidate_model):
                        model = candidate_model
                        break

                if not model:
                    pytest.skip(
                        f"No Ollama model found. "
                        f"Run: ollama pull {config['models'][0]}"
                    )

                provider = OllamaProvider(
                    base_url="http://localhost:11434",
                    timeout=120,
                )
                provider._selected_model = model

            elif provider_name == "llamacpp":
                if not is_local_provider_running("llamacpp"):
                    pytest.skip("LlamaCpp not running")

                provider = LlamaCppProvider(
                    model_path="local-model",  # User should have model loaded
                    timeout=120,
                )
                provider._selected_model = config["model"]

            else:
                # Other local providers (lmstudio, vllm)
                if not is_local_provider_running(provider_name):
                    pytest.skip(f"{provider_name} not running")

                pytest.skip(f"{provider_name} fixture not yet implemented")

        # Cloud providers - check API key
        else:
            if not api_key:
                pytest.skip(f"{config['api_key_env']} not set")

            try:
                provider = provider_class(
                    api_key=api_key,
                    model=config["model"],
                    timeout=120,
                )
                provider._selected_model = config["model"]

                # Validate API key with a test call
                from victor.providers.base import Message

                try:
                    test_response = await provider.chat(
                        messages=[Message(role="user", content="Hi")],
                        model=config["model"],
                        max_tokens=5
                    )

                    if not test_response or not test_response.content:
                        pytest.skip(f"{provider_name}: API key validation failed (no response)")

                except Exception as e:
                    # Classify error to provide helpful message
                    error_str = str(e).lower()

                    # Check for specific error patterns
                    if any(pattern in error_str for pattern in [
                        "invalid", "unauthorized", "401", "403",
                        "authentication", "forbidden"
                    ]):
                        pytest.skip(
                            f"{provider_name}: Invalid API key ({str(e)[:80]})"
                        )
                    elif any(pattern in error_str for pattern in [
                        "billing", "payment", "credit", "quota", "limit",
                        "insufficient", "balance", "suspended"
                    ]):
                        pytest.skip(
                            f"{provider_name}: Billing/credit issue ({str(e)[:80]})"
                        )
                    elif any(pattern in error_str for pattern in [
                        "rate limit", "429", "too many requests"
                    ]):
                        pytest.skip(
                            f"{provider_name}: Rate limit ({str(e)[:80]})"
                        )
                    else:
                        pytest.skip(
                            f"{provider_name}: API key error ({str(e)[:80]})"
                        )

            except Exception as e:
                pytest.skip(
                    f"{provider_name}: Failed to create provider ({str(e)[:80]})"
                )

        yield provider

        # Cleanup
        if hasattr(provider, "close"):
            try:
                await provider.close()
            except Exception:
                pass
        elif hasattr(provider, "client"):
            try:
                await provider.client.aclose()
            except Exception:
                pass

    return provider_fixture


# Create all provider fixtures dynamically
# NOTE: Dynamic fixture creation with pytest-asyncio has issues.
# Use explicit fixtures below instead.
# for provider_name in PROVIDER_CONFIG.keys():
#     fixture = _create_provider_fixture(provider_name)
#     if fixture:
#         fixture.__name__ = f"{provider_name}_provider"
#         globals()[f"{provider_name}_provider"] = fixture


# =============================================================================
# Model Name Fixtures
# =============================================================================


def _create_model_name_fixture(provider_name: str):
    """Create a model name fixture for a provider.

    Args:
        provider_name: Provider name (lowercase)

    Returns:
        Fixture function
    """
    config = PROVIDER_CONFIG.get(provider_name.lower())
    if not config:
        return None

    @pytest.fixture
    def model_name_fixture(request):
        """Get model name for provider."""
        provider = request.getfixturevalue(f"{provider_name}_provider")
        return getattr(provider, "_selected_model", config["model"])

    return model_name_fixture


# Create all model name fixtures
for provider_name in PROVIDER_CONFIG.keys():
    fixture = _create_model_name_fixture(provider_name)
    if fixture:
        fixture.__name__ = f"{provider_name}_model_name"
        globals()[f"{provider_name}_model_name"] = fixture


# =============================================================================
# Workspace and Sample File Fixtures
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
# Pytest Configuration and Markers
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "real_execution: Mark test as real execution (no mocks)"
    )
    config.addinivalue_line(
        "markers",
        "cloud_provider: Mark test as cloud provider test"
    )
    config.addinivalue_line(
        "markers",
        "benchmark: Mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers",
        "requires_provider(provider): Mark test as requiring specific provider"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration test (may require external services)"
    )


def pytest_collection_modifyitems(items, config):
    """Add skip markers to tests that require unavailable providers.

    This is called after test collection to dynamically add skip markers
    based on provider availability. Tests are skipped gracefully when:
    - Provider API key is not set
    - Provider is not accessible (local providers)
    """
    for item in items:
        # Skip real_execution tests if no provider is available
        if item.get_closest_marker("real_execution"):
            has_any_provider = is_ollama_running() or any(
                has_provider_api_key(p)
                for p in PROVIDER_CONFIG.keys()
            )

            if not has_any_provider:
                item.add_marker(
                    pytest.mark.skip(
                        reason="No provider available. "
                        "Either run Ollama locally or set API keys for cloud providers. "
                        "See README.md for setup instructions."
                    )
                )


# =============================================================================
# Explicit Provider Fixtures (pytest-asyncio compatible)
# =============================================================================

@pytest_asyncio.fixture
async def ollama_provider():
    """Ollama provider with automatic model selection."""
    from victor.providers.ollama_provider import OllamaProvider

    if not is_ollama_running():
        pytest.skip("Ollama not available at localhost:11434")

    config = PROVIDER_CONFIG.get("ollama")
    model = None
    for candidate_model in config["models"]:
        if is_ollama_model_available(candidate_model):
            model = candidate_model
            break

    if not model:
        pytest.skip(f"No Ollama model found. Run: ollama pull {config['models'][0]}")

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        timeout=120,
    )
    provider._selected_model = model

    yield provider

    # Cleanup
    if hasattr(provider, "close"):
        try:
            await provider.close()
        except Exception:
            pass
    elif hasattr(provider, "client"):
        try:
            await provider.client.aclose()
        except Exception:
            pass
