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

"""Fixtures for real execution integration tests."""

import os
import socket
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import httpx

from victor.providers.ollama_provider import OllamaProvider
from victor.providers.zai_provider import ZAIProvider


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
        import httpx

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


def has_zai_api_key() -> bool:
    """Check if ZAI API key is configured."""
    return bool(os.getenv("ZAI_API_KEY"))


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Check if Ollama is available for testing."""
    return is_ollama_running()


@pytest.fixture(scope="session")
def ollama_model_available() -> bool:
    """Check if Ollama model is available for testing."""
    return (
        is_ollama_model_available("qwen2.5-coder:14b")
        or is_ollama_model_available("gpt-oss-tools:20b-64K")
        or is_ollama_model_available("qwen2.5-coder:7b")
    )


@pytest.fixture(scope="session")
def zai_available() -> bool:
    """Check if ZAI API key is available."""
    return has_zai_api_key()


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

    # Try to find an available model (prioritize speed over size)
    model = None
    for candidate_model in [
        "qwen2.5-coder:14b",  # Fast and capable
        "gpt-oss-tools:20b-64K",  # Alternative with good tool support
        "qwen2.5-coder:7b",  # Fastest fallback
    ]:
        if is_ollama_model_available(candidate_model):
            model = candidate_model
            break

    if not model:
        pytest.skip("No suitable Ollama model found. Run: ollama pull qwen2.5-coder:14b")

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        timeout=120,  # 2 minutes for commodity hardware
    )

    # Store the selected model for tests to use
    provider._selected_model = model

    yield provider

    # Cleanup
    if hasattr(provider, "client"):
        await provider.client.aclose()


@pytest.fixture
def ollama_model_name(ollama_provider: OllamaProvider) -> str:
    """Get the selected Ollama model name for testing.

    Uses the same model selection logic as ollama_provider fixture.
    Tests should use this fixture instead of hardcoding model names.

    Example:
        settings.model = ollama_model_name
    """
    return getattr(ollama_provider, "_selected_model", "qwen2.5-coder:14b")


@pytest.fixture
async def zai_provider() -> AsyncGenerator[ZAIProvider, None]:
    """Create ZAI provider for testing."""
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        pytest.skip("ZAI_API_KEY not set")

    provider = ZAIProvider(
        api_key=api_key,
        base_url="https://api.z.ai/api/paas/v4/",
        timeout=60,
    )

    yield provider

    # Cleanup
    if hasattr(provider, "client"):
        await provider.client.aclose()


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


# Timeout configurations for commodity hardware (M1 Max)
TIMEOUT_SHORT = 30  # Simple queries, no tools
TIMEOUT_MEDIUM = 60  # Single tool calls
TIMEOUT_LONG = 120  # Multi-turn, multiple tools
TIMEOUT_XLONG = 180  # Complex workflows, file operations


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "real_execution: Mark test as real execution (no mocks)")
    config.addinivalue_line("markers", "cloud_provider: Mark test as cloud provider test")
    config.addinivalue_line("markers", "benchmark: Mark test as performance benchmark")


def pytest_collection_modifyitems(items, config):
    """Add skip markers to tests that require unavailable providers.

    This is called after test collection to dynamically add skip markers
    based on provider availability. This allows CI/CD to skip tests gracefully
    when Ollama or models are not available.
    """
    for item in items:
        # Skip real_execution tests if Ollama not available
        if item.get_closest_marker("real_execution"):
            if not is_ollama_running():
                item.add_marker(
                    pytest.mark.skip(
                        reason="Ollama not available at localhost:11434. "
                        "Install: brew install ollama && ollama serve && ollama pull qwen2.5-coder:14b"
                    )
                )
            elif (
                not is_ollama_model_available("qwen2.5-coder:14b")
                and not is_ollama_model_available("gpt-oss-tools:20b-64K")
                and not is_ollama_model_available("qwen2.5-coder:7b")
            ):
                item.add_marker(
                    pytest.mark.skip(
                        reason="Ollama model not available. " "Run: ollama pull qwen2.5-coder:14b"
                    )
                )

        # Skip cloud_provider tests if ZAI not available
        if item.get_closest_marker("cloud_provider"):
            if not has_zai_api_key():
                item.add_marker(
                    pytest.mark.skip(
                        reason="ZAI_API_KEY not set. "
                        "Set environment variable to run cloud provider tests."
                    )
                )
