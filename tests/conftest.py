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

"""Shared pytest fixtures and configuration."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_code_execution_manager():
    """Mock CodeExecutionManager to avoid Docker startup during tests."""
    with patch("victor.agent.orchestrator.CodeExecutionManager") as mock_cem:
        mock_instance = MagicMock()
        mock_instance.start.return_value = None
        mock_instance.stop.return_value = None
        mock_instance.docker_available = False
        mock_instance.container = None
        mock_cem.return_value = mock_instance
        yield mock_cem


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for tests that need Docker functionality."""
    with patch("docker.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        yield mock_client


@pytest.fixture(autouse=True)
def isolate_environment_variables(monkeypatch):
    """Isolate tests from environment variables and .env files.

    This fixture prevents tests from loading actual API keys from:
    - Environment variables
    - .env files
    - System keyring
    - profiles.yaml

    This ensures tests are deterministic and don't leak credentials.
    """
    # Mock env file loading to prevent .env file from being loaded
    monkeypatch.setenv("VICTOR_SKIP_ENV_FILE", "1")

    # Clear API key environment variables
    api_key_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "MOONSHOT_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "VICTOR_ANTHROPIC_KEY",
        "VICTOR_GOOGLE_KEY",
        "VICTOR_OPENAI_KEY",
    ]

    for var in api_key_vars:
        monkeypatch.delenv(var, raising=False)

    # Mock the API key manager to return None for all providers
    monkeypatch.setattr("victor.config.api_keys.get_api_key", lambda provider: None)


@pytest.fixture(autouse=True)
def auto_mock_docker_for_orchestrator(request):
    """Automatically mock Docker for tests that create AgentOrchestrator.

    This fixture is auto-used but only applies mocking when the test
    fixture requests 'orchestrator' or matches certain patterns.
    """
    # Check if test needs orchestrator
    test_name = request.node.name
    test_path = str(request.node.fspath)

    # Tests that create orchestrator need Docker mocking
    needs_mock = any(
        [
            "orchestrator" in test_name.lower(),
            "tool_selection" in test_path,
            "tool_cache" in test_path,
            "goal_inference" in test_path,
            "tool_dependency" in test_path,
            "tool_call_matrix" in test_path,
            "thinking_mode" in test_path,
            "model_capability" in test_path,
            "test_orchestrator" in test_path,
            "integration" in test_path,
            "test_file_editor_tool.py" in test_path,
        ]
    )

    if needs_mock:
        with patch("victor.agent.orchestrator.CodeExecutionManager") as mock_cem:
            mock_instance = MagicMock()
            mock_instance.start.return_value = None
            mock_instance.stop.return_value = None
            mock_instance.docker_available = False
            mock_instance.container = None
            mock_cem.return_value = mock_instance
            yield
    else:
        yield
