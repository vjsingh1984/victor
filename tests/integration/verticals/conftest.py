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

"""Local conftest for vertical integration tests.

This conftest provides minimal fixtures without the complex
auto-mocking that causes issues in the main conftest.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from victor.config.settings import Settings

    settings = Settings()
    return settings


@pytest.fixture
def mock_provider():
    """Mock provider for testing."""
    provider = MagicMock()
    provider.chat = MagicMock(return_value=MagicMock(
        content="Test response",
        usage=MagicMock(total_tokens=100)
    ))
    provider.stream_chat = MagicMock()
    provider.supports_tools = MagicMock(return_value=True)
    provider.name = "test_provider"
    return provider


@pytest.fixture
def mock_container():
    """Mock DI container for testing."""
    from victor.core.container import ServiceContainer

    container = ServiceContainer()
    return container


@pytest.fixture
async def cross_vertical_orchestrator(mock_settings, mock_provider, mock_container):
    """Create orchestrator with multiple verticals enabled."""
    from victor.agent.orchestrator_factory import OrchestratorFactory

    factory = OrchestratorFactory(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-sonnet-4-5",
        temperature=0.7,
        max_tokens=4096,
    )

    orchestrator = factory.create_orchestrator()
    return orchestrator
