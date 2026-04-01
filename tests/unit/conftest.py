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

"""Pytest fixtures for unit tests."""

import logging

import pytest


@pytest.fixture(autouse=True)
def reset_api_keys_logger():
    """Reset the api_keys logger to ensure log propagation works for caplog.

    This fixes test isolation issues where prior tests may have configured
    the logger differently (e.g., added handlers, changed propagate flag).
    """
    # Get the logger used by api_keys module
    logger = logging.getLogger("victor.config.api_keys")

    # Store original state
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    original_propagate = logger.propagate

    # Reset to clean state for test (propagate=True ensures caplog captures)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.DEBUG)

    yield

    # Restore original state after test
    logger.handlers = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate


@pytest.fixture(autouse=True)
def reset_embedding_singleton():
    """Reset EmbeddingService singleton between tests to prevent state leakage.

    The singleton pattern means first caller wins model choice. Without
    this fixture, test ordering can cause stale embeddings from a different
    model to be returned.
    """
    yield
    # Only import and reset if the module was loaded (avoid importing heavy deps)
    import sys

    if "victor.storage.embeddings.service" in sys.modules:
        from victor.storage.embeddings.service import EmbeddingService

        if EmbeddingService._instance is not None:
            EmbeddingService.reset_instance()


@pytest.fixture(autouse=True)
def reset_global_state_manager():
    """Reset GlobalStateManager singleton between tests to prevent state leakage."""
    yield
    import sys

    if "victor.state.factory" in sys.modules:
        from victor.state.factory import reset_global_manager

        reset_global_manager()


@pytest.fixture(autouse=True)
def isolate_environment_variables():
    """Isolate environment variables between tests.

    Clears API key env vars and mocks get_api_key to prevent
    tests from loading real credentials.
    """
    import os
    from unittest.mock import patch

    api_key_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "XAI_API_KEY",
        "REPLICATE_API_TOKEN",
        "HUGGINGFACE_API_KEY",
    ]
    saved = {k: os.environ.pop(k, None) for k in api_key_vars}

    with patch("victor.config.api_keys.get_api_key", return_value=None):
        yield

    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def reset_service_container():
    """Reset DI service container between tests to prevent state leakage."""
    yield
    import sys

    if "victor.core.bootstrap" in sys.modules:
        try:
            from victor.core.bootstrap import reset_container

            reset_container()
        except ImportError:
            pass  # Module not loaded, nothing to reset
