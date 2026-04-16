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
import os
from pathlib import Path

import pytest

_UNIT_TEST_REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe_current_working_directory() -> Path:
    """Return a valid current working directory, recovering from deleted cwd state."""
    try:
        return Path.cwd()
    except FileNotFoundError:
        os.chdir(_UNIT_TEST_REPO_ROOT)
        return _UNIT_TEST_REPO_ROOT


@pytest.fixture(autouse=True)
def isolate_working_directory():
    """Ensure each unit test starts and ends with a valid working directory.

    Some tests chdir into temporary directories that are later deleted. When that
    leaks into the next test, imports and subprocess helpers can fail with
    FileNotFoundError during collection or runtime. Restore a safe directory after
    every test, defaulting to the repo root if the original cwd vanished.
    """
    original_cwd = _safe_current_working_directory()

    yield

    target = original_cwd if original_cwd.exists() else _UNIT_TEST_REPO_ROOT
    current_cwd = _safe_current_working_directory()
    if current_cwd != target:
        os.chdir(target)


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

    # Reset via container module directly (not bootstrap which may not be imported)
    if "victor.core.container" in sys.modules:
        try:
            from victor.core.container import reset_container

            reset_container()
        except ImportError:
            pass  # Module not loaded, nothing to reset
    elif "victor.core.bootstrap" in sys.modules:
        try:
            from victor.core.bootstrap import reset_container

            reset_container()
        except ImportError:
            pass


@pytest.fixture(autouse=True)
def reset_plugin_registry():
    """Reset PluginRegistry between tests to prevent plugin state leakage.

    The PluginRegistry is a singleton that caches discovered plugins.
    Without reset, the first test triggering discovery pollutes the
    registry for all subsequent tests in the same xdist worker.
    """
    yield
    import sys

    if "victor.core.plugins.registry" in sys.modules:
        try:
            from victor.core.plugins.registry import PluginRegistry

            PluginRegistry._instance = None
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_capability_registry():
    """Reset CapabilityRegistry between tests to prevent capability leakage.

    The CapabilityRegistry singleton caches registered capabilities.
    Without reset, tests that register an EditorProtocol (via vertical
    loading) pollute the registry for file_editor_tool tests.
    """
    yield
    import sys

    if "victor.core.capability_registry" in sys.modules:
        try:
            from victor.core.capability_registry import CapabilityRegistry

            instance = CapabilityRegistry.get_instance()
            if hasattr(instance, "_capabilities"):
                instance._capabilities.clear()
            if hasattr(instance, "_enhanced"):
                instance._enhanced.clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_vertical_registry():
    """Save and restore VerticalRegistry between tests.

    Full snapshot/restore prevents both pollution (new keys leaking) and
    degradation (prior tests deleting built-in entries).
    """
    import sys

    saved = None
    if "victor.core.verticals.base" in sys.modules:
        try:
            from victor.core.verticals.base import VerticalRegistry

            saved = dict(VerticalRegistry._registry)
        except (ImportError, AttributeError):
            pass
    yield
    if "victor.core.verticals.base" in sys.modules:
        try:
            from victor.core.verticals.base import VerticalRegistry

            if saved is not None:
                VerticalRegistry._registry.clear()
                VerticalRegistry._registry.update(saved)
            else:
                # Module loaded during test — remove everything it added
                VerticalRegistry._registry.clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_feature_flags():
    """Reset FeatureFlagManager singleton between tests."""
    yield
    import sys

    if "victor.core.feature_flags" in sys.modules:
        try:
            from victor.core.feature_flags import get_feature_flag_manager

            mgr = get_feature_flag_manager()
            if hasattr(mgr, "_instance"):
                type(mgr)._instance = None
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def isolate_conversation_database(tmp_path):
    """Redirect conversation.db to a temp directory during tests.

    Without this, any code path that instantiates ConversationStore
    without an explicit db_path will write to the real project .victor/
    directory, polluting it with test-model sessions (37K+ observed).

    Only patches the conversation_db property — other ProjectPaths
    properties (MCP config, context files) still resolve normally.
    """
    from unittest.mock import PropertyMock, patch

    from victor.config.settings import ProjectPaths

    temp_victor = tmp_path / ".victor"
    temp_victor.mkdir(exist_ok=True)
    temp_conv_db = temp_victor / "conversation.db"

    with patch.object(
        ProjectPaths,
        "conversation_db",
        new_callable=PropertyMock,
        return_value=temp_conv_db,
    ):
        yield


@pytest.fixture(autouse=True)
def reset_extension_cache():
    """Reset VerticalExtensionLoader caches between tests."""
    yield
    import sys

    if "victor.core.verticals.extension_loader" in sys.modules:
        try:
            from victor.core.verticals.extension_loader import VerticalExtensionLoader

            VerticalExtensionLoader._cache_manager.clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_change_tracker():
    """Reset FileChangeHistory singleton between tests."""
    yield
    import sys

    if "victor.agent.change_tracker" in sys.modules:
        try:
            from victor.agent.change_tracker import reset_change_tracker

            reset_change_tracker()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_mode_controller():
    """Reset AgentModeController singleton between tests."""
    yield
    import sys

    if "victor.agent.mode_controller" in sys.modules:
        try:
            from victor.agent.mode_controller import reset_mode_controller

            reset_mode_controller()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_rl_coordinator():
    """Reset RLCoordinator singleton between tests."""
    yield
    import sys

    if "victor.framework.rl.coordinator" in sys.modules:
        try:
            from victor.framework.rl.coordinator import reset_rl_coordinator

            reset_rl_coordinator()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_snapshot_store():
    """Reset FileSnapshotStore singleton between tests."""
    yield
    import sys

    if "victor.agent.snapshot_store" in sys.modules:
        try:
            from victor.agent.snapshot_store import reset_snapshot_store

            reset_snapshot_store()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_tool_cache_manager():
    """Reset ToolCacheManager singleton between tests."""
    yield
    import sys

    if "victor.tools.cache_manager" in sys.modules:
        try:
            from victor.tools.cache_manager import reset_tool_cache_manager

            reset_tool_cache_manager()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_tool_result_cache():
    """Reset ToolResultCache singleton between tests."""
    yield
    import sys

    if "victor.agent.tool_result_cache" in sys.modules:
        try:
            from victor.agent.tool_result_cache import reset_tool_cache

            reset_tool_cache()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_task_analyzer():
    """Reset TaskAnalyzer singleton between tests."""
    yield
    import sys

    if "victor.agent.task_analyzer" in sys.modules:
        try:
            from victor.agent.task_analyzer import reset_task_analyzer

            reset_task_analyzer()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_compiled_graph_cache():
    """Reset CompiledGraphCache singleton between tests."""
    yield
    import sys

    if "victor.framework.graph_cache" in sys.modules:
        try:
            from victor.framework.graph_cache import reset_compiled_graph_cache

            reset_compiled_graph_cache()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_tool_metadata_registry():
    """Reset ToolMetadataRegistry singleton between tests."""
    yield
    import sys

    if "victor.tools.metadata_registry" in sys.modules:
        try:
            from victor.tools.metadata_registry import reset_global_registry

            reset_global_registry()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def cleanup_dangling_asyncio_tasks():
    """Cancel leftover asyncio tasks after each test to prevent xdist hangs.

    Tests that create InMemoryEventBackend instances with connect() spawn
    background dispatch tasks. If not properly disconnected, these tasks
    keep the event loop alive, hanging xdist worker shutdown.
    """
    yield
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return  # Can't cancel tasks from inside a running loop
        if loop.is_closed():
            return
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                task.cancel()
    except RuntimeError:
        pass  # No event loop


@pytest.fixture(autouse=True, scope="session")
def set_test_mode():
    """Set TEST_MODE environment variable to redirect test telemetry.

    This fixture runs once per test session and sets TEST_MODE to prevent
    MagicMock events and test telemetry from polluting the global usage.jsonl
    file. Test events will be redirected to /tmp/victor_test_telemetry/test_usage.jsonl
    instead.

    Priority: P1 - Prevents MagicMock leakage in global logs (HIGH criticality)
    """
    import os

    original = os.environ.get("TEST_MODE")
    os.environ["TEST_MODE"] = "1"

    yield

    if original is not None:
        os.environ["TEST_MODE"] = original
    else:
        os.environ.pop("TEST_MODE", None)
