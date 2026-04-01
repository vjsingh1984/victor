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

from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.coordinators.session_coordinator import SessionCoordinator
from victor.agent.runtime.memory_runtime import (
    create_memory_runtime_components,
    initialize_conversation_embedding_store,
)
from victor.core.async_utils import run_sync as real_run_sync


def test_create_memory_runtime_components_delegates_to_factory():
    factory = MagicMock()
    memory_manager = object()
    factory.create_memory_components.return_value = (memory_manager, "session-123")

    components = create_memory_runtime_components(
        factory=factory,
        provider_name="ollama",
        native_tool_calls=True,
    )

    factory.create_memory_components.assert_called_once_with("ollama", True)
    assert components.memory_manager is memory_manager
    assert components.memory_session_id == "session-123"


def test_initialize_conversation_embedding_store_delegates_to_session_coordinator():
    memory_manager = object()
    expected_store = object()
    expected_cache = object()

    with patch(
        "victor.agent.coordinators.session_coordinator.SessionCoordinator.init_conversation_embedding_store"
    ) as init_store:
        init_store.return_value = (expected_store, expected_cache)
        store, cache = initialize_conversation_embedding_store(memory_manager=memory_manager)

    init_store.assert_called_once_with(memory_manager=memory_manager)
    assert store is expected_store
    assert cache is expected_cache


def test_session_coordinator_init_embedding_store_bridges_sync_initialization():
    memory_manager = MagicMock()
    embedding_service = object()
    store = MagicMock()
    store.is_initialized = False
    store.initialize = AsyncMock(return_value=None)
    semantic_cache = object()

    with (
        patch(
            "victor.storage.embeddings.service.EmbeddingService.get_instance",
            return_value=embedding_service,
        ),
        patch(
            "victor.agent.conversation_embedding_store.ConversationEmbeddingStore",
            return_value=store,
        ),
        patch("victor.agent.conversation_embedding_store._embedding_store", None),
        patch("victor.agent.tool_result_cache.ToolResultCache", return_value=semantic_cache),
        patch(
            "victor.agent.coordinators.session_coordinator.run_sync",
            wraps=real_run_sync,
        ) as run_sync_mock,
    ):
        result_store, result_cache = SessionCoordinator.init_conversation_embedding_store(
            memory_manager=memory_manager
        )

    assert result_store is store
    assert result_cache is semantic_cache
    memory_manager.set_embedding_store.assert_called_once_with(store)
    memory_manager.set_embedding_service.assert_called_once_with(embedding_service)
    run_sync_mock.assert_called_once()
    store.initialize.assert_awaited_once_with()


def test_session_coordinator_init_embedding_store_schedules_on_running_loop():
    memory_manager = MagicMock()
    embedding_service = object()
    store = MagicMock()
    store.is_initialized = False
    store.initialize = AsyncMock(return_value=None)
    semantic_cache = object()
    scheduled = []
    loop = MagicMock()

    def capture_task(coro):
        scheduled.append(coro)
        coro.close()
        return MagicMock()

    loop.create_task.side_effect = capture_task

    with (
        patch(
            "victor.storage.embeddings.service.EmbeddingService.get_instance",
            return_value=embedding_service,
        ),
        patch(
            "victor.agent.conversation_embedding_store.ConversationEmbeddingStore",
            return_value=store,
        ),
        patch("victor.agent.conversation_embedding_store._embedding_store", None),
        patch("victor.agent.tool_result_cache.ToolResultCache", return_value=semantic_cache),
        patch(
            "victor.agent.coordinators.session_coordinator.asyncio.get_running_loop",
            return_value=loop,
        ),
        patch("victor.agent.coordinators.session_coordinator.run_sync") as run_sync_mock,
    ):
        result_store, result_cache = SessionCoordinator.init_conversation_embedding_store(
            memory_manager=memory_manager
        )

    assert result_store is store
    assert result_cache is semantic_cache
    assert len(scheduled) == 1
    run_sync_mock.assert_not_called()
    store.initialize.assert_called_once_with()
