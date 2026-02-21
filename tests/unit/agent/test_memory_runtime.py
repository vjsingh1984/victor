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

from unittest.mock import MagicMock, patch

from victor.agent.runtime.memory_runtime import (
    create_memory_runtime_components,
    initialize_conversation_embedding_store,
)


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
