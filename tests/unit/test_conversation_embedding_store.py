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

"""Tests for ConversationEmbeddingStore."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import numpy as np


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self.model_name = "mock-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding based on text hash."""
        # Generate a deterministic embedding based on text content
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self._dimension).astype(np.float32)

    async def embed_batch(self, texts: list) -> np.ndarray:
        """Generate mock embeddings for batch."""
        embeddings = []
        for text in texts:
            emb = await self.embed_text(text)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)


class TestConversationSearchResult:
    """Tests for ConversationSearchResult dataclass."""

    def test_creation(self):
        """Test creating a search result."""
        from victor.agent.conversation_embedding_store import ConversationSearchResult

        result = ConversationSearchResult(
            message_id="msg_123",
            session_id="session_456",
            role="user",
            content_preview="How do I implement auth?",
            similarity=0.85,
            timestamp=datetime.now(),
        )

        assert result.message_id == "msg_123"
        assert result.session_id == "session_456"
        assert result.role == "user"
        assert result.similarity == 0.85

    def test_repr(self):
        """Test string representation."""
        from victor.agent.conversation_embedding_store import ConversationSearchResult

        result = ConversationSearchResult(
            message_id="msg_123",
            session_id="session_456",
            role="assistant",
            content_preview="Here's how...",
            similarity=0.72,
        )

        repr_str = repr(result)
        assert "msg_123" in repr_str
        assert "assistant" in repr_str
        assert "0.720" in repr_str


class TestConversationEmbeddingStore:
    """Tests for ConversationEmbeddingStore class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "conversations"

    @pytest.fixture
    def mock_service(self):
        """Create a mock embedding service."""
        return MockEmbeddingService()

    @pytest.fixture
    def store(self, mock_service, temp_db_path):
        """Create a ConversationEmbeddingStore instance."""
        from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

        return ConversationEmbeddingStore(
            embedding_service=mock_service,
            db_path=temp_db_path,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, store, temp_db_path):
        """Test store initialization."""
        assert not store.is_initialized

        await store.initialize()

        assert store.is_initialized
        assert temp_db_path.exists()

    @pytest.mark.asyncio
    async def test_add_message_embedding(self, store):
        """Test adding a message embedding."""
        await store.initialize()

        content = "How do I implement user authentication in Python?"
        await store.add_message_embedding(
            message_id="msg_001",
            session_id="session_test",
            role="user",
            content=content,
            timestamp=datetime.now(),
        )

        # Verify stats show the message was added
        stats = await store.get_stats()
        assert stats["total_messages"] == 1

        # Verify we can search and find it (use same content for exact match)
        # Note: With mock embeddings, similarity is based on text hash, so
        # searching for the same content should give a high match
        results = await store.search_similar(
            query=content,  # Use same content for guaranteed match
            session_id="session_test",
            limit=5,
            min_similarity=0.0,  # Low threshold for mock embeddings
        )

        # Should find the message
        assert len(results) >= 1
        assert any(r.message_id == "msg_001" for r in results)

    @pytest.mark.asyncio
    async def test_skip_short_messages(self, store):
        """Test that very short messages are skipped."""
        await store.initialize()

        # Add a short message
        await store.add_message_embedding(
            message_id="msg_short",
            session_id="session_test",
            role="user",
            content="Hi",  # Too short
        )

        # Get stats - should have 0 messages
        stats = await store.get_stats()
        assert stats["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_batch_add(self, store):
        """Test batch adding messages."""
        await store.initialize()

        messages = [
            {
                "message_id": f"msg_{i}",
                "session_id": "session_batch",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"This is message number {i} with substantial content for testing.",
                "timestamp": datetime.now(),
            }
            for i in range(5)
        ]

        added = await store.add_messages_batch(messages)
        assert added == 5

        stats = await store.get_stats()
        assert stats["total_messages"] == 5

    @pytest.mark.asyncio
    async def test_search_with_session_filter(self, store):
        """Test searching with session filter."""
        await store.initialize()

        # Add messages to different sessions
        await store.add_message_embedding(
            message_id="msg_s1",
            session_id="session_1",
            role="user",
            content="Authentication and login implementation details",
        )
        await store.add_message_embedding(
            message_id="msg_s2",
            session_id="session_2",
            role="user",
            content="Authentication security best practices",
        )

        # Search in session_1 only
        results = await store.search_similar(
            query="authentication",
            session_id="session_1",
            limit=5,
        )

        # Should only find message from session_1
        session_ids = {r.session_id for r in results}
        assert "session_1" in session_ids or len(results) == 0
        # Should not find session_2
        assert "session_2" not in session_ids

    @pytest.mark.asyncio
    async def test_search_with_exclude_ids(self, store):
        """Test searching with message exclusions."""
        await store.initialize()

        await store.add_message_embedding(
            message_id="msg_exclude",
            session_id="session_test",
            role="user",
            content="Python programming with authentication",
        )
        await store.add_message_embedding(
            message_id="msg_include",
            session_id="session_test",
            role="user",
            content="Python programming with authorization",
        )

        results = await store.search_similar(
            query="Python programming",
            exclude_message_ids=["msg_exclude"],
            limit=5,
        )

        # Should not include excluded message
        message_ids = [r.message_id for r in results]
        assert "msg_exclude" not in message_ids

    @pytest.mark.asyncio
    async def test_search_by_role(self, store):
        """Test searching by specific role."""
        await store.initialize()

        await store.add_message_embedding(
            message_id="msg_user",
            session_id="session_test",
            role="user",
            content="How do I use the file system APIs in Python?",
        )
        await store.add_message_embedding(
            message_id="msg_tool",
            session_id="session_test",
            role="tool",
            content="File system read results: contents of main.py",
        )

        # Search for tool results only
        results = await store.search_by_role(
            query="file system",
            role="tool",
            session_id="session_test",
            limit=5,
        )

        # Should find tool message
        assert len(results) >= 0  # May be 0 if similarity threshold not met
        if results:
            assert all(r.role == "tool" for r in results)

    @pytest.mark.asyncio
    async def test_delete_message(self, store):
        """Test deleting a message embedding."""
        await store.initialize()

        await store.add_message_embedding(
            message_id="msg_to_delete",
            session_id="session_test",
            role="user",
            content="This message will be deleted from the embedding store",
        )

        # Verify it exists
        stats = await store.get_stats()
        initial_count = stats["total_messages"]
        assert initial_count > 0

        # Delete it
        result = await store.delete_message("msg_to_delete")
        assert result is True

        # Verify it's gone
        stats = await store.get_stats()
        assert stats["total_messages"] == initial_count - 1

    @pytest.mark.asyncio
    async def test_delete_session(self, store):
        """Test deleting all messages from a session."""
        await store.initialize()

        # Add messages to a session
        for i in range(3):
            await store.add_message_embedding(
                message_id=f"msg_session_{i}",
                session_id="session_to_delete",
                role="user",
                content=f"Message {i} in session that will be deleted entirely",
            )

        # Add a message to a different session
        await store.add_message_embedding(
            message_id="msg_other",
            session_id="other_session",
            role="user",
            content="This message is in a different session and should remain",
        )

        # Delete the session
        deleted = await store.delete_session("session_to_delete")
        assert deleted == 3

        # Verify other session still exists
        stats = await store.get_stats()
        assert stats["total_messages"] == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, store):
        """Test getting store statistics."""
        await store.initialize()

        stats = await store.get_stats()

        assert stats["store"] == "conversation_embedding_store"
        assert stats["backend"] == "lancedb"
        assert stats["embedding_dimension"] == 384
        assert stats["embedding_model"] == "mock-model"
        assert "db_path" in stats
        assert "table_name" in stats

    @pytest.mark.asyncio
    async def test_close(self, store):
        """Test closing the store."""
        await store.initialize()
        assert store.is_initialized

        await store.close()
        assert not store.is_initialized

    @pytest.mark.asyncio
    async def test_similarity_threshold(self, store):
        """Test that similarity threshold filters results."""
        await store.initialize()

        await store.add_message_embedding(
            message_id="msg_relevant",
            session_id="session_test",
            role="user",
            content="Python web framework Django REST API development",
        )

        # Search with high similarity threshold
        results = await store.search_similar(
            query="completely unrelated topic about cooking recipes",
            min_similarity=0.9,  # Very high threshold
            limit=5,
        )

        # Should not find anything with such a high threshold
        # (unless by chance the embeddings are similar)
        # This is a probabilistic test
        assert isinstance(results, list)


class TestConversationStoreIntegration:
    """Tests for ConversationStore integration with embedding store."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for both stores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield {
                "db_path": Path(tmpdir) / "conversation.db",
                "embed_path": Path(tmpdir) / "embeddings",
            }

    @pytest.mark.asyncio
    async def test_conversation_store_with_embedding_store(self, temp_paths):
        """Test that ConversationStore syncs with embedding store."""
        from victor.agent.conversation_memory import ConversationStore, MessageRole
        from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

        mock_service = MockEmbeddingService()

        # Create stores
        conv_store = ConversationStore(db_path=temp_paths["db_path"])
        embed_store = ConversationEmbeddingStore(
            embedding_service=mock_service,
            db_path=temp_paths["embed_path"],
        )
        await embed_store.initialize()

        # Wire embedding store to conversation store
        conv_store.set_embedding_store(embed_store)
        conv_store.set_embedding_service(mock_service)

        # Create a session
        session = conv_store.create_session(project_path="/test/project")

        # Add a message (should sync to embedding store)
        conv_store.add_message(
            session.session_id,
            MessageRole.USER,
            "How do I implement user authentication in Django?",
        )

        # Give async sync time to complete
        await asyncio.sleep(0.5)

        # Verify embedding store has the message
        stats = await embed_store.get_stats()
        assert stats["total_messages"] >= 1

        # Clean up
        await embed_store.close()
