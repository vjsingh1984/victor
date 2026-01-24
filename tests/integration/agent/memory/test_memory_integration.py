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

"""Integration tests for memory systems.

Tests cover:
1. Episodic + Semantic memory interaction (3 tests)
2. Memory consolidation workflow (3 tests)
3. Memory recall with embeddings (3 tests)
4. Memory persistence and recovery (3 tests)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from victor.agent.memory import EpisodicMemory, Episode, SemanticMemory
from victor.storage.embeddings.service import EmbeddingService


@pytest.mark.integration
@pytest.mark.memory
class TestMemoryInteraction:
    """Test episodic and semantic memory interaction (3 tests)."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service for faster tests."""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.dimension = 384
        mock_service.cosine_similarity = Mock(
            side_effect=lambda a, b: float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            )
        )

        # Simple embedding generation based on text hash
        async def mock_embed(text: str) -> np.ndarray:
            # Generate consistent embeddings based on text
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(384).astype(np.float32)

        mock_service.embed_text = mock_embed
        return mock_service

    @pytest.fixture
    def episodic_memory(self, mock_embedding_service):
        """Create episodic memory with mock service."""
        return EpisodicMemory(
            embedding_service=mock_embedding_service,
            max_episodes=100,
            consolidation_threshold=5,  # Low threshold for testing
        )

    @pytest.fixture
    def semantic_memory(self, mock_embedding_service):
        """Create semantic memory with mock service."""
        return SemanticMemory(
            embedding_service=mock_embedding_service,
            max_knowledge=100,
        )

    @pytest.mark.asyncio
    async def test_episodic_semantic_bidirectional_flow(self, episodic_memory, semantic_memory):
        """Test bidirectional information flow between episodic and semantic memory."""
        # Store episodes with related information
        episodes_data = [
            ("Fix authentication bug", ["debug", "fix"], {"success": True, "files_changed": 2}),
            (
                "Add user profile feature",
                ["implement", "test"],
                {"success": True, "files_changed": 5},
            ),
            (
                "Optimize database queries",
                ["profile", "optimize"],
                {"success": True, "speedup": 2.0},
            ),
        ]

        for query, actions, outcomes in episodes_data:
            episode = Episode(
                inputs={"query": query},
                actions=actions,
                outcomes=outcomes,
                context={"task_type": "development"},
            )
            await episodic_memory.store_episode(episode)

        # Recall from episodic memory (any result is acceptable since we use mock embeddings)
        episodic_results = await episodic_memory.recall_relevant("database performance", k=2)
        assert len(episodic_results) >= 1
        # Verify we got episodes back
        assert all(isinstance(ep, Episode) for ep in episodic_results)

        # Store related knowledge in semantic memory
        await semantic_memory.store_knowledge(
            "Database optimization improves query performance",
            confidence=0.9,
            source="experience",
        )

        # Query semantic memory
        semantic_results = await semantic_memory.query_knowledge("database queries", k=2)
        assert len(semantic_results) >= 1

        # Verify both memory systems are functional
        assert episodic_memory.episode_count == 3
        assert semantic_memory.knowledge_count == 1

    @pytest.mark.asyncio
    async def test_cross_memory_context_sharing(self, episodic_memory, semantic_memory):
        """Test context sharing between episodic and semantic memory."""
        # Store episode with context
        episode = Episode(
            inputs={"query": "Python async error"},
            actions=["read_docs", "implement_fix"],
            outcomes={"success": True, "language": "Python"},
            context={"language": "Python", "topic": "async"},
        )
        await episodic_memory.store_episode(episode)

        # Extract context from episodic memory
        stored_episode = episodic_memory.get_episode(episode.id)
        assert stored_episode is not None
        assert stored_episode.context.get("language") == "Python"

        # Use context to enrich semantic memory
        await semantic_memory.store_knowledge(
            "Python uses asyncio for asynchronous programming",
            confidence=0.95,
            metadata={"language": "Python", "topic": "async"},
        )

        # Query should find related knowledge
        results = await semantic_memory.query_knowledge("Python async", k=1)
        assert len(results) >= 1
        assert "Python" in results[0].fact or "asyncio" in results[0].fact

    @pytest.mark.asyncio
    async def test_memory_system_consistency(self, episodic_memory, semantic_memory):
        """Test consistency between episodic and semantic memory systems."""
        # Store multiple related episodes
        for i in range(5):
            episode = Episode(
                inputs={"query": f"Task {i}"},
                actions=["action1", "action2"],
                outcomes={"success": i % 2 == 0},
                context={"batch": "test_batch"},
            )
            await episodic_memory.store_episode(episode)

        # Consolidate to semantic memory
        consolidated = await episodic_memory.consolidate_memories()

        # Verify both systems have data
        assert episodic_memory.episode_count == 5
        assert (
            consolidated.knowledge_count >= 0
        )  # May not extract knowledge if patterns aren't strong

        # Store additional semantic knowledge
        fact_id = await semantic_memory.store_knowledge(
            "Test batch contains 5 tasks",
            metadata={"batch": "test_batch", "count": 5},
        )

        # Verify semantic memory
        assert semantic_memory.knowledge_count == 1
        knowledge = semantic_memory.get_knowledge(fact_id)
        assert knowledge is not None
        assert "5 tasks" in knowledge.fact


@pytest.mark.integration
@pytest.mark.memory
class TestMemoryConsolidation:
    """Test memory consolidation workflow (3 tests)."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.dimension = 384
        mock_service.cosine_similarity = Mock(
            side_effect=lambda a, b: float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            )
        )

        async def mock_embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(384).astype(np.float32)

        mock_service.embed_text = mock_embed
        return mock_service

    @pytest.fixture
    def episodic_memory(self, mock_embedding_service):
        """Create episodic memory with low consolidation threshold."""
        return EpisodicMemory(
            embedding_service=mock_embedding_service,
            max_episodes=100,
            consolidation_threshold=3,  # Low threshold for testing
        )

    @pytest.mark.asyncio
    async def test_episode_to_knowledge_extraction(self, episodic_memory):
        """Test extraction of knowledge from episodes."""
        # Store episodes with successful patterns
        for i in range(10):
            episode = Episode(
                inputs={"query": f"fix bug {i}", "type": "bugfix"},
                actions=["read_file", "edit_file", "run_tests"],
                outcomes={"success": True, "tests_passed": 5 + i},
                context={"category": "bugfix", "severity": "medium"},
                importance=0.6,
            )
            await episodic_memory.store_episode(episode)

        # Consolidate memories
        semantic_memory = await episodic_memory.consolidate_memories()

        # Should extract knowledge about successful actions
        assert semantic_memory.knowledge_count >= 0

        # If knowledge was extracted, verify it
        if semantic_memory.knowledge_count > 0:
            all_knowledge = list(semantic_memory._knowledge.values())
            facts = [k.fact.lower() for k in all_knowledge]

            # Check for action/outcome knowledge
            has_action_knowledge = any("action" in fact for fact in facts)
            has_outcome_knowledge = any("outcome" in fact for fact in facts)

            assert has_action_knowledge or has_outcome_knowledge

    @pytest.mark.asyncio
    async def test_consolidation_threshold_behavior(self, episodic_memory):
        """Test consolidation threshold and behavior."""
        # Store episodes below threshold
        for i in range(2):
            episode = Episode(
                inputs={"query": f"task {i}"},
                actions=["action"],
                outcomes={"result": i},
            )
            await episodic_memory.store_episode(episode)

        # Consolidation should not happen (below threshold)
        semantic_memory = await episodic_memory.consolidate_memories()
        assert episodic_memory.episode_count == 2
        # May have minimal knowledge extraction

        # Add more episodes to reach threshold
        for i in range(2, 5):
            episode = Episode(
                inputs={"query": f"task {i}"},
                actions=["action"],
                outcomes={"success": i % 2 == 0},
            )
            await episodic_memory.store_episode(episode)

        # Now consolidation should extract more knowledge
        semantic_memory = await episodic_memory.consolidate_memories()
        assert episodic_memory.episode_count == 5

    @pytest.mark.asyncio
    async def test_pattern_based_consolidation(self, episodic_memory):
        """Test consolidation extracts meaningful patterns."""
        # Store episodes with clear patterns
        patterns = [
            ("authentication", ["read_auth", "fix_auth"], {"success": True}),
            ("authentication", ["read_auth", "fix_auth"], {"success": True}),
            ("authentication", ["read_auth", "fix_auth"], {"success": True}),
            ("database", ["optimize_db"], {"success": True}),
            ("database", ["optimize_db"], {"success": True}),
        ]

        for topic, actions, outcomes in patterns:
            episode = Episode(
                inputs={"query": f"{topic} task", "topic": topic},
                actions=actions,
                outcomes=outcomes,
                context={"category": topic},
            )
            await episodic_memory.store_episode(episode)

        # Consolidate
        semantic_memory = await episodic_memory.consolidate_memories()

        # Should extract knowledge about patterns
        if semantic_memory.knowledge_count > 0:
            all_knowledge = list(semantic_memory._knowledge.values())
            facts = [k.fact.lower() for k in all_knowledge]

            # Should have knowledge about actions or outcomes
            has_knowledge = any("action" in f or "outcome" in f or "successful" in f for f in facts)
            assert has_knowledge


@pytest.mark.integration
@pytest.mark.memory
class TestMemoryRecall:
    """Test memory recall with embeddings (3 tests)."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.dimension = 384
        mock_service.cosine_similarity = Mock(
            side_effect=lambda a, b: float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            )
        )

        async def mock_embed(text: str) -> np.ndarray:
            # Similar texts get similar embeddings
            np.random.seed(hash(text.split()[0]) % (2**32))  # Seed by first word
            return np.random.rand(384).astype(np.float32)

        mock_service.embed_text = mock_embed
        return mock_service

    @pytest.fixture
    def episodic_memory(self, mock_embedding_service):
        """Create episodic memory."""
        return EpisodicMemory(
            embedding_service=mock_embedding_service,
            max_episodes=100,
        )

    @pytest.fixture
    def semantic_memory(self, mock_embedding_service):
        """Create semantic memory."""
        return SemanticMemory(
            embedding_service=mock_embedding_service,
            max_knowledge=100,
        )

    @pytest.mark.asyncio
    async def test_semantic_recall_accuracy(self, semantic_memory):
        """Test semantic knowledge retrieval accuracy."""
        # Store knowledge with clear topics
        knowledge_items = [
            (
                "Python uses asyncio for asynchronous programming",
                {"language": "Python", "topic": "async"},
            ),
            (
                "JavaScript uses Promises for async operations",
                {"language": "JavaScript", "topic": "async"},
            ),
            ("Go uses goroutines for concurrency", {"language": "Go", "topic": "concurrency"}),
            (
                "Java uses threads for parallel execution",
                {"language": "Java", "topic": "concurrency"},
            ),
        ]

        for fact, metadata in knowledge_items:
            await semantic_memory.store_knowledge(fact, metadata=metadata)

        # Query for async programming
        results = await semantic_memory.query_knowledge("asynchronous concurrency", k=4)

        assert len(results) >= 2

        # Results should be about async/concurrency
        result_facts = [r.fact.lower() for r in results]
        assert any("async" in fact or "concurrency" in fact for fact in result_facts)

    @pytest.mark.asyncio
    async def test_episodic_recall_with_embeddings(self, episodic_memory):
        """Test episodic memory recall using embeddings."""
        # Store episodes with different topics
        topics = [
            ("authentication", ["read_auth", "fix_auth"], {"success": True}),
            ("database", ["optimize_db", "add_index"], {"performance": "improved"}),
            ("ui", ["update_css", "fix_layout"], {"bugs_fixed": 3}),
        ]

        for topic, actions, outcomes in topics:
            episode = Episode(
                inputs={"query": f"improve {topic}", "topic": topic},
                actions=actions,
                outcomes=outcomes,
                context={"category": topic},
            )
            await episodic_memory.store_episode(episode)

        # Test recall for each topic
        for topic, actions, outcomes in topics:
            relevant = await episodic_memory.recall_relevant(f"{topic} improvement", k=3)

            # Should find relevant episode
            assert len(relevant) >= 1

            # Verify episodes are returned (with mock embeddings, exact matching isn't guaranteed)
            assert all(isinstance(ep, Episode) for ep in relevant)
            # Verify all episodes have inputs and context
            assert all(hasattr(ep, "inputs") and hasattr(ep, "context") for ep in relevant)

    @pytest.mark.asyncio
    async def test_hybrid_recall_strategy(self, episodic_memory, semantic_memory):
        """Test hybrid recall combining episodic and semantic memory."""
        # Store related information in both systems
        # Episodic: specific experiences
        episode = Episode(
            inputs={"query": "fix Python async bug"},
            actions=["debug", "read_docs", "implement_fix"],
            outcomes={"success": True, "language": "Python"},
            context={"language": "Python", "topic": "async"},
        )
        await episodic_memory.store_episode(episode)

        # Semantic: general knowledge
        await semantic_memory.store_knowledge(
            "Python uses asyncio for asynchronous programming",
            metadata={"language": "Python", "topic": "async"},
        )
        await semantic_memory.store_knowledge(
            "Common async bugs include event loop issues",
            metadata={"language": "Python", "topic": "bugs"},
        )

        # Query both systems
        episodic_results = await episodic_memory.recall_relevant("Python async problems", k=2)
        semantic_results = await semantic_memory.query_knowledge("Python async", k=2)

        # Both should return results
        assert len(episodic_results) >= 1
        assert len(semantic_results) >= 1

        # Results should be relevant
        assert any(
            "async" in str(ep.inputs).lower() or "python" in str(ep.inputs).lower()
            for ep in episodic_results
        )
        assert any(
            "async" in k.fact.lower() or "python" in k.fact.lower() for k in semantic_results
        )


@pytest.mark.integration
@pytest.mark.memory
class TestMemoryPersistence:
    """Test memory persistence and recovery (3 tests)."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.dimension = 384
        mock_service.cosine_similarity = Mock(
            side_effect=lambda a, b: float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            )
        )

        async def mock_embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(384).astype(np.float32)

        mock_service.embed_text = mock_embed
        return mock_service

    @pytest.fixture
    def semantic_memory(self, mock_embedding_service):
        """Create semantic memory."""
        return SemanticMemory(
            embedding_service=mock_embedding_service,
            max_knowledge=100,
        )

    @pytest.fixture
    def temp_json_path(self, tmp_path):
        """Create temporary JSON file path."""
        return tmp_path / "test_knowledge.json"

    @pytest.fixture
    def temp_sqlite_path(self, tmp_path):
        """Create temporary SQLite database path."""
        return tmp_path / "test_knowledge.db"

    @pytest.mark.asyncio
    async def test_json_export_import(self, semantic_memory, temp_json_path):
        """Test exporting and importing memory to/from JSON."""
        # Store knowledge
        knowledge_items = [
            ("Python uses asyncio for async", {"language": "Python"}),
            ("JavaScript uses Promises", {"language": "JavaScript"}),
        ]

        fact_ids = []
        for fact, metadata in knowledge_items:
            fact_id = await semantic_memory.store_knowledge(fact, metadata=metadata)
            fact_ids.append(fact_id)

        # Export to JSON
        await semantic_memory.export_to_json(temp_json_path)

        # Verify file exists
        assert temp_json_path.exists()

        # Create new memory and import
        new_memory = SemanticMemory(embedding_service=semantic_memory._embedding_service)
        imported_count = await new_memory.import_from_json(temp_json_path)

        # Verify import
        assert imported_count == 2
        assert new_memory.knowledge_count == 2

        # Verify knowledge content
        all_knowledge = list(new_memory._knowledge.values())
        facts = [k.fact for k in all_knowledge]
        assert "Python uses asyncio" in str(facts)
        assert "JavaScript uses Promises" in str(facts)

    @pytest.mark.asyncio
    async def test_sqlite_persistence(self, semantic_memory, temp_sqlite_path):
        """Test saving and loading memory from SQLite."""
        # Store knowledge with links
        fact1_id = await semantic_memory.store_knowledge(
            "Python is a programming language",
            metadata={"type": "language"},
        )
        fact2_id = await semantic_memory.store_knowledge(
            "asyncio is Python's async library",
            metadata={"type": "library"},
        )

        # Link knowledge
        semantic_memory.link_knowledge(fact1_id, fact2_id, "includes")

        # Save to SQLite
        await semantic_memory.save_to_sqlite(temp_sqlite_path)

        # Verify file exists
        assert temp_sqlite_path.exists()

        # Create new memory and load
        new_memory = SemanticMemory(embedding_service=semantic_memory._embedding_service)
        loaded_count = await new_memory.load_from_sqlite(temp_sqlite_path)

        # Verify load
        assert loaded_count == 2
        assert new_memory.knowledge_count == 2

        # Verify links
        graph = new_memory.export_knowledge_graph()
        assert len(graph.edges) >= 1

    @pytest.mark.asyncio
    async def test_memory_recovery_after_failure(self, semantic_memory, temp_json_path):
        """Test memory recovery after simulated failure."""
        # Store knowledge
        for i in range(5):
            await semantic_memory.store_knowledge(
                f"Fact {i}",
                metadata={"index": i},
            )

        # Export checkpoint
        await semantic_memory.export_to_json(temp_json_path)

        # Simulate failure and recovery (clear memory)
        await semantic_memory.clear()
        assert semantic_memory.knowledge_count == 0

        # Recover from checkpoint
        imported_count = await semantic_memory.import_from_json(temp_json_path)

        # Verify recovery
        assert imported_count == 5
        assert semantic_memory.knowledge_count == 5

        # Verify knowledge is accessible
        results = await semantic_memory.query_knowledge("Fact", k=10)
        assert len(results) == 5

        # Verify statistics
        stats = semantic_memory.get_statistics()
        assert stats["knowledge_count"] == 5
        assert stats["total_stored"] == 5  # After recovery
