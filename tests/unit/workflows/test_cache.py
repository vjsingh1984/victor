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

"""Tests for victor.workflows.cache module."""

import tempfile
from unittest.mock import MagicMock


from victor.workflows.cache import (
    DependencyGraph,
    CascadingInvalidator,
    WorkflowCacheConfig,
    WorkflowNodeCacheEntry,
    WorkflowCache,
    WorkflowCacheManager,
    DefinitionCacheConfig,
    WorkflowDefinitionCache,
)


# =============================================================================
# DependencyGraph Tests
# =============================================================================


class TestDependencyGraph:
    """Test DependencyGraph class."""

    def test_initialization(self):
        """Test DependencyGraph initialization."""
        graph = DependencyGraph()
        assert graph.dependents == {}
        assert graph.dependencies == {}

    def test_add_dependency(self):
        """Test adding a dependency relationship."""
        graph = DependencyGraph()
        graph.add_dependency("node_b", "node_a")

        assert "node_b" in graph.dependencies
        assert "node_a" in graph.dependencies["node_b"]
        assert "node_b" in graph.dependents["node_a"]

    def test_add_multiple_dependencies(self):
        """Test adding multiple dependencies for a node."""
        graph = DependencyGraph()
        graph.add_dependency("node_c", "node_a")
        graph.add_dependency("node_c", "node_b")

        deps = graph.get_dependencies("node_c")
        assert "node_a" in deps
        assert "node_b" in deps
        assert len(deps) == 2

    def test_add_multiple_dependents(self):
        """Test adding multiple dependents for a node."""
        graph = DependencyGraph()
        graph.add_dependency("node_b", "node_a")
        graph.add_dependency("node_c", "node_a")

        deps = graph.get_dependents("node_a")
        assert "node_b" in deps
        assert "node_c" in deps
        assert len(deps) == 2

    def test_get_dependencies_empty(self):
        """Test getting dependencies for node with none."""
        graph = DependencyGraph()
        deps = graph.get_dependencies("node_a")
        assert deps == set()

    def test_get_dependents_empty(self):
        """Test getting dependents for node with none."""
        graph = DependencyGraph()
        deps = graph.get_dependents("node_a")
        assert deps == set()

    def test_get_cascade_set_linear(self):
        """Test cascade set for linear dependency chain."""
        graph = DependencyGraph()
        # A -> B -> C -> D
        graph.add_dependency("node_b", "node_a")
        graph.add_dependency("node_c", "node_b")
        graph.add_dependency("node_d", "node_c")

        cascade = graph.get_cascade_set("node_a")
        assert "node_b" in cascade
        assert "node_c" in cascade
        assert "node_d" in cascade

    def test_get_cascade_set_branching(self):
        """Test cascade set for branching dependencies."""
        graph = DependencyGraph()
        #     A
        #    / \
        #   B   C
        #   |   |
        #   D   E
        graph.add_dependency("node_b", "node_a")
        graph.add_dependency("node_c", "node_a")
        graph.add_dependency("node_d", "node_b")
        graph.add_dependency("node_e", "node_c")

        cascade = graph.get_cascade_set("node_a")
        assert "node_b" in cascade
        assert "node_c" in cascade
        assert "node_d" in cascade
        assert "node_e" in cascade
        assert len(cascade) == 4

    def test_get_cascade_set_empty(self):
        """Test cascade set for node with no dependents."""
        graph = DependencyGraph()
        cascade = graph.get_cascade_set("node_a")
        assert cascade == set()

    def test_get_all_upstream(self):
        """Test getting all upstream dependencies."""
        graph = DependencyGraph()
        # D -> C -> B -> A
        graph.add_dependency("node_b", "node_a")
        graph.add_dependency("node_c", "node_b")
        graph.add_dependency("node_d", "node_c")

        upstream = graph.get_all_upstream("node_d")
        assert "node_a" in upstream
        assert "node_b" in upstream
        assert "node_c" in upstream

    def test_get_all_upstream_empty(self):
        """Test getting all upstream for node with no dependencies."""
        graph = DependencyGraph()
        upstream = graph.get_all_upstream("node_a")
        assert upstream == set()

    def test_from_workflow(self):
        """Test creating dependency graph from workflow."""
        # Create mock workflow
        mock_workflow = MagicMock()
        mock_workflow.nodes = [
            MagicMock(name="node1", next_nodes=["node2"]),
            MagicMock(name="node2", next_nodes=["node3"]),
            MagicMock(name="node3", next_nodes=[]),
        ]

        graph = DependencyGraph.from_workflow(mock_workflow)
        assert graph is not None
        # node2 should depend on node1
        assert "node2" in graph.get_dependents("node1")


# =============================================================================
# CascadingInvalidator Tests
# =============================================================================


class TestCascadingInvalidator:
    """Test CascadingInvalidator class."""

    def test_initialization(self):
        """Test CascadingInvalidator initialization."""
        graph = DependencyGraph()
        invalidator = CascadingInvalidator(graph)
        assert invalidator.dependency_graph == graph

    def test_invalidate_cascade(self):
        """Test cascading invalidation."""
        graph = DependencyGraph()
        # A -> B -> C
        graph.add_dependency("node_b", "node_a")
        graph.add_dependency("node_c", "node_b")

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.keys.return_value = ["node_a", "node_b", "node_c"]

        invalidator = CascadingInvalidator(graph)
        invalidator.invalidate_cascade(mock_cache, "node_a")

        # Should invalidate all three nodes
        assert mock_cache.invalidate.called

    def test_invalidate_node(self):
        """Test invalidating a single node."""
        graph = DependencyGraph()
        mock_cache = MagicMock()

        invalidator = CascadingInvalidator(graph)
        invalidator.invalidate_node(mock_cache, "node_a")

        mock_cache.invalidate.assert_called_once_with("node_a")


# =============================================================================
# WorkflowCacheConfig Tests
# =============================================================================


class TestWorkflowCacheConfig:
    """Test WorkflowCacheConfig dataclass."""

    def test_default_config(self):
        """Test default cache configuration."""
        config = WorkflowCacheConfig()
        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.max_size == 1000
        assert config.persist_to_disk is False

    def test_custom_config(self):
        """Test custom cache configuration."""
        config = WorkflowCacheConfig(
            enabled=False,
            ttl_seconds=7200,
            max_size=2000,
            persist_to_disk=True,
            disk_cache_path="/tmp/cache",
        )
        assert config.enabled is False
        assert config.ttl_seconds == 7200
        assert config.max_size == 2000
        assert config.persist_to_disk is True
        assert config.disk_cache_path == "/tmp/cache"


# =============================================================================
# WorkflowNodeCacheEntry Tests
# =============================================================================


class TestWorkflowNodeCacheEntry:
    """Test WorkflowNodeCacheEntry dataclass."""

    def test_initialization(self):
        """Test cache entry initialization."""
        entry = WorkflowNodeCacheEntry(
            node_id="test_node",
            context_hash="abc123",
            result={"status": "success"},
            timestamp=1234567890,
        )
        assert entry.node_id == "test_node"
        assert entry.context_hash == "abc123"
        assert entry.result == {"status": "success"}
        assert entry.timestamp == 1234567890


# =============================================================================
# WorkflowCache Tests
# =============================================================================


class TestWorkflowCache:
    """Test WorkflowCache class."""

    def test_initialization(self):
        """Test cache initialization with default config."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)
        assert cache.config == config
        assert cache._cache is not None

    def test_initialization_disabled(self):
        """Test cache initialization when disabled."""
        config = WorkflowCacheConfig(enabled=False)
        cache = WorkflowCache(config)
        assert cache.config.enabled is False

    def test_generate_cache_key(self):
        """Test generating cache key from node and context."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"

        context = {"key": "value", "number": 42}

        key = cache._generate_cache_key(mock_node, context)
        assert key is not None
        assert isinstance(key, str)

    def test_generate_cache_key_different_contexts(self):
        """Test that different contexts generate different keys."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"

        context1 = {"key": "value1"}
        context2 = {"key": "value2"}

        key1 = cache._generate_cache_key(mock_node, context1)
        key2 = cache._generate_cache_key(mock_node, context2)

        assert key1 != key2

    def test_get_cache_hit(self):
        """Test getting value from cache (hit)."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"
        context = {"key": "value"}

        result = {"status": "success"}
        cache.set(mock_node, context, result)

        cached_result = cache.get(mock_node, context)
        assert cached_result is not None
        assert cached_result == result

    def test_get_cache_miss(self):
        """Test getting value from cache (miss)."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"
        context = {"key": "value"}

        cached_result = cache.get(mock_node, context)
        assert cached_result is None

    def test_set_and_get(self):
        """Test setting and getting cache values."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"
        context = {"task": "test"}

        result = {"output": "done", "status": "completed"}
        cache.set(mock_node, context, result)

        retrieved = cache.get(mock_node, context)
        assert retrieved == result

    def test_invalidate(self):
        """Test invalidating a cache entry."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"
        context = {"key": "value"}

        result = {"status": "success"}
        cache.set(mock_node, context, result)
        assert cache.get(mock_node, context) is not None

        cache.invalidate("test_node")
        assert cache.get(mock_node, context) is None

    def test_clear(self):
        """Test clearing all cache entries."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"
        context = {"key": "value"}

        result = {"status": "success"}
        cache.set(mock_node, context, result)

        cache.clear()
        assert cache.get(mock_node, context) is None

    def test_get_stats(self):
        """Test getting cache statistics."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats

    def test_cache_disabled_operations(self):
        """Test that operations are no-ops when cache is disabled."""
        config = WorkflowCacheConfig(enabled=False)
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"
        context = {"key": "value"}
        result = {"status": "success"}

        # Should not raise
        cache.set(mock_node, context, result)
        retrieved = cache.get(mock_node, context)

        # Should return None when disabled
        assert retrieved is None

    def test_persist_to_disk(self):
        """Test persisting cache to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorkflowCacheConfig(
                persist_to_disk=True,
                disk_cache_path=tmpdir,
            )
            cache = WorkflowCache(config)

            mock_node = MagicMock()
            mock_node.name = "test_node"
            context = {"key": "value"}
            result = {"status": "success"}

            cache.set(mock_node, context, result)

            # Should not raise
            cache.persist()

    def test_load_from_disk(self):
        """Test loading cache from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorkflowCacheConfig(
                persist_to_disk=True,
                disk_cache_path=tmpdir,
            )
            cache = WorkflowCache(config)

            # Should not raise even if no cache file exists
            cache.load()

    def test_context_hash_consistency(self):
        """Test that same context produces same hash."""
        config = WorkflowCacheConfig()
        cache = WorkflowCache(config)

        mock_node = MagicMock()
        mock_node.name = "test_node"
        context = {"key": "value", "nested": {"data": [1, 2, 3]}}

        key1 = cache._generate_cache_key(mock_node, context)
        key2 = cache._generate_cache_key(mock_node, context)

        assert key1 == key2


# =============================================================================
# WorkflowCacheManager Tests
# =============================================================================


class TestWorkflowCacheManager:
    """Test WorkflowCacheManager class."""

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = WorkflowCacheManager()
        assert manager._caches == {}

    def test_create_cache(self):
        """Test creating a new cache."""
        manager = WorkflowCacheManager()
        config = WorkflowCacheConfig()

        cache = manager.create_cache("workflow_1", config)
        assert cache is not None
        assert "workflow_1" in manager._caches

    def test_get_cache_exists(self):
        """Test getting an existing cache."""
        manager = WorkflowCacheManager()
        config = WorkflowCacheConfig()

        manager.create_cache("workflow_1", config)
        cache = manager.get_cache("workflow_1")

        assert cache is not None

    def test_get_cache_not_exists(self):
        """Test getting a non-existent cache."""
        manager = WorkflowCacheManager()
        cache = manager.get_cache("nonexistent")
        assert cache is None

    def test_remove_cache(self):
        """Test removing a cache."""
        manager = WorkflowCacheManager()
        config = WorkflowCacheConfig()

        manager.create_cache("workflow_1", config)
        assert "workflow_1" in manager._caches

        manager.remove_cache("workflow_1")
        assert "workflow_1" not in manager._caches

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        manager = WorkflowCacheManager()
        config = WorkflowCacheConfig()

        manager.create_cache("workflow_1", config)
        manager.create_cache("workflow_2", config)

        manager.clear_all()
        assert len(manager._caches) == 0

    def test_get_global_stats(self):
        """Test getting global cache statistics."""
        manager = WorkflowCacheManager()
        config = WorkflowCacheConfig()

        manager.create_cache("workflow_1", config)

        stats = manager.get_global_stats()
        assert stats is not None
        assert "total_caches" in stats


# =============================================================================
# DefinitionCacheConfig Tests
# =============================================================================


class TestDefinitionCacheConfig:
    """Test DefinitionCacheConfig dataclass."""

    def test_default_config(self):
        """Test default definition cache config."""
        config = DefinitionCacheConfig()
        assert config.enabled is True
        assert config.ttl_seconds == 7200
        assert config.max_size == 500

    def test_custom_config(self):
        """Test custom definition cache config."""
        config = DefinitionCacheConfig(
            enabled=False,
            ttl_seconds=3600,
            max_size=1000,
        )
        assert config.enabled is False
        assert config.ttl_seconds == 3600
        assert config.max_size == 1000


# =============================================================================
# WorkflowDefinitionCache Tests
# =============================================================================


class TestWorkflowDefinitionCache:
    """Test WorkflowDefinitionCache class."""

    def test_initialization(self):
        """Test definition cache initialization."""
        config = DefinitionCacheConfig()
        cache = WorkflowDefinitionCache(config)
        assert cache.config == config

    def test_get_definition_hit(self):
        """Test getting cached definition (hit)."""
        config = DefinitionCacheConfig()
        cache = WorkflowDefinitionCache(config)

        mock_definition = MagicMock()
        mock_definition.name = "test_workflow"

        cache.set("test_workflow", mock_definition)
        retrieved = cache.get("test_workflow")

        assert retrieved is not None
        assert retrieved.name == "test_workflow"

    def test_get_definition_miss(self):
        """Test getting cached definition (miss)."""
        config = DefinitionCacheConfig()
        cache = WorkflowDefinitionCache(config)

        retrieved = cache.get("nonexistent")
        assert retrieved is None

    def test_set_definition(self):
        """Test setting a cached definition."""
        config = DefinitionCacheConfig()
        cache = WorkflowDefinitionCache(config)

        mock_definition = MagicMock()
        mock_definition.name = "test_workflow"

        cache.set("test_workflow", mock_definition)
        retrieved = cache.get("test_workflow")

        assert retrieved == mock_definition

    def test_invalidate_definition(self):
        """Test invalidating a cached definition."""
        config = DefinitionCacheConfig()
        cache = WorkflowDefinitionCache(config)

        mock_definition = MagicMock()
        mock_definition.name = "test_workflow"

        cache.set("test_workflow", mock_definition)
        assert cache.get("test_workflow") is not None

        cache.invalidate("test_workflow")
        assert cache.get("test_workflow") is None

    def test_clear_all(self):
        """Test clearing all cached definitions."""
        config = DefinitionCacheConfig()
        cache = WorkflowDefinitionCache(config)

        mock_def1 = MagicMock()
        mock_def2 = MagicMock()

        cache.set("workflow_1", mock_def1)
        cache.set("workflow_2", mock_def2)

        cache.clear()
        assert cache.get("workflow_1") is None
        assert cache.get("workflow_2") is None

    def test_cache_disabled(self):
        """Test behavior when cache is disabled."""
        config = DefinitionCacheConfig(enabled=False)
        cache = WorkflowDefinitionCache(config)

        mock_definition = MagicMock()
        mock_definition.name = "test_workflow"

        cache.set("test_workflow", mock_definition)
        retrieved = cache.get("test_workflow")

        # Should return None when disabled
        assert retrieved is None

    def test_get_stats(self):
        """Test getting cache statistics."""
        config = DefinitionCacheConfig()
        cache = WorkflowDefinitionCache(config)

        stats = cache.get_stats()
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
