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

"""Tests for victor.framework.chain_registry module.

These tests verify the ChainRegistry functionality for cross-vertical
chain discovery and access.
"""

import pytest
from unittest.mock import MagicMock

from victor.framework.chain_registry import (
    ChainRegistry,
    ChainMetadata,
    get_chain_registry,
    register_chain,
    get_chain,
)


# =============================================================================
# ChainMetadata Tests
# =============================================================================


class TestChainMetadata:
    """Tests for ChainMetadata dataclass."""

    def test_metadata_basic_attributes(self):
        """ChainMetadata should store basic attributes."""
        metadata = ChainMetadata(
            name="test_chain",
            vertical="coding",
            description="Test chain for testing",
            input_type="str",
            output_type="dict",
            tags=["test", "example"],
        )

        assert metadata.name == "test_chain"
        assert metadata.vertical == "coding"
        assert metadata.description == "Test chain for testing"
        assert metadata.input_type == "str"
        assert metadata.output_type == "dict"
        assert metadata.tags == ["test", "example"]

    def test_metadata_full_name_with_vertical(self):
        """ChainMetadata should return full name with vertical namespace."""
        metadata = ChainMetadata(name="my_chain", vertical="coding")
        assert metadata.full_name == "coding:my_chain"

    def test_metadata_full_name_without_vertical(self):
        """ChainMetadata should return name as full_name when no vertical."""
        metadata = ChainMetadata(name="my_chain")
        assert metadata.full_name == "my_chain"

    def test_metadata_defaults(self):
        """ChainMetadata should have sensible defaults."""
        metadata = ChainMetadata(name="simple")

        assert metadata.name == "simple"
        assert metadata.vertical is None
        assert metadata.description == ""
        assert metadata.input_type is None
        assert metadata.output_type is None
        assert metadata.tags == []


# =============================================================================
# ChainRegistry Basic Tests
# =============================================================================


class TestChainRegistry:
    """Tests for ChainRegistry class."""

    def test_registry_register_and_get(self):
        """Registry should register and retrieve chains."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("test_chain", mock_chain)
        result = registry.get("test_chain")

        assert result is mock_chain

    def test_registry_register_with_vertical(self):
        """Registry should register chains with vertical namespace."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("my_chain", mock_chain, vertical="coding")
        result = registry.get("my_chain", vertical="coding")

        assert result is mock_chain

    def test_registry_get_with_vertical_returns_namespaced(self):
        """Registry should return vertical-namespaced chain when vertical specified."""
        registry = ChainRegistry()
        global_chain = MagicMock(name="GlobalChain")
        coding_chain = MagicMock(name="CodingChain")

        registry.register("my_chain", global_chain)
        registry.register("my_chain", coding_chain, vertical="coding")

        # Without vertical, should get global
        assert registry.get("my_chain") is global_chain
        # With vertical, should get namespaced
        assert registry.get("my_chain", vertical="coding") is coding_chain

    def test_registry_register_duplicate_skips(self):
        """Registry should skip duplicate registration without replace=True."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("test_chain", chain1)
        registry.register("test_chain", chain2)  # Should be skipped

        assert registry.get("test_chain") is chain1

    def test_registry_register_duplicate_with_replace(self):
        """Registry should allow replacing with replace=True."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("test_chain", chain1)
        registry.register("test_chain", chain2, replace=True)

        assert registry.get("test_chain") is chain2

    def test_registry_unregister(self):
        """Registry should unregister chains."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("test_chain", mock_chain)
        result = registry.unregister("test_chain")

        assert result is True
        assert registry.get("test_chain") is None

    def test_registry_unregister_with_vertical(self):
        """Registry should unregister chains with vertical namespace."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("my_chain", mock_chain, vertical="coding")
        result = registry.unregister("my_chain", vertical="coding")

        assert result is True
        assert registry.get("my_chain", vertical="coding") is None

    def test_registry_unregister_nonexistent(self):
        """Registry should return False when unregistering nonexistent chain."""
        registry = ChainRegistry()
        result = registry.unregister("nonexistent_chain")
        assert result is False

    def test_registry_get_metadata(self):
        """Registry should return metadata for registered chains."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register(
            "my_chain",
            mock_chain,
            vertical="coding",
            description="Test chain",
            input_type="str",
            output_type="dict",
            tags=["test"],
        )

        metadata = registry.get_metadata("my_chain", vertical="coding")

        assert metadata is not None
        assert metadata.name == "my_chain"
        assert metadata.vertical == "coding"
        assert metadata.description == "Test chain"
        assert metadata.input_type == "str"
        assert metadata.output_type == "dict"
        assert metadata.tags == ["test"]

    def test_registry_get_metadata_nonexistent(self):
        """Registry should return None for nonexistent chain metadata."""
        registry = ChainRegistry()
        metadata = registry.get_metadata("nonexistent")
        assert metadata is None


# =============================================================================
# ChainRegistry Listing Tests
# =============================================================================


class TestChainRegistryListing:
    """Tests for ChainRegistry listing methods."""

    def test_registry_list_chains(self):
        """Registry should list all chain names."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock(), vertical="coding")
        registry.register("chain3", MagicMock(), vertical="devops")

        chains = registry.list_chains()

        assert len(chains) == 3
        assert "chain1" in chains
        assert "coding:chain2" in chains
        assert "devops:chain3" in chains

    def test_registry_list_chains_by_vertical(self):
        """Registry should list chains filtered by vertical."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock(), vertical="coding")
        registry.register("chain3", MagicMock(), vertical="coding")
        registry.register("chain4", MagicMock(), vertical="devops")

        coding_chains = registry.list_chains(vertical="coding")

        assert len(coding_chains) == 2
        assert "coding:chain2" in coding_chains
        assert "coding:chain3" in coding_chains

    def test_registry_list_metadata(self):
        """Registry should list all chain metadata."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock(), description="First")
        registry.register("chain2", MagicMock(), vertical="coding", description="Second")

        metadata_list = registry.list_metadata()

        assert len(metadata_list) == 2
        assert any(m.name == "chain1" and m.description == "First" for m in metadata_list)
        assert any(m.name == "chain2" and m.description == "Second" for m in metadata_list)

    def test_registry_list_metadata_by_vertical(self):
        """Registry should list metadata filtered by vertical."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock(), vertical="coding")
        registry.register("chain3", MagicMock(), vertical="devops")

        coding_metadata = registry.list_metadata(vertical="coding")

        assert len(coding_metadata) == 1
        assert coding_metadata[0].name == "chain2"


# =============================================================================
# ChainRegistry Find Tests
# =============================================================================


class TestChainRegistryFind:
    """Tests for ChainRegistry find methods."""

    def test_registry_find_by_vertical(self):
        """Registry should find chains by vertical."""
        registry = ChainRegistry()
        coding_chain = MagicMock(name="CodingChain")
        devops_chain = MagicMock(name="DevOpsChain")

        registry.register("chain1", coding_chain, vertical="coding")
        registry.register("chain2", devops_chain, vertical="devops")

        coding_chains = registry.find_by_vertical("coding")

        assert len(coding_chains) == 1
        assert "coding:chain1" in coding_chains
        assert coding_chains["coding:chain1"] is coding_chain

    def test_registry_find_by_tag(self):
        """Registry should find chains by tag."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("chain1", chain1, tags=["review", "quality"])
        registry.register("chain2", chain2, tags=["implementation"])

        review_chains = registry.find_by_tag("review")

        assert len(review_chains) == 1
        assert "chain1" in review_chains

    def test_registry_find_by_tags_match_any(self):
        """Registry should find chains matching any of multiple tags."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")
        chain3 = MagicMock(name="Chain3")

        registry.register("chain1", chain1, tags=["review", "quality"])
        registry.register("chain2", chain2, tags=["implementation"])
        registry.register("chain3", chain3, tags=["quality", "testing"])

        chains = registry.find_by_tags(["quality", "testing"], match_all=False)

        assert len(chains) == 2
        assert "chain1" in chains  # Has quality
        assert "chain3" in chains  # Has both

    def test_registry_find_by_tags_match_all(self):
        """Registry should find chains matching all tags."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("chain1", chain1, tags=["review", "quality"])
        registry.register("chain2", chain2, tags=["quality", "testing", "review"])

        chains = registry.find_by_tags(["quality", "review"], match_all=True)

        assert len(chains) == 2  # Both have quality and review


# =============================================================================
# ChainRegistry Bulk Operations Tests
# =============================================================================


class TestChainRegistryBulkOperations:
    """Tests for ChainRegistry bulk operations."""

    def test_registry_clear(self):
        """Registry should clear all chains."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock())

        registry.clear()

        assert len(registry.list_chains()) == 0

    def test_registry_register_from_vertical(self):
        """Registry should register multiple chains from a vertical."""
        registry = ChainRegistry()
        chains = {
            "review_chain": MagicMock(name="ReviewChain"),
            "build_chain": MagicMock(name="BuildChain"),
        }

        count = registry.register_from_vertical("coding", chains)

        assert count == 2
        assert registry.get("review_chain", vertical="coding") is not None
        assert registry.get("build_chain", vertical="coding") is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the global registry before each test."""
        get_chain_registry().clear()
        yield
        get_chain_registry().clear()

    def test_register_chain_function(self):
        """register_chain should add to global registry."""
        mock_chain = MagicMock(name="MockChain")
        register_chain("test_chain", mock_chain, description="Test chain")

        result = get_chain("test_chain")
        assert result is mock_chain

    def test_register_chain_with_vertical(self):
        """register_chain should add to global registry with vertical."""
        mock_chain = MagicMock(name="MockChain")
        register_chain("test_chain", mock_chain, vertical="coding")

        result = get_chain("test_chain", vertical="coding")
        assert result is mock_chain

    def test_get_chain_function(self):
        """get_chain should retrieve from global registry."""
        mock_chain = MagicMock(name="MockChain")
        register_chain("test_chain", mock_chain)

        result = get_chain("test_chain")
        assert result is mock_chain

    def test_get_chain_returns_none_for_missing(self):
        """get_chain should return None for missing chain."""
        result = get_chain("nonexistent_chain")
        assert result is None

    def test_get_chain_registry_singleton(self):
        """get_chain_registry should return the same instance."""
        registry1 = get_chain_registry()
        registry2 = get_chain_registry()
        assert registry1 is registry2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestChainRegistryThreadSafety:
    """Tests for ChainRegistry thread safety."""

    def test_concurrent_registration(self):
        """Registry should handle concurrent registrations safely."""
        import threading

        registry = ChainRegistry()
        errors = []

        def register_chains(prefix: str, count: int):
            try:
                for i in range(count):
                    registry.register(f"{prefix}_chain_{i}", MagicMock(), replace=True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_chains, args=("thread1", 100)),
            threading.Thread(target=register_chains, args=("thread2", 100)),
            threading.Thread(target=register_chains, args=("thread3", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_chains()) == 300

    def test_concurrent_read_write(self):
        """Registry should handle concurrent reads and writes safely."""
        import threading

        registry = ChainRegistry()
        errors = []

        def writer():
            try:
                for i in range(50):
                    registry.register(f"chain_{i}", MagicMock(), replace=True)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    registry.list_chains()
                    registry.get("chain_0")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
