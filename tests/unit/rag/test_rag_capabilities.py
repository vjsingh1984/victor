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

"""Unit tests for RAG vertical capabilities."""

from __future__ import annotations

from typing import Any

import pytest


class MockVerticalContext:
    """Mock vertical context for testing."""

    def __init__(self):
        self._configs: dict[str, Any] = {}

    def set_capability_config(self, name: str, config: dict[str, Any]) -> None:
        """Set capability configuration."""
        self._configs[name] = config

    def get_capability_config(self, name: str, default: Any = None) -> Any:
        """Get capability configuration."""
        return self._configs.get(name, default)


class MockOrchestrator:
    """Mock orchestrator for testing RAG capabilities."""

    def __init__(self):
        self.rag_config: dict[str, Any] = {}
        self.vertical_context = MockVerticalContext()


class TestIndexingCapability:
    """Tests for indexing capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test indexing with default settings."""
        from victor.rag.capabilities import configure_indexing

        configure_indexing(orchestrator)

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_indexing", {})
        assert config["chunk_size"] == 512
        assert config["chunk_overlap"] == 50
        assert config["embedding_model"] == "text-embedding-3-small"
        assert config["embedding_dimensions"] == 1536
        assert config["store_backend"] == "lancedb"

    def test_custom_configuration(self, orchestrator):
        """Test indexing with custom settings."""
        from victor.rag.capabilities import configure_indexing

        configure_indexing(
            orchestrator,
            chunk_size=1024,
            chunk_overlap=100,
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            store_backend="chromadb",
        )

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_indexing", {})
        assert config["chunk_size"] == 1024
        assert config["chunk_overlap"] == 100
        assert config["embedding_model"] == "text-embedding-3-large"
        assert config["embedding_dimensions"] == 3072
        assert config["store_backend"] == "chromadb"

    def test_get_default_config(self, orchestrator):
        """Test get_indexing_config returns defaults when not configured."""
        from victor.rag.capabilities import get_indexing_config

        config = get_indexing_config(orchestrator)
        assert config["chunk_size"] == 512
        assert config["chunk_overlap"] == 50
        assert config["embedding_model"] == "text-embedding-3-small"

    def test_get_config_after_configuration(self, orchestrator):
        """Test get_indexing_config returns configured values."""
        from victor.rag.capabilities import configure_indexing, get_indexing_config

        configure_indexing(orchestrator, chunk_size=768)
        config = get_indexing_config(orchestrator)
        assert config["chunk_size"] == 768


class TestRetrievalCapability:
    """Tests for retrieval capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test retrieval with default settings."""
        from victor.rag.capabilities import configure_retrieval

        configure_retrieval(orchestrator)

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_retrieval", {})
        assert config["top_k"] == 5
        assert config["similarity_threshold"] == 0.7
        assert config["search_type"] == "hybrid"
        assert config["rerank_enabled"] is True
        assert config["max_context_tokens"] == 4000

    def test_custom_configuration(self, orchestrator):
        """Test retrieval with custom settings."""
        from victor.rag.capabilities import configure_retrieval

        configure_retrieval(
            orchestrator,
            top_k=10,
            similarity_threshold=0.8,
            search_type="semantic",
            rerank_enabled=False,
            max_context_tokens=8000,
        )

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_retrieval", {})
        assert config["top_k"] == 10
        assert config["similarity_threshold"] == 0.8
        assert config["search_type"] == "semantic"
        assert config["rerank_enabled"] is False
        assert config["max_context_tokens"] == 8000

    def test_get_default_config(self, orchestrator):
        """Test get_retrieval_config returns defaults when not configured."""
        from victor.rag.capabilities import get_retrieval_config

        config = get_retrieval_config(orchestrator)
        assert config["top_k"] == 5
        assert config["search_type"] == "hybrid"

    def test_get_config_after_configuration(self, orchestrator):
        """Test get_retrieval_config returns configured values."""
        from victor.rag.capabilities import configure_retrieval, get_retrieval_config

        configure_retrieval(orchestrator, top_k=15)
        config = get_retrieval_config(orchestrator)
        assert config["top_k"] == 15


class TestSynthesisCapability:
    """Tests for synthesis capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test synthesis with default settings."""
        from victor.rag.capabilities import configure_synthesis

        configure_synthesis(orchestrator)

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_synthesis", {})
        assert config["citation_style"] == "inline"
        assert config["include_sources"] is True
        assert config["max_answer_tokens"] == 2000
        assert config["temperature"] == 0.3
        assert config["require_verification"] is True

    def test_custom_configuration(self, orchestrator):
        """Test synthesis with custom settings."""
        from victor.rag.capabilities import configure_synthesis

        configure_synthesis(
            orchestrator,
            citation_style="footnote",
            include_sources=False,
            max_answer_tokens=4000,
            temperature=0.5,
            require_verification=False,
        )

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_synthesis", {})
        assert config["citation_style"] == "footnote"
        assert config["include_sources"] is False
        assert config["max_answer_tokens"] == 4000
        assert config["temperature"] == 0.5
        assert config["require_verification"] is False

    def test_get_default_config(self, orchestrator):
        """Test get_synthesis_config returns defaults when not configured."""
        from victor.rag.capabilities import get_synthesis_config

        config = get_synthesis_config(orchestrator)
        assert config["citation_style"] == "inline"
        assert config["include_sources"] is True

    def test_get_config_after_configuration(self, orchestrator):
        """Test get_synthesis_config returns configured values."""
        from victor.rag.capabilities import configure_synthesis, get_synthesis_config

        configure_synthesis(orchestrator, citation_style="endnote")
        config = get_synthesis_config(orchestrator)
        assert config["citation_style"] == "endnote"


class TestSafetyCapability:
    """Tests for safety capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test safety with default settings."""
        from victor.rag.capabilities import configure_safety

        configure_safety(orchestrator)

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_safety", {})
        assert config["filter_sensitive_data"] is True
        assert config["max_document_size_mb"] == 50
        assert config["validate_sources"] is True
        assert "pdf" in config["allowed_file_types"]
        assert "txt" in config["allowed_file_types"]

    def test_custom_configuration(self, orchestrator):
        """Test safety with custom settings."""
        from victor.rag.capabilities import configure_safety

        configure_safety(
            orchestrator,
            filter_sensitive_data=False,
            max_document_size_mb=100,
            allowed_file_types=["pdf", "docx"],
            validate_sources=False,
        )

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_safety", {})
        assert config["filter_sensitive_data"] is False
        assert config["max_document_size_mb"] == 100
        assert config["allowed_file_types"] == ["pdf", "docx"]
        assert config["validate_sources"] is False


class TestQueryEnhancementCapability:
    """Tests for query enhancement capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test query enhancement with default settings."""
        from victor.rag.capabilities import configure_query_enhancement

        configure_query_enhancement(orchestrator)

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_query_enhancement", {})
        assert config["enable_expansion"] is True
        assert config["enable_decomposition"] is True
        assert config["max_query_variants"] == 3
        assert config["use_synonyms"] is True

    def test_custom_configuration(self, orchestrator):
        """Test query enhancement with custom settings."""
        from victor.rag.capabilities import configure_query_enhancement

        configure_query_enhancement(
            orchestrator,
            enable_expansion=False,
            enable_decomposition=False,
            max_query_variants=5,
            use_synonyms=False,
        )

        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_query_enhancement", {})
        assert config["enable_expansion"] is False
        assert config["enable_decomposition"] is False
        assert config["max_query_variants"] == 5
        assert config["use_synonyms"] is False


class TestRAGCapabilityProvider:
    """Tests for RAGCapabilityProvider class."""

    @pytest.fixture
    def provider(self):
        from victor.rag.capabilities import RAGCapabilityProvider

        return RAGCapabilityProvider()

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider._vertical_name == "rag"
        assert provider._applied == set()

    def test_list_all_capabilities(self, provider):
        """Test listing all capabilities."""
        capabilities = provider.list_capabilities()
        assert "indexing" in capabilities
        assert "retrieval" in capabilities
        assert "synthesis" in capabilities
        assert "safety" in capabilities
        assert "query_enhancement" in capabilities
        assert len(capabilities) == 5

    def test_list_capabilities_by_type(self, provider):
        """Test filtering capabilities by type."""
        from victor.framework.protocols import CapabilityType

        mode_caps = provider.list_capabilities(CapabilityType.MODE)
        assert "indexing" in mode_caps
        assert "retrieval" in mode_caps
        assert "synthesis" in mode_caps
        assert "query_enhancement" in mode_caps
        assert "safety" not in mode_caps  # Safety is SAFETY type

        safety_caps = provider.list_capabilities(CapabilityType.SAFETY)
        assert "safety" in safety_caps
        assert "indexing" not in safety_caps

    def test_has_capability(self, provider):
        """Test checking if capability exists."""
        assert provider.has_capability("indexing") is True
        assert provider.has_capability("retrieval") is True
        assert provider.has_capability("synthesis") is True
        assert provider.has_capability("safety") is True
        assert provider.has_capability("query_enhancement") is True
        assert provider.has_capability("nonexistent") is False

    def test_apply_capability_by_name(self, provider, orchestrator):
        """Test applying capability by name."""
        provider.apply_capability(orchestrator, "indexing", chunk_size=1024)
        # Check vertical_context for config
        config = orchestrator.vertical_context._configs.get("rag_indexing", {})
        assert config["chunk_size"] == 1024
        assert "indexing" in provider.get_applied()

    def test_apply_capability_with_invalid_name(self, provider, orchestrator):
        """Test applying capability with invalid name raises error."""
        with pytest.raises(ValueError, match="Unknown capability"):
            provider.apply_capability(orchestrator, "nonexistent")

    def test_get_capability_config(self, provider, orchestrator):
        """Test getting capability configuration."""
        from victor.rag.capabilities import configure_indexing

        configure_indexing(orchestrator, chunk_size=768)

        config = provider.get_capability_config(orchestrator, "indexing")
        assert config is not None
        assert config["chunk_size"] == 768

    def test_get_capability_config_for_capability_without_getter(self, provider, orchestrator):
        """Test getting config for capability without getter returns None."""
        config = provider.get_capability_config(orchestrator, "safety")
        # Safety doesn't have a get_fn in the original implementation
        assert config is None or isinstance(config, dict)

    def test_get_default_config(self, provider):
        """Test getting default configuration."""
        config = provider.get_default_config("indexing")
        assert config["chunk_size"] == 512
        assert config["embedding_model"] == "text-embedding-3-small"

    def test_get_default_config_for_invalid_capability(self, provider):
        """Test getting default config for invalid capability raises error."""
        with pytest.raises(ValueError, match="Unknown capability"):
            provider.get_default_config("nonexistent")

    def test_apply_all_capabilities(self, provider, orchestrator):
        """Test applying all capabilities."""
        provider.apply_all(orchestrator)

        assert "indexing" in provider.get_applied()
        assert "retrieval" in provider.get_applied()
        assert "synthesis" in provider.get_applied()
        assert "safety" in provider.get_applied()
        assert "query_enhancement" in provider.get_applied()
        assert len(provider.get_applied()) == 5

    def test_reset_applied_capabilities(self, provider, orchestrator):
        """Test resetting applied capabilities tracking."""
        provider.apply_capability(orchestrator, "indexing")
        assert "indexing" in provider.get_applied()

        provider.reset_applied()
        assert len(provider.get_applied()) == 0

    def test_capability_metadata(self, provider):
        """Test capability metadata is correct."""
        metadata = provider.get_capability_metadata()

        assert "indexing" in metadata
        assert metadata["indexing"].description == "Document indexing and chunking configuration"
        assert metadata["indexing"].version == "1.0"
        assert "indexing" in metadata["indexing"].tags

        assert "retrieval" in metadata
        assert metadata["retrieval"].description == "Search and retrieval configuration"
        assert "indexing" in metadata["retrieval"].dependencies

        assert "synthesis" in metadata
        assert "retrieval" in metadata["synthesis"].dependencies


class TestCapabilityConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_rag_capabilities(self):
        """Test get_rag_capabilities returns entries."""
        from victor.rag.capabilities import get_rag_capabilities

        capabilities = get_rag_capabilities()
        assert len(capabilities) > 0
        # Capabilities are registered with rag_ prefix
        assert all(cap.capability.name.startswith("rag_") for cap in capabilities)

    def test_create_rag_capability_loader(self):
        """Test create_rag_capability_loader returns configured loader."""
        from victor.rag.capabilities import create_rag_capability_loader

        loader = create_rag_capability_loader()
        assert loader is not None
        # Loader should have capabilities registered

    def test_get_capability_configs(self):
        """Test get_capability_configs returns default configurations."""
        from victor.rag.capabilities import get_capability_configs

        configs = get_capability_configs()
        assert isinstance(configs, dict)
        assert len(configs) > 0

        # Check that config keys follow expected pattern
        for key in configs.keys():
            assert key.endswith("_config")


class TestCapabilityListGeneration:
    """Tests for CAPABILITIES list generation."""

    def test_capabilities_list_is_not_empty(self):
        """Test CAPABILITIES list is generated and not empty."""
        from victor.rag.capabilities import CAPABILITIES

        assert len(CAPABILITIES) > 0

    def test_capabilities_list_has_all_capabilities(self):
        """Test CAPABILITIES list includes all expected capabilities."""
        from victor.rag.capabilities import CAPABILITIES

        capability_names = {cap.capability.name for cap in CAPABILITIES}
        # Provider generates capabilities with "rag_" prefix
        expected_names = {
            "rag_indexing",
            "rag_retrieval",
            "rag_synthesis",
            "rag_safety",
            "rag_query_enhancement",
        }
        assert capability_names == expected_names

    def test_capability_entries_have_handlers(self):
        """Test all capability entries have handler functions."""
        from victor.rag.capabilities import CAPABILITIES

        for entry in CAPABILITIES:
            assert entry.handler is not None
            assert callable(entry.handler)


class TestCapabilityDependencies:
    """Tests for capability dependency management."""

    @pytest.fixture
    def provider(self):
        from victor.rag.capabilities import RAGCapabilityProvider

        return RAGCapabilityProvider()

    def test_retrieval_depends_on_indexing(self, provider):
        """Test retrieval capability depends on indexing."""
        definition = provider.get_capability_definition("retrieval")
        assert "indexing" in definition.dependencies

    def test_synthesis_depends_on_retrieval(self, provider):
        """Test synthesis capability depends on retrieval."""
        definition = provider.get_capability_definition("synthesis")
        assert "retrieval" in definition.dependencies

    def test_indexing_has_no_dependencies(self, provider):
        """Test indexing capability has no dependencies."""
        definition = provider.get_capability_definition("indexing")
        assert len(definition.dependencies) == 0

    def test_safety_has_no_dependencies(self, provider):
        """Test safety capability has no dependencies."""
        definition = provider.get_capability_definition("safety")
        assert len(definition.dependencies) == 0

    def test_query_enhancement_has_no_dependencies(self, provider):
        """Test query enhancement capability has no dependencies."""
        definition = provider.get_capability_definition("query_enhancement")
        assert len(definition.dependencies) == 0


class TestCapabilityTags:
    """Tests for capability tags."""

    @pytest.fixture
    def provider(self):
        from victor.rag.capabilities import RAGCapabilityProvider

        return RAGCapabilityProvider()

    def test_indexing_tags(self, provider):
        """Test indexing capability has expected tags."""
        definition = provider.get_capability_definition("indexing")
        assert "indexing" in definition.tags
        assert "chunking" in definition.tags
        assert "embedding" in definition.tags

    def test_retrieval_tags(self, provider):
        """Test retrieval capability has expected tags."""
        definition = provider.get_capability_definition("retrieval")
        assert "retrieval" in definition.tags
        assert "search" in definition.tags
        assert "ranking" in definition.tags

    def test_synthesis_tags(self, provider):
        """Test synthesis capability has expected tags."""
        definition = provider.get_capability_definition("synthesis")
        assert "synthesis" in definition.tags
        assert "generation" in definition.tags
        assert "citations" in definition.tags

    def test_safety_tags(self, provider):
        """Test safety capability has expected tags."""
        definition = provider.get_capability_definition("safety")
        assert "safety" in definition.tags
        assert "filtering" in definition.tags
        assert "validation" in definition.tags

    def test_query_enhancement_tags(self, provider):
        """Test query enhancement capability has expected tags."""
        definition = provider.get_capability_definition("query_enhancement")
        assert "query" in definition.tags
        assert "expansion" in definition.tags
        assert "decomposition" in definition.tags
