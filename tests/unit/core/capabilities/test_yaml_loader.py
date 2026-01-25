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

"""Tests for YAML capability loader (Phase 5.2)."""

import tempfile
from pathlib import Path

import pytest

from victor.core.capabilities import (
    CapabilityRegistry,
    CapabilityType,
    get_vertical_capability_path,
    register_vertical_capabilities,
    register_all_vertical_capabilities,
)


@pytest.fixture
def fresh_registry():
    """Create a fresh registry for testing."""
    return CapabilityRegistry()


class TestGetVerticalCapabilityPath:
    """Tests for get_vertical_capability_path function."""

    def test_coding_vertical_path(self):
        """Test path generation for coding vertical."""
        path = get_vertical_capability_path("coding")
        assert path.name == "capabilities.yaml"
        assert "coding" in str(path)
        assert "config" in str(path)

    def test_devops_vertical_path(self):
        """Test path generation for devops vertical."""
        path = get_vertical_capability_path("devops")
        assert path.name == "capabilities.yaml"
        assert "devops" in str(path)

    def test_research_vertical_path(self):
        """Test path generation for research vertical."""
        path = get_vertical_capability_path("research")
        assert path.name == "capabilities.yaml"
        assert "research" in str(path)

    def test_rag_vertical_path(self):
        """Test path generation for rag vertical."""
        path = get_vertical_capability_path("rag")
        assert path.name == "capabilities.yaml"
        assert "rag" in str(path)

    def test_dataanalysis_vertical_path(self):
        """Test path generation for dataanalysis vertical."""
        path = get_vertical_capability_path("dataanalysis")
        assert path.name == "capabilities.yaml"
        assert "dataanalysis" in str(path)


class TestRegisterVerticalCapabilities:
    """Tests for register_vertical_capabilities function."""

    def test_register_coding_capabilities(self, fresh_registry):
        """Test registering coding vertical capabilities."""
        count = register_vertical_capabilities("coding", registry=fresh_registry)
        assert count > 0
        # Check some expected capabilities
        cap = fresh_registry.get("coding_git_safety")
        assert cap is not None
        assert cap.capability_type == CapabilityType.SAFETY

    def test_register_devops_capabilities(self, fresh_registry):
        """Test registering devops vertical capabilities."""
        count = register_vertical_capabilities("devops", registry=fresh_registry)
        assert count > 0
        cap = fresh_registry.get("devops_deployment_safety")
        assert cap is not None
        assert cap.capability_type == CapabilityType.SAFETY

    def test_register_research_capabilities(self, fresh_registry):
        """Test registering research vertical capabilities."""
        count = register_vertical_capabilities("research", registry=fresh_registry)
        assert count > 0
        cap = fresh_registry.get("research_source_verification")
        assert cap is not None
        assert cap.capability_type == CapabilityType.SAFETY

    def test_register_rag_capabilities(self, fresh_registry):
        """Test registering rag vertical capabilities."""
        count = register_vertical_capabilities("rag", registry=fresh_registry)
        assert count > 0
        cap = fresh_registry.get("rag_indexing")
        assert cap is not None
        assert cap.capability_type == CapabilityType.MODE

    def test_register_dataanalysis_capabilities(self, fresh_registry):
        """Test registering dataanalysis vertical capabilities."""
        count = register_vertical_capabilities("dataanalysis", registry=fresh_registry)
        assert count > 0
        cap = fresh_registry.get("dataanalysis_data_quality")
        assert cap is not None
        assert cap.capability_type == CapabilityType.MODE

    def test_register_nonexistent_vertical_returns_zero(self, fresh_registry):
        """Test registering from nonexistent vertical returns 0."""
        count = register_vertical_capabilities("nonexistent", registry=fresh_registry)
        assert count == 0

    def test_register_with_custom_path(self, fresh_registry):
        """Test registering from custom YAML path."""
        # Create a temp YAML file
        yaml_content = """
version: "1.0"
capabilities:
  - name: custom_capability
    capability_type: tool
    description: A custom capability
    default_config:
      enabled: true
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            count = register_vertical_capabilities(
                "custom", yaml_path=temp_path, registry=fresh_registry
            )
            assert count == 1
            cap = fresh_registry.get("custom_capability")
            assert cap is not None
            assert cap.capability_type == CapabilityType.TOOL
        finally:
            temp_path.unlink()

    def test_register_with_invalid_yaml_returns_zero(self, fresh_registry):
        """Test registering from invalid YAML returns 0."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            count = register_vertical_capabilities(
                "invalid", yaml_path=temp_path, registry=fresh_registry
            )
            assert count == 0
        finally:
            temp_path.unlink()


class TestRegisterAllVerticalCapabilities:
    """Tests for register_all_vertical_capabilities function."""

    def test_register_all_verticals(self, fresh_registry):
        """Test registering all vertical capabilities."""
        total = register_all_vertical_capabilities(registry=fresh_registry)
        # Should have capabilities from all 5 verticals
        assert total >= 20  # At least 4 capabilities per vertical

    def test_all_verticals_have_capabilities(self, fresh_registry):
        """Test that all known verticals have capabilities registered."""
        register_all_vertical_capabilities(registry=fresh_registry)

        # Check at least one capability from each vertical
        assert fresh_registry.get("coding_git_safety") is not None
        assert fresh_registry.get("devops_deployment_safety") is not None
        assert fresh_registry.get("research_source_verification") is not None
        assert fresh_registry.get("rag_indexing") is not None
        assert fresh_registry.get("dataanalysis_data_quality") is not None


class TestCapabilityYAMLContents:
    """Tests verifying the content of capability YAML files."""

    def test_coding_yaml_has_expected_capabilities(self, fresh_registry):
        """Test coding YAML has expected capability definitions."""
        register_vertical_capabilities("coding", registry=fresh_registry)

        # Verify specific capabilities exist
        git_safety = fresh_registry.get("coding_git_safety")
        assert git_safety is not None
        assert git_safety.capability_type == CapabilityType.SAFETY
        assert "git" in git_safety.description.lower()

        code_style = fresh_registry.get("coding_code_style")
        assert code_style is not None
        assert code_style.capability_type == CapabilityType.MODE

    def test_devops_yaml_has_expected_capabilities(self, fresh_registry):
        """Test devops YAML has expected capability definitions."""
        register_vertical_capabilities("devops", registry=fresh_registry)

        deployment_safety = fresh_registry.get("devops_deployment_safety")
        assert deployment_safety is not None
        assert deployment_safety.capability_type == CapabilityType.SAFETY
        assert "protected_environments" in str(deployment_safety.default_config)

        container = fresh_registry.get("devops_container")
        assert container is not None
        assert container.capability_type == CapabilityType.TOOL

    def test_rag_yaml_has_expected_capabilities(self, fresh_registry):
        """Test rag YAML has expected capability definitions."""
        register_vertical_capabilities("rag", registry=fresh_registry)

        indexing = fresh_registry.get("rag_indexing")
        assert indexing is not None
        assert indexing.capability_type == CapabilityType.MODE
        assert "chunk_size" in str(indexing.default_config)

        retrieval = fresh_registry.get("rag_retrieval")
        assert retrieval is not None
        assert "rag_indexing" in retrieval.dependencies

    def test_research_yaml_has_expected_capabilities(self, fresh_registry):
        """Test research YAML has expected capability definitions."""
        register_vertical_capabilities("research", registry=fresh_registry)

        source_verification = fresh_registry.get("research_source_verification")
        assert source_verification is not None
        assert source_verification.capability_type == CapabilityType.SAFETY

        quality = fresh_registry.get("research_quality")
        assert quality is not None
        assert "research_source_verification" in quality.dependencies

    def test_dataanalysis_yaml_has_expected_capabilities(self, fresh_registry):
        """Test dataanalysis YAML has expected capability definitions."""
        register_vertical_capabilities("dataanalysis", registry=fresh_registry)

        data_quality = fresh_registry.get("dataanalysis_data_quality")
        assert data_quality is not None
        assert data_quality.capability_type == CapabilityType.MODE

        ml_pipeline = fresh_registry.get("dataanalysis_ml_pipeline")
        assert ml_pipeline is not None
        assert ml_pipeline.capability_type == CapabilityType.TOOL
        assert "dataanalysis_data_quality" in ml_pipeline.dependencies


class TestCapabilityConfigSchemas:
    """Tests for config schema validation in YAML capabilities."""

    def test_coding_capability_has_config_schema(self, fresh_registry):
        """Test coding capabilities have config schemas."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        code_style = fresh_registry.get("coding_code_style")
        assert code_style is not None
        assert code_style.config_schema is not None
        assert "properties" in code_style.config_schema

    def test_rag_capability_has_config_schema(self, fresh_registry):
        """Test RAG capabilities have config schemas."""
        register_vertical_capabilities("rag", registry=fresh_registry)
        indexing = fresh_registry.get("rag_indexing")
        assert indexing is not None
        assert indexing.config_schema is not None
        assert "chunk_size" in str(indexing.config_schema)

    def test_dataanalysis_capability_has_config_schema(self, fresh_registry):
        """Test dataanalysis capabilities have config schemas."""
        register_vertical_capabilities("dataanalysis", registry=fresh_registry)
        stats = fresh_registry.get("dataanalysis_statistical_analysis")
        assert stats is not None
        assert stats.config_schema is not None
        assert "significance_level" in str(stats.config_schema)
