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

"""Tests for QualityCapability abstraction (Phase 8.1).

TDD tests for:
- QualityCapability dataclass
- Quality threshold checking
- Different enforcement modes
- Integration with CapabilityRegistry
"""

from __future__ import annotations

import pytest

from victor.core.capabilities.quality import (
    QualityCapability,
    QualityResult,
    Enforcement,
)
from victor.core.capabilities.types import CapabilityDefinition, CapabilityType
from victor.core.capabilities.registry import CapabilityRegistry


# =============================================================================
# Test QualityCapability Dataclass
# =============================================================================


class TestQualityCapability:
    """Tests for QualityCapability dataclass."""

    def test_quality_capability_has_required_fields(self):
        """Test that QualityCapability has all required fields."""
        cap = QualityCapability(
            name="test_coverage",
            metric_name="coverage",
            threshold=0.8,
        )

        assert cap.name == "test_coverage"
        assert cap.metric_name == "coverage"
        assert cap.threshold == 0.8
        assert cap.enforcement == Enforcement.WARN  # Default

    def test_check_passes_above_threshold(self):
        """Test that check passes when value is above threshold."""
        cap = QualityCapability(
            name="test_coverage",
            metric_name="coverage",
            threshold=0.8,
        )

        result = cap.check(0.85)

        assert result.passed is True
        assert result.value == 0.85
        assert result.threshold == 0.8
        assert result.enforcement is None

    def test_check_fails_below_threshold(self):
        """Test that check fails when value is below threshold."""
        cap = QualityCapability(
            name="test_coverage",
            metric_name="coverage",
            threshold=0.8,
            enforcement=Enforcement.BLOCK,
        )

        result = cap.check(0.5)

        assert result.passed is False
        assert result.value == 0.5
        assert result.threshold == 0.8
        assert result.enforcement == Enforcement.BLOCK

    def test_check_at_threshold(self):
        """Test that check passes when value equals threshold."""
        cap = QualityCapability(
            name="test_coverage",
            metric_name="coverage",
            threshold=0.8,
        )

        result = cap.check(0.8)

        assert result.passed is True

    def test_different_enforcement_modes(self):
        """Test different enforcement modes."""
        block_cap = QualityCapability(
            name="critical_coverage",
            metric_name="coverage",
            threshold=0.9,
            enforcement=Enforcement.BLOCK,
        )

        warn_cap = QualityCapability(
            name="warn_coverage",
            metric_name="coverage",
            threshold=0.7,
            enforcement=Enforcement.WARN,
        )

        log_cap = QualityCapability(
            name="log_coverage",
            metric_name="coverage",
            threshold=0.5,
            enforcement=Enforcement.LOG,
        )

        # All fail with value 0.3
        assert block_cap.check(0.3).enforcement == Enforcement.BLOCK
        assert warn_cap.check(0.3).enforcement == Enforcement.WARN
        assert log_cap.check(0.3).enforcement == Enforcement.LOG

    def test_quality_capability_for_coding(self):
        """Test QualityCapability for coding vertical (test coverage)."""
        coverage_cap = QualityCapability(
            name="coding_test_coverage",
            metric_name="test_coverage",
            threshold=0.8,
            enforcement=Enforcement.WARN,
            description="Minimum test coverage for code changes",
        )

        # 75% coverage should fail with warning
        result = coverage_cap.check(0.75)
        assert result.passed is False
        assert result.enforcement == Enforcement.WARN

        # 85% coverage should pass
        result = coverage_cap.check(0.85)
        assert result.passed is True

    def test_quality_capability_for_rag(self):
        """Test QualityCapability for RAG vertical (retrieval precision)."""
        precision_cap = QualityCapability(
            name="rag_retrieval_precision",
            metric_name="precision@k",
            threshold=0.7,
            enforcement=Enforcement.BLOCK,
            description="Minimum retrieval precision for RAG queries",
        )

        # Low precision should block
        result = precision_cap.check(0.5)
        assert result.passed is False
        assert result.enforcement == Enforcement.BLOCK

    def test_quality_capability_for_dataanalysis(self):
        """Test QualityCapability for DataAnalysis vertical (data quality)."""
        quality_cap = QualityCapability(
            name="data_completeness",
            metric_name="completeness_ratio",
            threshold=0.95,
            enforcement=Enforcement.LOG,
            description="Minimum data completeness for analysis",
        )

        # Low completeness should log
        result = quality_cap.check(0.9)
        assert result.passed is False
        assert result.enforcement == Enforcement.LOG


# =============================================================================
# Test QualityResult
# =============================================================================


class TestQualityResult:
    """Tests for QualityResult dataclass."""

    def test_result_has_all_fields(self):
        """Test that QualityResult has all expected fields."""
        result = QualityResult(
            passed=True,
            value=0.85,
            threshold=0.8,
            metric_name="coverage",
            enforcement=None,
        )

        assert result.passed is True
        assert result.value == 0.85
        assert result.threshold == 0.8
        assert result.metric_name == "coverage"
        assert result.enforcement is None

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = QualityResult(
            passed=False,
            value=0.5,
            threshold=0.8,
            metric_name="coverage",
            enforcement=Enforcement.BLOCK,
        )

        d = result.to_dict()

        assert d["passed"] is False
        assert d["value"] == 0.5
        assert d["threshold"] == 0.8
        assert d["metric_name"] == "coverage"
        assert d["enforcement"] == "block"

    def test_result_message_for_pass(self):
        """Test message generation for passing result."""
        result = QualityResult(
            passed=True,
            value=0.85,
            threshold=0.8,
            metric_name="coverage",
        )

        message = result.get_message()
        assert "passed" in message.lower() or "met" in message.lower()

    def test_result_message_for_fail(self):
        """Test message generation for failing result."""
        result = QualityResult(
            passed=False,
            value=0.5,
            threshold=0.8,
            metric_name="coverage",
            enforcement=Enforcement.BLOCK,
        )

        message = result.get_message()
        assert "below" in message.lower() or "failed" in message.lower()


# =============================================================================
# Test Integration with CapabilityRegistry
# =============================================================================


class TestQualityCapabilityRegistryIntegration:
    """Tests for QualityCapability integration with CapabilityRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        CapabilityRegistry.reset_instance()
        yield
        CapabilityRegistry.reset_instance()

    def test_to_capability_definition(self):
        """Test converting QualityCapability to CapabilityDefinition."""
        cap = QualityCapability(
            name="test_coverage",
            metric_name="coverage",
            threshold=0.8,
            enforcement=Enforcement.WARN,
            description="Minimum test coverage",
        )

        definition = cap.to_capability_definition()

        assert isinstance(definition, CapabilityDefinition)
        assert definition.name == "test_coverage"
        assert definition.capability_type == CapabilityType.MODE
        assert definition.description == "Minimum test coverage"
        assert "threshold" in definition.default_config
        assert definition.default_config["threshold"] == 0.8

    def test_register_quality_capability(self):
        """Test registering QualityCapability with registry."""
        registry = CapabilityRegistry.get_instance()

        cap = QualityCapability(
            name="quality_check",
            metric_name="coverage",
            threshold=0.8,
        )

        definition = cap.to_capability_definition()
        registry.register(definition)

        # Should be retrievable
        retrieved = registry.get("quality_check")
        assert retrieved is not None
        assert retrieved.name == "quality_check"

    def test_different_verticals_use_same_pattern(self):
        """Test that different verticals can use the same QualityCapability pattern."""
        coding_cap = QualityCapability(
            name="coding_coverage",
            metric_name="test_coverage",
            threshold=0.8,
        )

        rag_cap = QualityCapability(
            name="rag_precision",
            metric_name="precision",
            threshold=0.7,
        )

        data_cap = QualityCapability(
            name="data_completeness",
            metric_name="completeness",
            threshold=0.95,
        )

        # All should produce valid CapabilityDefinitions
        for cap in [coding_cap, rag_cap, data_cap]:
            definition = cap.to_capability_definition()
            assert isinstance(definition, CapabilityDefinition)
            assert "threshold" in definition.default_config

    def test_quality_registry_integration(self):
        """Test full registry integration with multiple quality capabilities."""
        registry = CapabilityRegistry.get_instance()

        # Register multiple quality capabilities
        caps = [
            QualityCapability(
                name="coding_coverage",
                metric_name="test_coverage",
                threshold=0.8,
            ),
            QualityCapability(
                name="rag_recall",
                metric_name="recall",
                threshold=0.75,
            ),
        ]

        for cap in caps:
            registry.register(cap.to_capability_definition())

        # All should be registered
        assert registry.get("coding_coverage") is not None
        assert registry.get("rag_recall") is not None


# =============================================================================
# Test Enforcement Enum
# =============================================================================


class TestEnforcement:
    """Tests for Enforcement enum."""

    def test_enforcement_values(self):
        """Test that Enforcement has expected values."""
        assert Enforcement.BLOCK.value == "block"
        assert Enforcement.WARN.value == "warn"
        assert Enforcement.LOG.value == "log"

    def test_enforcement_from_string(self):
        """Test creating Enforcement from string."""
        assert Enforcement.from_string("block") == Enforcement.BLOCK
        assert Enforcement.from_string("WARN") == Enforcement.WARN
        assert Enforcement.from_string("Log") == Enforcement.LOG

    def test_enforcement_from_string_invalid(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Enforcement.from_string("invalid")
