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

"""Unit tests for DataAnalysis capabilities module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestDataQualityCapability:
    """Tests for data quality capability configuration."""

    def test_configure_data_quality_default(self):
        """Test configure_data_quality with default values."""
        from victor.dataanalysis.capabilities import configure_data_quality

        orchestrator = MagicMock()
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config
        orchestrator.vertical_context = mock_context
        configure_data_quality(orchestrator)

       # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "data_quality"
        config = call_args[0][1]
        assert config["min_completeness"] == 0.9
        assert config["max_outlier_ratio"] == 0.05
        assert config["require_type_validation"] is True
        assert config["handle_missing"] == "impute"

    def test_configure_data_quality_custom(self):
        """Test configure_data_quality with custom values."""
        from victor.dataanalysis.capabilities import configure_data_quality

        orchestrator = MagicMock()
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config
        orchestrator.vertical_context = mock_context
        configure_data_quality(
            orchestrator,
            min_completeness=0.8,
            max_outlier_ratio=0.1,
            require_type_validation=False,
            handle_missing="drop",
        )

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "data_quality"
        config = call_args[0][1]
        assert config["min_completeness"] == 0.8
        assert config["max_outlier_ratio"] == 0.1
        assert config["require_type_validation"] is False
        assert config["handle_missing"] == "drop"

    def test_get_data_quality_default(self):
        """Test get_data_quality returns default when not configured."""
        from victor.dataanalysis.capabilities import get_data_quality

        mock_context = MagicMock()
        default_config = {
            "min_completeness": 0.9,
            "max_outlier_ratio": 0.05,
            "require_type_validation": True,
            "handle_missing": "impute",
        }
        mock_context.get_capability_config.return_value = default_config
        
        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        config = get_data_quality(orchestrator)

        assert config["min_completeness"] == 0.9
        assert config["handle_missing"] == "impute"


class TestVisualizationStyleCapability:
    """Tests for visualization style capability."""

    def test_configure_visualization_style_default(self):
        """Test configure_visualization_style with defaults."""
        from victor.dataanalysis.capabilities import configure_visualization_style

        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config

        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        configure_visualization_style(orchestrator)

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "visualization_style"
        config = call_args[0][1]
        assert config["backend"] == "matplotlib"
        assert config["theme"] == "seaborn-v0_8-whitegrid"
        assert config["figure_size"] == (10, 6)
        assert config["dpi"] == 100
        assert config["save_format"] == "png"

    def test_configure_visualization_style_custom(self):
        """Test configure_visualization_style with custom values."""
        from victor.dataanalysis.capabilities import configure_visualization_style

        orchestrator = MagicMock()
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config
        orchestrator.vertical_context = mock_context
        configure_visualization_style(
            orchestrator,
            default_backend="plotly",
            theme="dark",
            figure_size=(12, 8),
            dpi=150,
            save_format="svg",
        )

       
        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "visualization_style"
        config = call_args[0][1]
        assert config["backend"] == "plotly"
        assert config["theme"] == "dark"
        assert config["figure_size"] == (12, 8)
        assert config["dpi"] == 150
        assert config["save_format"] == "svg"


class TestStatisticalAnalysisCapability:
    """Tests for statistical analysis capability."""

    def test_configure_statistical_analysis_default(self):
        """Test configure_statistical_analysis with defaults."""
        from victor.dataanalysis.capabilities import configure_statistical_analysis

        orchestrator = MagicMock()
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config
        orchestrator.vertical_context = mock_context
        configure_statistical_analysis(orchestrator)

       
        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "statistical_analysis"
        config = call_args[0][1]
        assert config["significance_level"] == 0.05
        assert config["confidence_interval"] == 0.95
        assert config["multiple_testing_correction"] == "bonferroni"
        assert config["effect_size_threshold"] == 0.2

    def test_configure_statistical_analysis_custom(self):
        """Test configure_statistical_analysis with custom values."""
        from victor.dataanalysis.capabilities import configure_statistical_analysis

        orchestrator = MagicMock()
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config
        orchestrator.vertical_context = mock_context
        configure_statistical_analysis(
            orchestrator,
            significance_level=0.01,
            confidence_interval=0.99,
            multiple_testing_correction="fdr_bh",
            effect_size_threshold=0.3,
        )

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "statistical_analysis"
        config = call_args[0][1]
        assert config["significance_level"] == 0.01
        assert config["confidence_interval"] == 0.99
        assert config["multiple_testing_correction"] == "fdr_bh"
        assert config["effect_size_threshold"] == 0.3


class TestMLPipelineCapability:
    """Tests for ML pipeline capability."""

    def test_configure_ml_pipeline_default(self):
        """Test configure_ml_pipeline with defaults."""
        from victor.dataanalysis.capabilities import configure_ml_pipeline

        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config

        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        configure_ml_pipeline(orchestrator)

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "ml_pipeline"
        config = call_args[0][1]
        assert config["framework"] == "sklearn"
        assert config["cv_folds"] == 5
        assert config["test_size"] == 0.2
        assert config["random_state"] == 42
        assert config["hyperparameter_tuning"] is True
        assert config["tuning_method"] == "grid"

    def test_configure_ml_pipeline_custom(self):
        """Test configure_ml_pipeline with custom values."""
        from victor.dataanalysis.capabilities import configure_ml_pipeline

        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config

        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        configure_ml_pipeline(
            orchestrator,
            default_framework="xgboost",
            cross_validation_folds=10,
            test_size=0.3,
            random_state=123,
            enable_hyperparameter_tuning=False,
            tuning_method="bayesian",
        )

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "ml_pipeline"
        config = call_args[0][1]
        assert config["framework"] == "xgboost"
        assert config["cv_folds"] == 10
        assert config["test_size"] == 0.3
        assert config["random_state"] == 123
        assert config["hyperparameter_tuning"] is False
        assert config["tuning_method"] == "bayesian"


class TestDataPrivacyCapability:
    """Tests for data privacy capability."""

    def test_configure_data_privacy_default(self):
        """Test configure_data_privacy with defaults."""
        from victor.dataanalysis.capabilities import configure_data_privacy

        orchestrator = MagicMock()
        orchestrator.privacy_config = {}

        configure_data_privacy(orchestrator)

        config = orchestrator.privacy_config
        assert config["anonymize_pii"] is True
        assert config["pii_columns"] == []
        assert config["hash_identifiers"] is True
        assert config["log_access"] is True

    def test_configure_data_privacy_custom(self):
        """Test configure_data_privacy with custom values."""
        from victor.dataanalysis.capabilities import configure_data_privacy

        orchestrator = MagicMock()
        orchestrator.privacy_config = {}

        configure_data_privacy(
            orchestrator,
            anonymize_pii=False,
            pii_columns=["name", "email", "ssn"],
            hash_identifiers=False,
            log_access=False,
        )

        config = orchestrator.privacy_config
        assert config["anonymize_pii"] is False
        assert config["pii_columns"] == ["name", "email", "ssn"]
        assert config["hash_identifiers"] is False
        assert config["log_access"] is False


class TestDataAnalysisCapabilityProvider:
    """Tests for DataAnalysisCapabilityProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        from victor.dataanalysis.capabilities import DataAnalysisCapabilityProvider

        return DataAnalysisCapabilityProvider()

    def test_get_capabilities(self, provider):
        """Test get_capabilities returns all capabilities."""
        capabilities = provider.get_capabilities()

        assert "data_quality" in capabilities
        assert "visualization_style" in capabilities
        assert "statistical_analysis" in capabilities
        assert "ml_pipeline" in capabilities
        assert "data_privacy" in capabilities
        assert len(capabilities) == 5

    def test_get_capability_metadata(self, provider):
        """Test get_capability_metadata returns metadata for all capabilities."""
        metadata = provider.get_capability_metadata()

        assert "data_quality" in metadata
        assert metadata["data_quality"].name == "data_quality"
        assert "quality" in metadata["data_quality"].tags

        assert "ml_pipeline" in metadata
        assert "data_quality" in metadata["ml_pipeline"].dependencies

    def test_list_capabilities(self, provider):
        """Test list_capabilities returns capability names."""
        names = provider.list_capabilities()

        assert len(names) == 5
        assert "data_quality" in names
        assert "ml_pipeline" in names

    def test_has_capability(self, provider):
        """Test has_capability returns correct boolean."""
        assert provider.has_capability("data_quality") is True
        assert provider.has_capability("nonexistent") is False

    def test_get_capability(self, provider):
        """Test get_capability returns the capability function."""
        cap = provider.get_capability("data_quality")

        assert cap is not None
        assert callable(cap)

    def test_get_capability_nonexistent(self, provider):
        """Test get_capability returns None for nonexistent capability."""
        cap = provider.get_capability("nonexistent")

        assert cap is None

    def test_get_capability_definition(self, provider):
        """Test get_capability_definition returns capability definition."""
        definition = provider.get_capability_definition("data_quality")

        assert definition is not None
        assert definition.name == "data_quality"
        assert definition.type.name == "MODE"
        assert definition.configure_fn == "configure_data_quality"
        assert definition.get_fn == "get_data_quality"

    def test_apply_capability_by_name(self, provider):
        """Test apply_capability applies capability by name."""
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config

        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        provider.apply_capability(orchestrator, "data_quality")

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "data_quality"
        config = call_args[0][1]
        assert config["min_completeness"] == 0.9
        assert "data_quality" in provider.get_applied()

    def test_apply_capability_with_custom_config(self, provider):
        """Test apply_capability merges custom config with defaults."""
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config

        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        provider.apply_capability(orchestrator, "data_quality", min_completeness=0.95)

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        assert call_args[0][0] == "data_quality"
        config = call_args[0][1]
        assert config["min_completeness"] == 0.95
        # Other defaults should still be present
        assert config["handle_missing"] == "impute"

    def test_get_capability_config(self, provider):
        """Test get_capability_config retrieves current config."""
        mock_context = MagicMock()
        existing_config = {
            "min_completeness": 0.95,
            "max_outlier_ratio": 0.1,
        }
        mock_context.get_capability_config.return_value = existing_config

        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        config = provider.get_capability_config(orchestrator, "data_quality")

        assert config["min_completeness"] == 0.95
        assert config["max_outlier_ratio"] == 0.1

    def test_get_default_config(self, provider):
        """Test get_default_config returns default configuration."""
        config = provider.get_default_config("data_quality")

        assert config["min_completeness"] == 0.9
        assert config["handle_missing"] == "impute"

    def test_apply_data_quality(self, provider):
        """Test apply_capability with data_quality sets config and tracks application."""
        mock_context = MagicMock()
        mock_capability_config = {}
        mock_context.get_capability_config.return_value = mock_capability_config

        orchestrator = MagicMock()
        orchestrator.vertical_context = mock_context

        provider.apply_capability(orchestrator, "data_quality")

        # Verify config was set
        mock_context.set_capability_config.assert_called_once()
        call_args = mock_context.set_capability_config.call_args
        config = call_args[0][1]
        assert config["min_completeness"] == 0.9
        assert "data_quality" in provider.get_applied()

    def test_apply_all(self, provider):
        """Test apply_all applies all capabilities."""
        orchestrator = MagicMock()
        orchestrator.data_quality_config = {}
        orchestrator.visualization_config = {}
        orchestrator.statistics_config = {}
        orchestrator.ml_config = {}
        orchestrator.privacy_config = {}

        provider.apply_all(orchestrator)

        applied = provider.get_applied()
        assert len(applied) == 5
        assert "data_quality" in applied
        assert "visualization_style" in applied
        assert "statistical_analysis" in applied
        assert "ml_pipeline" in applied
        assert "data_privacy" in applied

    def test_reset_applied(self, provider):
        """Test reset_applied clears applied tracking."""
        orchestrator = MagicMock()
        orchestrator.data_quality_config = {}

        provider.apply_capability(orchestrator, "data_quality")
        assert len(provider.get_applied()) == 1

        provider.reset_applied()
        assert len(provider.get_applied()) == 0

    def test_generate_capabilities_list(self, provider):
        """Test generate_capabilities_list creates CapabilityEntry list."""
        capabilities_list = provider.generate_capabilities_list()

        assert len(capabilities_list) == 5
        for entry in capabilities_list:
            assert entry.capability is not None
            assert entry.handler is not None
            assert callable(entry.handler)

    def test_generate_capability_configs(self, provider):
        """Test generate_capability_configs creates config dict."""
        configs = provider.generate_capability_configs()

        assert "data_quality_config" in configs
        assert "visualization_style_config" in configs
        assert "statistical_analysis_config" in configs
        assert "ml_pipeline_config" in configs
        assert "data_privacy_config" in configs

        # Verify default values are present
        assert configs["data_quality_config"]["min_completeness"] == 0.9
        assert configs["ml_pipeline_config"]["default_framework"] == "sklearn"

    def test_create_capability_loader(self, provider):
        """Test create_capability_loader creates configured loader."""
        loader = provider.create_capability_loader()

        assert loader is not None
        # Loader should have capabilities registered
        assert hasattr(loader, "_capabilities")


class TestCAPABILITIESList:
    """Tests for CAPABILITIES list for CapabilityLoader discovery."""

    def test_capabilities_list_exists(self):
        """Test CAPABILITIES list is defined and populated."""
        from victor.dataanalysis.capabilities import CAPABILITIES

        assert len(CAPABILITIES) == 5

    def test_capabilities_have_required_fields(self):
        """Test all capability entries have required fields."""
        from victor.dataanalysis.capabilities import CAPABILITIES

        for entry in CAPABILITIES:
            assert entry.capability is not None
            assert entry.capability.name is not None
            assert entry.capability.capability_type is not None
            assert entry.handler is not None

    def test_get_data_analysis_capabilities(self):
        """Test get_data_analysis_capabilities returns copy of list."""
        from victor.dataanalysis.capabilities import (
            CAPABILITIES,
            get_data_analysis_capabilities,
        )

        caps = get_data_analysis_capabilities()

        assert len(caps) == len(CAPABILITIES)
        assert caps is not CAPABILITIES  # Should be a copy


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_capability_configs(self):
        """Test get_capability_configs returns default configs."""
        from victor.dataanalysis.capabilities import get_capability_configs

        configs = get_capability_configs()

        assert isinstance(configs, dict)
        assert "data_quality_config" in configs
        assert "ml_pipeline_config" in configs

    def test_create_data_analysis_capability_loader(self):
        """Test create_data_analysis_capability_loader creates loader."""
        from victor.dataanalysis.capabilities import create_data_analysis_capability_loader

        loader = create_data_analysis_capability_loader()

        assert loader is not None


class TestAssistantIntegration:
    """Tests for assistant integration with capability provider."""

    def test_get_capability_provider_exists(self):
        """Test DataAnalysisAssistant has get_capability_provider method."""
        from victor.dataanalysis.assistant import DataAnalysisAssistant

        provider = DataAnalysisAssistant.get_capability_provider()

        assert provider is not None

    def test_get_capability_provider_returns_correct_type(self):
        """Test get_capability_provider returns DataAnalysisCapabilityProvider."""
        from victor.dataanalysis.assistant import DataAnalysisAssistant
        from victor.dataanalysis.capabilities import DataAnalysisCapabilityProvider

        provider = DataAnalysisAssistant.get_capability_provider()

        assert isinstance(provider, DataAnalysisCapabilityProvider)


class TestModuleExports:
    """Tests for module exports."""

    def test_capability_provider_exported_from_init(self):
        """Test DataAnalysisCapabilityProvider is exported from __init__."""
        from victor.dataanalysis import DataAnalysisCapabilityProvider

        assert DataAnalysisCapabilityProvider is not None

    def test_all_exports_available(self):
        """Test all expected exports are available from capabilities module."""
        from victor.dataanalysis.capabilities import (
            configure_data_quality,
            configure_visualization_style,
            configure_statistical_analysis,
            configure_ml_pipeline,
            configure_data_privacy,
            get_data_quality,
            get_visualization_style,
            get_statistical_config,
            get_ml_config,
            get_privacy_config,
            DataAnalysisCapabilityProvider,
            CAPABILITIES,
            get_data_analysis_capabilities,
        )

        # Just verifying these imports work
        assert configure_data_quality is not None
        assert configure_visualization_style is not None
        assert configure_statistical_analysis is not None
        assert configure_ml_pipeline is not None
        assert configure_data_privacy is not None
        assert get_data_quality is not None
        assert get_visualization_style is not None
        assert get_statistical_config is not None
        assert get_ml_config is not None
        assert get_privacy_config is not None
        assert DataAnalysisCapabilityProvider is not None
        assert CAPABILITIES is not None
        assert get_data_analysis_capabilities is not None
