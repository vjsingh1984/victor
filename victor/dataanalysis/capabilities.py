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

"""Dynamic capability definitions for the data analysis vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the data analysis vertical with custom functionality.

Refactored to use BaseVerticalCapabilityProvider, reducing from
816 lines to ~300 lines by eliminating duplicated patterns.

Example:
    # Use provider
    from victor.dataanalysis.capabilities import DataAnalysisCapabilityProvider

    provider = DataAnalysisCapabilityProvider()

    # Apply capabilities
    provider.apply_data_quality(orchestrator, min_completeness=0.95)
    provider.apply_visualization_style(orchestrator, backend="plotly")

    # Get configurations
    config = provider.get_capability_config(orchestrator, "data_quality")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.framework.capabilities.base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
)
from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Handlers (configure_*, get_* functions)
# =============================================================================


def configure_data_quality(
    orchestrator: Any,
    *,
    min_completeness: float = 0.9,
    max_outlier_ratio: float = 0.05,
    require_type_validation: bool = True,
    handle_missing: str = "impute",
) -> None:
    """Configure data quality rules for the orchestrator.

    This capability configures data quality checks and handling
    strategies for data analysis tasks.

    Args:
        orchestrator: Target orchestrator
        min_completeness: Minimum data completeness ratio (0-1)
        max_outlier_ratio: Maximum allowed outlier ratio
        require_type_validation: Require data type validation
        handle_missing: Strategy for missing values: "impute", "drop", "flag"
    """
    if hasattr(orchestrator, "data_quality_config"):
        orchestrator.data_quality_config = {
            "min_completeness": min_completeness,
            "max_outlier_ratio": max_outlier_ratio,
            "require_type_validation": require_type_validation,
            "handle_missing": handle_missing,
        }

    logger.info(
        f"Configured data quality: completeness>={min_completeness:.0%}, "
        f"missing={handle_missing}"
    )


def get_data_quality(orchestrator: Any) -> Dict[str, Any]:
    """Get current data quality configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Data quality configuration dict
    """
    return getattr(
        orchestrator,
        "data_quality_config",
        {
            "min_completeness": 0.9,
            "max_outlier_ratio": 0.05,
            "require_type_validation": True,
            "handle_missing": "impute",
        },
    )


def configure_visualization_style(
    orchestrator: Any,
    *,
    default_backend: str = "matplotlib",
    theme: str = "seaborn-v0_8-whitegrid",
    figure_size: tuple = (10, 6),
    dpi: int = 100,
    save_format: str = "png",
) -> None:
    """Configure visualization style for data analysis.

    Args:
        orchestrator: Target orchestrator
        default_backend: Plotting backend (matplotlib, plotly, seaborn)
        theme: Plot theme/style
        figure_size: Default figure size (width, height) in inches
        dpi: Resolution in dots per inch
        save_format: Default save format (png, svg, pdf)
    """
    if hasattr(orchestrator, "visualization_config"):
        orchestrator.visualization_config = {
            "backend": default_backend,
            "theme": theme,
            "figure_size": figure_size,
            "dpi": dpi,
            "save_format": save_format,
        }

    logger.info(f"Configured visualization: backend={default_backend}, theme={theme}")


def get_visualization_style(orchestrator: Any) -> Dict[str, Any]:
    """Get current visualization style configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Visualization configuration dict
    """
    return getattr(
        orchestrator,
        "visualization_config",
        {
            "backend": "matplotlib",
            "theme": "seaborn-v0_8-whitegrid",
            "figure_size": (10, 6),
            "dpi": 100,
            "save_format": "png",
        },
    )


def configure_statistical_analysis(
    orchestrator: Any,
    *,
    significance_level: float = 0.05,
    confidence_interval: float = 0.95,
    multiple_testing_correction: str = "bonferroni",
    effect_size_threshold: float = 0.2,
) -> None:
    """Configure statistical analysis parameters.

    Args:
        orchestrator: Target orchestrator
        significance_level: P-value threshold for significance
        confidence_interval: Confidence interval level
        multiple_testing_correction: Correction method (bonferroni, holm, fdr_bh)
        effect_size_threshold: Minimum effect size for practical significance
    """
    if hasattr(orchestrator, "statistics_config"):
        orchestrator.statistics_config = {
            "significance_level": significance_level,
            "confidence_interval": confidence_interval,
            "multiple_testing_correction": multiple_testing_correction,
            "effect_size_threshold": effect_size_threshold,
        }

    logger.info(
        f"Configured statistics: alpha={significance_level}, " f"CI={confidence_interval:.0%}"
    )


def get_statistical_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current statistical configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Statistical configuration dict
    """
    return getattr(
        orchestrator,
        "statistics_config",
        {
            "significance_level": 0.05,
            "confidence_interval": 0.95,
            "multiple_testing_correction": "bonferroni",
            "effect_size_threshold": 0.2,
        },
    )


def configure_ml_pipeline(
    orchestrator: Any,
    *,
    default_framework: str = "sklearn",
    cross_validation_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    enable_hyperparameter_tuning: bool = True,
    tuning_method: str = "grid",
) -> None:
    """Configure machine learning pipeline settings.

    Args:
        orchestrator: Target orchestrator
        default_framework: ML framework (sklearn, xgboost, lightgbm)
        cross_validation_folds: Number of CV folds
        test_size: Test set proportion
        random_state: Random seed for reproducibility
        enable_hyperparameter_tuning: Enable hyperparameter optimization
        tuning_method: Tuning method (grid, random, bayesian)
    """
    if hasattr(orchestrator, "ml_config"):
        orchestrator.ml_config = {
            "framework": default_framework,
            "cv_folds": cross_validation_folds,
            "test_size": test_size,
            "random_state": random_state,
            "hyperparameter_tuning": enable_hyperparameter_tuning,
            "tuning_method": tuning_method,
        }

    logger.info(
        f"Configured ML pipeline: framework={default_framework}, "
        f"cv_folds={cross_validation_folds}"
    )


def get_ml_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current ML configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        ML configuration dict
    """
    return getattr(
        orchestrator,
        "ml_config",
        {
            "framework": "sklearn",
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
            "hyperparameter_tuning": True,
            "tuning_method": "grid",
        },
    )


def configure_data_privacy(
    orchestrator: Any,
    *,
    anonymize_pii: bool = True,
    pii_columns: Optional[List[str]] = None,
    hash_identifiers: bool = True,
    log_access: bool = True,
) -> None:
    """Configure data privacy settings.

    Delegates to framework PrivacyCapabilityProvider for cross-vertical privacy management.

    Args:
        orchestrator: Target orchestrator
        anonymize_pii: Whether to anonymize PII columns
        pii_columns: List of column names containing PII
        hash_identifiers: Hash identifier columns
        log_access: Log data access for audit trail
    """
    # Delegate to framework privacy capability
    from victor.framework.capabilities.privacy import configure_data_privacy as framework_privacy

    framework_privacy(
        orchestrator,
        anonymize_pii=anonymize_pii,
        pii_columns=pii_columns,
        hash_identifiers=hash_identifiers,
        log_access=log_access,
    )


def get_privacy_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current privacy configuration.

    Delegates to framework PrivacyCapabilityProvider for cross-vertical privacy management.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Privacy configuration dict
    """
    # Delegate to framework privacy capability
    from victor.framework.capabilities.privacy import get_privacy_config as framework_get_privacy

    return framework_get_privacy(orchestrator)


# =============================================================================
# Capability Provider Class (Refactored to use BaseVerticalCapabilityProvider)
# =============================================================================


class DataAnalysisCapabilityProvider(BaseVerticalCapabilityProvider):
    """Provider for data analysis-specific capabilities.

    Refactored to inherit from BaseVerticalCapabilityProvider, eliminating
    ~500 lines of duplicated boilerplate code.

    Example:
        provider = DataAnalysisCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_data_quality(orchestrator, min_completeness=0.95)
        provider.apply_visualization_style(orchestrator, backend="plotly")

        # Get configurations
        config = provider.get_capability_config(orchestrator, "data_quality")
    """

    def __init__(self):
        """Initialize the data analysis capability provider."""
        super().__init__("dataanalysis")

    def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
        """Define data analysis capability definitions.

        Returns:
            Dictionary of data analysis capability definitions
        """
        return {
            "data_quality": CapabilityDefinition(
                name="data_quality",
                type=CapabilityType.MODE,
                description="Data quality rules and validation settings",
                version="1.0",
                configure_fn="configure_data_quality",
                get_fn="get_data_quality",
                default_config={
                    "min_completeness": 0.9,
                    "max_outlier_ratio": 0.05,
                    "require_type_validation": True,
                    "handle_missing": "impute",
                },
                tags=["quality", "validation", "data-cleaning"],
            ),
            "visualization_style": CapabilityDefinition(
                name="visualization_style",
                type=CapabilityType.MODE,
                description="Visualization and plotting configuration",
                version="1.0",
                configure_fn="configure_visualization_style",
                get_fn="get_visualization_style",
                default_config={
                    "default_backend": "matplotlib",
                    "theme": "seaborn-v0_8-whitegrid",
                    "figure_size": (10, 6),
                    "dpi": 100,
                    "save_format": "png",
                },
                tags=["visualization", "charts", "plotting"],
            ),
            "statistical_analysis": CapabilityDefinition(
                name="statistical_analysis",
                type=CapabilityType.MODE,
                description="Statistical analysis configuration",
                version="1.0",
                configure_fn="configure_statistical_analysis",
                get_fn="get_statistical_config",
                default_config={
                    "significance_level": 0.05,
                    "confidence_interval": 0.95,
                    "multiple_testing_correction": "bonferroni",
                    "effect_size_threshold": 0.2,
                },
                tags=["statistics", "hypothesis-testing", "analysis"],
            ),
            "ml_pipeline": CapabilityDefinition(
                name="ml_pipeline",
                type=CapabilityType.TOOL,
                description="Machine learning pipeline configuration",
                version="1.0",
                configure_fn="configure_ml_pipeline",
                get_fn="get_ml_config",
                default_config={
                    "default_framework": "sklearn",
                    "cross_validation_folds": 5,
                    "test_size": 0.2,
                    "random_state": 42,
                    "enable_hyperparameter_tuning": True,
                    "tuning_method": "grid",
                },
                dependencies=["data_quality"],
                tags=["ml", "machine-learning", "training"],
            ),
            "data_privacy": CapabilityDefinition(
                name="data_privacy",
                type=CapabilityType.SAFETY,
                description="Data privacy and anonymization settings",
                version="1.0",
                configure_fn="configure_data_privacy",
                get_fn="get_privacy_config",
                default_config={
                    "anonymize_pii": True,
                    "pii_columns": [],
                    "hash_identifiers": True,
                    "log_access": True,
                },
                tags=["privacy", "pii", "anonymization", "safety"],
            ),
        }

    # Delegate to handler functions (required by BaseVerticalCapabilityProvider)
    def configure_data_quality(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure data quality capability."""
        configure_data_quality(orchestrator, **kwargs)

    def configure_visualization_style(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure visualization style capability."""
        configure_visualization_style(orchestrator, **kwargs)

    def configure_statistical_analysis(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure statistical analysis capability."""
        configure_statistical_analysis(orchestrator, **kwargs)

    def configure_ml_pipeline(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure ML pipeline capability."""
        configure_ml_pipeline(orchestrator, **kwargs)

    def configure_data_privacy(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure data privacy capability."""
        configure_data_privacy(orchestrator, **kwargs)

    def get_data_quality(self, orchestrator: Any) -> Dict[str, Any]:
        """Get data quality configuration."""
        return get_data_quality(orchestrator)

    def get_visualization_style(self, orchestrator: Any) -> Dict[str, Any]:
        """Get visualization configuration."""
        return get_visualization_style(orchestrator)

    def get_statistical_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get statistical analysis configuration."""
        return get_statistical_config(orchestrator)

    def get_ml_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get ML pipeline configuration."""
        return get_ml_config(orchestrator)

    def get_privacy_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get privacy configuration."""
        return get_privacy_config(orchestrator)


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


# Create singleton instance for generating CAPABILITIES list
_provider_instance: Optional[DataAnalysisCapabilityProvider] = None


def _get_provider() -> DataAnalysisCapabilityProvider:
    """Get or create provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = DataAnalysisCapabilityProvider()
    return _provider_instance


# Generate CAPABILITIES list from provider
CAPABILITIES: List[CapabilityEntry] = []


def _generate_capabilities_list() -> None:
    """Generate CAPABILITIES list from provider."""
    global CAPABILITIES
    if not CAPABILITIES:
        provider = _get_provider()
        CAPABILITIES.extend(provider.generate_capabilities_list())


_generate_capabilities_list()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_data_analysis_capabilities() -> List[CapabilityEntry]:
    """Get all data analysis capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_data_analysis_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for data analysis vertical.

    Returns:
        CapabilityLoader with data analysis capabilities registered
    """
    from victor.framework.capability_loader import CapabilityLoader

    provider = _get_provider()
    return provider.create_capability_loader()


def get_capability_configs() -> Dict[str, Any]:
    """Get data analysis capability configurations for centralized storage.

    Returns default data analysis configuration for VerticalContext storage.
    This replaces direct orchestrator data_quality/visualization/ml_config assignment.

    Returns:
        Dict with default data analysis capability configurations
    """
    provider = _get_provider()
    return provider.generate_capability_configs()


__all__ = [
    # Handlers
    "configure_data_quality",
    "configure_visualization_style",
    "configure_statistical_analysis",
    "configure_ml_pipeline",
    "configure_data_privacy",
    # Getters
    "get_data_quality",
    "get_visualization_style",
    "get_statistical_config",
    "get_ml_config",
    "get_privacy_config",
    # Provider class
    "DataAnalysisCapabilityProvider",
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_data_analysis_capabilities",
    "create_data_analysis_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]
