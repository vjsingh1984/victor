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

"""Vertical metadata capabilities.

This module provides metadata-related functionality for verticals, including:
- Provider hints for model selection
- Evaluation criteria for performance assessment
- Capability configurations for vertical-specific features

Extracted from VerticalBase for SRP compliance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Type

from victor.core.vertical_types import TieredToolTemplate

if TYPE_CHECKING:
    from victor.core.vertical_types import TieredToolConfig

logger = logging.getLogger(__name__)


class VerticalMetadataProvider(ABC):
    """Provider of vertical metadata.

    Handles metadata-related functionality including provider hints,
    evaluation criteria, and capability configurations.

    This is a mix-in class that can be used with VerticalBase or standalone.
    """

    # Subclasses must define these as ClassVar to avoid MyPy errors
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    version: ClassVar[str] = "0.5.0"

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Get hints for provider selection.

        Returns:
            Dictionary with provider preferences.

        Default implementation uses VerticalConfigRegistry. Override in subclasses
        for vertical-specific requirements.
        """
        from victor.core.verticals.config_registry import VerticalConfigRegistry

        # Try to get from registry based on vertical name
        try:
            return VerticalConfigRegistry.get_provider_hints(cls.name)
        except KeyError:
            # Fallback to generic defaults
            return {
                "preferred_providers": ["anthropic", "openai"],
                "min_context_window": 100000,
                "requires_tool_calling": True,
            }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Get criteria for evaluating agent performance.

        Returns:
            List of evaluation criteria descriptions.

        Default implementation uses VerticalConfigRegistry. Override in subclasses
        for vertical-specific requirements.
        """
        from victor.core.verticals.config_registry import VerticalConfigRegistry

        # Try to get from registry based on vertical name
        try:
            return VerticalConfigRegistry.get_evaluation_criteria(cls.name)
        except KeyError:
            # Fallback to generic defaults
            return [
                "Task completion accuracy",
                "Tool usage efficiency",
                "Response relevance",
            ]

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        """Get capability configurations for this vertical.

        Override to provide vertical-specific capability configurations
        that will be stored in VerticalContext instead of direct
        orchestrator attribute assignment.

        This replaces the previous pattern of setting attributes like:
        - orchestrator.rag_config = {...}
        - orchestrator.source_verification_config = {...}

        Example:
            @classmethod
            def get_capability_configs(cls) -> Dict[str, Any]:
                return {
                    "rag_config": {
                        "indexing": {"chunk_size": 512, ...},
                        "retrieval": {"top_k": 5, ...},
                    },
                }

        Returns:
            Dict mapping config names to configuration values
        """
        return {}

    @classmethod
    def get_tiered_tool_config(cls) -> Optional["TieredToolConfig"]:
        """Get tiered tool configuration for this vertical.

        TieredToolConfig defines mandatory, vertical_core, and semantic_pool
        tool sets for intelligent tool filtering by ToolAccessController.

        Default implementation uses TieredToolTemplate to generate config
        based on vertical name. Override for custom configurations.

        Returns:
            TieredToolConfig or None if vertical doesn't use tiered config.
        """
        # Try to get pre-built config from template
        return TieredToolTemplate.for_vertical(cls.name)


class VerticalMetadata(VerticalMetadataProvider):
    """Concrete implementation of vertical metadata.

    This class provides the default implementation of metadata-related
    functionality. Verticals can inherit from this class or use it as
    a mix-in to get metadata capabilities.
    """

    pass
