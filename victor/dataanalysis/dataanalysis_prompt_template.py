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

"""Data Analysis-specific prompt template using PromptBuilderTemplate.

This module provides the Template Method pattern for consistent prompt structure
for the data analysis vertical.

Usage:
    from victor.dataanalysis.dataanalysis_prompt_template import DataAnalysisPromptTemplate

    template = DataAnalysisPromptTemplate()
    builder = template.get_prompt_builder()
    prompt = builder.build()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.prompt_builder_template import PromptBuilderTemplate

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class DataAnalysisPromptTemplate(PromptBuilderTemplate):
    """Template Method pattern for data analysis vertical prompts.

    Provides consistent prompt structure with hook methods that can be
    customized for data analysis-specific requirements.

    Attributes:
        vertical_name: "data_analysis"
    """

    vertical_name: str = "data_analysis"

    def get_grounding(self) -> Optional[Dict[str, Any]]:
        """Get grounding configuration for the prompt.

        Returns:
            Dictionary with 'template', 'variables', and optional 'priority'
        """
        return {
            "template": "Context: You are conducting data analysis for {project}.",
            "variables": {"project": "a data analysis project"},
            "priority": 10,
        }

    def get_rules(self) -> List[str]:
        """Get list of rules for the prompt.

        Returns:
            List of rule strings
        """
        return [
            "Always explore and understand data before analysis",
            "Handle missing values explicitly",
            "Use appropriate statistical methods",
            "Create clear and informative visualizations",
            "Document analysis methodology",
            "Consider data privacy and ethics",
            "Validate assumptions and check for biases",
            "Provide reproducible code",
        ]

    def get_rules_priority(self) -> int:
        """Get priority for rules section.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return 20

    def get_checklist(self) -> List[str]:
        """Get checklist items for the prompt.

        Returns:
            List of checklist item strings
        """
        return [
            "Data is properly loaded and validated",
            "Missing values are handled appropriately",
            "Statistical methods are appropriate for the data",
            "Visualizations are clear and informative",
            "Results are well-documented",
            "Code is reproducible",
            "Limitations and assumptions are noted",
        ]

    def get_checklist_priority(self) -> int:
        """Get priority for checklist section.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return 30

    def get_vertical_prompt(self) -> str:
        """Get vertical-specific prompt content.

        Returns:
            Vertical-specific prompt content
        """
        return """You are an expert data analyst with strong skills in:
- Data exploration and profiling
- Statistical analysis and hypothesis testing
- Data visualization and storytelling
- Machine learning and predictive modeling
- Using Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Communicating insights clearly"""

    def pre_build(self, builder: "PromptBuilder") -> "PromptBuilder":
        """Hook called before building the prompt.

        Args:
            builder: The configured PromptBuilder

        Returns:
            The modified PromptBuilder
        """
        # Add custom sections or modify builder before building
        # This is where vertical-specific customizations can go
        return builder


__all__ = ["DataAnalysisPromptTemplate"]
