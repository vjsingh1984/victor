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

"""DevOps-specific prompt template using PromptBuilderTemplate.

This module provides the Template Method pattern for consistent prompt structure
for the DevOps vertical.

Usage:
    from victor.devops.devops_prompt_template import DevOpsPromptTemplate

    template = DevOpsPromptTemplate()
    builder = template.get_prompt_builder()
    prompt = builder.build()
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.framework.prompt_builder_template import PromptBuilderTemplate

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class DevOpsPromptTemplate(PromptBuilderTemplate):
    """Template Method pattern for DevOps vertical prompts.

    Provides consistent prompt structure with hook methods that can be
    customized for DevOps-specific requirements.

    Attributes:
        vertical_name: "devops"
    """

    vertical_name: str = "devops"

    def get_grounding(self) -> Optional[dict[str, Any]]:
        """Get grounding configuration for the prompt.

        Returns:
            Dictionary with 'template', 'variables', and optional 'priority'
        """
        return {
            "template": "Context: You are assisting with DevOps infrastructure automation for {project}.",
            "variables": {"project": "an infrastructure project"},
            "priority": 10,
        }

    def get_rules(self) -> list[str]:
        """Get list of rules for the prompt.

        Returns:
            List of rule strings
        """
        return [
            "Never commit secrets or credentials",
            "Use infrastructure as code principles",
            "Implement proper security measures",
            "Test changes in staging before production",
            "Document infrastructure decisions",
            "Follow least privilege access principles",
            "Use resource limits and quotas",
            "Implement monitoring and observability",
        ]

    def get_rules_priority(self) -> int:
        """Get priority for rules section.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return 20

    def get_checklist(self) -> list[str]:
        """Get checklist items for the prompt.

        Returns:
            List of checklist item strings
        """
        return [
            "Configuration is secure and follows best practices",
            "Secrets are properly managed",
            "Resource limits are configured",
            "Monitoring and logging are set up",
            "Changes are tested before deployment",
            "Documentation is up to date",
            "Rollback plan is available",
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
        return """You are an expert DevOps engineer with deep knowledge of:
- Container orchestration (Kubernetes, Docker)
- Infrastructure as Code (Terraform, Ansible, CloudFormation)
- CI/CD pipelines and automation
- Cloud platforms (AWS, GCP, Azure)
- Monitoring and observability (Prometheus, Grafana, ELK)
- Security best practices and compliance"""

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


__all__ = ["DevOpsPromptTemplate"]
