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

"""Research vertical workflows.

This package provides workflow definitions for common research tasks:
- Deep research with source verification
- Fact-checking
- Literature review
- Competitive analysis

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = ResearchWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("deep_research", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")

Available workflows (all YAML-defined):
- deep_research: Comprehensive research with source validation
- quick_research: Fast research for simple queries
- fact_check: Systematic fact verification
- literature_review: Academic literature review
- competitive_analysis: Market and competitive research
- competitive_scan: Quick competitive overview
"""

from typing import List, Optional, Tuple

from victor.framework.workflows import BaseYAMLWorkflowProvider


class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides research-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading and caching
    - Escape hatches registration from victor.research.escape_hatches
    - Streaming execution via StreamingWorkflowExecutor
    - Standard workflow execution

    Example:
        provider = ResearchWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream research execution
        async for chunk in provider.astream("deep_research", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for research escape hatches.

        Returns:
            Module path string for CONDITIONS and TRANSFORMS dictionaries
        """
        return "victor.research.escape_hatches"

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns.

        Returns:
            List of (regex_pattern, workflow_name) tuples for auto-triggering
        """
        return [
            (r"deep\s+research", "deep_research"),
            (r"research\s+.*\s+thoroughly", "deep_research"),
            (r"comprehensive\s+research", "deep_research"),
            (r"fact\s*check", "fact_check"),
            (r"verify\s+(claim|statement)", "fact_check"),
            (r"is\s+it\s+true", "fact_check"),
            (r"literature\s+review", "literature_review"),
            (r"academic\s+review", "literature_review"),
            (r"papers?\s+on", "literature_review"),
            (r"competitive?\s+analysis", "competitive_analysis"),
            (r"market\s+research", "competitive_analysis"),
            (r"competitor", "competitive_analysis"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get appropriate workflow for task type.

        Args:
            task_type: Type of task (e.g., "research", "fact_check")

        Returns:
            Workflow name string or None if no mapping exists
        """
        mapping = {
            "research": "deep_research",
            "fact_check": "fact_check",
            "verification": "fact_check",
            "literature": "literature_review",
            "academic": "literature_review",
            "competitive": "competitive_analysis",
            "market": "competitive_analysis",
        }
        return mapping.get(task_type.lower())


# Register Research domain handlers when this module is loaded
from victor.research.handlers import register_handlers as _register_handlers

_register_handlers()

__all__ = [
    # YAML-first workflow provider
    "ResearchWorkflowProvider",
]
