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

"""Preset workflow templates for common use cases.

This module provides ready-to-use workflow definitions for common agent
patterns and multi-agent collaborations. These presets demonstrate best
practices and can be used as-is or as starting points for customization.

Usage:
    from victor.workflows.presets import (
        get_code_review_workflow,
        get_refactoring_workflow,
        get_research_workflow,
    )

    # Use preset workflow
    workflow = get_code_review_workflow()

    # Execute with orchestrator
    from victor.workflows.executor import WorkflowExecutor
    executor = WorkflowExecutor(orchestrator)
    result = await executor.execute(workflow, initial_context)
"""

from victor.workflows.presets.agent_templates import (
    AgentPreset,
    get_agent_preset,
    list_agent_presets,
)
from victor.workflows.presets.workflow_templates import (
    WorkflowPreset,
    get_workflow_preset,
    list_workflow_presets,
)

__all__ = [
    # Agent templates
    "AgentPreset",
    "get_agent_preset",
    "list_agent_presets",
    # Workflow templates
    "WorkflowPreset",
    "get_workflow_preset",
    "list_workflow_presets",
]
