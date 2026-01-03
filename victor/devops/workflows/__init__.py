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

"""DevOps vertical workflows.

This package provides workflow definitions for common DevOps tasks:
- Infrastructure deployment
- Container management
- CI/CD pipeline setup
- Monitoring configuration

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = DevOpsWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("deploy", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")

Available workflows (all YAML-defined):
- deploy: Safe deployment with validation and rollback
- cicd: CI/CD pipeline with security scanning
- container_setup: Container setup with Dockerfile optimization
- container_quick: Quick container build
"""

from typing import List, Tuple

from victor.framework.workflows import BaseYAMLWorkflowProvider


class DevOpsWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides DevOps-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Includes support for streaming execution via StreamingWorkflowExecutor
    for real-time progress updates during long-running DevOps workflows.

    Example:
        provider = DevOpsWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream deployment execution
        async for chunk in provider.astream("deploy", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for DevOps escape hatches.

        Returns:
            Module path to victor.devops.escape_hatches
        """
        return "victor.devops.escape_hatches"

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers for DevOps tasks.

        Returns:
            List of (regex_pattern, workflow_name) tuples for auto-triggering
        """
        return [
            (r"deploy\s+infrastructure", "deploy_infrastructure"),
            (r"terraform\s+apply", "deploy_infrastructure"),
            (r"container(ize)?", "container_setup"),
            (r"docker(file)?", "container_setup"),
            (r"ci/?cd", "cicd_pipeline"),
            (r"pipeline", "cicd_pipeline"),
            (r"github\s+actions", "cicd_pipeline"),
        ]


# Register DevOps domain handlers when this module is loaded
from victor.devops.handlers import register_handlers as _register_handlers

_register_handlers()

__all__ = [
    # YAML-first workflow provider
    "DevOpsWorkflowProvider",
]
