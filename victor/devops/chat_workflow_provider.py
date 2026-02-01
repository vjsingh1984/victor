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

"""DevOps Chat Workflow Provider.

This module provides chat workflows for the DevOps vertical.
"""

from __future__ import annotations

from pathlib import Path

from victor.framework.workflows import BaseYAMLWorkflowProvider


class DevOpsChatWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides chat workflows for DevOps vertical.

    This provider loads YAML workflow definitions from the workflows
    directory and registers them with the workflow coordinator.
    """

    def _get_workflows_directory(self) -> Path:
        """Get the directory containing workflow YAML files.

        Returns:
            Path to the workflows directory
        """
        return Path(__file__).parent / "workflows"

    def _get_escape_hatches_module(self) -> str:
        """Get the module containing escape hatches.

        Returns:
            Module path for escape hatches
        """
        return "victor.devops.escape_hatches"

    def get_auto_workflows(self) -> list[tuple[str, str]]:
        """Get automatic workflow registrations.

        Returns:
            List of (pattern, workflow_name) tuples for auto-registration
        """
        return [
            (r"(deploy|upgrade|rollback).*kubernetes", "devops_chat"),
            (r"(deploy|upgrade|rollback).*docker", "devops_chat"),
            (r"scale.*up.*down.*pods", "devops_chat"),
            (r"monitor.*logs.*metrics", "devops_chat"),
            (r"backup.*restore.*database", "devops_chat"),
            (r"build.*push.*image", "devops_chat"),
        ]


__all__ = [
    "DevOpsChatWorkflowProvider",
]
