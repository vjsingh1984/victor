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

"""Data Analysis Chat Workflow Provider.

This module provides chat workflows for the Data Analysis vertical.
"""

from __future__ import annotations

from pathlib import Path

from victor.framework.workflows import BaseYAMLWorkflowProvider


class DataAnalysisChatWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides chat workflows for Data Analysis vertical.

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
        return "victor.dataanalysis.escape_hatches"

    def get_auto_workflows(self) -> list[tuple[str, str]]:
        """Get automatic workflow registrations.

        Returns:
            List of (pattern, workflow_name) tuples for auto-registration
        """
        return [
            (r"(analyze|explore|visualize).*data", "data_analysis_chat"),
            (r"(statistical|statistics|test).*", "data_analysis_chat"),
            (r"(chart|graph|plot|visualization).*", "data_analysis_chat"),
            (r"(machine learning|ml|predict|forecast).*", "data_analysis_chat"),
        ]


__all__ = [
    "DataAnalysisChatWorkflowProvider",
]
