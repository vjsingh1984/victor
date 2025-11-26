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


import re
from typing import Any, Dict

from victor.workflows.base import BaseWorkflow
from victor.tools.base import ToolRegistry


class NewFeatureWorkflow(BaseWorkflow):
    """
    A workflow to automate the process of starting a new feature.
    """

    @property
    def name(self) -> str:
        return "new_feature"

    @property
    def description(self) -> str:
        return "Creates a new git branch and placeholder source/test files for a new feature."

    async def run(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        feature_name = kwargs.get("feature_name")
        if not feature_name:
            return {"error": "Missing required argument: feature_name"}

        tool_registry: ToolRegistry = context.get("tool_registry")
        if not tool_registry:
            return {"error": "ToolRegistry not found in context."}

        # 1. Sanitize feature name into a branch name
        branch_name = self._sanitize_branch_name(feature_name)
        
        # 2. Create a new git branch
        branch_result = await tool_registry.execute(
            "git", context, operation="branch", branch=branch_name
        )
        if not branch_result.success:
            return {
                "error": f"Failed to create git branch '{branch_name}'.",
                "details": branch_result.error,
            }

        # 3. Create source and test files
        source_filename = f"{branch_name.replace('-', '_')}.py"
        test_filename = f"tests/test_{source_filename}"

        write_source_result = await tool_registry.execute(
            "write_file",
            context,
            path=source_filename,
            content=f'''"""
Implementation for {feature_name}.
"""

print("Hello from {feature_name}!")
''',
        )
        if not write_source_result.success:
            return {"error": f"Failed to create source file '{source_filename}'."}

        write_test_result = await tool_registry.execute(
            "write_file",
            context,
            path=test_filename,
            content=f'''"""
Tests for {feature_name}.
"""

import pytest

def test_{source_filename.replace(".py", "")}():
    assert True
''',
        )
        if not write_test_result.success:
            return {"error": f"Failed to create test file '{test_filename}'."}
        
        return {
            "success": True,
            "message": "Successfully created new feature setup.",
            "branch_name": branch_name,
            "source_file": source_filename,
            "test_file": test_filename,
        }

    def _sanitize_branch_name(self, name: str) -> str:
        """Converts a feature name into a git-friendly branch name."""
        name = name.lower()
        name = re.sub(r'\s+', '-', name)      # Replace spaces with hyphens
        name = re.sub(r'[^a-z0-9-]', '', name) # Remove non-alphanumeric chars except hyphens
        return f"feature/{name}"

