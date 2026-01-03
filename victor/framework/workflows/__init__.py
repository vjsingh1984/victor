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

"""Framework workflow components.

This package provides base classes and utilities for building workflow providers
across verticals. The primary component is BaseYAMLWorkflowProvider, which
eliminates boilerplate in vertical-specific workflow provider implementations.

Key Components:
    - BaseYAMLWorkflowProvider: Template Method pattern base class for YAML
      workflow providers. Subclasses only need to specify the escape hatches
      module path to get full workflow loading, caching, and streaming support.

Example:
    from victor.framework.workflows import BaseYAMLWorkflowProvider

    class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
        '''Provides research-specific workflows.'''

        def _get_escape_hatches_module(self) -> str:
            return "victor.research.escape_hatches"

        def get_auto_workflows(self) -> List[Tuple[str, str]]:
            return [
                (r"deep\\s+research", "deep_research"),
                (r"fact\\s*check", "fact_check"),
            ]

    # Usage
    provider = ResearchWorkflowProvider()
    workflow = provider.get_workflow("deep_research")
    async for chunk in provider.astream("deep_research", orchestrator, {}):
        print(f"Progress: {chunk.progress:.0f}%")
"""

from victor.framework.workflows.base_yaml_provider import BaseYAMLWorkflowProvider

__all__ = [
    "BaseYAMLWorkflowProvider",
]
