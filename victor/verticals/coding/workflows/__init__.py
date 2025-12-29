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

"""Coding vertical workflows.

This package provides workflow definitions for common coding tasks:
- Feature implementation (full and quick)
- Bug fix (systematic and quick)
- Code review (comprehensive, quick, and PR)

Example:
    from victor.verticals.coding.workflows import CodingWorkflowProvider

    provider = CodingWorkflowProvider()
    workflows = provider.get_workflows()

    # Get specific workflow
    feature_wf = provider.get_workflow("feature_implementation")
    print(f"Workflow: {feature_wf.name}")
    print(f"Description: {feature_wf.description}")
    print(f"Agents: {feature_wf.get_agent_count()}")
    print(f"Total Budget: {feature_wf.get_total_budget()}")
"""

from victor.verticals.coding.workflows.provider import CodingWorkflowProvider

# Individual workflows for direct import
from victor.verticals.coding.workflows.feature import (
    feature_implementation_workflow,
    quick_feature_workflow,
)
from victor.verticals.coding.workflows.bugfix import (
    bug_fix_workflow,
    quick_fix_workflow,
)
from victor.verticals.coding.workflows.review import (
    code_review_workflow,
    quick_review_workflow,
    pr_review_workflow,
)

__all__ = [
    # Provider
    "CodingWorkflowProvider",
    # Feature workflows
    "feature_implementation_workflow",
    "quick_feature_workflow",
    # Bug fix workflows
    "bug_fix_workflow",
    "quick_fix_workflow",
    # Review workflows
    "code_review_workflow",
    "quick_review_workflow",
    "pr_review_workflow",
]
