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

Two workflow APIs are available:

1. **WorkflowBuilder** (YAML-like DSL):
   - Simple, declarative workflow definitions
   - Good for linear workflows with basic conditions
   - Example: feature_implementation_workflow, bug_fix_workflow

2. **StateGraph** (LangGraph-compatible):
   - Typed state management
   - Cyclic execution (test-fix loops, validation cycles)
   - Explicit retry limits and checkpoint/resume
   - Example: create_tdd_workflow(), create_bugfix_workflow()

Example (WorkflowBuilder):
    from victor.verticals.coding.workflows import CodingWorkflowProvider

    provider = CodingWorkflowProvider()
    feature_wf = provider.get_workflow("feature_implementation")
    print(f"Workflow: {feature_wf.name}")

Example (StateGraph):
    from victor.verticals.coding.workflows import (
        create_tdd_workflow,
        CodingState,
    )
    from victor.framework.graph import RLCheckpointerAdapter

    graph = create_tdd_workflow()
    checkpointer = RLCheckpointerAdapter("tdd")
    app = graph.compile(checkpointer=checkpointer)

    result = await app.invoke({"feature_description": "Add auth", "iteration": 0})
"""

from victor.verticals.coding.workflows.provider import CodingWorkflowProvider

# Individual workflows for direct import (WorkflowBuilder DSL)
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

# StateGraph-based workflows (LangGraph-compatible)
from victor.verticals.coding.workflows.graph_workflows import (
    # State types
    CodingState,
    TestState,
    BugFixState,
    # Workflow factories
    create_feature_workflow,
    create_tdd_workflow,
    create_bugfix_workflow,
    create_code_review_workflow,
    # Executor
    GraphWorkflowExecutor,
)

# LCEL-composed tool chains
from victor.verticals.coding.composed_chains import (
    # Pre-built chains
    explore_file_chain,
    analyze_function_chain,
    safe_edit_chain,
    git_status_chain,
    search_with_context_chain,
    lint_chain,
    test_discovery_chain,
    review_analysis_chain,
    # Factories
    create_exploration_chain,
    create_edit_verify_chain,
    create_refactor_chain,
    # Registry
    CODING_CHAINS,
    get_chain,
    list_chains,
    # Lazy tool loading
    lazy_tool,
    LazyToolRunnable,
)

__all__ = [
    # Provider
    "CodingWorkflowProvider",
    # Feature workflows (WorkflowBuilder)
    "feature_implementation_workflow",
    "quick_feature_workflow",
    # Bug fix workflows (WorkflowBuilder)
    "bug_fix_workflow",
    "quick_fix_workflow",
    # Review workflows (WorkflowBuilder)
    "code_review_workflow",
    "quick_review_workflow",
    "pr_review_workflow",
    # StateGraph state types
    "CodingState",
    "TestState",
    "BugFixState",
    # StateGraph workflow factories
    "create_feature_workflow",
    "create_tdd_workflow",
    "create_bugfix_workflow",
    "create_code_review_workflow",
    # StateGraph executor
    "GraphWorkflowExecutor",
    # LCEL-composed chains
    "explore_file_chain",
    "analyze_function_chain",
    "safe_edit_chain",
    "git_status_chain",
    "search_with_context_chain",
    "lint_chain",
    "test_discovery_chain",
    "review_analysis_chain",
    # Chain factories
    "create_exploration_chain",
    "create_edit_verify_chain",
    "create_refactor_chain",
    # Chain registry
    "CODING_CHAINS",
    "get_chain",
    "list_chains",
    # Lazy tool loading
    "lazy_tool",
    "LazyToolRunnable",
]
