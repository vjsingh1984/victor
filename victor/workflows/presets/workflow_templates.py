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

These are ready-to-use workflow definitions for common multi-agent patterns.
Each workflow is optimized for its specific use case.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from victor.workflows.definition import WorkflowBuilder, WorkflowDefinition


@dataclass
class WorkflowPreset:
    """Preset workflow configuration.

    Attributes:
        name: Unique workflow name
        description: What this workflow does
        category: Workflow category (code_review, research, etc.)
        builder_factory: Function that creates the workflow
        example_context: Example initial context
        estimated_duration_minutes: Approximate execution time
        complexity: Workflow complexity (simple, medium, complex)
    """

    name: str
    description: str
    category: str
    builder_factory: Callable[[], WorkflowDefinition]
    example_context: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_minutes: int = 10
    complexity: str = "medium"


# =============================================================================
# CODE REVIEW WORKFLOW
# =============================================================================()


def _build_code_review_workflow() -> WorkflowDefinition:
    """Build code review workflow with researcher → reviewer → triage."""

    def decide_next_node(context: Dict[str, Any]) -> str:
        """Decide next step based on review findings."""
        issues_found = context.get("issues_found", False)
        critical_issues = context.get("critical_issues", False)

        if critical_issues:
            return "request_changes"
        elif issues_found:
            return "suggest_improvements"
        else:
            return "approve"

    return (
        WorkflowBuilder("code_review", "Multi-stage code review workflow")
        .add_agent(
            "analyze",
            "code_researcher",
            "Analyze the code changes and identify potential issues",
            tool_budget=20,
        )
        .add_agent(
            "review",
            "code_reviewer",
            "Review code for quality, security, and best practices",
            tool_budget=15,
        )
        .add_condition(
            "decide",
            decide_next_node,
            {
                "request_changes": "request_changes",
                "suggest_improvements": "suggest_improvements",
                "approve": "approve",
            },
        )
        .add_agent(
            "suggest_improvements",
            "code_reviewer",
            "Provide improvement suggestions",
            tool_budget=10,
            next_nodes=["approve"],
        )
        .add_agent(
            "request_changes",
            "code_reviewer",
            "Request changes for critical issues",
            tool_budget=10,
            next_nodes=["approve"],
        )
        .add_agent("approve", "planner", "Prepare approval summary")
        .build()
    )


CODE_REVIEW_WORKFLOW = WorkflowPreset(
    name="code_review",
    description="Multi-stage code review with analysis, review, and triage",
    category="code_review",
    builder_factory=_build_code_review_workflow,
    example_context={
        "pr_number": 123,
        "target_files": ["src/auth.py", "src/user.py"],
        "review_focus": "security and performance",
    },
    estimated_duration_minutes=15,
    complexity="medium",
)


# =============================================================================
# REFACTORING WORKFLOW
# =============================================================================()


def _build_refactoring_workflow() -> WorkflowDefinition:
    """Build refactoring workflow with analysis → planning → execution → verification."""

    def decide_refactoring_approach(context: Dict[str, Any]) -> str:
        """Decide refactoring approach based on analysis."""
        complexity = context.get("complexity", "medium")
        if complexity == "high":
            return "incremental"
        else:
            return "direct"

    return (
        WorkflowBuilder("refactoring", "Safe refactoring with analysis and verification")
        .add_agent(
            "analyze",
            "code_researcher",
            "Analyze current implementation and dependencies",
            tool_budget=25,
        )
        .add_agent(
            "plan",
            "task_planner",
            "Create refactoring plan with safety checks",
            tool_budget=15,
        )
        .add_condition(
            "decide_approach",
            decide_refactoring_approach,
            {"direct": "execute", "incremental": "incremental"},
        )
        .add_agent(
            "execute",
            "refactoring_agent",
            "Execute refactoring changes",
            tool_budget=30,
            next_nodes=["verify"],
        )
        .add_agent(
            "incremental",
            "refactoring_agent",
            "Execute incremental refactoring steps",
            tool_budget=40,
            next_nodes=["verify"],
        )
        .add_agent(
            "verify",
            "test_triage_agent",
            "Verify refactoring with tests and analysis",
            tool_budget=20,
        )
        .add_agent("report", "planner", "Generate refactoring report")
        .build()
    )


REFACTORING_WORKFLOW = WorkflowPreset(
    name="refactoring",
    description="Safe refactoring with analysis, planning, execution, and verification",
    category="refactoring",
    builder_factory=_build_refactoring_workflow,
    example_context={
        "target_component": "UserService",
        "refactoring_goal": "extract validation logic",
        "safety_level": "high",
    },
    estimated_duration_minutes=25,
    complexity="complex",
)


# =============================================================================
# RESEARCH WORKFLOW
# =============================================================================()


def _build_research_workflow() -> WorkflowDefinition:
    """Build research workflow with exploration → synthesis → reporting."""

    return (
        WorkflowBuilder("research", "Deep research with exploration and synthesis")
        .add_agent(
            "explore",
            "code_researcher",
            "Explore codebase and gather information",
            tool_budget=30,
        )
        .add_agent(
            "analyze",
            "code_researcher",
            "Analyze findings and identify patterns",
            tool_budget=20,
        )
        .add_agent(
            "synthesize",
            "planner",
            "Synthesize findings into actionable insights",
            tool_budget=15,
        )
        .add_agent("report", "planner", "Generate research report")
        .build()
    )


RESEARCH_WORKFLOW = WorkflowPreset(
    name="research",
    description="Deep codebase research with exploration and synthesis",
    category="research",
    builder_factory=_build_research_workflow,
    example_context={
        "research_topic": "authentication mechanisms",
        "depth": "comprehensive",
        "output_format": "markdown_report",
    },
    estimated_duration_minutes=20,
    complexity="medium",
)


# =============================================================================
# BUG INVESTIGATION WORKFLOW
# =============================================================================()


def _build_bug_investigation_workflow() -> WorkflowDefinition:
    """Build bug investigation workflow with reproduction → analysis → fix."""

    def decide_fix_approach(context: Dict[str, Any]) -> str:
        """Decide fix approach based on bug analysis."""
        bug_type = context.get("bug_type", "unknown")
        if bug_type == "critical":
            return "hotfix"
        else:
            return "standard_fix"

    return (
        WorkflowBuilder("bug_investigation", "Systematic bug investigation and resolution")
        .add_agent(
            "reproduce",
            "test_triage_agent",
            "Reproduce the bug and gather symptoms",
            tool_budget=15,
        )
        .add_agent(
            "analyze",
            "code_researcher",
            "Analyze root cause and affected code",
            tool_budget=25,
        )
        .add_condition(
            "decide_fix",
            decide_fix_approach,
            {"hotfix": "hotfix", "standard_fix": "standard_fix"},
        )
        .add_agent(
            "hotfix",
            "bug_fixer",
            "Apply urgent hotfix",
            tool_budget=20,
            next_nodes=["verify"],
        )
        .add_agent(
            "standard_fix",
            "bug_fixer",
            "Implement standard fix with tests",
            tool_budget=30,
            next_nodes=["verify"],
        )
        .add_agent(
            "verify",
            "test_triage_agent",
            "Verify fix and prevent regression",
            tool_budget=20,
        )
        .add_agent("report", "planner", "Document bug and fix")
        .build()
    )


BUG_INVESTIGATION_WORKFLOW = WorkflowPreset(
    name="bug_investigation",
    description="Systematic bug investigation and resolution",
    category="debugging",
    builder_factory=_build_bug_investigation_workflow,
    example_context={
        "bug_description": "User authentication fails after session timeout",
        "severity": "high",
        "environment": "production",
    },
    estimated_duration_minutes=30,
    complexity="complex",
)


# =============================================================================
# FEATURE DEVELOPMENT WORKFLOW
# =============================================================================()


def _build_feature_development_workflow() -> WorkflowDefinition:
    """Build feature development workflow with planning → implementation → testing."""

    return (
        WorkflowBuilder("feature_development", "End-to-end feature development")
        .add_agent(
            "plan",
            "architecture_planner",
            "Design feature architecture and implementation plan",
            tool_budget=20,
        )
        .add_agent(
            "implement",
            "code_writer",
            "Implement the feature",
            tool_budget=40,
        )
        .add_agent(
            "test",
            "test_writer",
            "Write and execute tests",
            tool_budget=25,
        )
        .add_agent(
            "review",
            "code_reviewer",
            "Review implementation",
            tool_budget=15,
        )
        .add_agent(
            "document",
            "documentation_researcher",
            "Update documentation",
            tool_budget=10,
        )
        .build()
    )


FEATURE_DEVELOPMENT_WORKFLOW = WorkflowPreset(
    name="feature_development",
    description="End-to-end feature development from planning to documentation",
    category="development",
    builder_factory=_build_feature_development_workflow,
    example_context={
        "feature_name": "user_profile_management",
        "requirements": ["create", "read", "update", "delete profiles"],
        "priority": "high",
    },
    estimated_duration_minutes=45,
    complexity="complex",
)


# =============================================================================
# SECURITY AUDIT WORKFLOW
# =============================================================================()


def _build_security_audit_workflow() -> WorkflowDefinition:
    """Build security audit workflow with scanning → analysis → recommendations."""

    return (
        WorkflowBuilder("security_audit", "Comprehensive security audit")
        .add_agent(
            "scan",
            "security_researcher",
            "Scan codebase for security issues",
            tool_budget=30,
        )
        .add_agent(
            "deep_analysis",
            "security_auditor",
            "Perform deep security analysis",
            tool_budget=25,
        )
        .add_agent(
            "recommend",
            "security_auditor",
            "Generate security recommendations",
            tool_budget=15,
        )
        .build()
    )


SECURITY_AUDIT_WORKFLOW = WorkflowPreset(
    name="security_audit",
    description="Comprehensive security audit with vulnerability scanning",
    category="security",
    builder_factory=_build_security_audit_workflow,
    example_context={
        "target_components": ["authentication", "authorization", "data_validation"],
        "compliance_standards": ["OWASP", "SOC2"],
        "severity_threshold": "medium",
    },
    estimated_duration_minutes=35,
    complexity="complex",
)


# =============================================================================
# WORKFLOW REGISTRY
# =============================================================================()

_WORKFLOW_PRESETS: Dict[str, WorkflowPreset] = {
    "code_review": CODE_REVIEW_WORKFLOW,
    "refactoring": REFACTORING_WORKFLOW,
    "research": RESEARCH_WORKFLOW,
    "bug_investigation": BUG_INVESTIGATION_WORKFLOW,
    "feature_development": FEATURE_DEVELOPMENT_WORKFLOW,
    "security_audit": SECURITY_AUDIT_WORKFLOW,
}


def get_workflow_preset(name: str) -> Optional[WorkflowPreset]:
    """Get a workflow preset by name.

    Args:
        name: Workflow preset name

    Returns:
        WorkflowPreset if found, None otherwise

    Example:
        preset = get_workflow_preset("code_review")
        workflow = preset.builder_factory()
        result = await executor.execute(workflow, context)
    """
    preset = _WORKFLOW_PRESETS.get(name)
    if preset:
        return preset
    return None


def list_workflow_presets() -> List[str]:
    """List all available workflow preset names.

    Returns:
        List of preset names

    Example:
        presets = list_workflow_presets()
        for name in presets:
            preset = get_workflow_preset(name)
            print(f"{name}: {preset.description}")
    """
    return list(_WORKFLOW_PRESETS.keys())


def list_workflow_presets_by_category() -> Dict[str, List[str]]:
    """List workflow presets grouped by category.

    Returns:
        Dictionary mapping categories to lists of preset names

    Example:
        by_category = list_workflow_presets_by_category()
        for category, presets in by_category.items():
            print(f"{category}: {', '.join(presets)}")
    """
    by_category: Dict[str, List[str]] = {}
    for name, preset in _WORKFLOW_PRESETS.items():
        if preset.category not in by_category:
            by_category[preset.category] = []
        by_category[preset.category].append(name)
    return by_category


def create_workflow_from_preset(preset_name: str, **kwargs: Any) -> Optional[WorkflowDefinition]:
    """Create a workflow from a preset with optional customization.

    Args:
        preset_name: Name of the preset
        **kwargs: Additional parameters to customize the workflow

    Returns:
        WorkflowDefinition if preset found, None otherwise

    Example:
        workflow = create_workflow_from_preset(
            "code_review",
            tool_budget_multiplier=1.5,  # Increase budgets by 50%
        )
    """
    preset = get_workflow_preset(preset_name)
    if not preset:
        return None

    # For now, just return the default workflow
    # Future: Apply customization based on kwargs
    return preset.builder_factory()


__all__ = [
    "WorkflowPreset",
    "get_workflow_preset",
    "list_workflow_presets",
    "list_workflow_presets_by_category",
    "create_workflow_from_preset",
    # Individual workflows
    "CODE_REVIEW_WORKFLOW",
    "REFACTORING_WORKFLOW",
    "RESEARCH_WORKFLOW",
    "BUG_INVESTIGATION_WORKFLOW",
    "FEATURE_DEVELOPMENT_WORKFLOW",
    "SECURITY_AUDIT_WORKFLOW",
]
