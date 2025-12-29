# Victor Vertical Framework Integration Specification

## Implementation Status (December 2025)

**Status: ✅ COMPLETE**

All migration steps have been implemented:

| Step | Description | Status | Files Created |
|------|-------------|--------|---------------|
| 1.1 | Coding Workflow Provider | ✅ | `victor/verticals/coding/workflows/` |
| 1.2 | Feature/Bugfix/Review Workflows | ✅ | `feature.py`, `bugfix.py`, `review.py` |
| 1.3 | Workflow Provider Protocol | ✅ | `provider.py` |
| 1.4 | YAML Workflow Loader | ✅ | `victor/workflows/yaml_loader.py` |
| 2.1 | Coding RL Config | ✅ | `victor/verticals/coding/rl/config.py` |
| 2.2 | Coding RL Hooks | ✅ | `victor/verticals/coding/rl/hooks.py` |
| 2.3 | Middleware RL Integration | ✅ | `victor/verticals/coding/middleware.py` |
| 3.1 | Coding Team Specs | ✅ | `victor/verticals/coding/teams/specs.py` |
| 4 | CodingAssistant Integration | ✅ | `victor/verticals/coding/assistant.py` |
| 5 | Framework Pipeline Extension | ✅ | `victor/framework/vertical_integration.py` |
| 5.1 | Agent Workflow/Team Methods | ✅ | `victor/framework/agent.py` |

**Applied to All Verticals:**
- ✅ Coding (7 workflows, 5 learners, 6 teams)
- ✅ DevOps (3 workflows, 3 learners, 3 teams)
- ✅ Research (4 workflows, 3 learners, 5 teams)
- ✅ Data Analysis (4 workflows, 3 learners, 5 teams)

**Tests:**
- ✅ Unit tests: `tests/unit/framework/test_vertical_integration.py`
- ✅ Unit tests: `tests/unit/workflows/test_yaml_loader.py`
- ✅ Integration tests: `tests/integration/test_workflow_execution.py`

---

## Executive Summary

This document analyzes the current state of the victor-coding vertical and provides a migration plan to fully leverage the victor.framework capabilities including:
- Multi-agent Teams
- Reinforcement Learning (RL)
- Workflow DSL
- Framework Integration Pipelines

---

## 1. Current State Analysis

### 1.1 CodingAssistant Vertical

**Location**: `victor/verticals/coding/assistant.py`

**Current Capabilities**:
| Capability | Status | Implementation |
|------------|--------|----------------|
| Tools (45+) | ✅ Implemented | `get_tools()` returns tool list |
| System Prompt | ✅ Implemented | `get_system_prompt()` |
| Stages | ✅ Implemented | 7 stages (INITIAL → COMPLETION) |
| Middleware | ✅ Implemented | CodeCorrectionMiddleware, GitSafetyMiddleware |
| Safety Extension | ✅ Implemented | CodingSafetyExtension |
| Prompt Contributor | ✅ Implemented | CodingPromptContributor |
| Mode Config | ✅ Implemented | 8 modes (fast, thorough, debug, etc.) |
| Tool Dependencies | ✅ Implemented | CodingToolDependencyProvider |
| Tiered Tools | ✅ Implemented | mandatory, vertical_core, semantic_pool |
| **Workflows** | ✅ Implemented | 7 workflows via `CodingWorkflowProvider` |
| **RL Integration** | ✅ Implemented | 5 learners via `CodingRLConfig/Hooks` |
| **Teams** | ✅ Implemented | 6 teams via `CODING_TEAM_SPECS` |

### 1.2 Framework Capabilities (Now Fully Integrated)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAMEWORK CAPABILITIES                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │    Teams    │  │     RL      │  │  Workflows  │              │
│  │  AgentTeam  │  │  RLManager  │  │ WorkflowDef │              │
│  │  Formation  │  │ LearnerType │  │  Executor   │              │
│  │  TeamEvent  │  │ record_*()  │  │    HITL     │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         │     ✅ FULLY CONNECTED           │                      │
│         │                │                │                      │
│  ┌──────┴────────────────┴────────────────┴──────┐              │
│  │              CodingAssistant                   │              │
│  │  workflows + RL + teams fully integrated      │              │
│  └───────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Gap Analysis

### 2.1 Workflows Gap

**Current State**:
- `get_workflow_provider()` returns `None`
- No coding-specific workflows defined
- Workflow DSL exists in `victor/workflows/definition.py` but unused

**Framework Available**:
```python
# victor/workflows/definition.py
@workflow("code_review", "Comprehensive code review")
def code_review_workflow():
    return (
        WorkflowBuilder("code_review")
        .add_agent("analyze", "researcher", "Find code patterns")
        .add_condition("decide", lambda ctx: "fix" if ctx.get("issues") else "done")
        .add_agent("fixer", "executor", "Fix issues")
        .add_agent("report", "planner", "Summarize findings")
        .build()
    )
```

**Recommended Workflows for Coding**:
1. `feature_implementation` - End-to-end feature development
2. `bug_fix` - Systematic bug investigation and fix
3. `code_review` - Comprehensive code review
4. `refactoring` - Safe refactoring with tests
5. `test_coverage` - Increase test coverage
6. `documentation` - Generate/update documentation

### 2.2 RL Integration Gap

**Current State**:
- No RL recording in tool execution
- No learning from successful/failed patterns
- No adaptive behavior based on task type

**Framework Available**:
```python
from victor.framework.rl import RLManager, LearnerType, record_tool_success

# Record successful tool selection
rl.record_success(
    learner=LearnerType.TOOL_SELECTOR,
    provider="anthropic",
    model="claude-sonnet",
    task_type="refactoring",
    quality_score=0.95,
    metadata={"tool": "extract_function"},
)

# Get recommendation
rec = rl.get_tool_recommendation(
    task_type="debugging",
    available_tools=["read", "grep", "shell"],
)
```

**RL Use Cases for Coding**:
1. **Tool Selection** - Learn which tools work best for task types
2. **Continuation Patience** - Learn optimal retry patterns per provider
3. **Quality Thresholds** - Adapt grounding thresholds by task
4. **Mode Transitions** - Learn when to switch modes (explore → build)

### 2.3 Teams Gap

**Current State**:
- All coding tasks handled by single agent
- No role specialization
- No parallel execution of subtasks

**Framework Available**:
```python
team = await AgentTeam.create(
    name="Feature Implementation",
    goal="Implement user authentication",
    members=[
        TeamMemberSpec(role="researcher", goal="Find patterns"),
        TeamMemberSpec(role="planner", goal="Design approach"),
        TeamMemberSpec(role="executor", goal="Write code"),
        TeamMemberSpec(role="reviewer", goal="Review and test"),
    ],
    formation=TeamFormation.PIPELINE,
)
```

**Team Use Cases for Coding**:
1. **Feature Implementation** - researcher → planner → executor → reviewer
2. **Large Refactoring** - analyzer || (file1_executor, file2_executor) → integrator
3. **Bug Investigation** - parallel (log_analyzer, code_analyzer, test_analyzer) → fixer
4. **Code Review** - security_reviewer || style_reviewer || logic_reviewer → summarizer

---

## 3. Recommended Architecture

### 3.1 Enhanced CodingAssistant

```python
# victor/verticals/coding/assistant.py (enhanced)
class CodingAssistant(VerticalBase):
    # ... existing methods ...

    @classmethod
    def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
        """Get coding-specific workflow provider."""
        from victor.verticals.coding.workflows import CodingWorkflowProvider
        return CodingWorkflowProvider()

    @classmethod
    def get_rl_config(cls) -> Optional[RLConfig]:
        """Get RL configuration for coding vertical."""
        from victor.verticals.coding.rl_config import CodingRLConfig
        return CodingRLConfig()

    @classmethod
    def get_team_specs(cls) -> Dict[str, TeamSpec]:
        """Get team specifications for complex coding tasks."""
        from victor.verticals.coding.teams import CODING_TEAM_SPECS
        return CODING_TEAM_SPECS
```

### 3.2 New Directory Structure

```
victor/verticals/coding/
├── assistant.py           # Main vertical (enhanced)
├── middleware.py          # Existing
├── mode_config.py         # Existing
├── prompts.py             # Existing
├── safety.py              # Existing
├── service_provider.py    # Existing
├── tool_dependencies.py   # Existing
│
├── workflows/             # NEW: Workflow definitions
│   ├── __init__.py
│   ├── provider.py        # CodingWorkflowProvider
│   ├── feature.py         # FeatureImplementationWorkflow
│   ├── bugfix.py          # BugFixWorkflow
│   ├── review.py          # CodeReviewWorkflow
│   ├── refactor.py        # RefactoringWorkflow
│   └── workflows.yaml     # YAML DSL definitions (optional)
│
├── teams/                 # NEW: Team configurations
│   ├── __init__.py
│   ├── specs.py           # Team specifications
│   ├── roles.py           # Coding-specific roles
│   └── formations.py      # Task-specific formations
│
└── rl/                    # NEW: RL integration
    ├── __init__.py
    ├── config.py          # RLConfig for coding
    ├── hooks.py           # RL recording hooks
    └── strategies.py      # Learner strategies
```

---

## 4. Implementation Plan

### Phase 1: Workflow DSL Integration

**Files to Create**:

#### 4.1.1 `victor/verticals/coding/workflows/provider.py`

```python
"""Coding workflow provider."""
from typing import Dict, List
from victor.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import WorkflowDefinition

class CodingWorkflowProvider(WorkflowProviderProtocol):
    """Provides coding-specific workflows."""

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Return all coding workflows."""
        from victor.verticals.coding.workflows.feature import feature_workflow
        from victor.verticals.coding.workflows.bugfix import bugfix_workflow
        from victor.verticals.coding.workflows.review import review_workflow
        return {
            "feature_implementation": feature_workflow(),
            "bug_fix": bugfix_workflow(),
            "code_review": review_workflow(),
        }

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        return self.get_workflows().get(name)
```

#### 4.1.2 `victor/verticals/coding/workflows/feature.py`

```python
"""Feature implementation workflow."""
from victor.workflows.definition import (
    WorkflowBuilder,
    WorkflowDefinition,
)

def feature_workflow() -> WorkflowDefinition:
    """Multi-step feature implementation workflow."""
    return (
        WorkflowBuilder("feature_implementation")
        .set_description("End-to-end feature development with review")

        # Step 1: Research existing patterns
        .add_agent(
            "research",
            role="researcher",
            goal="Analyze codebase for relevant patterns and dependencies",
            tool_budget=20,
            allowed_tools=["read", "grep", "code_search", "overview"],
            output_key="research_findings",
        )

        # Step 2: Plan implementation
        .add_agent(
            "plan",
            role="planner",
            goal="Create implementation plan based on research",
            tool_budget=10,
            input_mapping={"findings": "research_findings"},
            output_key="implementation_plan",
        )

        # Step 3: Implement feature
        .add_agent(
            "implement",
            role="executor",
            goal="Implement the feature according to plan",
            tool_budget=40,
            allowed_tools=["read", "write", "edit", "shell", "git"],
            input_mapping={"plan": "implementation_plan"},
            output_key="implementation_result",
        )

        # Step 4: Review and test
        .add_agent(
            "review",
            role="reviewer",
            goal="Review implementation and run tests",
            tool_budget=20,
            allowed_tools=["read", "shell", "run_tests", "git_diff"],
            input_mapping={"code": "implementation_result"},
            output_key="review_result",
        )

        # Condition: fix issues or complete
        .add_condition(
            "check_review",
            condition=lambda ctx: "fix" if ctx.get("review_result", {}).get("issues") else "done",
            branches={"fix": "implement", "done": "finalize"},
        )

        # Step 5: Finalize
        .add_agent(
            "finalize",
            role="executor",
            goal="Commit changes and summarize",
            tool_budget=5,
            allowed_tools=["git_commit", "git_status"],
        )

        .build()
    )
```

#### 4.1.3 YAML Workflow Alternative

```yaml
# victor/verticals/coding/workflows/workflows.yaml
workflows:
  feature_implementation:
    description: "End-to-end feature development with review"
    nodes:
      - id: research
        type: agent
        role: researcher
        goal: "Analyze codebase for relevant patterns"
        tool_budget: 20
        tools: [read, grep, code_search, overview]
        output: research_findings
        next: [plan]

      - id: plan
        type: agent
        role: planner
        goal: "Create implementation plan"
        tool_budget: 10
        input:
          findings: research_findings
        output: implementation_plan
        next: [implement]

      - id: implement
        type: agent
        role: executor
        goal: "Implement the feature"
        tool_budget: 40
        tools: [read, write, edit, shell, git]
        input:
          plan: implementation_plan
        output: implementation_result
        next: [review]

      - id: review
        type: agent
        role: reviewer
        goal: "Review and test implementation"
        tool_budget: 20
        tools: [read, shell, run_tests, git_diff]
        output: review_result
        next: [check_review]

      - id: check_review
        type: condition
        branches:
          has_issues: implement
          no_issues: finalize

      - id: finalize
        type: agent
        role: executor
        goal: "Commit and summarize"
        tool_budget: 5
        tools: [git_commit, git_status]

  bug_fix:
    description: "Systematic bug investigation and fix"
    nodes:
      - id: investigate
        type: agent
        role: researcher
        goal: "Investigate bug root cause"
        tool_budget: 25
        tools: [read, grep, shell, git_log]
        output: investigation
        next: [diagnose]

      - id: diagnose
        type: agent
        role: planner
        goal: "Diagnose issue and plan fix"
        tool_budget: 10
        output: fix_plan
        next: [fix]

      - id: fix
        type: agent
        role: executor
        goal: "Apply the fix"
        tool_budget: 20
        tools: [read, edit, shell]
        output: fix_result
        next: [verify]

      - id: verify
        type: agent
        role: reviewer
        goal: "Verify fix with tests"
        tool_budget: 15
        tools: [run_tests, git_diff]
        next: [check_verify]

      - id: check_verify
        type: condition
        branches:
          tests_pass: commit
          tests_fail: fix

      - id: commit
        type: agent
        role: executor
        goal: "Commit the fix"
        tool_budget: 5
        tools: [git_commit]
```

### Phase 2: RL Integration

#### 4.2.1 `victor/verticals/coding/rl/config.py`

```python
"""RL configuration for coding vertical."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from victor.framework.rl import LearnerType

@dataclass
class CodingRLConfig:
    """RL configuration for coding vertical."""

    # Learners to activate
    active_learners: List[LearnerType] = field(default_factory=lambda: [
        LearnerType.TOOL_SELECTOR,
        LearnerType.CONTINUATION_PATIENCE,
        LearnerType.GROUNDING_THRESHOLD,
        LearnerType.MODE_TRANSITION,
    ])

    # Task type mappings for tool selection learning
    task_type_mappings: Dict[str, List[str]] = field(default_factory=lambda: {
        "refactoring": ["refactor", "rename_symbol", "extract_function", "edit"],
        "debugging": ["read", "grep", "shell", "run_tests", "git_log"],
        "feature": ["read", "write", "edit", "shell", "git"],
        "exploration": ["read", "grep", "code_search", "overview", "symbols"],
        "testing": ["run_tests", "test_file", "shell", "read"],
        "documentation": ["read", "write", "edit"],
    })

    # Quality thresholds by task type
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "refactoring": 0.90,  # High bar for refactoring
        "debugging": 0.85,
        "feature": 0.80,
        "exploration": 0.70,  # Lower bar for exploration
        "testing": 0.85,
        "documentation": 0.75,
    })

    # Continuation patience by provider
    default_patience: Dict[str, int] = field(default_factory=lambda: {
        "anthropic": 3,
        "openai": 3,
        "deepseek": 5,  # More patient with DeepSeek
        "ollama": 7,     # Most patient with local models
    })
```

#### 4.2.2 `victor/verticals/coding/rl/hooks.py`

```python
"""RL recording hooks for coding vertical."""
from typing import Any, Dict, Optional
from victor.framework.rl import (
    RLManager,
    LearnerType,
    create_outcome,
    record_tool_success,
)

class CodingRLHooks:
    """RL recording hooks for coding middleware."""

    def __init__(self, rl_manager: Optional[RLManager] = None):
        self._rl = rl_manager

    @property
    def rl(self) -> RLManager:
        if self._rl is None:
            from victor.framework.rl import get_rl_coordinator
            self._rl = RLManager(get_rl_coordinator())
        return self._rl

    def on_tool_success(
        self,
        tool_name: str,
        task_type: str,
        provider: str,
        model: str,
        duration_ms: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record successful tool execution."""
        self.rl.record_success(
            learner=LearnerType.TOOL_SELECTOR,
            provider=provider,
            model=model,
            task_type=task_type,
            quality_score=1.0,
            metadata={
                "tool": tool_name,
                "duration_ms": duration_ms,
                **(context or {}),
            },
            vertical="coding",
        )

    def on_tool_failure(
        self,
        tool_name: str,
        task_type: str,
        provider: str,
        model: str,
        error: str,
    ) -> None:
        """Record failed tool execution."""
        self.rl.record_failure(
            learner=LearnerType.TOOL_SELECTOR,
            provider=provider,
            model=model,
            task_type=task_type,
            error=error,
            metadata={"tool": tool_name},
            vertical="coding",
        )

    def on_continuation_success(
        self,
        provider: str,
        model: str,
        attempts: int,
    ) -> None:
        """Record successful continuation."""
        self.rl.record_success(
            learner=LearnerType.CONTINUATION_PATIENCE,
            provider=provider,
            model=model,
            task_type="continuation",
            metadata={"attempts": attempts},
            vertical="coding",
        )

    def on_mode_transition(
        self,
        from_mode: str,
        to_mode: str,
        success: bool,
        task_type: str,
    ) -> None:
        """Record mode transition outcome."""
        if success:
            self.rl.record_success(
                learner=LearnerType.MODE_TRANSITION,
                task_type=task_type,
                metadata={"from_mode": from_mode, "to_mode": to_mode},
                vertical="coding",
            )
        else:
            self.rl.record_failure(
                learner=LearnerType.MODE_TRANSITION,
                task_type=task_type,
                error="Mode transition ineffective",
                metadata={"from_mode": from_mode, "to_mode": to_mode},
                vertical="coding",
            )

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: List[str],
    ) -> Optional[List[str]]:
        """Get RL-recommended tools for task."""
        return self.rl.get_tool_recommendation(
            task_type=task_type,
            available_tools=available_tools,
            vertical="coding",
        )

    def get_patience_recommendation(
        self,
        provider: str,
        model: str,
    ) -> Optional[int]:
        """Get recommended continuation patience."""
        return self.rl.get_patience_recommendation(
            provider=provider,
            model=model,
        )
```

### Phase 3: Teams Integration

#### 4.3.1 `victor/verticals/coding/teams/specs.py`

```python
"""Team specifications for coding tasks."""
from typing import Dict, List
from victor.framework.teams import TeamMemberSpec, TeamFormation

# Coding-specific roles with tool allocations
CODING_ROLES = {
    "code_researcher": {
        "base_role": "researcher",
        "tools": ["read", "grep", "code_search", "overview", "symbols", "references"],
        "tool_budget": 25,
    },
    "code_planner": {
        "base_role": "planner",
        "tools": ["read", "overview", "plan_files"],
        "tool_budget": 15,
    },
    "code_executor": {
        "base_role": "executor",
        "tools": ["read", "write", "edit", "shell", "git"],
        "tool_budget": 40,
    },
    "code_reviewer": {
        "base_role": "reviewer",
        "tools": ["read", "git_diff", "run_tests", "shell"],
        "tool_budget": 20,
    },
    "test_writer": {
        "base_role": "executor",
        "tools": ["read", "write", "run_tests", "shell"],
        "tool_budget": 30,
    },
    "doc_writer": {
        "base_role": "executor",
        "tools": ["read", "write", "edit"],
        "tool_budget": 20,
    },
}


# Pre-defined team specifications
CODING_TEAM_SPECS: Dict[str, Dict] = {
    "feature_team": {
        "name": "Feature Implementation Team",
        "formation": TeamFormation.PIPELINE,
        "members": [
            TeamMemberSpec(role="researcher", goal="Analyze codebase for patterns"),
            TeamMemberSpec(role="planner", goal="Design implementation approach"),
            TeamMemberSpec(role="executor", goal="Implement the feature"),
            TeamMemberSpec(role="reviewer", goal="Review code and run tests"),
        ],
    },
    "bug_fix_team": {
        "name": "Bug Fix Team",
        "formation": TeamFormation.PIPELINE,
        "members": [
            TeamMemberSpec(role="researcher", goal="Investigate bug root cause"),
            TeamMemberSpec(role="executor", goal="Apply the fix"),
            TeamMemberSpec(role="reviewer", goal="Verify fix with tests"),
        ],
    },
    "refactoring_team": {
        "name": "Refactoring Team",
        "formation": TeamFormation.HIERARCHICAL,
        "members": [
            TeamMemberSpec(role="planner", goal="Plan refactoring approach"),
            TeamMemberSpec(role="executor", goal="Execute refactoring"),
            TeamMemberSpec(role="reviewer", goal="Ensure tests pass"),
        ],
    },
    "review_team": {
        "name": "Code Review Team",
        "formation": TeamFormation.PARALLEL,
        "members": [
            TeamMemberSpec(role="researcher", goal="Check security issues"),
            TeamMemberSpec(role="researcher", goal="Check code style"),
            TeamMemberSpec(role="researcher", goal="Check logic correctness"),
            TeamMemberSpec(role="planner", goal="Synthesize findings"),
        ],
    },
}


def get_team_for_task(task_type: str) -> Dict:
    """Get appropriate team specification for task type."""
    mapping = {
        "feature": "feature_team",
        "implement": "feature_team",
        "bug": "bug_fix_team",
        "fix": "bug_fix_team",
        "refactor": "refactoring_team",
        "review": "review_team",
    }
    spec_name = mapping.get(task_type, "feature_team")
    return CODING_TEAM_SPECS[spec_name]
```

---

## 5. Integration with Existing Infrastructure

### 5.1 Middleware Enhancement

Update `CodeCorrectionMiddleware` to include RL hooks:

```python
# victor/verticals/coding/middleware.py (enhanced)
class CodeCorrectionMiddleware(MiddlewareProtocol):
    def __init__(self, rl_hooks: Optional[CodingRLHooks] = None):
        self._rl_hooks = rl_hooks or CodingRLHooks()

    async def after_tool_call(
        self,
        tool_name: str,
        result: ToolResult,
        context: Dict[str, Any],
    ) -> ToolResult:
        # Record RL outcome
        if result.success:
            self._rl_hooks.on_tool_success(
                tool_name=tool_name,
                task_type=context.get("task_type", "general"),
                provider=context.get("provider", "unknown"),
                model=context.get("model", "unknown"),
                duration_ms=result.metadata.get("duration_ms", 0),
            )
        else:
            self._rl_hooks.on_tool_failure(
                tool_name=tool_name,
                task_type=context.get("task_type", "general"),
                provider=context.get("provider", "unknown"),
                model=context.get("model", "unknown"),
                error=str(result.error),
            )

        # Existing correction logic...
        return result
```

### 5.2 VerticalIntegrationPipeline Enhancement

The `VerticalIntegrationPipeline` should be extended to apply:
- Workflow registration
- RL configuration
- Team specifications

```python
# victor/framework/vertical_integration.py (enhanced steps)
class VerticalIntegrationPipeline:
    def _apply_step_10_workflows(
        self,
        orchestrator: OrchestratorProtocol,
        vertical: Type[VerticalBase],
        result: IntegrationResult,
    ) -> None:
        """Apply workflow provider."""
        provider = vertical.get_workflow_provider()
        if provider:
            workflows = provider.get_workflows()
            orchestrator.register_workflows(workflows)
            result.add_info(f"Registered {len(workflows)} workflows")

    def _apply_step_11_rl(
        self,
        orchestrator: OrchestratorProtocol,
        vertical: Type[VerticalBase],
        result: IntegrationResult,
    ) -> None:
        """Apply RL configuration."""
        rl_config = getattr(vertical, 'get_rl_config', lambda: None)()
        if rl_config:
            orchestrator.set_rl_config(rl_config)
            result.add_info(f"Applied RL config with {len(rl_config.active_learners)} learners")

    def _apply_step_12_teams(
        self,
        orchestrator: OrchestratorProtocol,
        vertical: Type[VerticalBase],
        result: IntegrationResult,
    ) -> None:
        """Register team specifications."""
        team_specs = getattr(vertical, 'get_team_specs', lambda: {})()
        if team_specs:
            orchestrator.register_team_specs(team_specs)
            result.add_info(f"Registered {len(team_specs)} team specs")
```

---

## 6. Usage Examples

### 6.1 Using Workflows

```python
from victor.framework import Agent
from victor.verticals.coding import CodingAssistant

# Create agent with coding vertical
agent = await Agent.create(
    provider="anthropic",
    vertical=CodingAssistant,
)

# Run a workflow
result = await agent.run_workflow(
    "feature_implementation",
    context={
        "feature_description": "Add user authentication",
        "target_files": ["src/auth/"],
    },
)
print(result.final_output)
```

### 6.2 Using Teams

```python
from victor.framework.teams import AgentTeam
from victor.verticals.coding.teams import get_team_for_task

# Get team spec for task type
spec = get_team_for_task("feature")

# Create and run team
team = await AgentTeam.create(
    orchestrator=agent.get_orchestrator(),
    name=spec["name"],
    goal="Implement user authentication",
    members=spec["members"],
    formation=spec["formation"],
)

async for event in team.stream():
    if event.type == TeamEventType.MEMBER_COMPLETE:
        print(f"Completed: {event.member_name}")

print(team.result.final_output)
```

### 6.3 Using RL Recommendations

```python
from victor.verticals.coding.rl import CodingRLHooks

rl = CodingRLHooks()

# Get tool recommendations for debugging
recommended = rl.get_tool_recommendation(
    task_type="debugging",
    available_tools=["read", "grep", "shell", "edit", "run_tests"],
)
print(f"Recommended tools: {recommended}")

# Get patience for provider
patience = rl.get_patience_recommendation("deepseek", "deepseek-chat")
print(f"Recommended patience: {patience}")
```

---

## 7. Migration Steps

### Step 1: Create Workflow Infrastructure
1. Create `victor/verticals/coding/workflows/` directory
2. Implement `CodingWorkflowProvider`
3. Define core workflows (feature, bugfix, review)
4. Add YAML loader for workflow definitions

### Step 2: Add RL Integration
1. Create `victor/verticals/coding/rl/` directory
2. Implement `CodingRLConfig`
3. Implement `CodingRLHooks`
4. Integrate hooks into middleware

### Step 3: Add Teams Support
1. Create `victor/verticals/coding/teams/` directory
2. Define coding-specific roles
3. Create team specifications
4. Add helper functions

### Step 4: Update CodingAssistant
1. Add `get_workflow_provider()` implementation
2. Add `get_rl_config()` method
3. Add `get_team_specs()` method
4. Update service provider registration

### Step 5: Extend Framework Integration
1. Add workflow registration to pipeline
2. Add RL config application to pipeline
3. Add team spec registration to pipeline
4. Update Agent class with workflow/team methods

---

## 8. Benefits of Migration

| Benefit | Description |
|---------|-------------|
| **Structured Workflows** | Complex tasks become reproducible DAGs |
| **Multi-Agent Coordination** | Parallelize analysis, specialize roles |
| **Adaptive Learning** | System improves tool selection over time |
| **YAML Configuration** | Non-developers can modify workflows |
| **Observability** | Rich events from teams/workflows |
| **Testability** | Workflows can be tested in isolation |
| **Reusability** | Share workflows across verticals |

---

## 9. Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Complex task success rate | ~70% | ~85% (projected) |
| Tool selection efficiency | Manual | RL-optimized |
| Parallel execution | None | 2-4x on suitable tasks |
| Task reproducibility | Low | High (workflow-based) |
| Customization | Code-only | Code + YAML DSL |

---

*Specification version: 1.0*
*Last updated: December 2025*
