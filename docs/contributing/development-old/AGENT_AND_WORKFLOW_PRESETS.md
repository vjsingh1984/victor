# Agent and Workflow Presets

**Victor Framework**

This guide covers the preset system for agents and workflows, which provides ready-to-use configurations for common patterns.

## Overview

Victor's preset system offers:

- **12+ Agent Presets**: Pre-configured agents for common roles (researcher, reviewer, tester, etc.)
- **6+ Workflow Presets**: Complete workflow templates for common use cases
- **CLI Integration**: Easy discovery and usage via command-line tools
- **Customizable**: Use presets as-is or customize for your needs

## Quick Start

### Using Presets via CLI

```bash
# List all available presets
victor workflow presets

# List only agent presets
victor workflow presets --type agents

# List workflow presets by category
victor workflow presets --type workflows --category code_review

# Show details for a specific preset
victor workflow preset-info code_reviewer --type agent
victor workflow preset-info bug_investigation --type workflow
```

### Using Presets in Python

#### Agent Presets

```python
from victor.workflows.presets import get_agent_preset
from victor.workflows.definition import WorkflowBuilder

# Get a preset
preset = get_agent_preset("code_reviewer")

# Use in workflow builder
workflow = (
    WorkflowBuilder("my_review")
    .add_agent("review", **preset.to_workflow_builder_params(
        "Review the authentication code"
    ))
    .build()
)
```

#### Workflow Presets

```python
from victor.workflows.presets import create_workflow_from_preset

# Create workflow from preset
workflow = create_workflow_from_preset("code_review")

# Execute with executor
from victor.workflows.executor import WorkflowExecutor

executor = WorkflowExecutor(orchestrator)
result = await executor.execute(
    workflow,
    context={"pr_number": 123, "target_files": ["src/auth.py"]}
)
```

## Agent Presets

### Researcher Agents

#### code_researcher
- **Role**: `researcher`
- **Purpose**: Explore codebase to find patterns, implementations, and dependencies
- **Tools**: `read`, `code_search`, `search`, `list_directory`, `git_diff`, `web_search`
- **Budget**: 20 tool calls
- **Example Use**: Research authentication patterns in the codebase

#### documentation_researcher
- **Role**: `researcher`
- **Purpose**: Research documentation, README files, and inline comments
- **Tools**: `read`, `search`, `web_search`, `list_directory`
- **Budget**: 15 tool calls
- **Example Use**: Research API documentation for endpoints

#### security_researcher
- **Role**: `researcher`
- **Purpose**: Analyze code for security vulnerabilities and best practices
- **Tools**: `read`, `code_search`, `search`, `grep`, `web_search`
- **Budget**: 25 tool calls
- **LLM Config**: `temperature=0.1` (lower for consistent analysis)
- **Example Use**: Analyze authentication flow for security issues

### Planner Agents

#### task_planner
- **Role**: `planner`
- **Purpose**: Break down complex tasks into actionable steps
- **Tools**: `read`, `search`, `code_search`, `list_directory`
- **Budget**: 10 tool calls
- **Example Use**: Create plan for implementing user authentication

#### architecture_planner
- **Role**: `planner`
- **Purpose**: Design system architecture and component integration
- **Tools**: `read`, `code_search`, `search`, `list_directory`, `web_search`
- **Budget**: 15 tool calls
- **Example Use**: Design microservices architecture for the API

### Executor Agents

#### code_writer
- **Role**: `executor`
- **Purpose**: Write new code and implement features
- **Tools**: `read`, `write`, `edit`, `create_file`, `search`, `code_search`, `shell`, `test`
- **Budget**: 25 tool calls
- **Example Use**: Implement user registration endpoint

#### bug_fixer
- **Role**: `executor`
- **Purpose**: Diagnose and fix bugs in existing code
- **Tools**: `read`, `write`, `edit`, `search`, `code_search`, `shell`, `test`, `git_diff`, `git_log`
- **Budget**: 30 tool calls
- **Example Use**: Fix the authentication token expiration bug

#### refactoring_agent
- **Role**: `executor`
- **Purpose**: Refactor code for improved maintainability and performance
- **Tools**: `read`, `write`, `edit`, `code_search`, `search`, `test`, `shell`, `lint`
- **Budget**: 35 tool calls
- **LLM Config**: `temperature=0.2` (lower temp for consistent refactoring)
- **Example Use**: Refactor user service to follow SOLID principles

### Reviewer Agents

#### code_reviewer
- **Role**: `reviewer`
- **Purpose**: Review code changes for quality, security, and best practices
- **Tools**: `read`, `code_search`, `search`, `git_diff`, `test`, `lint`
- **Budget**: 15 tool calls
- **LLM Config**: `temperature=0.1` (very low temp for consistent reviews)
- **Example Use**: Review pull request for user authentication feature

#### security_auditor
- **Role**: `reviewer`
- **Purpose**: Perform security audits and identify vulnerabilities
- **Tools**: `read`, `code_search`, `search`, `grep`, `git_diff`, `web_search`
- **Budget**: 20 tool calls
- **LLM Config**: `temperature=0.0` (zero temp for rigorous security analysis)
- **Example Use**: Audit authentication flow for OWASP Top 10

#### performance_reviewer
- **Role**: `reviewer`
- **Purpose**: Analyze code for performance bottlenecks and optimization opportunities
- **Tools**: `read`, `code_search`, `search`, `profiler`, `shell`
- **Budget**: 18 tool calls
- **Example Use**: Analyze database query performance in reporting module

### Tester Agents

#### test_writer
- **Role**: `tester`
- **Purpose**: Write comprehensive unit and integration tests
- **Tools**: `read`, `write`, `edit`, `create_file`, `code_search`, `search`, `test`, `shell`
- **Budget**: 20 tool calls
- **Example Use**: Write unit tests for payment processing module

#### test_triage_agent
- **Role**: `tester`
- **Purpose**: Analyze test failures and create actionable bug reports
- **Tools**: `read`, `test`, `shell`, `git_log`, `git_diff`, `search`
- **Budget**: 15 tool calls
- **Example Use**: Analyze test failures in authentication suite

## Workflow Presets

### code_review
- **Category**: `code_review`
- **Description**: Multi-stage code review with analysis, review, and triage
- **Complexity**: Medium
- **Duration**: ~15 minutes
- **Workflow**: `analyze → review → decide → (suggest_improvements | request_changes) → approve`
- **Example Context**:
  ```json
  {
    "pr_number": 123,
    "target_files": ["src/auth.py", "src/user.py"],
    "review_focus": "security and performance"
  }
  ```

### refactoring
- **Category**: `refactoring`
- **Description**: Safe refactoring with analysis, planning, execution, and verification
- **Complexity**: Complex
- **Duration**: ~25 minutes
- **Workflow**: `analyze → plan → decide_approach → (execute | incremental) → verify → report`
- **Example Context**:
  ```json
  {
    "target_component": "UserService",
    "refactoring_goal": "extract validation logic",
    "safety_level": "high"
  }
  ```

### research
- **Category**: `research`
- **Description**: Deep codebase research with exploration and synthesis
- **Complexity**: Medium
- **Duration**: ~20 minutes
- **Workflow**: `explore → analyze → synthesize → report`
- **Example Context**:
  ```json
  {
    "research_topic": "authentication mechanisms",
    "depth": "comprehensive",
    "output_format": "markdown_report"
  }
  ```

### bug_investigation
- **Category**: `debugging`
- **Description**: Systematic bug investigation and resolution
- **Complexity**: Complex
- **Duration**: ~30 minutes
- **Workflow**: `reproduce → analyze → decide_fix → (hotfix | standard_fix) → verify → report`
- **Example Context**:
  ```json
  {
    "bug_description": "User authentication fails after session timeout",
    "severity": "high",
    "environment": "production"
  }
  ```

### feature_development
- **Category**: `development`
- **Description**: End-to-end feature development from planning to documentation
- **Complexity**: Complex
- **Duration**: ~45 minutes
- **Workflow**: `plan → implement → test → review → document`
- **Example Context**:
  ```json
  {
    "feature_name": "user_profile_management",
    "requirements": ["create", "read", "update", "delete profiles"],
    "priority": "high"
  }
  ```

### security_audit
- **Category**: `security`
- **Description**: Comprehensive security audit with vulnerability scanning
- **Complexity**: Complex
- **Duration**: ~35 minutes
- **Workflow**: `scan → deep_analysis → recommend`
- **Example Context**:
  ```json
  {
    "target_components": ["authentication", "authorization", "data_validation"],
    "compliance_standards": ["OWASP", "SOC2"],
    "severity_threshold": "medium"
  }
  ```

## Creating Custom Presets

### Custom Agent Preset

```python
from victor.workflows.presets.agent_templates import AgentPreset
from victor.workflows.presets import _AGENT_PRESETS

# Define custom preset
MY_CUSTOM_AGENT = AgentPreset(
    name="my_custom_agent",
    role="executor",
    description="Custom agent for my specific use case",
    goal_template="Perform {task} with custom logic",
    default_tool_budget=20,
    allowed_tools=["read", "write", "edit", "custom_tool"],
    example_goals=[
        "Custom task 1",
        "Custom task 2",
    ],
)

# Register it
_AGENT_PRESETS["my_custom_agent"] = MY_CUSTOM_AGENT
```

### Custom Workflow Preset

```python
from victor.workflows.presets.workflow_templates import (
    WorkflowPreset,
    _WORKFLOW_PRESETS,
)
from victor.workflows.definition import WorkflowBuilder

def _build_my_custom_workflow() -> WorkflowDefinition:
    """Build custom workflow."""
    return (
        WorkflowBuilder("my_workflow", "My custom workflow")
        .add_agent("step1", "researcher", "Research topic")
        .add_agent("step2", "executor", "Execute task")
        .add_agent("step3", "reviewer", "Review results")
        .build()
    )

MY_CUSTOM_WORKFLOW = WorkflowPreset(
    name="my_custom_workflow",
    description="My custom workflow for specific use case",
    category="custom",
    builder_factory=_build_my_custom_workflow,
    example_context={"param1": "value1"},
    estimated_duration_minutes=10,
    complexity="simple",
)

# Register it
_WORKFLOW_PRESETS["my_custom_workflow"] = MY_CUSTOM_WORKFLOW
```

## Best Practices

### When to Use Presets

**Use presets when:**
- You need a standard agent role (reviewer, researcher, tester, etc.)
- You're implementing a common workflow pattern
- You want to ensure consistent behavior across your codebase
- You're starting a new project and need baseline configurations

**Create custom configs when:**
- You have domain-specific requirements
- You need highly specialized tool combinations
- Presets don't match your workflow patterns
- You need fine-tuned LLM parameters

### Preset Customization

Presets are starting points, not rigid requirements. Customize them:

```python
# Start with preset
preset = get_agent_preset("code_reviewer")

# Customize for your needs
custom_params = preset.to_workflow_builder_params("Review my code")
custom_params["tool_budget"] = 25  # Increase budget
custom_params["allowed_tools"] = preset.allowed_tools + ["custom_linter"]

# Use customized params
workflow = (
    WorkflowBuilder("custom_review")
    .add_agent("review", **custom_params)
    .build()
)
```

### Workflow Composition

Combine presets with custom nodes:

```python
from victor.workflows.presets import create_workflow_from_preset
from victor.workflows.definition import WorkflowBuilder

# Get base workflow from preset
base_workflow = create_workflow_from_preset("code_review")

# Use as starting point, then customize
custom_workflow = WorkflowBuilder("extended_review")
for node_id, node in base_workflow.nodes.items():
    # Copy nodes from preset workflow
    custom_workflow._nodes[node_id] = node

# Add custom nodes
custom_workflow.add_agent("security_check", "security_auditor", "Deep security scan")
custom_workflow.add_agent("performance_check", "performance_reviewer", "Performance analysis")

# Build customized workflow
final_workflow = custom_workflow.build()
```

## Architecture Notes

### Victor vs CrewAI

Victor takes a different architectural approach than CrewAI:

| Aspect | CrewAI | Victor |
|--------|--------|--------|
| **Agent Definition** | Separate `Agent` class | `AgentNode` in workflow DAG |
| **Crew Orchestration** | `Crew` class | `TeamNodeWorkflow` + `TeamFormation` |
| **Integration** | Agents → Crew → Task | Agents as first-class workflow nodes |

**Victor's advantages:**
- Hybrid workflows (agents + compute + conditions in one DAG)
- Dynamic team formation within workflows
- HITL integration with teams
- No separate "crew" abstraction needed

### Preset System Design

The preset system is designed for:
1. **Discovery**: Easy CLI listing and info commands
2. **Consistency**: Standardized configurations across projects
3. **Customization**: Simple modification while maintaining benefits
4. **Extensibility**: Easy to add new presets

## API Reference

### Agent Preset API

```python
@dataclass
class AgentPreset:
    name: str
    role: str
    description: str
    goal_template: str
    default_tool_budget: int = 15
    allowed_tools: Optional[List[str]] = None
    system_prompt_template: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None
    example_goals: List[str] = field(default_factory=list)

    def to_workflow_builder_params(self, goal: str) -> Dict[str, Any]:
        """Convert to WorkflowBuilder.add_agent() parameters."""
```

### Workflow Preset API

```python
@dataclass
class WorkflowPreset:
    name: str
    description: str
    category: str
    builder_factory: Callable[[], WorkflowDefinition]
    example_context: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_minutes: int = 10
    complexity: str = "medium"
```

### Preset Functions

```python
# Agent presets
get_agent_preset(name: str) -> Optional[AgentPreset]
list_agent_presets() -> List[str]
list_agent_presets_by_role() -> Dict[str, List[str]]

# Workflow presets
get_workflow_preset(name: str) -> Optional[WorkflowPreset]
list_workflow_presets() -> List[str]
list_workflow_presets_by_category() -> Dict[str, List[str]]
create_workflow_from_preset(name: str, **kwargs) -> Optional[WorkflowDefinition]
```

## See Also

- [Workflow Definition Guide](../workflows/README.md)
- [Multi-Agent Teams](../teams/README.md)
- [CLI Reference](../cli/README.md)
- [Workflow Visualization](../workflows/visualization.md)
