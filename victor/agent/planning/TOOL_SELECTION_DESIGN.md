# Context-Aware Tool Selection for TaskPlanner

## Overview

This document describes the strategic alignment between TaskPlanner's step-based planning and Victor's existing tool selection infrastructure.

## Current State Analysis

### Existing Infrastructure

1. **Tool Selection Layers** (victor/agent/tool_selection.py):
   - `ToolSelector.select_semantic()` - Embedding-based semantic selection
   - `ToolSelector.select_keywords()` - Keyword-based category matching
   - `ToolSelector.prioritize_by_stage()` - Stage-aware filtering
   - `ToolSelector.prioritize_by_task()` - Task-type filtering

2. **Task-Aware Configuration** (victor/agent/task_tool_config_loader.py):
   - YAML-based task tool mappings (edit, search, create, analyze, design)
   - Stage-specific tools (initial, reading, executing, verifying)
   - Force-action hints and thresholds

3. **Schema-Level Broadcasting** (victor/agent/tool_selection.py):
   - `get_tools_with_levels()` - Tiered schema selection
   - FULL, COMPACT, STUB schema levels
   - Vertical-specific core tools

### TaskPlanner Step Types

From `victor/agent/planning/readable_schema.py`:

| Step Type | Description | Current Tool Mapping |
|-----------|-------------|---------------------|
| `research` | Information gathering, reading code | → StepType.RESEARCH |
| `planning` | Create a detailed plan | → StepType.PLANNING |
| `feature` | Writing/creating code | → StepType.IMPLEMENTATION |
| `bugfix` | Debugging and fixing bugs | → StepType.IMPLEMENTATION |
| `refactor` | Restructuring code | → StepType.IMPLEMENTATION |
| `test` | Writing and running tests | → StepType.TESTING |
| `review` | Code review and validation | → StepType.REVIEW |
| `deploy` | Deployment and destructive actions | → StepType.DEPLOYMENT |
| `analyze` | Analysis and investigation | → StepType.RESEARCH |
| `doc` | Writing documentation | → StepType.RESEARCH |

## Strategic Alignment

### Step Type → Tool Set Mapping

```python
STEP_TOOL_MAPPING: Dict[str, Set[str]] = {
    # Research steps need read-only exploration tools
    "research": {
        "read", "grep", "code_search", "overview", "ls",
        "git_readonly",  # For reading git history
    },

    # Planning needs exploration + analysis tools
    "planning": {
        "read", "grep", "code_search", "overview",
        "ls", "analyze",
    },

    # Feature implementation needs full toolset
    "feature": {
        "read", "write", "edit", "grep", "test",
        "code_search", "git", "shell",
    },

    # Bugfix needs debugging tools
    "bugfix": {
        "read", "grep", "edit", "test", "debugger",
        "code_search", "shell",
    },

    # Refactor needs code analysis + modification
    "refactor": {
        "read", "edit", "grep", "test", "analyze",
        "code_search",
    },

    # Testing needs test execution tools
    "test": {
        "test", "read", "grep", "shell",
    },

    # Review needs read-only + analysis
    "review": {
        "read", "grep", "analyze", "lint",
        "code_search",
    },

    # Deploy needs deployment + verification tools
    "deploy": {
        "shell", "git", "docker", "kubectl",
        "read", "test",
    },

    # Analyze needs exploration tools
    "analyze": {
        "read", "grep", "code_search", "overview",
        "analyze", "shell_readonly",
    },

    # Doc needs reading + minimal writing
    "doc": {
        "read", "grep", "write", "code_search",
    },
}
```

### Complexity-Based Tool Exposure

```python
COMPLEXITY_TOOL_LIMITS: Dict[str, int] = {
    "simple": 5,      # Auto mode, minimal tools
    "moderate": 10,   # Plan-mode, balanced tools
    "complex": 15,    # Plan-mode, comprehensive tools
}
```

## Implementation Design

### Phase 1: Step-Aware Tool Selection

Create `victor/agent/planning/tool_selection.py`:

```python
class StepAwareToolSelector:
    """Selects tools based on TaskPlanner step types."""

    def __init__(
        self,
        tool_selector: ToolSelector,
        task_config_loader: TaskToolConfigLoader,
    ):
        self.tool_selector = tool_selector
        self.task_config_loader = task_config_loader

    def get_tools_for_step(
        self,
        step_type: str,  # From ReadableTaskPlan step data
        complexity: TaskComplexity,
        step_description: str,
        conversation_stage: Optional[ConversationStage] = None,
    ) -> List[ToolDefinition]:
        """Get context-appropriate tools for a planning step.

        Args:
            step_type: Step type from plan (e.g., "research", "feature")
            complexity: Task complexity level
            step_description: What this step does
            conversation_stage: Optional conversation stage

        Returns:
            List of ToolDefinition objects
        """
        # 1. Get base tool set for step type
        base_tools = STEP_TOOL_MAPPING.get(step_type, set())

        # 2. Adjust for complexity (simple = fewer tools)
        max_tools = COMPLEXITY_TOOL_LIMITS[complexity.value]

        # 3. Add task-type specific tools if available
        # (Uses existing TaskToolConfigLoader)
        stage_name = conversation_stage.name.lower() if conversation_stage else "initial"

        # 4. Filter available tools to step-relevant set
        available_tools = self._filter_by_step_type(
            self.tool_selector.tools,
            base_tools,
            step_description,
        )

        # 5. Apply complexity limit
        if len(available_tools) > max_tools:
            available_tools = self._prioritize_core_tools(
                available_tools,
                base_tools,
                max_tools,
            )

        return available_tools

    def _filter_by_step_type(
        self,
        tools: ToolRegistry,
        step_tools: Set[str],
        description: str,
    ) -> List[ToolDefinition]:
        """Filter registry tools to step-relevant set."""
        from victor.providers.base import ToolDefinition

        result = []
        all_tools_map = {t.name: t for t in tools.list_tools()}

        # Always include critical tools
        critical_tools = get_critical_tools(tools)
        step_tools = step_tools | critical_tools

        # Add tools matching step description keywords
        for tool_name, tool in all_tools_map.items():
            if tool_name in step_tools:
                result.append(ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ))

        return result
```

### Phase 2: Integration with ReadableTaskPlan

Extend `victor/agent/planning/readable_schema.py`:

```python
class ReadableTaskPlan:
    # ... existing code ...

    def get_contextual_tools(
        self,
        tool_selector: ToolSelector,
        step_index: int,
    ) -> List[ToolDefinition]:
        """Get tools for a specific step based on context.

        Args:
            tool_selector: Tool selector instance
            step_index: Index of step in plan

        Returns:
            List of tools for this step
        """
        if step_index >= len(self.steps):
            return []

        step_data = self.steps[step_index]
        step_type = step_data[1]  # [id, type, desc, tools, deps]

        # Map step type to tool set
        step_selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            task_config_loader=TaskToolConfigLoader(),
        )

        return step_selector.get_tools_for_step(
            step_type=step_type,
            complexity=self.complexity,
            step_description=step_data[2],
        )
```

### Phase 3: Execution Context Integration

Extend `victor/agent/planning/autonomous.py`:

```python
class AutonomousPlanner:
    # ... existing code ...

    async def _execute_step(
        self,
        step: PlanStep,
        tool_selector: ToolSelector,
    ) -> StepResult:
        """Execute a step with context-aware tool selection."""
        # Get tools for this step type
        step_tools = self._get_step_tools(
            step.step_type,
            tool_selector,
        )

        # Build step prompt with tool hints
        tool_names = [t.name for t in step_tools]
        prompt = self._build_step_prompt(
            step,
            available_tools=tool_names,
        )

        # Execute via orchestrator with restricted tool set
        response = await self.orchestrator.chat(
            prompt,
            tools=step_tools,  # Pass filtered tools
        )

        return StepResult(...)

    def _get_step_tools(
        self,
        step_type: StepType,
        tool_selector: ToolSelector,
    ) -> List[ToolDefinition]:
        """Map step type to appropriate tools."""
        from victor.agent.planning.tool_selection import StepAwareToolSelector

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            task_config_loader=TaskToolConfigLoader(),
        )

        # Convert StepType to string
        step_type_str = {
            StepType.RESEARCH: "research",
            StepType.PLANNING: "planning",
            StepType.IMPLEMENTATION: "feature",
            StepType.TESTING: "test",
            StepType.REVIEW: "review",
            StepType.DEPLOYMENT: "deploy",
        }.get(step_type, "feature")

        return selector.get_tools_for_step(
            step_type=step_type_str,
            complexity=TaskComplexity.MODERATE,
            step_description=f"Step type: {step_type_str}",
        )
```

## Benefits

### 1. Token Efficiency
- **Before**: All tools exposed to LLM (30+ tools)
- **After**: Step-relevant tools only (5-15 tools)
- **Savings**: 50-80% reduction in tool schema tokens

### 2. Improved LLM Focus
- LLM sees only relevant tools for current step
- Reduces hallucination and tool misuse
- Faster decision-making

### 3. Better Alignment
- Step types directly map to tool capabilities
- Existing infrastructure reused (TaskToolConfigLoader)
- Consistent with task-type aware tool selection

### 4. Progressive Disclosure
- Research steps: read-only tools
- Implementation steps: write tools enabled
- Testing steps: test tools prioritized
- Deployment steps: deployment tools exposed

## Migration Path

### Phase 1: Foundation (Current)
- Create StepAwareToolSelector
- Define STEP_TOOL_MAPPING
- Add unit tests

### Phase 2: Integration
- Integrate with ReadableTaskPlan
- Add get_contextual_tools() method
- Update AutonomousPlanner

### Phase 3: Optimization
- Add caching for tool sets
- Implement tool set pre-computation
- Add observability/metrics

### Phase 4: Enhancement
- Dynamic tool adaptation based on execution results
- Learning from tool usage patterns
- User feedback integration

## Configuration Example

```yaml
# victor/config/step_tool_mapping.yaml
step_tool_mapping:
  research:
    core_tools:
      - read
      - grep
      - code_search
    optional_tools:
      - overview
      - ls
      - git_readonly
    excluded_tools:
      - write
      - edit
      - shell
      - deploy

  feature:
    core_tools:
      - read
      - write
      - edit
    optional_tools:
      - test
      - grep
      - code_search
    excluded_tools:
      - deploy
      - kubectl

  test:
    core_tools:
      - test
      - read
    optional_tools:
      - grep
      - shell_readonly
    excluded_tools:
      - write
      - edit
```

## Testing Strategy

1. **Unit Tests**: Test step → tool mapping
2. **Integration Tests**: Test with real tool selector
3. **End-to-End Tests**: Test full plan execution
4. **Token Efficiency Tests**: Measure token savings

## Metrics

Track:
- Average tools exposed per step type
- Token usage per step type
- Step execution success rate
- Tool selection accuracy

## Open Questions

1. Should step tools be configurable per vertical?
2. How to handle tools needed across multiple steps?
3. Should we allow dynamic tool addition during execution?
4. How to cache/pre-compute tool sets for performance?
