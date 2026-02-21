# TaskPlanner with Compact JSON Schema

## Overview

The TaskPlanner provides token-efficient task planning using a compact JSON schema that reduces token usage by ~50% compared to verbose JSON while maintaining type safety and easy validation.

## Key Features

### 🎯 Token Efficiency
- **50% fewer tokens** than verbose JSON
- Short field names (n, c, d, s, e, a)
- List-based step representation
- Optimized for LLM generation

### 📋 Compact Schema

```json
{
  "n": "task name",           // task_name
  "c": "simple|moderate|complex",  // complexity
  "d": "description",          // description
  "s": [                      // steps (list format)
    [step_id, type, description, tools, dependencies]
  ],
  "e": "30min",               // estimated_duration (optional)
  "a": false                  // requires_approval (optional)
}
```

### 🔄 Step Types

| Code | Type | Description |
|------|------|-------------|
| `res` | RESEARCH | Information gathering |
| `feat` | FEATURE | Feature implementation |
| `bug` | BUGFIX | Bug fixing/debugging |
| `ref` | REFACTOR | Code refactoring |
| `test` | TESTING | Writing/running tests |
| `rev` | REVIEW | Code review |
| `dep` | DEPLOYMENT | Deployment/destructive |
| `anl` | ANALYSIS | Analysis/investigation |
| `doc` | DOCUMENTATION | Documentation |

### 📊 Token Comparison

```
Verbose JSON: ~180 tokens
{
  "task_name": "Add authentication",
  "complexity": "moderate",
  "description": "Implement OAuth2 login",
  "steps": [
    {
      "id": "1",
      "type": "feature",
      "description": "Create auth module",
      "tools": ["write", "test"]
    }
  ]
}

Compact JSON: ~90 tokens (50% savings!)
{"n":"Add auth","c":"moderate","d":"OAuth2","s":[[1,"feat","Create module","write,test"]]}
```

## Usage

### Basic Usage

```python
from victor.agent.planning import CompactTaskPlan, generate_compact_plan

# Generate plan from LLM
provider = YourProvider()
compact_plan = await generate_compact_plan(
    provider,
    "Add user authentication with JWT"
)

# Convert to execution plan
execution_plan = compact_plan.to_execution_plan()

# Display to user
print(compact_plan.to_markdown())

# Convert to YAML for storage
yaml_str = compact_plan.to_yaml()
```

### Session Context Integration

```python
from victor.agent.planning import TaskPlannerContext, plan_to_session_context

# Initialize context
context = TaskPlannerContext()
context.set_plan(execution_plan)

# Add to session context for persistence
session_context = plan_to_session_context(
    compact_plan,
    session_id="user-session-123"
)

# Export for LLM context
llm_context = context.to_context_dict()
```

### Auto vs Plan Mode

```python
from victor.agent.planning import TaskComplexity

# Simple tasks: Auto mode (direct execution)
simple_plan = CompactTaskPlan(
    n="Fix typo",
    c=TaskComplexity.SIMPLE,
    d="Fix typo in README",
    s=[[1, "bug", "Fix typo", "write"]]
)

# Complex tasks: Plan mode (user approval)
complex_plan = CompactTaskPlan(
    n="Deploy to prod",
    c=TaskComplexity.COMPLEX,
    d="Deploy API to production",
    s=[[1, "dep", "Deploy", "kubectl"], [2, "dep", "Verify", "curl"]],
    a=True  # Requires approval
)
```

## LLM Prompts

### Complexity Classification Prompt

```python
prompt = CompactTaskPlan.get_complexity_prompt()
```

Output:
```json
{
  "complexity": "moderate",
  "reason": "Multiple files involved with some uncertainty"
}
```

### Plan Generation Prompt

```python
prompt = CompactTaskPlan.get_llm_prompt()
```

Output:
```json
{
  "n": "Add auth",
  "c": "moderate",
  "d": "OAuth2 login",
  "s": [
    [1, "res", "Research patterns", "overview"],
    [2, "feat", "Create module", "write,test"]
  ]
}
```

## API Reference

### CompactTaskPlan

**Methods:**
- `to_execution_plan()` - Convert to ExecutionPlan
- `to_yaml()` - Convert to YAML workflow
- `to_markdown()` - Convert to markdown display
- `from_execution_plan()` - Create from ExecutionPlan
- `get_llm_prompt()` - Get LLM prompt
- `get_complexity_prompt()` - Get classification prompt

### TaskPlannerContext

**Methods:**
- `set_plan(plan)` - Set current active plan
- `approve_plan()` - Mark current plan as approved
- `archive_plan()` - Archive current plan to history
- `get_plan_summary()` - Get summary of all plans
- `to_context_dict()` - Export for LLM context

### Helper Functions

- `generate_compact_plan(provider, request, complexity)` - Generate plan using LLM
- `plan_to_workflow_yaml(plan)` - Convert to YAML workflow
- `plan_to_session_context(plan, session_id)` - Add to session context

## Token Efficiency Metrics

| Complexity | Verbose JSON | Compact JSON | Savings |
|------------|--------------|--------------|---------|
| Simple (2-3 steps) | ~150 tokens | ~80 tokens | 47% |
| Moderate (3-5 steps) | ~200 tokens | ~110 tokens | 45% |
| Complex (5-8 steps) | ~280 tokens | ~150 tokens | 46% |

## Design Decisions

### Why JSON Instead of TOML/YAML?

1. **LLM Training**: LLMs are trained extensively on JSON
2. **Validation**: Direct Pydantic validation
3. **Error Messages**: Clear, actionable error messages
4. **Conversion**: Easy JSON ↔ YAML conversion for storage
5. **Short Keys**: Achieve same token efficiency as TOML/YAML

### Why Short Keys?

| Full Name | Short Key | Token Savings |
|-----------|------------|----------------|
| task_name | n | 9 chars |
| complexity | c | 11 chars |
| description | d | 11 chars |
| steps | s | 6 chars |
| estimated_duration | e | 18 chars |
| requires_approval | a | 18 chars |

**Total savings per plan: ~70-80 characters**

## Integration with Session Context

The TaskPlanner is designed to work with conversation sessions:

```python
# In your orchestrator or session manager
from victor.agent.planning import TaskPlannerContext

class ConversationSession:
    def __init__(self):
        self.planner_context = TaskPlannerContext()

    async def handle_task_request(self, user_request: str):
        # Generate plan
        compact_plan = await generate_compact_plan(
            self.provider,
            user_request
        )

        # Add to context
        execution_plan = compact_plan.to_execution_plan()
        self.planner_context.set_plan(execution_plan)

        # For plan-mode tasks, get approval
        if compact_plan.a:
            approved = await self.request_approval(compact_plan.to_markdown())
            if not approved:
                return

        # Execute plan
        result = await self.execute_plan(execution_plan)
        self.planner_context.archive_plan()
```

## Best Practices

1. **Always validate** LLM output with Pydantic
2. **Use compact JSON** for LLM generation
3. **Convert to YAML** for storage and human editing
4. **Add to session context** for multi-turn conversations
5. **Use complexity classification** to determine auto vs plan mode
