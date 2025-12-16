# Sub-Agent Architecture Design

**Status**: Phase 1 Complete (Implementation)
**Created**: 2025-12-16
**Updated**: 2025-12-16
**Related**: ROADMAP.md P2.1, P2.2

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Core Infrastructure | ✅ Complete | SubAgent, SubAgentOrchestrator, role prompts |
| Phase 2: Integration | ⏳ Pending | AgentOrchestrator integration |
| Phase 3: Advanced Features | ⏳ Pending | Parallel execution, hierarchical spawning |
| Phase 4: UX & Polish | ⏳ Pending | CLI commands, TUI visualization |

### Phase 1 Deliverables
- `victor/agent/subagents/__init__.py` - Package exports
- `victor/agent/subagents/base.py` - SubAgent, SubAgentConfig, SubAgentResult, SubAgentRole
- `victor/agent/subagents/orchestrator.py` - SubAgentOrchestrator, FanOutResult, role defaults
- `victor/agent/subagents/prompts.py` - Role-specific system prompts
- `tests/unit/test_subagents.py` - 31 unit tests (all passing)

### P2.2 Autonomous Planning Mode
- `victor/agent/planning/__init__.py` - Package exports
- `victor/agent/planning/base.py` - ExecutionPlan, PlanStep, StepResult, PlanResult
- `victor/agent/planning/autonomous.py` - AutonomousPlanner with plan_for_goal(), execute_plan()
- `tests/unit/test_planning.py` - 43 unit tests (all passing)

## Executive Summary

This document proposes a sub-agent architecture for Victor that enables hierarchical task delegation and parallel execution. Sub-agents are specialized, isolated instances of AgentOrchestrator with constrained scopes, allowing the main agent to delegate subtasks (research, planning, execution, review) to focused helpers.

## Problem Statement

### Current Limitations

1. **No Task Delegation**: Victor must handle all subtasks sequentially in a single context
2. **Context Saturation**: Complex tasks fill the context window with intermediate work
3. **No Specialization**: Same agent handles research, planning, coding, and testing
4. **No Parallelization**: Cannot execute independent subtasks concurrently
5. **Workflow Rigidity**: Cannot dynamically spawn helpers for specific needs

### Competitive Landscape

**Claude Code (Anthropic)**:
- Cooperative sub-agents for specialized tasks
- Main agent delegates to research/execution sub-agents
- Hierarchical context management

**Cursor**:
- Multiple agent modes (chat, apply, fix)
- Each mode is effectively a specialized sub-agent

**Aider**:
- Single-agent architecture (no sub-agents)
- Linear execution only

## Design Goals

1. **Hierarchical Delegation**: Main agent can spawn sub-agents for subtasks
2. **Role Specialization**: Sub-agents have specific roles (researcher, planner, executor, reviewer, tester)
3. **Context Isolation**: Sub-agents have independent contexts, avoiding pollution
4. **Parallel Execution**: Multiple sub-agents can work concurrently
5. **Resource Management**: Sub-agents have constrained budgets and scopes
6. **Backward Compatibility**: Existing code continues working without sub-agents

## Architecture

### Three-Layer Model

```
┌──────────────────────────────────────────────────────────────┐
│  MAIN AGENT (Parent Orchestrator)                            │
│  - Full context and tool access                              │
│  - Spawns and coordinates sub-agents                         │
│  - Aggregates sub-agent results                              │
└────────────────┬─────────────────────────────────────────────┘
                 │ Spawns
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  SUB-AGENT LAYER (Child Orchestrators)                       │
│  - Constrained context and tool access                       │
│  - Specialized roles (researcher, executor, reviewer)        │
│  - Report results back to parent                             │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  Researcher  │  Planner     │  Executor    │  Reviewer       │
│  SubAgent    │  SubAgent    │  SubAgent    │  SubAgent       │
└──────────────┴──────────────┴──────────────┴─────────────────┘
```

### Sub-Agent Roles

#### Researcher SubAgent
**Purpose**: Gather information from codebase, docs, web
**Tools**: read, search, code_search, web_search, web_fetch, overview
**Constraints**:
- Read-only tools only (no write/execute)
- Tool budget: 10-15 calls
- Context limit: 50K chars
- No sub-agent spawning (leaf node)

**Use Cases**:
- "Research how authentication works in this codebase"
- "Find all files related to database migrations"
- "Look up API documentation for library X"

#### Planner SubAgent
**Purpose**: Break down tasks into steps, generate execution plans
**Tools**: read, ls, search, plan_files
**Constraints**:
- Read-only tools only
- Tool budget: 5-10 calls
- Context limit: 30K chars
- No sub-agent spawning

**Use Cases**:
- "Create a plan to implement feature X"
- "Analyze dependencies for refactoring Y"
- "Generate step-by-step migration plan"

#### Executor SubAgent
**Purpose**: Make changes (write code, run tests, commit)
**Tools**: read, write, edit, shell, test, git
**Constraints**:
- Full tool access (write/execute)
- Tool budget: 20-30 calls
- Context limit: 80K chars
- Can spawn Reviewer sub-agents

**Use Cases**:
- "Implement function X based on this plan"
- "Refactor module Y following these rules"
- "Fix bug Z in file A"

#### Reviewer SubAgent
**Purpose**: Review changes, run tests, suggest improvements
**Tools**: read, search, test, git_diff, shell (for linting)
**Constraints**:
- Read + execute tools only (no write)
- Tool budget: 10-15 calls
- Context limit: 50K chars
- No sub-agent spawning

**Use Cases**:
- "Review these changes for bugs"
- "Check if tests pass"
- "Suggest improvements to this code"

#### Tester SubAgent
**Purpose**: Write and run tests
**Tools**: read, write (test files only), test, shell
**Constraints**:
- Can write only to test/ directories
- Tool budget: 15-20 calls
- Context limit: 50K chars
- Can spawn Reviewer for test review

**Use Cases**:
- "Write unit tests for function X"
- "Add integration test for feature Y"
- "Fix failing test in test_file.py"

### Sub-Agent Lifecycle

```
1. SPAWN
   Main agent: "I need to research authentication patterns"
   → SubAgentOrchestrator.spawn(
       role=SubAgentRole.RESEARCHER,
       task="Research authentication implementation",
       budget=10,
       allowed_tools=["read", "search", "code_search"]
     )

2. EXECUTION
   Sub-agent runs in isolated context with:
   - Independent message history
   - Constrained tool access
   - Limited budget
   - Scoped working directory (optional)

3. COMPLETION
   Sub-agent returns structured result:
   {
     "success": true,
     "summary": "Found 3 authentication patterns...",
     "findings": [...],
     "tool_calls_used": 7,
     "context_size": 12000
   }

4. AGGREGATION
   Main agent incorporates result:
   - Add summary to context
   - Optionally include detailed findings
   - Continue with next step
```

### Implementation Design

#### Core Classes

```python
# victor/agent/subagents/base.py

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from victor.agent.orchestrator import AgentOrchestrator

class SubAgentRole(Enum):
    """Role specialization for sub-agents."""
    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    TESTER = "tester"

@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""
    role: SubAgentRole
    task: str
    allowed_tools: List[str]
    tool_budget: int
    context_limit: int
    can_spawn_subagents: bool = False
    working_directory: Optional[str] = None
    timeout_seconds: int = 300

@dataclass
class SubAgentResult:
    """Result from sub-agent execution."""
    success: bool
    summary: str
    details: Dict[str, Any]
    tool_calls_used: int
    context_size: int
    duration_seconds: float
    error: Optional[str] = None

class SubAgent:
    """Represents a spawned sub-agent instance.

    This is a wrapper around AgentOrchestrator with:
    - Constrained tool access
    - Independent context
    - Limited budget
    - Role-specific system prompt
    """

    def __init__(self, config: SubAgentConfig, parent_orchestrator: AgentOrchestrator):
        self.config = config
        self.parent = parent_orchestrator

        # Create isolated orchestrator with constraints
        self.orchestrator = self._create_constrained_orchestrator()

    def _create_constrained_orchestrator(self) -> AgentOrchestrator:
        """Create orchestrator with role-specific constraints."""
        # Copy settings from parent but apply constraints
        settings = self.parent.settings.copy(deep=True)
        settings.tool_budget = self.config.tool_budget
        settings.max_context_chars = self.config.context_limit

        # Create new orchestrator instance
        orchestrator = AgentOrchestrator(
            settings=settings,
            provider=self.parent.provider,
            model=self.parent.model,
            temperature=self.parent.temperature,
            # DI container with limited services
            container=self._create_constrained_container(),
        )

        # Set role-specific system prompt
        orchestrator.set_system_prompt(self._get_role_prompt())

        # Register only allowed tools
        orchestrator.tool_registry.clear()
        for tool_name in self.config.allowed_tools:
            tool = self.parent.tool_registry.get(tool_name)
            if tool:
                orchestrator.tool_registry.register(tool)

        return orchestrator

    def _get_role_prompt(self) -> str:
        """Get system prompt for this role."""
        base = f"You are a {self.config.role.value} sub-agent."

        prompts = {
            SubAgentRole.RESEARCHER: """
You are a researcher sub-agent. Your job is to gather information.
- Use read-only tools to explore the codebase
- Summarize your findings clearly
- Focus on answering the specific question
- Do not make any changes
            """,
            SubAgentRole.PLANNER: """
You are a planner sub-agent. Your job is to create execution plans.
- Break tasks into clear, actionable steps
- Consider dependencies and risks
- Output structured plans
- Do not execute the plan
            """,
            SubAgentRole.EXECUTOR: """
You are an executor sub-agent. Your job is to make changes.
- Follow the plan provided
- Write clean, tested code
- Commit changes atomically
- Report what you did
            """,
            SubAgentRole.REVIEWER: """
You are a reviewer sub-agent. Your job is to check quality.
- Review code for bugs and style issues
- Run tests and linters
- Suggest improvements
- Do not make changes yourself
            """,
            SubAgentRole.TESTER: """
You are a tester sub-agent. Your job is to write tests.
- Write comprehensive test cases
- Ensure good coverage
- Follow testing best practices
- Only write to test directories
            """,
        }

        return base + prompts.get(self.config.role, "")

    async def execute(self) -> SubAgentResult:
        """Execute the sub-agent task."""
        import time
        start_time = time.time()

        try:
            # Run the task
            response = await self.orchestrator.chat(self.config.task)

            # Extract result
            return SubAgentResult(
                success=True,
                summary=response.content[:500],  # First 500 chars
                details={
                    "full_response": response.content,
                    "tool_calls": response.tool_calls or [],
                },
                tool_calls_used=self.orchestrator.tool_calls_used,
                context_size=len(str(self.orchestrator.messages)),
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return SubAgentResult(
                success=False,
                summary=f"Sub-agent failed: {str(e)}",
                details={},
                tool_calls_used=self.orchestrator.tool_calls_used,
                context_size=len(str(self.orchestrator.messages)),
                duration_seconds=time.time() - start_time,
                error=str(e),
            )


class SubAgentOrchestrator:
    """Orchestrates sub-agent lifecycle and coordination."""

    def __init__(self, parent_orchestrator: AgentOrchestrator):
        self.parent = parent_orchestrator
        self.active_subagents: Dict[str, SubAgent] = {}

    async def spawn(
        self,
        role: SubAgentRole,
        task: str,
        budget: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> SubAgentResult:
        """Spawn a sub-agent for a specific task.

        Args:
            role: Sub-agent role (researcher, executor, etc.)
            task: Task description for the sub-agent
            budget: Tool call budget (uses role default if not specified)
            allowed_tools: Specific tools to allow (uses role default if not specified)

        Returns:
            SubAgentResult with execution outcome
        """
        # Get role-specific defaults
        config = self._get_role_config(role, task, budget, allowed_tools)

        # Create sub-agent
        subagent = SubAgent(config, self.parent)

        # Execute
        result = await subagent.execute()

        return result

    async def fan_out(
        self,
        tasks: List[Tuple[SubAgentRole, str]],
        max_concurrent: int = 3,
    ) -> List[SubAgentResult]:
        """Execute multiple sub-agent tasks in parallel.

        Args:
            tasks: List of (role, task) tuples
            max_concurrent: Maximum concurrent sub-agents

        Returns:
            List of SubAgentResult in same order as tasks
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_task(role: SubAgentRole, task: str) -> SubAgentResult:
            async with semaphore:
                return await self.spawn(role, task)

        # Run all tasks with concurrency limit
        results = await asyncio.gather(
            *[_run_task(role, task) for role, task in tasks],
            return_exceptions=True,
        )

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                role, task = tasks[i]
                final_results.append(SubAgentResult(
                    success=False,
                    summary=f"Task failed: {str(result)}",
                    details={},
                    tool_calls_used=0,
                    context_size=0,
                    duration_seconds=0,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

    def _get_role_config(
        self,
        role: SubAgentRole,
        task: str,
        budget: Optional[int],
        allowed_tools: Optional[List[str]],
    ) -> SubAgentConfig:
        """Get configuration for a role."""
        # Role-specific defaults
        defaults = {
            SubAgentRole.RESEARCHER: SubAgentConfig(
                role=role,
                task=task,
                allowed_tools=allowed_tools or [
                    "read", "ls", "search", "code_search", "grep",
                    "web_search", "web_fetch", "overview", "docs_coverage"
                ],
                tool_budget=budget or 15,
                context_limit=50000,
                can_spawn_subagents=False,
            ),
            SubAgentRole.PLANNER: SubAgentConfig(
                role=role,
                task=task,
                allowed_tools=allowed_tools or [
                    "read", "ls", "search", "plan_files", "overview"
                ],
                tool_budget=budget or 10,
                context_limit=30000,
                can_spawn_subagents=False,
            ),
            SubAgentRole.EXECUTOR: SubAgentConfig(
                role=role,
                task=task,
                allowed_tools=allowed_tools or [
                    "read", "write", "edit", "ls", "search", "shell",
                    "test", "git", "git_commit"
                ],
                tool_budget=budget or 30,
                context_limit=80000,
                can_spawn_subagents=True,  # Can spawn reviewers
            ),
            SubAgentRole.REVIEWER: SubAgentConfig(
                role=role,
                task=task,
                allowed_tools=allowed_tools or [
                    "read", "search", "test", "git_diff", "shell"
                ],
                tool_budget=budget or 15,
                context_limit=50000,
                can_spawn_subagents=False,
            ),
            SubAgentRole.TESTER: SubAgentConfig(
                role=role,
                task=task,
                allowed_tools=allowed_tools or [
                    "read", "write", "test", "shell"
                ],
                tool_budget=budget or 20,
                context_limit=50000,
                can_spawn_subagents=True,  # Can spawn reviewers
            ),
        }

        return defaults[role]
```

### Integration with Main Agent

```python
# In victor/agent/orchestrator.py

class AgentOrchestrator:
    def __init__(self, ...):
        # ...existing code...

        # Add sub-agent orchestrator
        self.subagent_orchestrator = SubAgentOrchestrator(self)

    async def _handle_complex_task(self, task: str) -> str:
        """Handle complex task using sub-agents.

        Example workflow:
        1. Spawn Planner to create plan
        2. Spawn Executor to implement each step
        3. Spawn Reviewer to check quality
        4. Aggregate results
        """
        # Step 1: Plan
        plan_result = await self.subagent_orchestrator.spawn(
            SubAgentRole.PLANNER,
            f"Create a plan to: {task}"
        )

        if not plan_result.success:
            return f"Planning failed: {plan_result.error}"

        # Step 2: Execute plan steps
        # (parse plan and execute each step)

        # Step 3: Review
        review_result = await self.subagent_orchestrator.spawn(
            SubAgentRole.REVIEWER,
            "Review the changes made and suggest improvements"
        )

        # Step 4: Aggregate
        return f"""
Task completed using sub-agents:

Planning: {plan_result.summary}
Execution: (implementation details)
Review: {review_result.summary}

Total tool calls: {plan_result.tool_calls_used + review_result.tool_calls_used}
        """
```

## Use Cases

### Use Case 1: Research Then Implement

```python
# User: "Add JWT authentication to the API"

# Main agent workflow:
1. Spawn RESEARCHER sub-agent:
   Task: "Research current authentication implementation"
   Result: "Found basic auth in auth.py, no JWT"

2. Spawn PLANNER sub-agent:
   Task: "Plan JWT authentication implementation based on research"
   Result: "Steps: 1) Add jwt library 2) Create token generation 3) Add middleware..."

3. Spawn EXECUTOR sub-agent:
   Task: "Implement JWT auth following this plan: {plan}"
   Result: "Added jwt_utils.py, updated auth.py, created middleware"

4. Spawn REVIEWER sub-agent:
   Task: "Review JWT implementation for security issues"
   Result: "Looks good, but add token expiration check"

5. Spawn EXECUTOR sub-agent:
   Task: "Add token expiration check as suggested"
   Result: "Added expiration validation"

6. Main agent aggregates and reports to user
```

### Use Case 2: Parallel Research

```python
# User: "Research error handling patterns across the codebase"

# Main agent spawns 3 concurrent researchers:
results = await subagent_orchestrator.fan_out([
    (SubAgentRole.RESEARCHER, "Find try/except patterns in victor/agent/"),
    (SubAgentRole.RESEARCHER, "Find error classes in victor/core/errors.py"),
    (SubAgentRole.RESEARCHER, "Find logging of errors across codebase"),
], max_concurrent=3)

# Aggregate findings from all researchers
summary = aggregate_research(results)
```

### Use Case 3: Test-Driven Development

```python
# User: "Implement calculate_discount function with tests"

1. Spawn TESTER sub-agent:
   Task: "Write tests for calculate_discount(price, discount_percent)"
   Result: "Created test_pricing.py with 5 test cases"

2. Spawn EXECUTOR sub-agent:
   Task: "Implement calculate_discount to pass these tests: {test_code}"
   Result: "Created pricing.py with implementation"

3. Spawn TESTER sub-agent:
   Task: "Run the tests"
   Result: "All 5 tests passing"
```

## Implementation Plan

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Create `victor/agent/subagents/base.py` with SubAgent classes
- [x] Implement SubAgentOrchestrator with spawn() method
- [x] Add role-specific system prompts
- [x] Create constrained orchestrator factory
- [x] Unit tests for sub-agent creation (31 tests passing)

### Phase 2: Integration (Week 2)
- [ ] Integrate SubAgentOrchestrator into AgentOrchestrator
- [ ] Add sub-agent awareness to tool pipeline
- [ ] Implement result aggregation
- [ ] Add sub-agent metrics/logging
- [ ] Integration tests for spawn/execute flow

### Phase 3: Advanced Features (Week 3)
- [ ] Implement fan_out() for parallel execution
- [ ] Add hierarchical sub-agent spawning (Executor → Reviewer)
- [ ] Implement sub-agent context isolation
- [ ] Add sub-agent budget enforcement
- [ ] Performance tests for parallel execution

### Phase 4: UX & Polish (Week 4)
- [ ] Add /spawn command to CLI
- [ ] Show sub-agent activity in TUI
- [ ] Document sub-agent patterns
- [ ] Create examples and tutorials
- [ ] User testing and feedback

## Security Considerations

1. **Sandbox Isolation**: Sub-agents should not access parent's sensitive data
2. **Tool Restrictions**: Strict enforcement of allowed_tools list
3. **Budget Limits**: Hard limits on tool calls to prevent runaway sub-agents
4. **Context Leakage**: Sub-agent contexts should not leak into parent
5. **Privilege Escalation**: Sub-agents cannot elevate their permissions

## Performance Considerations

1. **Memory Overhead**: Each sub-agent is a full orchestrator (~10MB)
2. **Concurrency**: Limit concurrent sub-agents (default: 3)
3. **Context Duplication**: Sub-agents duplicate parent settings/config
4. **Provider Rate Limits**: Multiple sub-agents share same API limits
5. **Timeout Management**: Sub-agents must timeout gracefully

## Open Questions

1. Should sub-agents share tool cache with parent?
2. How to handle sub-agent failures in fan_out()?
3. Should sub-agents emit events to parent's event bus?
4. Maximum sub-agent nesting depth?
5. How to visualize sub-agent hierarchy in TUI?

## References

- ROADMAP.md: P2.1 Sub-Agent Architecture
- ADR-0001: Protocol-First Architecture (sub-agents use protocols)
- ADR-0004: Phase 10 DI Migration (sub-agents use DI container)
- victor/agent/orchestrator.py: Parent orchestrator implementation
