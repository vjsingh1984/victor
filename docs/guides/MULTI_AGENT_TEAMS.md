# Multi-Agent Teams Guide

This guide covers Victor's multi-agent team system for orchestrating collaborative AI agents.

## Overview

Victor supports multi-agent workflows where multiple specialized agents collaborate to solve complex tasks. Key features:

- **4 Team Formations**: Sequential, Parallel, Hierarchical, Pipeline
- **Rich Personas**: Communication styles, expertise levels, backstories
- **Inter-Agent Communication**: Message bus and shared memory
- **Pre-built Team Specs**: Feature implementation, code review, bug fix teams
- **Progress Tracking**: Real-time callbacks and observability

## Quick Start

```python
from victor.framework import Agent

# Create an agent
agent = await Agent.create(provider="anthropic")

# Create a team from preset spec
team = agent.create_team("feature_implementation")

# Run the team
result = await agent.run_team(team, task="Add user authentication")
print(result.final_output)
```

## Team Formations

### Sequential

Agents execute one after another, each receiving the previous agent's output.

```python
from victor.teams import TeamConfig, TeamFormation, TeamMember

config = TeamConfig(
    name="review_pipeline",
    formation=TeamFormation.SEQUENTIAL,
    members=[
        TeamMember(id="analyzer", role="Code Analyzer", ...),
        TeamMember(id="reviewer", role="Code Reviewer", ...),
        TeamMember(id="approver", role="Final Approver", ...),
    ],
    task="Review the authentication module"
)
```

**Use when**: Tasks have clear stages that must happen in order.

### Parallel

All agents work simultaneously on the same task.

```python
config = TeamConfig(
    name="multi_review",
    formation=TeamFormation.PARALLEL,
    members=[
        TeamMember(id="security", role="Security Reviewer", ...),
        TeamMember(id="style", role="Style Reviewer", ...),
        TeamMember(id="logic", role="Logic Reviewer", ...),
    ],
    task="Review this pull request"
)
```

**Use when**: Multiple perspectives needed independently.

### Hierarchical

A manager agent delegates to workers, then synthesizes results.

```python
config = TeamConfig(
    name="complex_feature",
    formation=TeamFormation.HIERARCHICAL,
    members=[
        TeamMember(id="manager", role="Tech Lead", is_manager=True, ...),
        TeamMember(id="dev1", role="Backend Developer", ...),
        TeamMember(id="dev2", role="Frontend Developer", ...),
    ],
    task="Implement user dashboard"
)
```

**Use when**: Complex tasks requiring planning, delegation, and synthesis.

### Pipeline

Each agent's output becomes the next agent's input, with handoff messages.

```python
config = TeamConfig(
    name="code_pipeline",
    formation=TeamFormation.PIPELINE,
    members=[
        TeamMember(id="researcher", role="Researcher", ...),
        TeamMember(id="implementer", role="Implementer", ...),
        TeamMember(id="tester", role="Tester", ...),
    ],
    task="Add caching to the API"
)
```

**Use when**: Tasks with clear input/output transformations.

## Personas

Define agent personalities with `PersonaTraits`:

```python
from victor.framework.multi_agent import (
    PersonaTraits,
    CommunicationStyle,
    ExpertiseLevel,
)

security_expert = PersonaTraits(
    name="SecurityBot",
    role="Security Analyst",
    description="Expert in application security and vulnerability detection",
    communication_style=CommunicationStyle.TECHNICAL,
    expertise_level=ExpertiseLevel.SPECIALIST,
    strengths=["vulnerability detection", "secure coding", "threat modeling"],
    weaknesses=["UI/UX design"],
    preferred_tools=["security_scan", "dependency_audit"],
    risk_tolerance=0.1,  # Very risk-averse
    creativity=0.3,      # Methodical
    verbosity=0.7,       # Detailed explanations
)
```

### Communication Styles

| Style | Description | Best For |
|-------|-------------|----------|
| `FORMAL` | Professional, structured | Documentation, reports |
| `CASUAL` | Friendly, conversational | User interactions |
| `TECHNICAL` | Precise, detailed | Code analysis, debugging |
| `CONCISE` | Brief, to-the-point | Quick tasks, summaries |

### Expertise Levels

| Level | Description | Tool Budget |
|-------|-------------|-------------|
| `NOVICE` | Learning, needs guidance | 5-10 |
| `INTERMEDIATE` | Competent, reliable | 10-20 |
| `EXPERT` | Deep knowledge | 20-30 |
| `SPECIALIST` | Domain authority | 30-50 |

## Team Members

Create team members with rich context:

```python
from victor.teams import TeamMember, MemoryConfig

researcher = TeamMember(
    id="researcher",
    role="Code Researcher",
    goal="Understand the codebase structure and find relevant patterns",
    tool_budget=25,
    persona=PersonaTraits(
        name="ResearchBot",
        communication_style=CommunicationStyle.TECHNICAL,
        expertise_level=ExpertiseLevel.EXPERT,
    ),
    backstory="You have analyzed thousands of codebases and can quickly identify patterns, anti-patterns, and architectural decisions.",
    memory=MemoryConfig(
        enabled=True,
        persist=True,
        namespace="research_findings"
    ),
    max_delegation_depth=1,  # Can delegate once
    can_delegate=True,
)
```

### Member Properties

| Property | Description | Default |
|----------|-------------|---------|
| `id` | Unique identifier | Required |
| `role` | Role description | Required |
| `goal` | Task-specific objective | None |
| `tool_budget` | Max tool calls | 20 |
| `persona` | PersonaTraits | None |
| `backstory` | Context/history | None |
| `memory` | Memory config | Disabled |
| `can_delegate` | Allow delegation | False |

## Pre-built Team Specs

Victor includes pre-configured teams in `victor/coding/teams/specs.py`:

### Feature Implementation Team

```python
from victor.coding.teams import FEATURE_IMPLEMENTATION_TEAM

team = FEATURE_IMPLEMENTATION_TEAM
# Pipeline: Researcher → Planner → Implementer → Reviewer
```

### Bug Fix Team

```python
from victor.coding.teams import BUG_FIX_TEAM

# Pipeline: Investigator → Fixer → Verifier
```

### Code Review Team

```python
from victor.coding.teams import CODE_REVIEW_TEAM

# Parallel: Security + Style + Logic + Synthesizer
```

### Refactoring Team

```python
from victor.coding.teams import REFACTORING_TEAM

# Hierarchical: Manager → Executors → Quality Verifier
```

## Inter-Agent Communication

### Message Bus

Agents can send messages to each other:

```python
from victor.teams import AgentMessage, MessageType

# Send a message
message = AgentMessage(
    sender_id="researcher",
    recipient_id="implementer",
    message_type=MessageType.HANDOFF,
    content="Found the pattern at src/auth/handler.py:45"
)

# Broadcast to all members
await coordinator.broadcast(message)
```

### Message Types

| Type | Description |
|------|-------------|
| `DISCOVERY` | Share a finding |
| `REQUEST` | Ask for help |
| `RESPONSE` | Reply to request |
| `STATUS` | Progress update |
| `ALERT` | Important notification |
| `HANDOFF` | Transfer task |
| `RESULT` | Final output |

### Shared Memory

Teams share discoveries across members:

```python
# Store a discovery
await team.remember(
    key="auth_pattern",
    value={"file": "auth.py", "pattern": "decorator-based"},
    metadata={"confidence": 0.9}
)

# Recall relevant memories
memories = await team.recall("authentication patterns")
```

## Progress Tracking

Monitor team execution in real-time:

```python
from victor.teams import TeamCoordinator

coordinator = TeamCoordinator(orchestrator)

def on_member_complete(member_id: str, result: MemberResult):
    print(f"{member_id} completed: {result.success}")

result = await coordinator.execute_team(
    config,
    on_member_complete=on_member_complete
)
```

### Team Result

```python
result = await coordinator.execute_team(config)

print(f"Success: {result.success}")
print(f"Final output: {result.final_output}")
print(f"Formation used: {result.formation_used}")
print(f"Total duration: {result.total_duration}s")

# Individual member results
for member_id, member_result in result.member_results.items():
    print(f"  {member_id}: {member_result.success}")

# Communication log
for message in result.communication_log:
    print(f"  {message.sender_id} → {message.recipient_id}: {message.content}")
```

## Team Registry

Register and discover teams:

```python
from victor.framework.team_registry import get_team_registry

registry = get_team_registry()

# Register a custom team
registry.register(
    name="my_team",
    spec=my_team_spec,
    vertical="coding",
    tags=["custom", "review"],
    description="My custom review team"
)

# Find teams by vertical
coding_teams = registry.find_by_vertical("coding")

# Find teams by tag
review_teams = registry.find_by_tag("review")

# List all teams
all_teams = registry.list_teams()
```

## Best Practices

### 1. Right-Size Your Teams

```python
# Good - focused team with clear roles
config = TeamConfig(
    formation=TeamFormation.PIPELINE,
    members=[
        TeamMember(id="analyzer", role="Analyzer", tool_budget=15),
        TeamMember(id="fixer", role="Fixer", tool_budget=25),
    ]
)

# Avoid - too many agents with overlapping roles
```

### 2. Use Appropriate Formations

| Task Type | Recommended Formation |
|-----------|----------------------|
| Multi-step process | PIPELINE |
| Independent reviews | PARALLEL |
| Complex planning | HIERARCHICAL |
| Simple handoffs | SEQUENTIAL |

### 3. Set Tool Budgets Wisely

```python
# Researchers need fewer tools
researcher = TeamMember(tool_budget=15, ...)

# Implementers need more
implementer = TeamMember(tool_budget=40, ...)
```

### 4. Enable Memory for Learning

```python
member = TeamMember(
    memory=MemoryConfig(
        enabled=True,
        persist=True,  # Persist across sessions
        namespace="findings"
    )
)
```

### 5. Use Backstories for Context

```python
member = TeamMember(
    backstory="""You are a senior security engineer with 10 years
    of experience. You've seen every type of vulnerability and
    know the OWASP Top 10 by heart. Your reviews have prevented
    countless breaches."""
)
```

## Observability

Team events are emitted to EventBus:

```python
from victor.observability.event_bus import get_event_bus, EventCategory

bus = get_event_bus()

bus.subscribe(EventCategory.LIFECYCLE, lambda e:
    print(f"Team event: {e.event_type} - {e.data}")
)

# Events emitted:
# - team_started
# - member_started
# - member_completed
# - team_completed
# - team_error
```

## Troubleshooting

### Team Not Making Progress

1. Check individual member tool budgets
2. Verify formation matches task structure
3. Review member goals for clarity

### Poor Collaboration

1. Enable shared memory
2. Add explicit handoff messages
3. Use hierarchical formation for complex tasks

### Inconsistent Results

1. Set lower creativity for deterministic tasks
2. Use SPECIALIST expertise for critical roles
3. Add verification member at end of pipeline

## Related Resources

- [Observability Guide](OBSERVABILITY.md) - Team event monitoring
- [Workflow DSL Guide](workflow-development/dsl.md) - Team nodes in workflows
- [User Guide](../user-guide/index.md) - General usage

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
