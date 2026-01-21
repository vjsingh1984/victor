# Agentic AI Integration Guide (Phase 3)

This guide explains how to use Victor's Phase 3 Agentic AI features, including hierarchical planning, memory systems, skill discovery, and self-improvement.

## Overview

Victor's Phase 3 Agentic AI features enable advanced autonomous behavior:

1. **Hierarchical Planning** - Break down complex tasks into subtasks
2. **Episodic Memory** - Store and retrieve agent experiences
3. **Semantic Memory** - Store and query factual knowledge
4. **Skill Discovery** - Dynamically discover and compose tools
5. **Skill Chaining** - Plan and execute multi-step workflows
6. **Self-Improvement** - Track proficiency and optimize performance
7. **RL Coordinator** - Reinforcement learning for decision optimization

## Feature Flags

All agentic AI features are **opt-in** via feature flags in settings:

```python
from victor.config.settings import Settings

settings = Settings()

# Enable individual features
settings.enable_hierarchical_planning = True
settings.enable_episodic_memory = True
settings.enable_semantic_memory = True
settings.enable_skill_discovery = True
settings.enable_skill_chaining = True
settings.enable_self_improvement = True
settings.enable_rl_coordinator = True  # Enabled by default
```

Or via environment variables:

```bash
export VICTOR_ENABLE_HIERARCHICAL_PLANNING=true
export VICTOR_ENABLE_EPISODIC_MEMORY=true
export VICTOR_ENABLE_SEMANTIC_MEMORY=true
export VICTOR_ENABLE_SKILL_DISCOVERY=true
export VICTOR_ENABLE_SKILL_CHAINING=true
export VICTOR_ENABLE_SELF_IMPROVEMENT=true
export VICTOR_ENABLE_RL_COORDINATOR=true
```

## Usage Examples

### 1. Hierarchical Planning

Break down complex tasks into executable subtasks:

```python
from victor.agent import HierarchicalPlanner
from victor.config.settings import Settings

# Enable hierarchical planning
settings = Settings()
settings.enable_hierarchical_planning = True

# Create planner (requires orchestrator)
planner = HierarchicalPlanner(orchestrator=orchestrator)

# Decompose a complex task
graph = await planner.decompose_task(
    "Implement user authentication with JWT tokens"
)

# Get next executable tasks
tasks = await planner.suggest_next_tasks(graph, max_tasks=5)

# Execute tasks and update plan
completed = ["task_1", "task_2"]
updated = await planner.update_plan(graph, completed_tasks=completed)

# Validate plan
validation = planner.validate_plan(graph)
if not validation.is_valid:
    print(f"Plan errors: {validation.errors}")
```

**Configuration:**

```python
settings.hierarchical_planning_max_depth = 5  # Maximum decomposition depth
settings.hierarchical_planning_min_subtasks = 2  # Minimum subtasks per decomposition
settings.hierarchical_planning_max_subtasks = 10  # Maximum subtasks per decomposition
```

### 2. Episodic Memory

Store and retrieve agent experiences:

```python
from victor.agent import EpisodicMemory, Episode, create_episodic_memory

# Enable episodic memory
settings = Settings()
settings.enable_episodic_memory = True

# Create episodic memory
memory = create_episodic_memory(
    max_episodes=1000,
    recall_threshold=0.3,
    consolidation_interval=100,
)

# Store an episode
episode_id = await memory.store_episode(
    inputs={"query": "fix authentication bug"},
    actions=["read_file", "edit_file", "run_tests"],
    outcomes={"success": True, "files_changed": 2, "tests_passed": 15},
    rewards=10.0,
    context={"task_type": "bugfix", "complexity": "medium"},
)

# Recall relevant episodes
relevant = await memory.recall_relevant(
    "authentication error",
    k=5,
    threshold=0.3,
)

# Recall recent episodes
recent = await memory.recall_recent(n=10)

# Recall by outcome pattern
successful = await memory.recall_by_outcome(
    outcome_key="success",
    outcome_value=True,
    n=10,
)

# Get memory statistics
stats = memory.get_memory_statistics()
print(f"Total episodes: {stats['total_episodes']}")
print(f"Average reward: {stats['average_reward']}")
```

**Configuration:**

```python
settings.episodic_memory_max_episodes = 1000
settings.episodic_memory_recall_threshold = 0.3
settings.episodic_memory_consolidation_interval = 100
```

### 3. Semantic Memory

Store and query factual knowledge:

```python
from victor.agent import SemanticMemory

# Enable semantic memory
settings = Settings()
settings.enable_semantic_memory = True

# Create semantic memory
memory = SemanticMemory(
    max_facts=5000,
    query_threshold=0.25,
    link_threshold=0.4,
)

# Store knowledge
fact_id = await memory.store_knowledge(
    "Python uses asyncio for concurrent programming",
    metadata={"category": "programming", "language": "python"},
    confidence=1.0,
)

# Query knowledge
facts = await memory.query_knowledge(
    "concurrency in Python",
    k=5,
    threshold=0.25,
)

for fact in facts:
    print(f"Fact: {fact.fact}")
    print(f"Similarity: {fact.similarity}")

# Link related facts
await memory.link_facts(
    fact_id_1=fact_id,
    fact_id_2=another_fact_id,
    link_type="related",
    strength=0.9,
)

# Get knowledge graph
graph = memory.get_knowledge_graph()
print(f"Total facts: {len(graph.facts)}")
print(f"Total links: {len(graph.links)}")
```

**Configuration:**

```python
settings.semantic_memory_max_facts = 5000
settings.semantic_memory_query_threshold = 0.25
settings.semantic_memory_link_threshold = 0.4
```

### 4. Skill Discovery

Dynamically discover and compose tools:

```python
from victor.agent import SkillDiscoveryEngine
from victor.config.settings import Settings

# Enable skill discovery
settings = Settings()
settings.enable_skill_discovery = True

# Create skill discovery engine
discovery = SkillDiscoveryEngine(
    tool_registry=tool_registry,
    max_tools=20,
    min_compatibility=0.5,
    auto_composition=True,
)

# Discover tools for a context
tools = await discovery.discover_tools(
    context="code analysis and refactoring",
    max_tools=10,
)

for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Compatibility: {tool.compatibility}")

# Compose multiple tools into a skill
skill = await discovery.compose_skill(
    skill_name="code_analyzer",
    tools=tools[:3],
    description="Analyzes code quality and suggests improvements",
)

print(f"Skill: {skill.name}")
print(f"Tools: {[t.name for t in skill.tools]}")

# Analyze tool compatibility
compatibility = await discovery.analyze_compatibility(
    tool_1=tools[0],
    tool_2=tools[1],
)
print(f"Compatibility score: {compatibility}")
```

**Configuration:**

```python
settings.skill_discovery_max_tools = 20
settings.skill_discovery_min_compatibility = 0.5
settings.skill_discovery_auto_composition = True
```

### 5. Skill Chaining

Plan and execute multi-step skill chains:

```python
from victor.agent import SkillChainer, Skill

# Enable skill chaining
settings = Settings()
settings.enable_skill_chaining = True

# Create skill chainer
chainer = SkillChainer(
    max_chain_length=10,
    validation_enabled=True,
    parallel_enabled=True,
)

# Create skills
skill1 = Skill(name="reader", tools=[read_tool], description="Reads files")
skill2 = Skill(name="analyzer", tools=[analyze_tool], description="Analyzes code")
skill3 = Skill(name="fixer", tools=[fix_tool], description="Fixes issues")

skills = [skill1, skill2, skill3]

# Plan a skill chain
chain = await chainer.plan_chain(
    goal="Analyze and fix code quality issues",
    skills=skills,
    max_length=5,
)

print(f"Chain: {chain.goal}")
print(f"Steps: {len(chain.steps)}")

# Execute the chain
result = await chainer.execute_chain(
    chain=chain,
    context={"file_path": "src/main.py"},
)

print(f"Status: {result.status}")
print(f"Outputs: {result.outputs}")

# Validate chain before execution
validation = chainer.validate_chain(chain)
if not validation.is_valid:
    print(f"Validation errors: {validation.errors}")
```

**Configuration:**

```python
settings.skill_chaining_max_chain_length = 10
settings.skill_chaining_validation_enabled = True
settings.skill_chaining_parallel_enabled = True
```

### 6. Proficiency Tracking

Track tool/task performance and get improvement suggestions:

```python
from victor.agent import ProficiencyTracker

# Enable self-improvement
settings = Settings()
settings.enable_self_improvement = True

# Create proficiency tracker
tracker = ProficiencyTracker(
    window_size=100,
    decay_rate=0.95,
    min_samples=5,
)

# Record outcomes
await tracker.record_outcome(
    task="code_review",
    tool="ast_analyzer",
    success=True,
    duration=1.5,
    cost=0.001,
    metadata={"file_count": 10, "complexity": "high"},
)

# Get proficiency score
score = tracker.get_proficiency(
    tool="ast_analyzer",
    task="code_review",
)

print(f"Success rate: {score.success_rate}")
print(f"Average duration: {score.avg_duration}")
print(f"Trend: {score.trend}")  # IMPROVING, DECLINING, STABLE

# Get improvement suggestions
suggestions = tracker.get_suggestions(n=5)

for suggestion in suggestions:
    print(f"Suggestion: {suggestion.description}")
    print(f"Reason: {suggestion.reason}")
    print(f"Priority: {suggestion.priority}")

# Get overall metrics
metrics = tracker.get_metrics()
print(f"Total outcomes: {metrics['total_outcomes']}")
print(f"Overall success rate: {metrics['overall_success_rate']}")
```

**Configuration:**

```python
settings.proficiency_tracker_window_size = 100
settings.proficiency_tracker_decay_rate = 0.95
settings.proficiency_tracker_min_samples = 5
```

### 7. RL Coordinator

Reinforcement learning for decision optimization:

```python
from victor.agent import EnhancedRLCoordinator

# Enable RL coordinator (enabled by default)
settings = Settings()
settings.enable_rl_coordinator = True

# Create RL coordinator
rl = EnhancedRLCoordinator(
    reward_shaping=RewardShapingStrategy.SPARSE,
    policy_update_interval=50,
    exploration_rate=0.3,
)

# Select action using current policy
state = {"task": "code_review", "files": 5, "complexity": "high"}
actions = ["read_file", "analyze_code", "edit_file", "run_tests"]

action = await rl.select_action(
    state=state,
    actions=actions,
    explore=True,  # Enable exploration
)

print(f"Selected action: {action.name}")

# Update policy based on reward
next_state = {"task": "code_review", "files": 5, "files_analyzed": 5}
reward = 1.0  # Positive reward for successful action

await rl.update_policy(
    state=state,
    action=action,
    reward=reward,
    next_state=next_state,
    done=False,
)

# Get current policy
policy = rl.get_policy()
print(f"Policy state: {policy}")
```

**Configuration:**

```python
settings.rl_reward_shaping = "sparse"  # sparse, dense, hybrid
settings.rl_policy_update_interval = 50
settings.rl_exploration_rate = 0.3
```

## Integration with AgentOrchestrator

All agentic AI features are integrated with the DI container and can be accessed via protocols:

```python
from victor.core.container import ServiceContainer
from victor.agent.service_provider import configure_orchestrator_services
from victor.config.settings import Settings
from victor.agent.protocols import (
    HierarchicalPlannerProtocol,
    EpisodicMemoryProtocol,
    SemanticMemoryProtocol,
    SkillDiscoveryProtocol,
    SkillChainerProtocol,
    ProficiencyTrackerProtocol,
    RLCoordinatorProtocol,
)

# Enable features
settings = Settings()
settings.enable_hierarchical_planning = True
settings.enable_episodic_memory = True
settings.enable_semantic_memory = True
settings.enable_skill_discovery = True
settings.enable_skill_chaining = True
settings.enable_self_improvement = True

# Configure services
container = ServiceContainer()
configure_orchestrator_services(container, settings)

# Resolve services via protocols
planner = container.get(HierarchicalPlannerProtocol)
episodic_memory = container.get(EpisodicMemoryProtocol)
semantic_memory = container.get(SemanticMemoryProtocol)
skill_discovery = container.get(SkillDiscoveryProtocol)
skill_chainer = container.get(SkillChainerProtocol)
proficiency_tracker = container.get(ProficiencyTrackerProtocol)
rl_coordinator = container.get(RLCoordinatorProtocol)

# Use services
graph = await planner.decompose_task("Implement feature")
episode_id = await episodic_memory.store_episode(...)
facts = await semantic_memory.query_knowledge(...)
tools = await skill_discovery.discover_tools(...)
chain = await skill_chainer.plan_chain(...)
await proficiency_tracker.record_outcome(...)
action = await rl_coordinator.select_action(...)
```

## End-to-End Workflow Example

Combine multiple agentic AI features in a workflow:

```python
import asyncio
from victor.agent import AgentOrchestrator
from victor.config.settings import Settings

async def autonomous_task_execution():
    # Enable all agentic AI features
    settings = Settings()
    settings.enable_hierarchical_planning = True
    settings.enable_episodic_memory = True
    settings.enable_semantic_memory = True
    settings.enable_skill_discovery = True
    settings.enable_skill_chaining = True
    settings.enable_self_improvement = True

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
    )

    # Get agentic AI services
    planner = orchestrator._hierarchical_planner
    episodic = orchestrator._episodic_memory
    semantic = orchestrator._semantic_memory
    discovery = orchestrator._skill_discovery
    chainer = orchestrator._skill_chainer
    tracker = orchestrator._proficiency_tracker

    # Complex task
    goal = "Implement user authentication with JWT tokens"

    # 1. Hierarchical planning
    graph = await planner.decompose_task(goal)
    tasks = await planner.suggest_next_tasks(graph)

    # 2. Skill discovery
    tools = await discovery.discover_tools("authentication implementation")
    skills = [await discovery.compose_skill(f"skill_{i}", [t], "") for i, t in enumerate(tools)]

    # 3. Skill chaining
    chain = await chainer.plan_chain(goal, skills)

    # 4. Execute chain
    result = await chainer.execute_chain(chain)

    # 5. Store episode in episodic memory
    episode_id = await episodic.store_episode(
        inputs={"goal": goal},
        actions=[s.name for s in skills],
        outcomes={"success": result.status == "completed"},
        rewards=10.0 if result.status == "completed" else -5.0,
    )

    # 6. Store knowledge in semantic memory
    await semantic.store_knowledge(
        f"JWT authentication requires: {', '.join([s.name for s in skills])}",
        confidence=0.9,
    )

    # 7. Record outcomes for proficiency tracking
    for skill in skills:
        await tracker.record_outcome(
            task="authentication",
            tool=skill.name,
            success=True,
            duration=1.0,
            cost=0.001,
        )

    # 8. Get improvement suggestions
    suggestions = tracker.get_suggestions()
    print(f"Improvement suggestions: {suggestions}")

    return result

# Run the workflow
result = asyncio.run(autonomous_task_execution())
```

## Performance Considerations

### Memory Usage

- **Episodic Memory**: ~1KB per episode (1000 episodes ≈ 1MB)
- **Semantic Memory**: ~0.5KB per fact (5000 facts ≈ 2.5MB)
- **Proficiency Tracker**: ~100 bytes per outcome

### CPU Overhead

- **Hierarchical Planning**: 500-2000ms per decomposition (LLM-based)
- **Memory Search**: 50-200ms per query (vector similarity)
- **Skill Discovery**: 100-500ms per context (tool matching)
- **Skill Chaining**: 50-200ms per chain (validation)

### Recommendations

1. **Enable features selectively** based on use case
2. **Configure cache sizes** to limit memory usage
3. **Use lazy loading** for expensive operations
4. **Monitor performance** with built-in statistics

## Troubleshooting

### Features Not Working

1. Check feature flags are enabled in settings
2. Verify services are registered in DI container
3. Check logs for import/initialization errors

### Memory Issues

1. Reduce max_episodes/max_facts in settings
2. Lower consolidation_interval
3. Increase recall_threshold to reduce matches

### Performance Issues

1. Disable unused features
2. Reduce hierarchical_planning_max_depth
3. Lower skill_discovery_max_tools
4. Enable parallel execution for skill chains

## Best Practices

1. **Start Simple**: Enable features one at a time
2. **Monitor Statistics**: Use get_memory_statistics() and get_metrics()
3. **Tune Thresholds**: Adjust similarity thresholds based on results
4. **Validate Plans**: Always validate plans before execution
5. **Handle Errors**: Use try/except for agentic operations
6. **Test Locally**: Test with small datasets before scaling

## Migration Guide

### From Manual Planning

**Before:**
```python
# Manual task breakdown
tasks = ["design", "implement", "test"]
```

**After:**
```python
# Automatic hierarchical planning
graph = await planner.decompose_task("Implement feature")
tasks = await planner.suggest_next_tasks(graph)
```

### From Manual Tool Selection

**Before:**
```python
# Manual tool selection
tools = [read_tool, analyze_tool]
```

**After:**
```python
# Automatic skill discovery
tools = await discovery.discover_tools("code analysis")
```

### From Static Execution

**Before:**
```python
# Manual execution
result1 = tool1.execute()
result2 = tool2.execute(result1)
```

**After:**
```python
# Automatic skill chaining
chain = await chainer.plan_chain(goal, skills)
result = await chainer.execute_chain(chain)
```

## Future Enhancements

Planned features for future releases:

1. **Multi-agent coordination** - Multiple agents working together
2. **Advanced RL algorithms** - PPO, A3C, DQN
3. **Transfer learning** - Learn from past projects
4. **Explainable AI** - Explain decisions and plans
5. **Meta-learning** - Learn how to learn

## References

- [Hierarchical Planning Documentation](../agent/planning/README.md)
- [Memory Systems Documentation](../agent/memory/README.md)
- [Skill Discovery Documentation](../agent/skills/README.md)
- [Self-Improvement Documentation](../agent/improvement/README.md)
- [Protocol Reference](../agent/protocols/README.md)
- [Settings Reference](../config/settings.md)
