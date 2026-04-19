# Credit Assignment for Victor Agent Framework

Based on [arXiv:2604.09459](https://arxiv.org/abs/2604.09459) "From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models".

## What is Credit Assignment?

Credit assignment solves the problem of determining **which actions within a long trajectory caused the final outcome**. This is critical for:

1. **Reasoning RL**: Distributing credit across tokens/steps in CoT generation (500-30K tokens)
2. **Agentic RL**: Attributing success/failure across multi-turn workflows (100K-1M tokens)
3. **Multi-Agent Coordination**: Fairly attributing credit when multiple agents collaborate

The paper surveys 47 methods across two axes: *granularity* (token → episode) and *methodology* (value-based, game-theoretic, information-theoretic). Our implementation covers the most practical methods from each category.

## Quick Start

```python
from victor.framework.rl import (
    CreditAssignmentIntegration,
    CreditMethodology,
    ActionMetadata,
)

# Create trajectory from execution log
trajectory = [
    ActionMetadata(agent_id="agent_1", action_id=f"action_{i}")
    for i in range(100)
]
rewards = [0.1 * (1 if i < 90 else -1) for i in range(100)]

# Assign credit
ca = CreditAssignmentIntegration()
signals = ca.assign_credit(trajectory, rewards, CreditMethodology.GAE)

# Get attribution for a specific agent
attribution = ca.get_agent_attribution("agent_1")
```

## Credit Assignment Methods

### Value-Based Methods

| Method | Enum | Best For | Complexity |
|--------|------|----------|------------|
| Monte Carlo | `MONTE_CARLO` | Short episodes with clear outcomes | O(n) |
| GAE | `GAE` | Most agentic workflows (default) | O(n) |
| Temporal Difference | `TEMPORAL_DIFFERENCE` | Long reasoning chains | O(n) |
| N-step Returns | `N_STEP_RETURNS` | Turn-level with bifurcation detection | O(n) |

### Game-Theoretic Methods

| Method | Enum | Best For | Complexity |
|--------|------|----------|------------|
| Shapley Values | `SHAPLEY` | Fair multi-agent attribution | O(m x n) |
| CARL | `CARL` | Critical action identification | O(n) |

### Information-Theoretic Methods

| Method | Enum | Best For | Complexity |
|--------|------|----------|------------|
| Hindsight | `HINDSIGHT` | Failed trajectories, exploration | O(n) |

Where n = trajectory length, m = number of agents.

**Key mathematical foundations:**
- **GAE**: `A_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}` where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`. Value function estimated via discounted future returns when no learned critic is available.
- **Shapley**: `φ_i = E_π[V(S∪{i}) - V(S)]` averaged over permutations. Satisfies efficiency axiom: `Σφ_i = V(N)`. Uses Monte Carlo permutation sampling with synergy redistribution for non-additive coalitions.
- **Hindsight (HER-style)**: Failed trajectories are relabeled as successes for achieved intermediate goals. Credit = blend of original reward and discounted distance to retrospective goals.

## Granularity Levels

```python
CreditGranularity.TOKEN    # Individual token in reasoning chain
CreditGranularity.SEGMENT  # Group of tokens (e.g., reasoning step)
CreditGranularity.STEP     # Single action within a turn
CreditGranularity.TURN     # Complete agent interaction (request-response)
CreditGranularity.AGENT    # Agent-level across multiple turns
CreditGranularity.EPISODE  # Full trajectory from start to terminal state
```

## StateGraph Integration

```python
from victor.framework.rl import create_credit_aware_workflow

credit_graph = create_credit_aware_workflow(
    graph=StateGraph(MyState),
    reward_key="reward",
    agent_key="agent_id",
)

app = credit_graph.compile(enable_credit=True)
result = await app.invoke(initial_state)
attribution = app.get_credit_attribution()
```

## Team Coordination

```python
from victor.teams.mixins.credit_assignment import CreditAssignmentMixin

class MyTeamCoordinator(CreditAssignmentMixin, UnifiedTeamCoordinator):
    pass

coordinator = MyTeamCoordinator(orchestrator)
coordinator.enable_credit_tracking(CreditMethodology.SHAPLEY)
result = await coordinator.execute_task("Build feature", context)
attribution = coordinator.get_team_credit_attribution()
```

## API Reference

### CreditAssignmentIntegration

Main orchestration class for credit assignment.

| Method | Description |
|--------|-------------|
| `assign_credit(trajectory, rewards, methodology, config)` | Assign credit to trajectory |
| `get_credit(action_id)` | Get credit for specific action |
| `get_agent_attribution(agent_id, granularity)` | Get attribution for agent |
| `identify_critical_actions(trajectory, rewards)` | Find bifurcation points |
| `get_trajectory_summary()` | Get execution statistics |
| `reset()` | Clear all state |

### CreditTracer

Tracks graph execution for credit assignment.

| Method | Description |
|--------|-------------|
| `start_trace(initial_state, trace_id)` | Start new execution trace |
| `record_transition(from_node, to_node, state_before, state_after, output)` | Record transition |
| `end_trace(final_state, success, error)` | Complete trace |
| `assign_credit_to_trace(trace, methodology, config)` | Assign credit |
| `get_agent_attribution(trace, agent_id)` | Get agent attribution |

### Configuration

```python
CreditAssignmentConfig(
    methodology=CreditMethodology.GAE,
    granularity=CreditGranularity.STEP,
    gamma=0.99,              # Discount factor
    lambda_gae=0.95,         # GAE parameter
    n_step=5,                # N-step returns
    hindsight_ratio=0.8,     # Hindsight relabeling ratio
    shapley_sampling_count=10,  # Shapley Monte Carlo samples
    enable_bifurcation_detection=True,
    bifurcation_threshold=0.3,
)
```

### Visualization & Export

```python
from victor.framework.rl import export_credit_report, create_interactive_report

# Export formats: json, csv, md, html
export_credit_report(signals, metrics, attribution, "report.html", format="html")

# Interactive HTML report with charts
create_interactive_report(signals, metrics, attribution, "report.html")
```

### Persistence

```python
from victor.framework.rl.credit_persistence import get_persistent_db

db = get_persistent_db()
db.save_session(session_id, methodology, granularity, signals, success, duration)
history = db.get_agent_history("agent_1", limit=100)
stats = db.get_methodology_stats()
```

### CLI Commands

```bash
victor credit analyze trajectory.json --methodology gae -o results.json
victor credit compare trajectory.json -m gae shapley hindsight
victor credit visualize results.json -o report.html
victor credit export results.json -f csv -o attribution.csv
victor credit template -o template.json
```

## Testing

```bash
# Unit tests
pytest tests/unit/framework/rl/test_credit_assignment.py -v

# Integration tests
pytest tests/integration/rl/test_credit_assignment_integration.py -v

# All RL tests
pytest tests/unit/framework/rl/ tests/integration/rl/ -v
```

## Files

| File | Purpose |
|------|---------|
| `victor/framework/rl/credit_assignment.py` | Core: 9 assigners (GAE, TD, MC, N-step, Shapley, Hindsight, C3, CARL, LLM-as-Critic) |
| `victor/framework/rl/credit_tracking_service.py` | Runtime bridge: ToolPipeline → credit assignment → ObservabilityBus |
| `victor/framework/rl/tool_reputation.py` | Online mid-turn EMA reputation tracking |
| `victor/framework/rl/credit_graph_integration.py` | StateGraph wrapper and tracing |
| `victor/framework/rl/credit_visualization.py` | HTML visualization and multi-format export |
| `victor/framework/rl/credit_persistence.py` | SQLite storage and analytics |
| `victor/teams/mixins/credit_assignment.py` | Team coordination mixin |
| `victor/teams/mixins/credit_aware_routing.py` | Shapley-driven agent rerouting |
| `victor/config/credit_assignment_settings.py` | Settings (enabled, methodology, gamma, etc.) |
| `victor/commands/credit_commands.py` | CLI commands |

## Feedback Loops

### Between-Turn (strategic, slower)

After all tool calls in a turn complete, `CreditTrackingService.assign_turn_credit()` runs credit assignment and generates tool effectiveness guidance that's injected into the next turn's prompt.

```python
# Automatic — wired in ToolCoordinator after execute_tool_calls()
# Credit signals → ObservabilityBus → GEPA trace enrichment
# generate_tool_guidance() → UnifiedPromptPipeline.compose_turn_prefix()
```

### Within-Turn (tactical, faster)

`ToolReputationTracker` updates an EMA reputation score after every tool execution. Its `get_selection_guidance()` is injected mid-stream, biasing the agent toward reliable tools.

```python
from victor.framework.rl.tool_reputation import ToolReputationTracker

tracker = ToolReputationTracker(alpha=0.3)
tracker.record("read_file", success=True, duration_ms=50)
tracker.record("shell", success=False, duration_ms=5000)

guidance = tracker.get_selection_guidance()
# → "Mid-turn tool reputation:
#    - read_file: reliable (1 calls, score +0.3)
#    - shell: unreliable (1/last 5 failed, score -0.3). Verify arguments."
```

## Credit-Driven Team Rerouting

```python
from victor.teams.mixins.credit_aware_routing import CreditAwareTeamCoordinator
from victor.teams import UnifiedTeamCoordinator

class MyTeam(CreditAwareTeamCoordinator, UnifiedTeamCoordinator):
    pass

team = MyTeam(orchestrator, reroute_threshold=0.5, min_rounds_before_reroute=3)
# After multiple rounds, underperforming agents' tasks are
# automatically rerouted to the highest-performing agent.

perf = team.get_agent_performance()
# → {"coder": {"avg_credit": 0.8, "is_rerouted": False},
#    "tester": {"avg_credit": -0.3, "is_rerouted": True, "rerouted_to": "coder"}}
```

## Extending

To add a new credit assignment method:

1. Inherit from `BaseCreditAssigner`
2. Implement `assign_credit()` method
3. Add to `CreditAssignmentIntegration._assigners` registry
4. Remove from `CreditAssignmentIntegration._UNIMPLEMENTED` if present
5. Add unit tests in `test_credit_assignment.py`
6. Add integration test in `test_credit_assignment_integration.py`
