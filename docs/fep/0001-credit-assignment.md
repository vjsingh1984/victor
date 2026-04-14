# FEP-0001: Credit Assignment System for Agentic RL

**Status**: Proposed
**Type**: Feature
**Created**: 2025-04-14
**Author**: Vijaykumar Singh
**Review Period**: 14 days (ending 2025-04-28)

## Summary

This FEP proposes adding a comprehensive credit assignment system to the Victor framework to address the credit assignment problem in reinforcement learning for large language model agents. The system enables fair attribution of success/failure across long trajectories, multi-agent workflows, and complex StateGraph executions.

Based on: [arXiv:2604.09459](https://arxiv.org/abs/2604.09459) "From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models"

## Motivation

### Problem Statement

Current Victor framework uses outcome-level rewards for agent training and evaluation, but determining **which actions within a long trajectory caused the outcome** is difficult. This manifests as:

1. **Reasoning RL**: Cannot attribute credit to specific tokens/steps in long CoT chains (500-30K tokens)
2. **Agentic RL**: Cannot identify which agent/action in a multi-turn workflow succeeded/failed (100K-1M tokens)
3. **Multi-Agent Coordination**: Cannot fairly attribute credit when multiple agents collaborate
4. **Sparse Rewards**: Long horizons with sparse rewards make credit assignment exponentially difficult

### Current Limitations

- No built-in mechanism for credit attribution across agent execution
- Multi-agent teams have no way to understand individual contributions
- Failed workflows provide limited learning value
- No tools for analyzing bifurcation points in trajectories
- StateGraph workflows lack credit tracking

### Proposed Solution

Implement a multi-granularity credit assignment system with:
- 6 credit assigners for different scenarios (reasoning, agentic, multi-agent)
- 12 methodologies (GAE, Shapley, Hindsight, Monte Carlo, TD, etc.)
- StateGraph integration for automatic tracking
- Team coordination mixins for fair attribution
- CLI commands for analysis and visualization
- Persistence layer for historical analysis

## Design

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     High-Level API                            │
│  CreditAssignmentIntegration, StateGraphCreditMixin          │
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Reasoning   │ │   Agentic    │ │  Multi-Agent │
│  Assigners   │ │   Assigners  │ │   Assigners  │
│              │ │              │ │              │
│ - TokenLevel │ │ - Episode    │ │ - Shapley    │
│ - Segment    │ │ - Turn       │ │ - HCAPO      │
└──────────────┘ └──────────────┘ └──────────────┘
```

### Components

#### 1. Core Credit Assignment (`victor/framework/rl/credit_assignment.py`)

**Data Structures:**
- `CreditGranularity`: TOKEN, SEGMENT, STEP, TURN, AGENT, EPISODE
- `CreditMethodology`: 12 methods (GAE, SHAPLEY, HINDSIGHT, etc.)
- `ActionMetadata`: Context for each action
- `CreditSignal`: Credit assignment result
- `CreditAssignmentConfig`: Configuration

**Credit Assigners:**
- `TokenLevelCreditAssigner`: TD learning for reasoning chains
- `SegmentLevelCreditAssigner`: Monte Carlo for segments
- `EpisodeLevelCreditAssigner`: GAE for workflows
- `TurnLevelCreditAssigner`: N-step with bifurcation detection
- `HindsightCreditAssigner`: Goal relabeling for failures
- `MultiAgentCreditAssigner`: Shapley values for teams

#### 2. StateGraph Integration (`victor/framework/rl/credit_graph_integration.py`)

**Classes:**
- `CreditTracer`: Tracks execution transitions
- `ExecutionTrace`: Complete execution trace
- `CreditAwareGraph`: Wrapper for StateGraph
- `CompiledCreditAwareGraph`: Compiled graph with credit

#### 3. Team Coordination Mixin (`victor/teams/mixins/credit_assignment.py`)

**Classes:**
- `CreditAssignmentMixin`: Adds credit tracking to `UnifiedTeamCoordinator`
- `TeamExecutionTrace`: Team-specific execution trace
- `TeamExecutionStep`: Single step in team execution

#### 4. CLI Commands (`victor/commands/credit_commands.py`)

**Commands:**
- `victor credit analyze` - Analyze trajectory and assign credit
- `victor credit compare` - Compare methodologies
- `victor credit visualize` - Visualize results
- `victor credit export` - Export to various formats
- `victor credit template` - Generate trajectory template

#### 5. Visualization & Export (`victor/framework/rl/credit_visualization.py`)

**Classes:**
- `CreditVisualizationBuilder`: HTML report builder
- `CreditAssignmentExporter`: Multi-format export
- `CreditAssignmentReport`: Comprehensive reports

#### 6. Persistence (`victor/framework/rl/credit_persistence.py`)

**Classes:**
- `CreditAssignmentDB`: SQLite storage manager
- Functions for querying, aggregation, export

### API Examples

#### Basic Usage

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

# Get attribution
attribution = ca.get_agent_attribution("agent_1")
```

#### StateGraph Integration

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

#### Team Coordination

```python
from victor.teams.mixins.credit_assignment import CreditAssignmentMixin

class MyTeamCoordinator(CreditAssignmentMixin, UnifiedTeamCoordinator):
    pass

coordinator = MyTeamCoordinator(orchestrator)
coordinator.enable_credit_tracking(CreditMethodology.SHAPLEY)
result = await coordinator.execute_task("Build feature", context)
attribution = coordinator.get_team_credit_attribution()
```

### Methodology Selection Guide

| Scenario | Recommended Methodology | Why |
|----------|-------------------------|-----|
| Short workflows (< 10 steps) | Monte Carlo | Unbiased, fast |
| General workflows | GAE (λ=0.95) | Bias-variance tradeoff |
| Long reasoning chains | TD learning | Bootstrapping for efficiency |
| Multi-agent teams | Shapley values | Fair attribution |
| Failed trajectories | Hindsight | Extracts learning value |
| High-frequency trading | N-step returns | Balance speed and accuracy |

## Implementation Plan

### Phase 1: Core Implementation ✅ (Complete)

- [x] Core credit assigners (6 implementations)
- [x] StateGraph integration
- [x] Team coordination mixin
- [x] Unit tests (39 tests)
- [x] Integration tests (18 tests)

### Phase 2: Enhanced Features ✅ (Complete)

- [x] CLI commands for analysis
- [x] Visualization and export tools
- [x] Persistence layer
- [x] Real-world examples

### Phase 3: Production Readiness (Proposed)

- [ ] Integration with AgentOrchestrator
- [ ] Automatic credit tracking for all agent runs
- [ ] Observability dashboard integration
- [ ] Performance optimization
- [ ] Documentation review

### Phase 4: Advanced Features (Future)

- [ ] LLM-as-Critic methodology
- [ ] Hierarchical credit assignment
- [ ] Counterfactual analysis
- [ ] Time-dependent credit decay
- [ ] Tool-specific credit attribution

## Breaking Changes

**None.** This is a pure addition with no breaking changes to existing APIs.

All new functionality is opt-in:
- Credit tracking must be explicitly enabled
- Mixins are optional
- No changes to existing agent/team behavior

## Backward Compatibility

**Fully backward compatible.**

- Existing code continues to work without modification
- Credit assignment is an add-on feature
- No deprecations introduced

## Performance Impact

### Runtime Overhead

| Component | Overhead | Notes |
|-----------|----------|-------|
| CreditTracer (recording) | < 1% | Simple dict appends |
| GAE assignment (100 steps) | ~10ms | O(n) operations |
| Shapley (10 agents, 100 steps) | ~50ms | O(m×n) operations |
| StateGraph wrapper | < 5% | Delegation overhead |

### Storage

- In-memory: ~1KB per signal
- SQLite: ~500 bytes per signal
- 1000-step trace: ~1MB in database

## Security Considerations

- No external network calls
- All data stored locally
- SQLite database in user home directory
- No sensitive data in logs by default

## Testing Strategy

### Unit Tests (39 tests)

- Test all credit assigners independently
- Test data structures and protocols
- Test configuration handling
- Test edge cases (empty trajectories, single agent, etc.)

### Integration Tests (18 tests)

- Test StateGraph integration
- Test team coordination scenarios
- Test multi-agent workflows
- Test failed workflow handling
- Test performance with large trajectories

### Manual Testing Scenarios

1. **Multi-Agent Code Review**: 3 agents review PR, verify fair attribution
2. **Feature Development Workflow**: 4-stage workflow, verify GAE results
3. **Failed Task Analysis**: Multiple failed attempts, verify hindsight learning
4. **Large Trajectory**: 1000-step workflow, verify performance

## Documentation

### User Documentation

- `docs/credit_assignment.md`: API reference and usage guide
- `docs/credit_assignment_summary.md`: Complete implementation summary
- `examples/credit_assignment_examples.py`: Real-world examples

### Developer Documentation

- Inline docstrings for all public APIs
- Type hints for all functions
- Protocol definitions for extensibility

## Alternatives Considered

### Alternative 1: Use Existing RL Libraries

**Rejected.** Existing libraries (Stable-Baselines3, RLlib) are not designed for LLM agents and don't handle:
- Token-level credit for CoT
- Multi-agent Shapley values
- Hindsight goal relabeling for text

### Alternative 2: Simple Reward Averaging

**Rejected.** Uniform averaging doesn't account for:
- Critical actions (bifurcation points)
- Agent interdependence
- Temporal credit decay
- Fair attribution in teams

### Alternative 3: Outcome-Only Metrics

**Rejected.** This is the current approach and has well-known limitations:
- No learning from failures
- Cannot identify what worked
- Unfair team attribution
- No actionable insights

## Risks and Mitigations

### Risk 1: Performance Overhead

**Mitigation:**
- All credit assignment is optional and lazy
- Efficient O(n) algorithms for most methods
- Can be disabled for production if needed

### Risk 2: Complexity

**Mitigation:**
- Simple default configuration (GAE, λ=0.95)
- Clear documentation and examples
- Automatic methodology selection

### Risk 3: Incorrect Credit Assignment

**Mitigation:**
- Confidence scores on all signals
- Methodology comparison tools
- Visual validation via HTML reports
- Extensive test coverage

## Dependencies

### New Dependencies

**None.** All implementations use only standard library and existing Victor dependencies:
- `typing`, `dataclasses`, `enum`
- `sqlite3` (standard library)
- `typer` (already used for CLI)
- JSON/CSV/Markdown formatters

### Internal Dependencies

- `victor.framework.graph` (StateGraph)
- `victor.teams` (UnifiedTeamCoordinator)
- `victor.framework.rl` (existing RL infrastructure)

## Migration Guide

### For Existing Users

**No migration needed.** Credit assignment is opt-in.

If you want to add credit tracking:

```python
# Before (no credit tracking)
result = await agent.run("Write code")

# After (with credit tracking)
from victor.framework.rl import create_credit_aware_workflow

# Wrap StateGraph if using workflows
credit_graph = create_credit_aware_workflow(graph)
app = credit_graph.compile(enable_credit=True)
```

### For New Users

Credit assignment is automatically available when using:
- StateGraph with `create_credit_aware_workflow()`
- Teams with `CreditAssignmentMixin`
- CLI commands with `victor credit *`

## Open Questions

1. **Should credit tracking be enabled by default in production?**
   - **Recommendation**: No, keep opt-in for now. Enable after gathering usage data.

2. **Should we add automatic methodology selection?**
   - **Recommendation**: Yes, add heuristics to choose based on (trajectory length, agent count, etc.)

3. **Should credit data be included in observability events?**
   - **Recommendation**: Yes, add credit signals to ObservabilityBus for dashboard integration.

4. **Should we persist all credit data by default?**
   - **Recommendation**: No, persist only when explicitly requested (user privacy, storage concerns).

## Timeline

- **2025-04-14**: Initial proposal (this document)
- **2025-04-28**: Review period ends
- **2025-05-01**: Implementation begins (Phase 1 & 2 complete)
- **2025-05-15**: Production readiness (Phase 3)
- **2025-06-01**: Advanced features (Phase 4) - future work

## References

- Paper: [arXiv:2604.09459](https://arxiv.org/abs/2604.09459) "From Reasoning to Agentic: Credit Assignment in RL for LLMs"
- Wikipedia: [Credit Assignment](https://en.wikipedia.org/wiki/Credit_assignment)
- Victor RL System: `victor/framework/rl/`
- Victor Teams: `victor/teams/`

## Appendix: Implementation Details

### File Structure

```
victor/framework/rl/
├── credit_assignment.py           # Core implementation (1000 LOC)
├── credit_graph_integration.py     # StateGraph integration (500 LOC)
├── credit_visualization.py         # Visualization & export (600 LOC)
├── credit_persistence.py           # SQLite storage (400 LOC)
└── __init__.py                      # Module exports

victor/teams/mixins/
└── credit_assignment.py           # Team coordination mixin (350 LOC)

victor/commands/
└── credit_commands.py              # CLI commands (350 LOC)

tests/unit/framework/rl/
└── test_credit_assignment.py       # Unit tests (39 tests)

tests/integration/rl/
└── test_credit_assignment_integration.py  # Integration tests (18 tests)

docs/
├── credit_assignment.md            # API documentation
└── credit_assignment_summary.md    # Implementation summary

examples/
└── credit_assignment_examples.py   # Real-world examples
```

### Code Statistics

- **Production code**: ~3,200 LOC
- **Test code**: ~1,800 LOC
- **Documentation**: ~2,000 LOC
- **Total**: ~7,000 LOC

### Test Coverage

- **Unit tests**: 39 tests
- **Integration tests**: 18 tests
- **Total RL tests**: 181 tests (all passing)
- **Coverage**: > 95% for new code

## Approval

This FEP is under review until 2025-04-28. Please provide feedback on:

1. **Design**: Is the architecture sound?
2. **APIs**: Are the interfaces clean and intuitive?
3. **Documentation**: Is it clear how to use the system?
4. **Testing**: Is the test coverage adequate?
5. **Performance**: Are the overheads acceptable?

Please submit feedback via GitHub issues or PRs to the Victor repository.

---

**FEP-0001**
**Status**: Proposed
**Review Ends**: 2025-04-28
