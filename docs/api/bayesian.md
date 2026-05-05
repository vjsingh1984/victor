# Bayesian Orchestration API Reference

This section provides detailed API documentation for Victor's Bayes-consistent orchestration system.

## Core Components

### BayesianTaskAnalysis

Extends `TaskAnalysis` with Bayesian belief states for decision-making under uncertainty.

**Location**: `victor.agent.bayesian_task_analysis`

**Class**: `BayesianTaskAnalysis`

```python
from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.agent.task_analyzer import TaskComplexity, UnifiedTaskType

analysis = BayesianTaskAnalysis(
    complexity=TaskComplexity.SIMPLE,
    tool_budget=10,
    complexity_confidence=0.8,
    unified_task_type=UnifiedTaskType.EDIT,
    outcome_belief={"success": 0.5, "failure": 0.5},
)
```

**Attributes**:
- `outcome_belief: Dict[str, float]` — Posterior distribution P(Y|D) over task outcomes
- `belief_entropy: float` — Shannon entropy H[Y|D] quantifying uncertainty
- `belief_variance: float` — Variance of the belief distribution
- `belief_id: str` — Unique identifier for orchestration service tracking
- `agent_reliability: Dict[str, float]` — Reliability weights α_i for agents

**Methods**:

##### `compute_posterior(prior, likelihood, agent_id=None)`

Compute posterior distribution using Bayes' rule with reliability weighting.

**Parameters**:
- `prior: Dict[str, float]` — Prior distribution P(Y|D)
- `likelihood: Dict[str, float]` — Likelihood P(z|Y) from observation model
- `agent_id: Optional[str]` — Agent ID for reliability weighting

**Returns**: `Dict[str, float]` — Posterior distribution P(Y|D,z)

**Example**:
```python
prior = {"success": 0.5, "failure": 0.5}
likelihood = {"success": 0.8, "failure": 0.2}
posterior = analysis.compute_posterior(prior, likelihood, agent_id="agent_a")
# Result: {"success": 0.8, "failure": 0.2} (shifted toward success)
```

##### `compute_entropy(distribution)`

Compute Shannon entropy for uncertainty quantification.

**Parameters**:
- `distribution: Dict[str, float]` — Probability distribution

**Returns**: `float` — Entropy in nats

**Example**:
```python
entropy = analysis.compute_entropy({"success": 0.5, "failure": 0.5})
# Result: 0.693 nats (maximum uncertainty)
```

##### `compute_voi(agent_id, query_cost)`

Compute Value of Information for querying an agent.

**Parameters**:
- `agent_id: str` — Agent to query
- `query_cost: float` — Cost of querying the agent

**Returns**: `float` — Expected information gain minus cost

**Example**:
```python
voi = analysis.compute_voi("agent_a", 0.1)
if voi > 0:
    print("Query this agent")
```

##### `get_most_likely_outcome()`

Get the most probable task outcome.

**Returns**: `Tuple[str, float]` — (outcome, probability)

**Example**:
```python
outcome, prob = analysis.get_most_likely_outcome()
print(f"Most likely: {outcome} with {prob:.2%} confidence")
```

---

### ObservationModelLearner

Learns P(agent_message | task_outcome) using Bayesian inference with Beta distributions.

**Location**: `victor.framework.rl.learners.observation_model`

**Class**: `ObservationModelLearner`

```python
from victor.framework.rl.learners.observation_model import ObservationModelLearner
import sqlite3

conn = sqlite3.connect("victor.db")
learner = ObservationModelLearner(
    name="observation_model",
    db_connection=conn,
)
```

**Methods**:

##### `record_observation(agent_id, message, actual_outcome, confidence)`

Record an observation for learning.

**Parameters**:
- `agent_id: str` — Agent identifier
- `message: str` — Agent's message
- `actual_outcome: str` — Actual task outcome
- `confidence: float` — Agent's confidence (0-1)

**Example**:
```python
learner.record_observation(
    agent_id="agent_a",
    message="This will work",
    actual_outcome="success",
    confidence=0.8,
)
```

##### `get_likelihood(agent_id, message, outcome)`

Get likelihood P(message | outcome) for Bayesian updates.

**Parameters**:
- `agent_id: str` — Agent identifier
- `message: str` — Agent's message
- `outcome: str` — Task outcome

**Returns**: `float` — Likelihood P(message | outcome)

**Example**:
```python
likelihood = learner.get_likelihood("agent_a", "This works", "success")
# Use in Bayesian update
posterior = analysis.compute_posterior(prior, likelihood)
```

##### `categorize_message(message)`

Categorize agent message into semantic categories.

**Parameters**:
- `message: str` — Agent's message

**Returns**: `str` — Category: "affirm", "deny", "uncertain", or "error"

**Example**:
```python
category = learner.categorize_message("Yes, this works")
# Result: "affirm"
```

##### `get_calibration_stats(agent_id=None)`

Get calibration statistics for assessing prediction quality.

**Parameters**:
- `agent_id: Optional[str]` — Specific agent or all agents

**Returns**: `Dict[str, Dict]` — Calibration metrics per agent

**Example**:
```python
stats = learner.get_calibration_stats("agent_a")
print(f"Mean calibration error: {stats['agent_a']['mean_error']:.3f}")
```

##### `get_recommendation(context)`

Get recommendation for most informative agent to query.

**Parameters**:
- `context: Dict[str, Any]` — Context with task_type, complexity, etc.

**Returns**: `Dict[str, Any]` — Recommendation with agent_id and expected_voi

**Example**:
```python
recommendation = learner.get_recommendation({
    "task_type": "code_edit",
    "complexity": "SIMPLE",
})
print(f"Query {recommendation['agent_id']} (VoI: {recommendation['expected_voi']:.3f})")
```

---

### AgentReliabilityLearner

Learns reliability weights α_i for each agent using Bayesian inference.

**Location**: `victor.framework.rl.learners.agent_reliability`

**Class**: `AgentReliabilityLearner`

```python
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner

learner = AgentReliabilityLearner(
    name="agent_reliability",
    db_connection=conn,
)
```

**Methods**:

##### `record_prediction_result(agent_id, was_correct, calibration_error)`

Record prediction result for reliability learning.

**Parameters**:
- `agent_id: str` — Agent identifier
- `was_correct: bool` — Whether prediction was correct
- `calibration_error: float` — Calibration error |expected - actual|

**Example**:
```python
learner.record_prediction_result(
    agent_id="agent_a",
    was_correct=True,
    calibration_error=0.1,
)
```

##### `get_reliability_weight(agent_id)`

Get reliability weight α_i for Bayesian evidence weighting.

**Parameters**:
- `agent_id: str` — Agent identifier

**Returns**: `float` — Reliability weight α_i

**Interpretation**:
- α_i > 1.0: Upweight agent's evidence (more reliable)
- α_i = 1.0: Use evidence as-is (neutral)
- α_i < 1.0: Downweight agent's evidence (less reliable)

**Example**:
```python
weight = learner.get_reliability_weight("agent_a")
if weight > 1.0:
    print("Agent is reliable")
```

##### `get_agent_reliability_stats(agent_id)`

Get detailed reliability statistics for an agent.

**Parameters**:
- `agent_id: str` — Agent identifier

**Returns**: `Dict[str, Any]` — Reliability metrics

**Example**:
```python
stats = learner.get_agent_reliability_stats("agent_a")
print(f"Sample count: {stats['sample_count']}")
print(f"Reliability: {stats['alpha_reliability']:.3f}")
print(f"Accuracy: {stats['accuracy']:.2%}")
```

##### `get_all_reliability_weights()`

Get reliability weights for all agents.

**Returns**: `Dict[str, float]` — Agent ID to reliability weight mapping

**Example**:
```python
weights = learner.get_all_reliability_weights()
for agent_id, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{agent_id}: {weight:.3f}")
```

---

### VoIController

Computes Expected Value of Information for intelligent query decisions.

**Location**: `victor.framework.rl.learners.voi_controller`

**Class**: `VoIController`

```python
from victor.framework.rl.learners.voi_controller import VoIController

controller = VoIController(
    name="voi_controller",
    db_connection=conn,
    observation_learner=observation_learner,
    reliability_learner=reliability_learner,
)
```

**Methods**:

##### `compute_voi(task_analysis, agent_id, query_cost)`

Compute Expected Value of Information.

**Parameters**:
- `task_analysis: BayesianTaskAnalysis` — Task belief state
- `agent_id: str` — Agent to query
- `query_cost: float` — Cost of querying

**Returns**: `float` — VoI = E[H[Y|D] - H[Y|D,z]] - cost

**Example**:
```python
voi = controller.compute_voi(belief, "agent_a", 0.1)
if voi > 0:
    print("Query agent_a")
```

##### `should_query(task_analysis, agent_id, query_cost)`

Decision rule: query if VoI > 0.

**Parameters**:
- `task_analysis: BayesianTaskAnalysis` — Task belief state
- `agent_id: str` — Agent to query
- `query_cost: float` — Cost of querying

**Returns**: `bool` — True if VoI > 0

**Example**:
```python
if controller.should_query(belief, "agent_a", 0.1):
    message = query_agent("agent_a")
```

##### `rank_agents_by_voi(task_analysis, agent_ids, query_cost)`

Rank agents by expected information gain.

**Parameters**:
- `task_analysis: BayesianTaskAnalysis` — Task belief state
- `agent_ids: List[str]` — Agents to rank
- `query_cost: float` — Cost per query

**Returns**: `List[Dict[str, Any]]` — Ranked agents with VoI scores

**Example**:
```python
ranked = controller.rank_agents_by_voi(belief, ["agent_a", "agent_b", "agent_c"], 0.1)
for agent_rank in ranked:
    print(f"{agent_rank['agent_id']}: VoI={agent_rank['voi']:.3f}")
```

##### `record_voi_outcome(agent_id, predicted_voi, actual_gain, query_cost, metadata=None)`

Record VoI prediction outcome for learning.

**Parameters**:
- `agent_id: str` — Agent queried
- `predicted_voi: float` — Predicted VoI
- `actual_gain: float` — Actual information gain
- `query_cost: float` — Query cost
- `metadata: Optional[Dict]` — Additional metadata

**Example**:
```python
controller.record_voi_outcome(
    agent_id="agent_a",
    predicted_voi=0.3,
    actual_gain=0.25,
    query_cost=0.1,
)
```

##### `get_voi_statistics(agent_id=None)`

Get VoI statistics for evaluation.

**Parameters**:
- `agent_id: Optional[str]` — Specific agent or all agents

**Returns**: `Dict[str, Any]` — VoI metrics

**Example**:
```python
stats = controller.get_voi_statistics("agent_a")
print(f"Mean VoI: {stats['mean_voi']:.3f}")
print(f"Beneficial query rate: {stats['beneficial_rate']:.2%}")
```

---

### BayesianOrchestrationService

Integration layer coordinating all Bayesian components.

**Location**: `victor.framework.rl.orchestration.bayesian_orchestrator`

**Class**: `BayesianOrchestrationService`

```python
from victor.framework.rl.orchestration.bayesian_orchestrator import BayesianOrchestrationService

service = BayesianOrchestrationService(
    db_connection=conn,
    observation_learner=observation_learner,
    reliability_learner=reliability_learner,
    voi_controller=voi_controller,
)
```

**Methods**:

##### `create_belief_state(task_type, complexity, tool_budget, initial_belief)`

Create belief state for a task.

**Parameters**:
- `task_type: str` — Type of task
- `complexity: TaskComplexity` — Task complexity
- `tool_budget: int` — Tool budget
- `initial_belief: Dict[str, float]` — Prior distribution

**Returns**: `BayesianTaskAnalysis` — Belief state

**Example**:
```python
belief = service.create_belief_state(
    task_type="code_edit",
    complexity=TaskComplexity.SIMPLE,
    tool_budget=10,
    initial_belief={"success": 0.5, "failure": 0.5},
)
```

##### `update_belief_with_message(belief_id, agent_id, message, confidence)`

Update belief state using Bayesian posterior update.

**Parameters**:
- `belief_id: str` — Belief state identifier
- `agent_id: str` — Agent ID
- `message: str` — Agent's message
- `confidence: float` — Agent's confidence

**Returns**: `BayesianTaskAnalysis` — Updated belief state

**Example**:
```python
updated = service.update_belief_with_message(
    belief_id=belief.belief_id,
    agent_id="agent_a",
    message="Yes, this works",
    confidence=0.8,
)
```

##### `should_query_agent(belief_id, agent_id, query_cost)`

Decide whether to query an agent using VoI.

**Parameters**:
- `belief_id: str` — Belief state identifier
- `agent_id: str` — Agent to query
- `query_cost: float` — Query cost

**Returns**: `bool` — True if VoI > 0

**Example**:
```python
if service.should_query_agent(belief.belief_id, "agent_a", 0.1):
    # Query agent
    pass
```

##### `select_best_agent_to_query(belief_id, agent_ids, query_cost)`

Select best agent to query by VoI.

**Parameters**:
- `belief_id: str` — Belief state identifier
- `agent_ids: List[str]` — Available agents
- `query_cost: float` — Query cost

**Returns**: `str` — Best agent ID

**Example**:
```python
best_agent = service.select_best_agent_to_query(
    belief_id=belief.belief_id,
    agent_ids=["agent_a", "agent_b", "agent_c"],
    query_cost=0.1,
)
```

##### `record_task_outcome(belief_id, agent_id, actual_outcome, agent_message, agent_confidence)`

Record task outcome for learning.

**Parameters**:
- `belief_id: str` — Belief state identifier
- `agent_id: str` — Agent ID
- `actual_outcome: str` — Actual outcome
- `agent_message: str` — Agent's message
- `agent_confidence: float` — Agent's confidence

**Example**:
```python
service.record_task_outcome(
    belief_id=belief.belief_id,
    agent_id="agent_a",
    actual_outcome="success",
    agent_message="Yes, this works",
    agent_confidence=0.8,
)
```

##### `get_belief_state(belief_id)`

Retrieve belief state by ID.

**Parameters**:
- `belief_id: str` — Belief state identifier

**Returns**: `Optional[BayesianTaskAnalysis]` — Belief state or None

**Example**:
```python
belief = service.get_belief_state(belief_id)
if belief:
    print(f"Success prob: {belief.outcome_belief['success']:.2%}")
```

##### `get_belief_history(belief_id)`

Get belief state evolution history.

**Parameters**:
- `belief_id: str` — Belief state identifier

**Returns**: `List[Dict[str, Any]]` — History of belief updates

**Example**:
```python
history = service.get_belief_history(belief.belief_id)
for entry in history:
    print(f"Entropy: {entry['entropy']:.3f}")
```

##### `cleanup_belief_state(belief_id)`

Clean up belief state from memory.

**Parameters**:
- `belief_id: str` — Belief state identifier

**Example**:
```python
service.cleanup_belief_state(belief.belief_id)
```

---

### BayesianConsensusBuilder

Build consensus from multiple agent opinions with reliability weighting.

**Location**: `victor.framework.rl.consensus.bayesian_consensus`

**Class**: `BayesianConsensusBuilder`

```python
from victor.framework.rl.consensus.bayesian_consensus import BayesianConsensusBuilder

builder = BayesianConsensusBuilder(
    orchestration_service=service,
)
```

**Methods**:

##### `compute_consensus(belief_id, agent_messages, strategy="weighted_bayesian")`

Compute consensus from multiple agent opinions.

**Parameters**:
- `belief_id: str` — Belief state identifier
- `agent_messages: Dict[str, str]` — Agent ID to message mapping
- `strategy: str` — Consensus strategy: "majority_vote" or "weighted_bayesian"

**Returns**: `Dict[str, Any]` — Consensus result

**Example**:
```python
agent_messages = {
    "agent_a": "Yes, this works",
    "agent_b": "This will work",
    "agent_c": "Agreed",
}

consensus = builder.compute_consensus(
    belief_id=belief.belief_id,
    agent_messages=agent_messages,
    strategy="weighted_bayesian",
)

print(f"Consensus: {consensus['recommended_outcome']}")
print(f"Confidence: {consensus['confidence']:.2%}")
print(f"Agreement: {consensus['agreement_level']}")
```

##### `compute_consensus_and_update_belief(belief_id, agent_messages, strategy="weighted_bayesian")`

Compute consensus and update belief state.

**Parameters**:
- `belief_id: str` — Belief state identifier
- `agent_messages: Dict[str, str]` — Agent messages
- `strategy: str` — Consensus strategy

**Returns**: `Dict[str, Any]` — Consensus result

**Example**:
```python
consensus = builder.compute_consensus_and_update_belief(
    belief_id=belief.belief_id,
    agent_messages=agent_messages,
)
```

##### `record_consensus_outcome(belief_id, consensus, actual_outcome)`

Record consensus outcome for learning.

**Parameters**:
- `belief_id: str` — Belief state identifier
- `consensus: Dict[str, Any]` — Consensus result
- `actual_outcome: str` — Actual outcome

**Example**:
```python
builder.record_consensus_outcome(
    belief_id=belief.belief_id,
    consensus=consensus,
    actual_outcome="success",
)
```

##### `get_consensus_stats()`

Get aggregate consensus statistics.

**Returns**: `Dict[str, Any]` — Consensus metrics

**Example**:
```python
stats = builder.get_consensus_stats()
print(f"Total consensus: {stats['total_consensus']}")
print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"Avg confidence: {stats['avg_confidence']:.2%}")
```

##### `get_agent_consensus_stats(agent_id)`

Get consensus statistics for a specific agent.

**Parameters**:
- `agent_id: str` — Agent identifier

**Returns**: `Dict[str, Any]` — Agent's consensus metrics

**Example**:
```python
stats = builder.get_agent_consensus_stats("agent_a")
print(f"Participations: {stats['participation_count']}")
```

---

## Database Schema

### rl_observation_model

Stores Beta distribution parameters for P(agent_message | task_outcome).

```sql
CREATE TABLE rl_observation_model (
    agent_id TEXT NOT NULL,
    outcome_type TEXT NOT NULL,
    message_category TEXT NOT NULL,
    alpha REAL NOT NULL,
    beta REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (agent_id, outcome_type, message_category)
);
```

### rl_agent_reliability

Stores reliability weights α_i for each agent.

```sql
CREATE TABLE rl_agent_reliability (
    agent_id TEXT NOT NULL PRIMARY KEY,
    alpha_reliability REAL NOT NULL,
    beta_reliability REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    last_updated TEXT NOT NULL
);
```

### rl_voi_history

Tracks Value of Information computations and outcomes.

```sql
CREATE TABLE rl_voi_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    predicted_voi REAL NOT NULL,
    actual_information_gain REAL NOT NULL,
    query_cost REAL NOT NULL,
    was_beneficial BOOLEAN NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT
);
```

### rl_belief_history

Tracks belief state evolution over time.

```sql
CREATE TABLE rl_belief_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id TEXT NOT NULL,
    success_prob REAL NOT NULL,
    failure_prob REAL NOT NULL,
    entropy REAL NOT NULL,
    agent_id TEXT,
    message TEXT,
    timestamp TEXT NOT NULL
);
```

### rl_bayesian_consensus

Tracks consensus decisions and outcomes.

```sql
CREATE TABLE rl_bayesian_consensus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id TEXT NOT NULL,
    recommended_outcome TEXT NOT NULL,
    confidence REAL NOT NULL,
    agreement_level TEXT NOT NULL,
    agent_contributions TEXT,
    timestamp TEXT NOT NULL,
    actual_outcome TEXT,
    was_correct BOOLEAN
);
```

---

## Type Definitions

### OutcomeBelief

```python
from typing import Dict

OutcomeBelief = Dict[str, float]  # outcome -> probability
```

Example: `{"success": 0.8, "failure": 0.2}`

### MessageCategory

```python
class MessageCategory(str):
    AFFIRM = "affirm"
    DENY = "deny"
    UNCERTAIN = "uncertain"
    ERROR = "error"
```

### AgreementLevel

```python
class AgreementLevel(str):
    UNANIMOUS = "unanimous"
    PARTIAL = "partial"
    DIVERGENT = "divergent"
```

### ConsensusStrategy

```python
class ConsensusStrategy(str):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_BAYESIAN = "weighted_bayesian"
```

---

## Configuration

### Settings

The Bayesian orchestration system uses standard Victor settings. No additional configuration is required beyond the RL framework setup.

### Database

All Bayesian components use SQLite databases:

- **Global Database** (`~/.victor/victor.db`): Cross-project learning
- **Project Database** (`./.victor/project.db`): Project-specific data

---

## Exceptions

### BeliefStateNotFound

Raised when attempting to access a non-existent belief state.

```python
from victor.framework.rl.orchestration.bayesian_orchestrator import BeliefStateNotFound

try:
    belief = service.get_belief_state("invalid_id")
except BeliefStateNotFound:
    print("Belief state not found")
```

### InvalidLikelihood

Raised when likelihood computation fails.

```python
from victor.framework.rl.learners.observation_model import InvalidLikelihood

try:
    likelihood = learner.get_likelihood("agent_a", "message", "outcome")
except InvalidLikelihood as e:
    print(f"Invalid likelihood: {e}")
```

---

## Performance Considerations

### Thompson Sampling

Thompson Sampling adds variance to predictions for exploration. For deterministic behavior, disable sampling:

```python
learner = ObservationModelLearner(
    name="observation_model",
    db_connection=conn,
    use_thompson_sampling=False,
)
```

### Caching

Belief states are cached in memory. Clean up when done:

```python
service.cleanup_belief_state(belief_id)
```

### Database Indexes

Indexes are automatically created on:
- `agent_id` (for all tables)
- `belief_id` (for history tables)
- `timestamp` (for time-series queries)

---

## See Also

- [Bayesian Orchestration Guide](../bayesian_orchestration.md) — Usage guide and examples
- [Architecture Documentation](../architecture/bayesian.md) — Architecture details
- [Paper Reference](https://arxiv.org/abs/2605.00742) — "Position: agentic AI orchestration should be Bayes-consistent"
