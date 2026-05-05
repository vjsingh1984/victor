# Bayes-Consistent Orchestration

This guide describes Victor's Bayes-consistent orchestration system, which implements principled decision-making under uncertainty using Bayesian decision theory.

## Overview

Victor's Bayesian orchestration system provides:

- **Belief State Tracking**: Maintain posterior distributions over task outcomes
- **Observation Models**: Learn P(agent_message | task_outcome) for likelihoods
- **Reliability Weighting**: Downweight noisy/biased agents using learned weights
- **Value of Information**: Make intelligent querying decisions
- **Bayesian Consensus**: Combine multi-agent opinions with reliability weighting

Based on: "Position: agentic AI orchestration should be Bayes-consistent" (arXiv:2605.00742, ICML 2026)

## Architecture

```
Task → BayesianOrchestrationService
       ↓
   Create Belief State P(Y|D)
       ↓
   Should Query Agent? (VoI > cost?)
       ↓ YES
   Select Best Agent (max VoI)
       ↓
   Query Agent → Get Message z
       ↓
   Update Belief: P(Y|D,z) ∝ P(Y|D) * P(z|Y)^α_i
       ↓
   Execute Task → Get Outcome Y
       ↓
   Record Outcome → Update All Learners
       ↓
   Consensus (if multiple agents)
```

## Core Components

### 1. BayesianTaskAnalysis

Extends TaskAnalysis with Bayesian belief states.

**Location**: `victor/agent/bayesian_task_analysis.py`

**Key Features**:
- Posterior distribution P(Y=y|D) over task outcomes
- Shannon entropy H[Y|D] for uncertainty quantification
- Variance metrics for uncertainty tracking
- Bayesian posterior updates with conjugate priors
- Value of Information computation

**Example**:
```python
from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.agent.task_analyzer import TaskComplexity, UnifiedTaskType

# Create belief state with uniform prior
analysis = BayesianTaskAnalysis(
    complexity=TaskComplexity.SIMPLE,
    tool_budget=10,
    complexity_confidence=0.8,
    unified_task_type=UnifiedTaskType.EDIT,
    outcome_belief={"success": 0.5, "failure": 0.5},
)

# Compute posterior after agent message
likelihood = {"success": 0.8, "failure": 0.2}
analysis.compute_posterior(analysis.outcome_belief, likelihood)

# Belief shifted toward success
assert analysis.outcome_belief["success"] > 0.5
```

### 2. ObservationModelLearner

Learns P(agent_message | task_outcome) using Beta distributions.

**Location**: `victor/framework/rl/learners/observation_model.py`

**Key Features**:
- Message categorization (affirm, deny, uncertain, error)
- Beta distribution for each (agent, outcome, category) triplet
- Thompson Sampling for exploration
- Calibration metrics per agent

**Example**:
```python
from victor.framework.rl.learners.observation_model import ObservationModelLearner
import sqlite3

conn = sqlite3.connect("victor.db")
learner = ObservationModelLearner(
    name="observation_model",
    db_connection=conn,
)

# Record observation
learner.record_observation(
    agent_id="agent_a",
    message="This will work",
    actual_outcome="success",
    confidence=0.8,
)

# Get likelihood for future predictions
likelihood = learner.get_likelihood(
    agent_id="agent_a",
    message="This works",
    outcome="success",
)
```

### 3. AgentReliabilityLearner

Learns reliability weights α_i for each agent.

**Location**: `victor/framework/rl/learners/agent_reliability.py`

**Key Features**:
- Calibration-error-weighted updates
- Thompson Sampling for reliability exploration
- Reliability weights in range [0.1, 5.0]
  - α > 1.0: Upweight (more reliable)
  - α = 1.0: Neutral
  - α < 1.0: Downweight (less reliable)

**Example**:
```python
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner

learner = AgentReliabilityLearner(
    name="agent_reliability",
    db_connection=conn,
)

# Record prediction result
learner.record_prediction_result(
    agent_id="agent_a",
    was_correct=True,
    calibration_error=0.1,
)

# Get reliability weight
weight = learner.get_reliability_weight("agent_a")
assert weight > 1.0  # Reliable agent
```

### 4. VoIController

Computes Expected Value of Information for query decisions.

**Location**: `victor/framework/rl/learners/voi_controller.py`

**Key Features**:
- VoI = E[H[Y|D] - H[Y|D,z]] - cost
- Integrates observation and reliability learners
- Agent ranking by expected information gain
- Query outcome tracking

**Example**:
```python
from victor.framework.rl.learners.voi_controller import VoIController

controller = VoIController(
    name="voi_controller",
    db_connection=conn,
    observation_learner=observation_learner,
    reliability_learner=reliability_learner,
)

# Compute VoI for querying agent
voi = controller.compute_voi(
    task_analysis=belief_state,
    agent_id="agent_a",
    query_cost=0.1,
)

if voi > 0:
    print("Query this agent")
```

### 5. BayesianOrchestrationService

Integration layer coordinating all Bayesian components.

**Location**: `victor/framework/rl/orchestration/bayesian_orchestrator.py`

**Key Features**:
- Belief state lifecycle management
- Bayesian posterior updates
- VoI-based agent selection
- Outcome recording to all learners
- Belief history tracking

**Example**:
```python
from victor.framework.rl.orchestration.bayesian_orchestrator import BayesianOrchestrationService

service = BayesianOrchestrationService(
    db_connection=conn,
    observation_learner=observation_learner,
    reliability_learner=reliability_learner,
    voi_controller=voi_controller,
)

# Create belief state for task
belief = service.create_belief_state(
    task_type="code_edit",
    complexity=TaskComplexity.SIMPLE,
    tool_budget=10,
    initial_belief={"success": 0.5, "failure": 0.5},
)

# Should we query an agent?
if service.should_query_agent(belief.belief_id, "agent_a", 0.1):
    # Query agent and update belief
    service.update_belief_with_message(
        belief_id=belief.belief_id,
        agent_id="agent_a",
        message="Yes, this works",
        confidence=0.8,
    )

# Record task outcome
service.record_task_outcome(
    belief_id=belief.belief_id,
    agent_id="agent_a",
    actual_outcome="success",
    agent_message="Yes, this works",
    agent_confidence=0.8,
)
```

### 6. BayesianConsensusBuilder

Builds consensus from multiple agent opinions.

**Location**: `victor/framework/rl/consensus/bayesian_consensus.py`

**Key Features**:
- Reliability-weighted consensus
- Multiple strategies (majority vote, Bayesian)
- Agreement level detection
- Belief state integration

**Example**:
```python
from victor.framework.rl.consensus.bayesian_consensus import BayesianConsensusBuilder

builder = BayesianConsensusBuilder(
    orchestration_service=service,
)

# Get messages from multiple agents
agent_messages = {
    "agent_a": "Yes, this works",
    "agent_b": "This will work",
    "agent_c": "Agreed",
}

# Compute consensus
consensus = builder.compute_consensus(
    belief_id=belief.belief_id,
    agent_messages=agent_messages,
    strategy="weighted_bayesian",
)

print(f"Consensus: {consensus['recommended_outcome']}")
print(f"Confidence: {consensus['confidence']}")
print(f"Agreement: {consensus['agreement_level']}")
```

## Bayesian Update Formula

The core Bayesian update is:

```
P(Y|D,z) ∝ P(Y|D) * P(z|Y)^α_i
```

Where:
- `P(Y|D)`: Current belief (prior)
- `P(z|Y)`: Observation model (likelihood)
- `α_i`: Reliability weight for agent i
- `z`: Agent's message
- `Y`: Task outcome

**Reliability Weighting**:
- α_i > 1.0: Amplify agent's evidence (more reliable)
- α_i = 1.0: Use evidence as-is (neutral)
- α_i < 1.0: Diminish agent's evidence (less reliable)

## Value of Information

VoI determines whether to query an agent:

```
VoI = E[H[Y|D] - H[Y|D,z]] - cost
```

Where:
- `H[Y|D]`: Current entropy (uncertainty)
- `E[H[Y|D,z]]`: Expected posterior entropy after observation
- `cost`: Query cost

**Decision Rule**: Query if VoI > 0

**Factors Influencing VoI**:
1. Current uncertainty (higher → more VoI)
2. Agent reliability (more reliable → more VoI)
3. Query cost (lower → more VoI)

## Usage Guide

### Basic Workflow

```python
from victor.framework.rl.orchestration.bayesian_orchestrator import BayesianOrchestrationService
from victor.framework.rl.learners.observation_model import ObservationModelLearner
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner
from victor.framework.rl.learners.voi_controller import VoIController
import sqlite3

# Initialize database connection
conn = sqlite3.connect("victor.db")

# Initialize learners
observation_learner = ObservationModelLearner(
    name="observation_model",
    db_connection=conn,
)
reliability_learner = AgentReliabilityLearner(
    name="agent_reliability",
    db_connection=conn,
)
voi_controller = VoIController(
    name="voi_controller",
    db_connection=conn,
    observation_learner=observation_learner,
    reliability_learner=reliability_learner,
)

# Initialize orchestration service
service = BayesianOrchestrationService(
    db_connection=conn,
    observation_learner=observation_learner,
    reliability_learner=reliability_learner,
    voi_controller=voi_controller,
)

# 1. Create belief state for task
belief = service.create_belief_state(
    task_type="code_edit",
    complexity=TaskComplexity.COMPLEX,
    tool_budget=50,
    initial_belief={"success": 0.5, "failure": 0.5},
)

# 2. Decide whether to query agents
available_agents = ["agent_a", "agent_b", "agent_c"]
for agent_id in available_agents:
    if service.should_query_agent(belief.belief_id, agent_id, query_cost=0.1):
        print(f"Querying {agent_id}...")
        # In practice, you would query the agent here
        # For now, simulate getting a message
        agent_message = simulate_agent_query(agent_id, task)
        
        # Update belief with agent's message
        service.update_belief_with_message(
            belief_id=belief.belief_id,
            agent_id=agent_id,
            message=agent_message,
            confidence=0.8,
        )

# 3. Make decision based on final belief
final_belief = service.get_belief_state(belief.belief_id)
most_likely_outcome = final_belief.get_most_likely_outcome()

print(f"Most likely outcome: {most_likely_outcome[0]}")
print(f"Probability: {most_likely_outcome[1]:.2%}")

# 4. Execute task and record outcome
actual_outcome = execute_task(task)
service.record_task_outcome(
    belief_id=belief.belief_id,
    agent_id="agent_a",
    actual_outcome=actual_outcome,
    agent_message=agent_message,
    agent_confidence=0.8,
)

# 5. Clean up
service.cleanup_belief_state(belief.belief_id)
```

### Multi-Agent Consensus

```python
from victor.framework.rl.consensus.bayesian_consensus import BayesianConsensusBuilder

builder = BayesianConsensusBuilder(
    orchestration_service=service,
)

# Get messages from multiple agents
agent_messages = {}
for agent_id in ["agent_a", "agent_b", "agent_c"]:
    agent_messages[agent_id] = query_agent(agent_id, task)

# Compute Bayesian consensus
consensus = builder.compute_consensus_and_update_belief(
    belief_id=belief.belief_id,
    agent_messages=agent_messages,
    strategy="weighted_bayesian",
)

print(f"Consensus: {consensus['recommended_outcome']}")
print(f"Confidence: {consensus['confidence']:.2f}")
print(f"Agreement: {consensus['agreement_level']}")

# Record outcome
builder.record_consensus_outcome(
    belief_id=belief.belief_id,
    consensus=consensus,
    actual_outcome=actual_outcome,
)
```

## Database Schema

### rl_observation_model
Stores observation model parameters P(agent_message | task_outcome).

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

## Configuration

### Settings

The Bayesian orchestration system is controlled via standard Victor settings. No additional configuration is required beyond the standard RL framework setup.

### Database

All Bayesian components use SQLite databases:
- **Global Database** (`~/.victor/victor.db`): Cross-project learning data
- **Project Database** (`./.victor/project.db`): Project-specific data

## CLI and Chat Usage

Victor now exposes Bayesian monitoring from both the top-level CLI and mid-session chat slash commands.

### Top-Level CLI

Use the `victor bayesian` command group to inspect historical Bayesian learning and orchestration data:

```bash
victor bayesian summary --days 14
victor bayesian reliability --agents agent_a,agent_b --days 30
victor bayesian consensus --days 30
victor bayesian voi --agent agent_a --days 14
victor bayesian correlations agent_a,agent_b,agent_c --days 30
victor bayesian belief belief-123
```

Exports are supported for the surfaces that already have file exporters:

```bash
victor bayesian summary --output /tmp/bayesian-summary.json
victor bayesian reliability --agents agent_a,agent_b --export /tmp/reliability.csv
victor bayesian belief belief-123 --export /tmp/belief.csv
```

### Mid-Session Slash Command

Inside `victor chat`, use `/bayesian` to inspect the same historical data without leaving the session:

```text
/bayesian summary
/bayesian summary --days 14
/bayesian reliability agent_a,agent_b 30
/bayesian consensus 30
/bayesian voi agent_a 14
/bayesian correlations agent_a,agent_b,agent_c 30
/bayesian belief belief-123
```

The slash command also supports `--days`, `--agents`, `--agent`, `--output`, and `--export` where relevant.

## Performance Considerations

### Thompson Sampling

Thompson Sampling is used for exploration in:
- ObservationModelLearner: Likelihood computation
- AgentReliabilityLearner: Reliability weight computation

This provides uncertainty estimates and enables exploration but adds variance to predictions.

### Caching Strategies

Belief states are cached in memory during task execution. Use `cleanup_belief_state()` to free memory when done.

### Database Performance

Indexes are automatically created on:
- agent_id (for all tables)
- belief_id (for history tables)
- timestamp (for time-series queries)

## Troubleshooting

### Belief State Not Found

**Problem**: `get_belief_state()` returns None

**Solution**: Ensure belief state was created with `create_belief_state()` and hasn't been cleaned up with `cleanup_belief_state()`.

### Low Reliability Weights

**Problem**: All agents have reliability ≈ 1.0 (neutral)

**Solution**: Agents need more prediction history. Reliability learning requires sufficient samples (10+ predictions) to differentiate agents.

### Zero VoI

**Problem**: VoI is always 0 or negative

**Possible Causes**:
1. Belief entropy is very low (already certain)
2. Query cost is too high
3. Agent reliability is too low

**Solution**: Check belief entropy with `belief.belief_entropy` and adjust query costs accordingly.

## Further Reading

- **Paper**: "Position: agentic AI orchestration should be Bayes-consistent" (arXiv:2605.00742, ICML 2026)
- **Theory**: Bayesian decision theory, conjugate priors, Thompson sampling
- **Application**: Multi-agent systems, uncertainty quantification

## API Reference

See API documentation for detailed reference:
- `docs/api/bayesian.md` - Bayesian API reference
- `docs/architecture/bayesian.md` - Architecture details
