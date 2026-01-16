# Advanced Team Formations Guide

Victor provides cutting-edge team coordination patterns beyond basic formations. This guide covers advanced formations including dynamic switching, negotiation, and voting.

## Table of Contents

- [Overview](#overview)
- [Switching Formation](#switching-formation)
- [Negotiation Formation](#negotiation-formation)
- [Voting Formation](#voting-formation)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Overview

Victor's team system supports 8 formation patterns:

### Basic Formations
- **SEQUENTIAL**: Execute members one after another
- **PARALLEL**: Execute all members simultaneously
- **HIERARCHICAL**: Manager delegates to workers
- **PIPELINE**: Output passes between members
- **CONSENSUS**: All members must agree

### Advanced Formations
- **SWITCHING**: Dynamically switch formations based on progress
- **NEGOTIATION**: Members negotiate on approach before execution
- **VOTING**: Democratic decision-making on next steps

## Switching Formation

The `SwitchingFormation` allows teams to adapt their coordination strategy in real-time based on execution metrics, errors, or custom conditions.

### Key Features

- **Dynamic Formation Switching**: Change coordination patterns mid-execution
- **Condition-Based Triggers**: Switch based on progress, errors, time, or custom logic
- **Configurable Criteria**: Define multiple switching rules with priorities
- **Switch History Tracking**: Track all formation switches for analysis

### Basic Usage

```python
from victor.teams import SwitchingFormation, SwitchingCriteria, TeamFormation

# Create switching formation with progress-based criteria
formation = SwitchingFormation(
    initial_formation=TeamFormation.SEQUENTIAL,
    switching_criteria=[
        SwitchingCriteria(
            trigger_type=TriggerType.THRESHOLD,
            threshold_key="progress",
            threshold_value=0.5,
            comparison=">=",
            target_formation=TeamFormation.PARALLEL,
            trigger_once=True
        ),
        SwitchingCriteria(
            trigger_type=TriggerType.ERROR_COUNT,
            threshold_value=3,
            target_formation=TeamFormation.SEQUENTIAL,
            trigger_once=False
        )
    ],
    max_switches=5
)
```

### Trigger Types

#### CONDITION Trigger
Evaluate Python expressions on context:

```python
SwitchingCriteria(
    trigger_type=TriggerType.CONDITION,
    condition="progress > 0.5 and error_count < 2",
    target_formation=TeamFormation.PARALLEL
)
```

#### THRESHOLD Trigger
Switch when a value exceeds threshold:

```python
SwitchingCriteria(
    trigger_type=TriggerType.THRESHOLD,
    threshold_key="progress",
    threshold_value=0.7,
    comparison=">=",
    target_formation=TeamFormation.PARALLEL
)
```

#### ERROR_COUNT Trigger
Switch when error count exceeded:

```python
SwitchingCriteria(
    trigger_type=TriggerType.ERROR_COUNT,
    threshold_value=5,
    target_formation=TeamFormation.SEQUENTIAL
)
```

#### TIME_ELAPSED Trigger
Switch after time threshold:

```python
SwitchingCriteria(
    trigger_type=TriggerType.TIME_ELAPSED,
    threshold_value=300,  # 5 minutes
    target_formation=TeamFormation.PARALLEL
)
```

### Advanced Example

```python
# Adaptive formation for complex tasks
formation = SwitchingFormation(
    initial_formation=TeamFormation.SEQUENTIAL,
    switching_criteria=[
        # Switch to parallel once we understand the task
        SwitchingCriteria(
            trigger_type=TriggerType.THRESHOLD,
            threshold_key="progress",
            threshold_value=0.3,
            comparison=">=",
            target_formation=TeamFormation.PARALLEL,
            priority=10
        ),
        # Fall back to sequential if too many errors
        SwitchingCriteria(
            trigger_type=TriggerType.ERROR_COUNT,
            threshold_value=3,
            target_formation=TeamFormation.SEQUENTIAL,
            priority=20,
            trigger_once=False
        ),
        # Switch to consensus for final review
        SwitchingCriteria(
            trigger_type=TriggerType.THRESHOLD,
            threshold_key="progress",
            threshold_value=0.9,
            comparison=">=",
            target_formation=TeamFormation.CONSENSUS,
            trigger_once=True
        )
    ],
    max_switches=10,
    track_switches=True
)

# Check switch history
result = await formation.execute(agents, context, task)
for switch_event in formation.get_switch_history():
    print(f"Switched: {switch_event['from']} -> {switch_event['to']}")
    print(f"Context: {switch_event['context']}")
```

## Negotiation Formation

The `NegotiationFormation` implements structured negotiation where team members discuss and agree on the best approach before executing.

### Key Features

- **Multi-Round Negotiation**: Multiple rounds to build consensus
- **Consensus Threshold**: Minimum agreement level required
- **Proposal Evaluation**: Members submit and evaluate proposals
- **Fallback Strategy**: Use default formation if no consensus

### Basic Usage

```python
from victor.teams import NegotiationFormation

formation = NegotiationFormation(
    max_rounds=3,
    consensus_threshold=0.7,
    fallback_formation=TeamFormation.SEQUENTIAL,
    voting_strategy="weighted_confidence",
    timeout_per_round=60.0
)

# Execute with negotiation
result = await formation.execute(agents, context, task)

# Check negotiation history
for negotiation_round in formation.get_negotiation_history():
    print(f"Round {negotiation_round.round_number}")
    print(f"Consensus: {negotiation_round.consensus_reached}")
    print(f"Proposals: {len(negotiation_round.proposals)}")
    if negotiation_round.selected_proposal:
        print(f"Selected: {negotiation_round.selected_proposal.content}")
```

### Negotiation Process

1. **Initial Round**: All members submit proposals
2. **Evaluation**: Proposals evaluated for consensus
3. **Discussion**: Proposals shared (if no consensus)
4. **Subsequent Rounds**: Members refine proposals
5. **Decision**: Select best proposal or use fallback

### Example with Custom Proposal Parsing

```python
from victor.teams.advanced_formations import NegotiationProposal

class CustomNegotiationFormation(NegotiationFormation):
    def _parse_proposal(self, member_id: str, response: str) -> NegotiationProposal:
        # Custom parsing logic for your use case
        import re

        # Extract approach
        approach_match = re.search(r"Approach: (.+)", response)
        approach = approach_match.group(1) if approach_match else response

        # Extract confidence
        confidence_match = re.search(r"Confidence: ([\d.]+)", response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        return NegotiationProposal(
            member_id=member_id,
            proposal_type="approach",
            content=approach,
            confidence=confidence,
            rationale=response
        )
```

## Voting Formation

The `VotingFormation` enables democratic decision-making where team members vote on important decisions during execution.

### Key Features

- **Multiple Voting Methods**: Majority, supermajority, unanimous, weighted
- **Tiebreaker Rules**: Handle ties with configurable strategies
- **Vote Distribution Tracking**: See how votes were distributed
- **Confidence Scoring**: Track agreement levels

### Voting Methods

#### MAJORITY
Simple majority (>50% wins):

```python
from victor.teams import VotingFormation, VotingMethod

formation = VotingFormation(
    voting_method=VotingMethod.MAJORITY,
    tiebreaker="first_vote"
)
```

#### SUPERMAJORITY
Requires 67% agreement (configurable):

```python
formation = VotingFormation(
    voting_method=VotingMethod.SUPERMAJORITY,
    supermajority_threshold=0.67,
    tiebreaker="manager_decides"
)
```

#### UNANIMOUS
All members must agree:

```python
formation = VotingFormation(
    voting_method=VotingMethod.UNANIMOUS,
    tiebreaker="random"
)
```

#### WEIGHTED
Votes weighted by member confidence/expertise:

```python
formation = VotingFormation(
    voting_method=VotingMethod.WEIGHTED,
    tiebreaker="first_vote"
)
```

### Basic Usage

```python
formation = VotingFormation(
    voting_method=VotingMethod.SUPERMAJORITY,
    supermajority_threshold=0.67,
    tiebreaker="first_vote",
    fallback_formation=TeamFormation.SEQUENTIAL
)

# Execute with voting
result = await formation.execute(agents, context, task)

# Check voting history
for vote_round in formation.get_voting_history():
    print(f"Winner: {vote_round.winner}")
    print(f"Consensus: {vote_round.consensus_level:.2%}")
    print(f"Distribution: {vote_round.vote_distribution}")
```

### Example with Custom Choices

```python
class CustomVotingFormation(VotingFormation):
    def _get_choices(self, task: AgentMessage) -> List[str]:
        # Define custom voting choices
        return [
            "sequential_with_review",
            "parallel_with_sync",
            "hierarchical_with_lead"
        ]
```

## Usage Examples

### Example 1: Adaptive Code Review Team

```python
from victor.teams import (
    SwitchingFormation,
    SwitchingCriteria,
    TeamFormation,
    TriggerType
)

# Start with sequential, switch based on findings
formation = SwitchingFormation(
    initial_formation=TeamFormation.SEQUENTIAL,
    switching_criteria=[
        # If many issues found, switch to parallel for faster review
        SwitchingCriteria(
            trigger_type=TriggerType.CONDITION,
            condition="issue_count > 10",
            target_formation=TeamFormation.PARALLEL
        ),
        # If critical issues found, switch to consensus
        SwitchingCriteria(
            trigger_type=TriggerType.CONDITION,
            condition="critical_issues > 0",
            target_formation=TeamFormation.CONSENSUS
        ),
        # If taking too long, switch to pipeline
        SwitchingCriteria(
            trigger_type=TriggerType.TIME_ELAPSED,
            threshold_value=600,  # 10 minutes
            target_formation=TeamFormation.PIPELINE
        )
    ]
)
```

### Example 2: Decision-Making for Architecture

```python
from victor.teams import NegotiationFormation

# Team negotiates architecture approach
formation = NegotiationFormation(
    max_rounds=5,
    consensus_threshold=0.8,  # High agreement needed
    fallback_formation=TeamFormation.HIERARCHICAL,  # Architect decides
    voting_strategy="weighted_confidence"
)
```

### Example 3: Democratic Task Prioritization

```python
from victor.teams import VotingFormation, VotingMethod

# Team votes on task priority and approach
formation = VotingFormation(
    voting_method=VotingMethod.SUPERMAJORITY,
    supermajority_threshold=0.75,
    tiebreaker="manager_decides"
)
```

## Best Practices

### 1. Choose the Right Formation

- **Use SWITCHING** when:
  - Task characteristics are uncertain
  - You need adaptive strategies
  - Different phases require different coordination

- **Use NEGOTIATION** when:
  - Multiple valid approaches exist
  - Team has diverse expertise
  - Consensus is important for success

- **Use VOTING** when:
  - Decisions are subjective
  - Democratic process is valued
  - No clear expert authority

### 2. Define Clear Criteria

For switching formations, be specific about triggers:

```python
# Good: Specific, measurable criteria
SwitchingCriteria(
    trigger_type=TriggerType.THRESHOLD,
    threshold_key="progress",
    threshold_value=0.7,
    comparison=">=",
    target_formation=TeamFormation.PARALLEL
)

# Avoid: Vague conditions
SwitchingCriteria(
    trigger_type=TriggerType.CONDITION,
    condition="task_is_hard",  # Too subjective
    target_formation=TeamFormation.PARALLEL
)
```

### 3. Set Appropriate Thresholds

```python
# For low-risk tasks: Lower threshold
formation = NegotiationFormation(
    consensus_threshold=0.5,  # Simple majority
    max_rounds=2
)

# For high-risk tasks: Higher threshold
formation = NegotiationFormation(
    consensus_threshold=0.9,  # Near-unanimous
    max_rounds=5
)
```

### 4. Limit Switches

```python
# Prevent excessive switching
formation = SwitchingFormation(
    max_switches=5,  # Reasonable limit
    track_switches=True  # Monitor
)
```

### 5. Handle Edge Cases

```python
# Always provide fallback
formation = NegotiationFormation(
    fallback_formation=TeamFormation.SEQUENTIAL  # Safe default
)

formation = VotingFormation(
    fallback_formation=TeamFormation.SEQUENTIAL
)
```

### 6. Track and Analyze

```python
# Monitor switch history
formation = SwitchingFormation(track_switches=True)
result = await formation.execute(agents, context, task)

# Analyze switches
for switch in formation.get_switch_history():
    print(f"Switch: {switch['from']} -> {switch['to']}")
    print(f"Progress: {switch['context']['progress']:.2%}")
```

## Performance Considerations

### Switching Formation
- **Overhead**: Each switch adds ~10-50ms overhead
- **Optimal**: 2-4 switches per execution
- **Avoid**: Rapid switching (throttle with cooldown)

### Negotiation Formation
- **Time**: Each round adds 30-120s
- **Optimal**: 2-3 rounds for most cases
- **Timeout**: Set reasonable timeout per round (60s default)

### Voting Formation
- **Complexity**: O(n) where n = team size
- **Scalability**: Works well up to 10 members
- **Parallel**: Can collect votes in parallel

## Troubleshooting

### Switching Not Triggering

```python
# Ensure context variables exist
context = {
    "progress": 0.5,  # Required for THRESHOLD trigger
    "error_count": 2,  # Required for ERROR_COUNT trigger
    "elapsed_time": 120.0  # Required for TIME_ELAPSED trigger
}
```

### Negotiation Not Reaching Consensus

```python
# Lower threshold or increase rounds
formation = NegotiationFormation(
    consensus_threshold=0.6,  # Lower from 0.7
    max_rounds=5  # Increase from 3
)
```

### Voting Ties

```python
# Configure tiebreaker
formation = VotingFormation(
    tiebreaker="manager_decides"  # Explicit tiebreaker
)
```

## Further Reading

- [ML-Powered Teams](ML_TEAMS.md) - Using ML to optimize teams
- [Team Analytics](TEAM_ANALYTICS.md) - Tracking team performance
- [Team Optimization Guide](../workflows/README.md) - Optimizing team configurations
