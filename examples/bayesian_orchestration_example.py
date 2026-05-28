#!/usr/bin/env python3
"""
Bayesian Orchestration Workflow Example

This example demonstrates the complete Bayes-consistent orchestration workflow:
1. Create belief state for a task
2. Decide whether to query agents using Value of Information
3. Update beliefs with agent messages using Bayesian posterior updates
4. Execute task and record outcome for learning
5. (Optional) Build multi-agent consensus with reliability weighting

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

import sqlite3
from pathlib import Path

from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.agent.task_analyzer import TaskComplexity, UnifiedTaskType
from victor.framework.rl.consensus.bayesian_consensus import BayesianConsensusBuilder
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner
from victor.framework.rl.learners.observation_model import ObservationModelLearner
from victor.framework.rl.learners.voi_controller import VoIController
from victor.framework.rl.orchestration.bayesian_orchestrator import (
    BayesianOrchestrationService,
)


def setup_bayesian_system(db_path: str = "bayesian_example.db"):
    """Initialize all Bayesian components."""
    conn = sqlite3.connect(db_path)

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

    # Initialize consensus builder
    consensus_builder = BayesianConsensusBuilder(
        orchestration_service=service,
    )

    return service, consensus_builder, conn


def example_1_single_agent_workflow():
    """Example 1: Single-agent Bayesian workflow."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single-Agent Bayesian Workflow")
    print("=" * 80)

    # Setup
    service, consensus_builder, conn = setup_bayesian_system("example1.db")

    # Step 1: Create belief state for a code editing task
    print("\n[Step 1] Creating belief state for code editing task...")
    belief = service.create_belief_state(
        task_type="code_edit",
        complexity=TaskComplexity.SIMPLE,
        tool_budget=10,
        initial_belief={"success": 0.5, "failure": 0.5},  # Uniform prior
    )

    print(f"  Belief ID: {belief.belief_id}")
    print(f"  Prior: P(success) = {belief.outcome_belief['success']:.2%}")
    print(f"  Entropy: {belief.belief_entropy:.3f} nats")

    # Step 2: Decide whether to query agent_a using VoI
    print("\n[Step 2] Computing Value of Information for agent_a...")
    should_query = service.should_query_agent(
        belief_id=belief.belief_id,
        agent_id="agent_a",
        query_cost=0.1,
    )

    print(f"  Should query agent_a: {should_query}")

    if should_query:
        # Step 3: Simulate querying agent_a
        print("\n[Step 3] Querying agent_a...")
        agent_message = "Yes, this will work"
        agent_confidence = 0.8
        print(
            f"  Agent response: '{agent_message}' (confidence: {agent_confidence:.2%})"
        )

        # Step 4: Update belief with agent's message
        print("\n[Step 4] Updating belief using Bayesian posterior update...")
        updated_belief = service.update_belief_with_message(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            message=agent_message,
            confidence=agent_confidence,
        )

        print(
            f"  Posterior: P(success) = {updated_belief.outcome_belief['success']:.2%}"
        )
        print(f"  Entropy: {updated_belief.belief_entropy:.3f} nats")
        print(
            f"  Entropy reduction: {belief.belief_entropy - updated_belief.belief_entropy:.3f} nats"
        )

    # Step 5: Execute task and record outcome
    print("\n[Step 5] Executing task...")
    actual_outcome = "success"  # Simulated outcome
    print(f"  Actual outcome: {actual_outcome}")

    print("\n[Step 6] Recording outcome for learning...")
    service.record_task_outcome(
        belief_id=belief.belief_id,
        agent_id="agent_a",
        actual_outcome=actual_outcome,
        agent_message=agent_message,
        agent_confidence=agent_confidence,
    )

    # Verify learning occurred
    print("\n[Verification] Checking that learning occurred...")
    reliability_weight = service.reliability_learner.get_reliability_weight("agent_a")
    print(f"  agent_a reliability weight: {reliability_weight:.3f}")

    # Cleanup
    service.cleanup_belief_state(belief.belief_id)
    conn.close()

    print("\n[✓] Example 1 complete!")


def example_2_multi_agent_consensus():
    """Example 2: Multi-agent consensus with reliability weighting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multi-Agent Consensus with Reliability Weighting")
    print("=" * 80)

    # Setup
    service, consensus_builder, conn = setup_bayesian_system("example2.db")

    # Train agents with different reliabilities
    print("\n[Setup] Training agents with different reliabilities...")
    for _ in range(9):
        service.reliability_learner.record_prediction_result(
            agent_id="agent_a",
            was_correct=True,
            calibration_error=0.1,
        )
    for _ in range(5):
        service.reliability_learner.record_prediction_result(
            agent_id="agent_b",
            was_correct=True,
            calibration_error=0.2,
        )
    for _ in range(2):
        service.reliability_learner.record_prediction_result(
            agent_id="agent_c",
            was_correct=True,
            calibration_error=0.4,
        )

    # Check reliability weights
    print("\n[Verification] Agent reliability weights:")
    for agent_id in ["agent_a", "agent_b", "agent_c"]:
        weight = service.reliability_learner.get_reliability_weight(agent_id)
        print(f"  {agent_id}: α = {weight:.3f}")

    # Step 1: Create belief state
    print("\n[Step 1] Creating belief state...")
    belief = service.create_belief_state(
        task_type="code_edit",
        complexity=TaskComplexity.COMPLEX,
        tool_budget=50,
        initial_belief={"success": 0.5, "failure": 0.5},
    )

    # Step 2: Query multiple agents
    print("\n[Step 2] Querying multiple agents...")
    agent_messages = {
        "agent_a": "Yes, this will work",
        "agent_b": "This should work",
        "agent_c": "Not sure, might fail",
    }

    for agent_id, message in agent_messages.items():
        print(f"  {agent_id}: '{message}'")

    # Step 3: Compute Bayesian consensus
    print("\n[Step 3] Computing reliability-weighted Bayesian consensus...")
    consensus = consensus_builder.compute_consensus(
        belief_id=belief.belief_id,
        agent_messages=agent_messages,
        strategy="weighted_bayesian",
    )

    print(f"  Consensus outcome: {consensus['recommended_outcome']}")
    print(f"  Confidence: {consensus['confidence']:.2%}")
    print(f"  Agreement level: {consensus['agreement_level']}")

    # Show agent contributions
    print("\n[Details] Agent contributions:")
    for agent_id, contribution in consensus["agent_contributions"].items():
        print(
            f"  {agent_id}: vote={contribution['vote']}, reliability={contribution['reliability']:.3f}"
        )

    # Step 4: Update belief with consensus
    print("\n[Step 4] Updating belief with all agent messages...")
    consensus = consensus_builder.compute_consensus_and_update_belief(
        belief_id=belief.belief_id,
        agent_messages=agent_messages,
        strategy="weighted_bayesian",
    )

    final_belief = service.get_belief_state(belief.belief_id)
    print(f"  Posterior: P(success) = {final_belief.outcome_belief['success']:.2%}")
    print(f"  Entropy: {final_belief.belief_entropy:.3f} nats")

    # Step 5: Execute task and record consensus outcome
    print("\n[Step 5] Executing task...")
    actual_outcome = "success"
    print(f"  Actual outcome: {actual_outcome}")

    print("\n[Step 6] Recording consensus outcome...")
    consensus_builder.record_consensus_outcome(
        belief_id=belief.belief_id,
        consensus=consensus,
        actual_outcome=actual_outcome,
    )

    # Verify consensus tracking
    print("\n[Verification] Consensus statistics:")
    stats = consensus_builder.get_consensus_stats()
    print(f"  Total consensus: {stats['total_consensus']}")
    print(f"  Accuracy: {stats['accuracy']:.2%}")

    # Cleanup
    service.cleanup_belief_state(belief.belief_id)
    conn.close()

    print("\n[✓] Example 2 complete!")


def example_3_voi_based_agent_selection():
    """Example 3: VoI-based agent selection."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Value of Information-Based Agent Selection")
    print("=" * 80)

    # Setup
    service, consensus_builder, conn = setup_bayesian_system("example3.db")

    # Train agents with different characteristics
    print("\n[Setup] Training agents with different characteristics...")

    # agent_a: Reliable but expensive
    for _ in range(10):
        service.observation_learner.record_observation(
            agent_id="agent_a",
            message="Yes",
            actual_outcome="success",
            confidence=0.9,
        )
    for _ in range(2):
        service.observation_learner.record_observation(
            agent_id="agent_a",
            message="No",
            actual_outcome="failure",
            confidence=0.9,
        )

    # agent_b: Less reliable but cheap
    for _ in range(6):
        service.observation_learner.record_observation(
            agent_id="agent_b",
            message="Yes",
            actual_outcome="success",
            confidence=0.7,
        )
    for _ in range(4):
        service.observation_learner.record_observation(
            agent_id="agent_b",
            message="No",
            actual_outcome="failure",
            confidence=0.7,
        )

    # Step 1: Create belief state with high uncertainty
    print("\n[Step 1] Creating belief state with high uncertainty...")
    belief = service.create_belief_state(
        task_type="code_edit",
        complexity=TaskComplexity.COMPLEX,
        tool_budget=50,
        initial_belief={"success": 0.5, "failure": 0.5},
    )

    print(f"  Initial entropy: {belief.belief_entropy:.3f} nats (high uncertainty)")

    # Step 2: Compute VoI for each agent
    print("\n[Step 2] Computing Value of Information for each agent...")

    agents_and_costs = [
        ("agent_a", 0.3),  # Expensive but reliable
        ("agent_b", 0.1),  # Cheap but less reliable
    ]

    for agent_id, query_cost in agents_and_costs:
        voi = service.voi_controller.compute_voi(
            task_analysis=belief,
            agent_id=agent_id,
            query_cost=query_cost,
        )
        print(f"  {agent_id}: VoI = {voi:.3f} (cost: {query_cost})")

    # Step 3: Rank agents by VoI
    print("\n[Step 3] Ranking agents by VoI...")
    ranked = service.voi_controller.rank_agents_by_voi(
        task_analysis=belief,
        agent_ids=["agent_a", "agent_b"],
        query_cost=0.1,
    )

    print("  Agent ranking:")
    for i, agent_rank in enumerate(ranked, 1):
        print(f"    {i}. {agent_rank['agent_id']}: VoI = {agent_rank['voi']:.3f}")

    # Step 4: Select best agent
    print("\n[Step 4] Selecting best agent to query...")
    best_agent = service.select_best_agent_to_query(
        belief_id=belief.belief_id,
        agent_ids=["agent_a", "agent_b"],
        query_cost=0.1,
    )

    print(f"  Selected agent: {best_agent}")

    # Step 5: Query selected agent and update belief
    print(f"\n[Step 5] Querying {best_agent}...")
    agent_message = "Yes, this will work"
    updated_belief = service.update_belief_with_message(
        belief_id=belief.belief_id,
        agent_id=best_agent,
        message=agent_message,
        confidence=0.8,
    )

    print(f"  Posterior: P(success) = {updated_belief.outcome_belief['success']:.2%}")
    print(
        f"  Entropy reduction: {belief.belief_entropy - updated_belief.belief_entropy:.3f} nats"
    )

    # Cleanup
    service.cleanup_belief_state(belief.belief_id)
    conn.close()

    print("\n[✓] Example 3 complete!")


def example_4_learning_from_execution():
    """Example 4: Learning from execution over multiple tasks."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Learning from Execution Over Multiple Tasks")
    print("=" * 80)

    # Setup
    service, consensus_builder, conn = setup_bayesian_system("example4.db")

    # Simulate multiple tasks
    print("\n[Simulation] Running 10 tasks and learning from outcomes...")

    for task_id in range(1, 11):
        print(f"\n[Task {task_id}]")

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # Query agent
        should_query = service.should_query_agent(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            query_cost=0.1,
        )

        if should_query:
            # Simulate agent message
            agent_message = "Yes, this will work"

            # Update belief
            service.update_belief_with_message(
                belief_id=belief.belief_id,
                agent_id="agent_a",
                message=agent_message,
                confidence=0.8,
            )

            # Simulate task outcome (80% success rate)
            import random

            actual_outcome = "success" if random.random() < 0.8 else "failure"

            # Record outcome
            service.record_task_outcome(
                belief_id=belief.belief_id,
                agent_id="agent_a",
                actual_outcome=actual_outcome,
                agent_message=agent_message,
                agent_confidence=0.8,
            )

            print(f"  Outcome: {actual_outcome}")

        # Cleanup
        service.cleanup_belief_state(belief.belief_id)

    # Check learned statistics
    print("\n[Learning Results] Statistics after 10 tasks:")

    # Get reliability weight directly
    reliability_weight = service.reliability_learner.get_reliability_weight("agent_a")
    print(f"  agent_a reliability weight: {reliability_weight:.3f}")

    # Get reliability stats if available
    reliability_stats = service.reliability_learner.get_agent_reliability_stats(
        "agent_a"
    )
    if reliability_stats:
        print(f"  agent_a sample count: {reliability_stats['sample_count']}")

    conn.close()

    print("\n[✓] Example 4 complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BAYES-CONSISTENT ORCHESTRATION WORKFLOW EXAMPLES")
    print("=" * 80)
    print("\nThis demonstrates the complete Bayesian orchestration system:")
    print("  • Belief state tracking with posterior distributions")
    print("  • Observation model learning P(agent_message | task_outcome)")
    print("  • Reliability weighting for noisy/biased agents")
    print("  • Value of Information for query decisions")
    print("  • Multi-agent consensus with reliability weighting")

    # Run examples
    example_1_single_agent_workflow()
    example_2_multi_agent_consensus()
    example_3_voi_based_agent_selection()
    example_4_learning_from_execution()

    # Cleanup databases
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)
    print("\nRemoving example databases...")
    for db_file in ["example1.db", "example2.db", "example3.db", "example4.db"]:
        Path(db_file).unlink(missing_ok=True)
        print(f"  Removed {db_file}")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Read the full guide: docs/bayesian_orchestration.md")
    print("  2. Explore the API: docs/api/bayesian.md")
    print("  3. Study the architecture: docs/architecture/bayesian.md")
    print("  4. Integrate into your Victor application")


if __name__ == "__main__":
    main()
