#!/usr/bin/env python3
"""
Correlation Tracking Example

This example demonstrates how to use the CorrelationTracker for
dependence-aware pooling in multi-agent consensus.

When agents' predictions are correlated, their messages don't provide
independent evidence. The correlation tracker detects these correlations
and adjusts weights accordingly to avoid overcounting evidence.
"""

import sqlite3
from pathlib import Path

from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.agent.task_analyzer import TaskComplexity
from victor.framework.rl.consensus.bayesian_consensus import BayesianConsensusBuilder
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner
from victor.framework.rl.learners.correlation_tracker import CorrelationTracker
from victor.framework.rl.learners.observation_model import ObservationModelLearner
from victor.framework.rl.learners.voi_controller import VoIController
from victor.framework.rl.orchestration.bayesian_orchestrator import BayesianOrchestrationService


def setup_bayesian_system_with_correlations(db_path: str = "correlation_example.db"):
    """Initialize Bayesian system with correlation tracking."""
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

    # Initialize correlation tracker
    correlation_tracker = CorrelationTracker(
        name="correlation_tracker",
        db_connection=conn,
    )

    # Initialize orchestration service
    service = BayesianOrchestrationService(
        db_connection=conn,
        observation_learner=observation_learner,
        reliability_learner=reliability_learner,
        voi_controller=voi_controller,
    )

    # Initialize consensus builder with correlation tracker
    consensus_builder = BayesianConsensusBuilder(
        orchestration_service=service,
        correlation_tracker=correlation_tracker,
    )

    return service, consensus_builder, correlation_tracker, conn


def example_1_correlation_detection():
    """Example 1: Detect correlations between agents."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Detecting Correlations Between Agents")
    print("=" * 80)

    # Setup
    service, consensus_builder, correlation_tracker, conn = setup_bayesian_system_with_correlations(
        "example1_correlation.db"
    )

    # Simulate tasks where agents always agree
    print("\n[Simulation] Running 10 tasks where agents always agree...")
    for task_id in range(10):
        # Both agents predict success
        correlation_tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

    # Check correlation
    correlation = correlation_tracker.get_correlation("agent_a", "agent_b")
    print(f"\n[Result] Correlation between agent_a and agent_b: {correlation:.3f}")
    print("  → Perfect positive correlation (1.0) means agents always agree")

    # Simulate tasks where agents always disagree
    print("\n[Simulation] Running 10 tasks where agents always disagree...")
    for task_id in range(10):
        correlation_tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_c",
            prediction_1="success",
            prediction_2="failure",
            actual_outcome="success",
        )

    # Check correlation
    correlation = correlation_tracker.get_correlation("agent_a", "agent_c")
    print(f"\n[Result] Correlation between agent_a and agent_c: {correlation:.3f}")
    print("  → Perfect negative correlation (-1.0) means agents always disagree")

    # Get correlation matrix
    print("\n[Correlation Matrix]")
    matrix = correlation_tracker.get_correlation_matrix(["agent_a", "agent_b", "agent_c"])
    for agent_1, correlations in matrix.items():
        for agent_2, corr in correlations.items():
            print(f"  {agent_1} ↔ {agent_2}: {corr:.3f}")

    conn.close()
    Path("example1_correlation.db").unlink(missing_ok=True)

    print("\n[✓] Example 1 complete!")


def example_2_effective_sample_size():
    """Example 2: Compute effective sample size accounting for correlations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Effective Sample Size with Correlations")
    print("=" * 80)

    # Setup
    service, consensus_builder, correlation_tracker, conn = setup_bayesian_system_with_correlations(
        "example2_ess.db"
    )

    # Create three agents with different correlations
    print("\n[Setup] Creating three agent relationships:")
    print("  - agent_a and agent_b: Highly correlated (always agree)")
    print("  - agent_c: Independent from both")

    # Train agent_a and agent_b to always agree
    for _ in range(10):
        correlation_tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

    # Compute effective sample size
    agent_ids = ["agent_a", "agent_b", "agent_c"]
    weights = {"agent_a": 1.0, "agent_b": 1.0, "agent_c": 1.0}

    ess = correlation_tracker.compute_effective_sample_size(agent_ids, weights)

    print("\n[Result] Effective Sample Size:")
    print(f"  Nominal sample size: {sum(weights.values()):.1f}")
    print(f"  Effective sample size: {ess:.2f}")
    print("  → ESS < nominal because agent_a and agent_b are correlated")
    print("  → Correlated agents provide less independent information")

    conn.close()
    Path("example2_ess.db").unlink(missing_ok=True)

    print("\n[✓] Example 2 complete!")


def example_3_correlation_adjusted_consensus():
    """Example 3: Correlation-adjusted consensus."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Correlation-Adjusted Consensus")
    print("=" * 80)

    # Setup
    service, consensus_builder, correlation_tracker, conn = setup_bayesian_system_with_correlations(
        "example3_consensus.db"
    )

    # Train agents with different reliabilities and correlations
    print("\n[Setup] Training agents:")
    print("  - agent_a: High reliability, correlated with agent_b")
    print("  - agent_b: Medium reliability, correlated with agent_a")
    print("  - agent_c: Medium reliability, independent")

    # Train reliabilities
    for _ in range(10):
        service.reliability_learner.record_prediction_result("agent_a", True, 0.1)
    for _ in range(7):
        service.reliability_learner.record_prediction_result("agent_b", True, 0.2)
    for _ in range(7):
        service.reliability_learner.record_prediction_result("agent_c", True, 0.2)

    # Create correlation between agent_a and agent_b
    for _ in range(10):
        correlation_tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

    # Check base reliabilities
    print("\n[Base Reliability Weights]")
    for agent_id in ["agent_a", "agent_b", "agent_c"]:
        weight = service.reliability_learner.get_reliability_weight(agent_id)
        print(f"  {agent_id}: {weight:.3f}")

    # Check adjusted weights
    print("\n[Correlation-Adjusted Weights]")
    base_weights = {
        "agent_a": service.reliability_learner.get_reliability_weight("agent_a"),
        "agent_b": service.reliability_learner.get_reliability_weight("agent_b"),
        "agent_c": service.reliability_learner.get_reliability_weight("agent_c"),
    }

    adjusted_weights = correlation_tracker.get_adjusted_reliability_weights(
        ["agent_a", "agent_b", "agent_c"], base_weights
    )

    for agent_id in ["agent_a", "agent_b", "agent_c"]:
        base = base_weights[agent_id]
        adjusted = adjusted_weights[agent_id]
        reduction = (base - adjusted) / base * 100 if base > 0 else 0
        print(f"  {agent_id}: {base:.3f} → {adjusted:.3f} ({reduction:.1f}% reduction)")

    # Compute consensus with correlation adjustment
    print("\n[Consensus]")
    belief = service.create_belief_state(
        task_type="code_edit",
        complexity=TaskComplexity.SIMPLE,
        tool_budget=10,
        initial_belief={"success": 0.5, "failure": 0.5},
    )

    agent_messages = {
        "agent_a": "Yes, this works",
        "agent_b": "This will work",
        "agent_c": "Agreed",
    }

    consensus = consensus_builder.compute_consensus(
        belief_id=belief.belief_id, agent_messages=agent_messages, strategy="weighted_bayesian"
    )

    print(f"  Recommended outcome: {consensus['recommended_outcome']}")
    print(f"  Confidence: {consensus['confidence']:.2%}")

    print("\n[Agent Contributions]")
    for agent_id, contribution in consensus["agent_contributions"].items():
        reliability = contribution["reliability"]
        adjusted_weight = contribution["adjusted_weight"]
        correlation_adjusted = contribution["correlation_adjusted"]
        print(f"  {agent_id}:")
        print(f"    Base reliability: {reliability:.3f}")
        print(f"    Adjusted weight: {adjusted_weight:.3f}")
        print(f"    Correlation adjusted: {correlation_adjusted}")

    conn.close()
    Path("example3_consensus.db").unlink(missing_ok=True)

    print("\n[✓] Example 3 complete!")


def example_4_identify_correlated_agents():
    """Example 4: Identify highly correlated agent pairs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Identifying Highly Correlated Agent Pairs")
    print("=" * 80)

    # Setup
    service, consensus_builder, correlation_tracker, conn = setup_bayesian_system_with_correlations(
        "example4_identify.db"
    )

    # Create various correlation patterns
    print("\n[Setup] Creating different correlation patterns:")

    # Highly correlated pair
    print("  - agent_a ↔ agent_b: Highly correlated (0.9)")
    for _ in range(9):
        correlation_tracker.record_prediction_pair("agent_a", "agent_b", "success", "success", "success")
    for _ in range(1):
        correlation_tracker.record_prediction_pair("agent_a", "agent_b", "success", "failure", "success")

    # Moderately correlated pair
    print("  - agent_b ↔ agent_c: Moderately correlated (0.6)")
    for _ in range(8):
        correlation_tracker.record_prediction_pair("agent_b", "agent_c", "success", "success", "success")
    for _ in range(2):
        correlation_tracker.record_prediction_pair("agent_b", "agent_c", "success", "failure", "success")

    # Weakly correlated pair
    print("  - agent_a ↔ agent_c: Weakly correlated (0.2)")
    for _ in range(6):
        correlation_tracker.record_prediction_pair("agent_a", "agent_c", "success", "success", "success")
    for _ in range(4):
        correlation_tracker.record_prediction_pair("agent_a", "agent_c", "success", "failure", "success")

    # Find highly correlated pairs (threshold > 0.7)
    print("\n[Highly Correlated Pairs (threshold > 0.7)]")
    highly_correlated = correlation_tracker.get_highly_correlated_pairs(threshold=0.7)

    for agent_1, agent_2, correlation in highly_correlated:
        print(f"  {agent_1} ↔ {agent_2}: {correlation:.3f}")

    # Get detailed statistics
    print("\n[Detailed Correlation Statistics]")
    stats = correlation_tracker.get_correlation_stats()

    for agent_id, agent_stats in stats.items():
        print(f"\n  {agent_id}:")
        for other_agent, correlation_info in agent_stats["correlations"].items():
            correlation = correlation_info["correlation"]
            agreement_rate = correlation_info["agreement_rate"]
            total_pairs = correlation_info["total_pairs"]
            print(f"    ↔ {other_agent}:")
            print(f"      Correlation: {correlation:.3f}")
            print(f"      Agreement rate: {agreement_rate:.2%}")
            print(f"      Total pairs: {total_pairs}")

    conn.close()
    Path("example4_identify.db").unlink(missing_ok=True)

    print("\n[✓] Example 4 complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CORRELATION TRACKING EXAMPLES")
    print("=" * 80)
    print("\nThis demonstrates correlation tracking for dependence-aware pooling:")
    print("  • Detect correlations between agent predictions")
    print("  • Compute effective sample size accounting for correlations")
    print("  • Adjust reliability weights to avoid overcounting evidence")
    print("  • Identify highly correlated agent pairs")

    # Run examples
    example_1_correlation_detection()
    example_2_effective_sample_size()
    example_3_correlation_adjusted_consensus()
    example_4_identify_correlated_agents()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  1. Correlation tracking prevents overcounting correlated evidence")
    print("  2. Effective sample size is reduced when agents are correlated")
    print("  3. Correlation-adjusted consensus downweights correlated agents")
    print("  4. Identifying correlated pairs helps improve agent diversity")
    print("\nNext steps:")
    print("  • Integrate correlation tracking into your multi-agent systems")
    print("  • Monitor correlation statistics to detect agent groupthink")
    print("  • Use effective sample size for resource allocation")


if __name__ == "__main__":
    main()
