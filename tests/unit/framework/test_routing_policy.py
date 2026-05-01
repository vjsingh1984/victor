from victor.framework.routing_policy import StructuredRoutingPolicy


def test_structured_routing_policy_merges_selector_and_planning_sections():
    policy = StructuredRoutingPolicy(
        scope_context={"task_type": "analysis"},
        topology_hints={"learned_topology_action": "team_plan"},
        team_hints={"learned_worktree_isolation_hint": True},
        degradation_hints={"learned_degradation_conservative_routing_hint": True},
        experiment_hints={"experiment_memory_selection_policy_bias": -0.4},
        planning_hints={"planning_force_llm": True},
        metadata={"source": "test"},
    )

    assert policy.selector_context() == {
        "learned_topology_action": "team_plan",
        "learned_worktree_isolation_hint": True,
        "learned_degradation_conservative_routing_hint": True,
        "experiment_memory_selection_policy_bias": -0.4,
    }
    assert policy.planning_context() == {"planning_force_llm": True}
    assert policy.combined_context()["planning_force_llm"] is True
    assert policy.to_dict()["metadata"]["source"] == "test"
    assert policy.is_empty() is False
