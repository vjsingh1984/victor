from victor.tools.base import CostTier
from victor.tools.dependency_graph import ToolDependencyGraph


def test_dependency_graph_finds_plan():
    graph = ToolDependencyGraph()
    graph.add_tool("code_search", inputs=["query"], outputs=["file_candidates"])
    graph.add_tool("read_file", inputs=["file_candidates"], outputs=["file_contents"])
    graph.add_tool("analyze_docs", inputs=["file_contents"], outputs=["summary"])

    plan = graph.plan(goals=["summary"], available=["query"])

    assert plan == ["code_search", "read_file", "analyze_docs"]


def test_dependency_graph_returns_empty_on_unmet_requirement():
    graph = ToolDependencyGraph()
    graph.add_tool("read_file", inputs=["file_candidates"], outputs=["file_contents"])

    plan = graph.plan(goals=["file_contents"], available=["query"])

    assert plan == []


def test_dependency_graph_get_total_plan_cost():
    """Test get_total_plan_cost calculates correct total (covers lines 124-128)."""
    graph = ToolDependencyGraph()
    graph.add_tool("tool1", inputs=[], outputs=["a"], cost_tier=CostTier.FREE)
    graph.add_tool("tool2", inputs=["a"], outputs=["b"], cost_tier=CostTier.MEDIUM)
    graph.add_tool("tool3", inputs=["b"], outputs=["c"], cost_tier=CostTier.HIGH)

    plan = ["tool1", "tool2", "tool3"]
    cost = graph.get_total_plan_cost(plan)

    # FREE=0.0, MEDIUM=1.5, HIGH=3.0
    expected = CostTier.FREE.weight + CostTier.MEDIUM.weight + CostTier.HIGH.weight
    assert cost == expected


def test_dependency_graph_get_total_plan_cost_with_unknown_tool():
    """Test get_total_plan_cost skips unknown tools."""
    graph = ToolDependencyGraph()
    graph.add_tool("tool1", inputs=[], outputs=["a"], cost_tier=CostTier.LOW)

    plan = ["tool1", "unknown_tool"]
    cost = graph.get_total_plan_cost(plan)

    assert cost == CostTier.LOW.weight


def test_dependency_graph_cycle_detection():
    """Test that cycles are detected and return empty plan (covers line 95)."""
    graph = ToolDependencyGraph()
    # Create a cycle: A needs B's output, B needs A's output
    graph.add_tool("tool_a", inputs=["output_b"], outputs=["output_a"])
    graph.add_tool("tool_b", inputs=["output_a"], outputs=["output_b"])

    # This should return empty due to cycle detection
    plan = graph.plan(goals=["output_a"], available=[])
    assert plan == []


def test_dependency_graph_memo_hit():
    """Test that memo prevents re-resolving same tool (covers line 93)."""
    graph = ToolDependencyGraph()
    graph.add_tool("base", inputs=[], outputs=["base_output"])
    # Both goals need the same base tool
    graph.add_tool("tool_a", inputs=["base_output"], outputs=["output_a"])
    graph.add_tool("tool_b", inputs=["base_output"], outputs=["output_b"])

    # Plan should only include base once due to memo
    plan = graph.plan(goals=["output_a", "output_b"], available=[])
    assert "base" in plan
    assert plan.count("base") == 1


def test_dependency_graph_cost_aware_selection():
    """Test cost-aware selection prefers lower-cost providers."""
    graph = ToolDependencyGraph(cost_aware=True)
    # Two tools provide the same output
    graph.add_tool("cheap_tool", inputs=[], outputs=["result"], cost_tier=CostTier.FREE)
    graph.add_tool("expensive_tool", inputs=[], outputs=["result"], cost_tier=CostTier.HIGH)

    plan = graph.plan(goals=["result"], available=[])

    # Should pick cheaper tool
    assert plan == ["cheap_tool"]


def test_dependency_graph_not_cost_aware():
    """Test non-cost-aware selection uses alphabetical order."""
    graph = ToolDependencyGraph(cost_aware=False)
    graph.add_tool("z_tool", inputs=[], outputs=["result"], cost_tier=CostTier.FREE)
    graph.add_tool("a_tool", inputs=[], outputs=["result"], cost_tier=CostTier.HIGH)

    plan = graph.plan(goals=["result"], available=[])

    # Should pick alphabetically first
    assert plan == ["a_tool"]
