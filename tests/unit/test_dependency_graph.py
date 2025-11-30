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
