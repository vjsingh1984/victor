#!/usr/bin/env python
"""Demonstration of Tool Execution Graph (Phase 6).

This script shows how the declarative tool execution graph system works.
"""

from victor.agent.tool_graph import (
    CacheStrategy,
    ToolDependency,
    ToolExecutionGraph,
    ToolExecutionNode,
    ValidationRule,
    ValidationRuleType,
)
from victor.agent.tool_compiler import ToolExecutionCompiler
from victor.tools.base import ToolRegistry


def main():
    """Demonstrate tool execution graph functionality."""

    print("=" * 70)
    print("Phase 6: Tool Execution Graph - Declarative & Cacheable")
    print("=" * 70)
    print()

    # 1. Create execution nodes
    print("1. Creating tool execution nodes...")
    read_node = ToolExecutionNode(
        tool_name="read_file",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.REQUIRED,
                parameter="path",
                constraint="required",
                error_message="File path is required",
            )
        ],
        cache_policy="idempotent",
        timeout_seconds=30.0,
    )

    grep_node = ToolExecutionNode(
        tool_name="grep",
        cache_policy="idempotent",
        timeout_seconds=30.0,
    )

    print(f"   - Created node: {read_node.tool_name}")
    print(f"   - Created node: {grep_node.tool_name}")
    print()

    # 2. Create execution graph
    print("2. Creating execution graph...")
    graph = ToolExecutionGraph(
        nodes=[read_node, grep_node],
        edges=[
            ToolDependency(from_node="read_file", to_node="grep"),
        ],
        cache_strategy=CacheStrategy.TTL,
        metadata={"description": "Read and search file"},
    )

    print(f"   - Graph has {len(graph.nodes)} nodes")
    print(f"   - Graph has {len(graph.edges)} edges")
    print(f"   - Cache strategy: {graph.cache_strategy.value}")
    print()

    # 3. Serialize graph
    print("3. Serializing graph...")
    serialized = graph.to_dict()
    print(f"   - Serialized to {len(serialized)} keys")
    print(f"   - Keys: {list(serialized.keys())}")
    print()

    # 4. Deserialize graph
    print("4. Deserializing graph...")
    restored = ToolExecutionGraph.from_dict(serialized)
    print(f"   - Restored {len(restored.nodes)} nodes")
    print(f"   - Restored {len(restored.edges)} edges")
    print()

    # 5. Query graph
    print("5. Querying graph structure...")
    read_node_restored = restored.get_node("read_file")
    print(f"   - Found node: {read_node_restored.tool_name}")
    print(f"   - Validation rules: {len(read_node_restored.validation_rules)}")

    dependencies = restored.get_dependencies("read_file")
    print(f"   - Dependencies from read_file: {len(dependencies)}")
    if dependencies:
        print(f"   -   → {dependencies[0].to_node}")

    dependents = restored.get_dependents("grep")
    print(f"   - Dependents on grep: {len(dependents)}")
    if dependents:
        print(f"   -   ← {dependents[0].from_node}")
    print()

    # 6. Demonstrate compilation
    print("6. Compiling from tool calls...")
    # Note: Using mock registry for demo purposes
    class MockTool:
        def __init__(self, name):
            self.name = name
            self.description = f"Mock {name} tool"
            self.parameters = {
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["path"],
            }

    class MockRegistry:
        def get_tool(self, name):
            return MockTool(name)

    compiler = ToolExecutionCompiler(MockRegistry())

    tool_calls = [
        {"name": "read_file", "arguments": {"path": "/test/file.py"}},
        {"name": "grep", "arguments": {"pattern": "import"}},
    ]

    compiled_graph = compiler.compile(tool_calls)
    print(f"   - Compiled {len(compiled_graph.nodes)} nodes")
    print(f"   - Compiler version: {compiled_graph.metadata.get('compiler_version')}")
    print()

    # 7. Compute hash for caching
    print("7. Computing graph hash for caching...")
    cache_key = compiler.compute_graph_hash(tool_calls)
    print(f"   - Cache key: {cache_key}")
    print(f"   - Key length: {len(cache_key)} characters")
    print()

    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
