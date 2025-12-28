# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the workflows module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.workflows import (
    NodeType,
    WorkflowNode,
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowBuilder,
    workflow,
    get_registered_workflows,
    WorkflowMetadata,
    WorkflowRegistry,
    get_global_registry,
    NodeStatus,
    NodeResult,
    WorkflowContext,
    WorkflowResult,
    WorkflowExecutor,
)


class TestNodeType:
    """Test NodeType enum."""

    def test_all_types_defined(self):
        """All expected node types are defined."""
        assert NodeType.AGENT.value == "agent"
        assert NodeType.CONDITION.value == "condition"
        assert NodeType.PARALLEL.value == "parallel"
        assert NodeType.TRANSFORM.value == "transform"
        assert NodeType.START.value == "start"
        assert NodeType.END.value == "end"


class TestAgentNode:
    """Test AgentNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = AgentNode(id="test", name="Test Node")
        assert node.id == "test"
        assert node.name == "Test Node"
        assert node.node_type == NodeType.AGENT
        assert node.role == "executor"
        assert node.tool_budget == 15

    def test_full_node(self):
        """Create node with all fields."""
        node = AgentNode(
            id="research",
            name="Research Node",
            role="researcher",
            goal="Find all API endpoints",
            tool_budget=20,
            allowed_tools=["read", "search"],
            input_mapping={"files": "target_files"},
            output_key="research_results",
            next_nodes=["review"],
        )
        assert node.role == "researcher"
        assert node.goal == "Find all API endpoints"
        assert node.tool_budget == 20
        assert node.allowed_tools == ["read", "search"]
        assert "review" in node.next_nodes

    def test_to_dict(self):
        """to_dict serializes correctly."""
        node = AgentNode(
            id="test",
            name="Test",
            role="executor",
            goal="Do something",
        )
        d = node.to_dict()
        assert d["id"] == "test"
        assert d["type"] == "agent"
        assert d["role"] == "executor"


class TestConditionNode:
    """Test ConditionNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = ConditionNode(id="decide", name="Decision")
        assert node.id == "decide"
        assert node.node_type == NodeType.CONDITION

    def test_evaluate_condition(self):
        """Condition evaluation works."""
        node = ConditionNode(
            id="decide",
            name="Decision",
            condition=lambda ctx: "fix" if ctx.get("issues", 0) > 0 else "done",
            branches={"fix": "fixer", "done": "report"},
        )

        # With issues
        result = node.evaluate({"issues": 5})
        assert result == "fixer"

        # Without issues
        result = node.evaluate({"issues": 0})
        assert result == "report"


class TestParallelNode:
    """Test ParallelNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = ParallelNode(id="parallel", name="Parallel")
        assert node.id == "parallel"
        assert node.node_type == NodeType.PARALLEL
        assert node.join_strategy == "all"

    def test_with_parallel_nodes(self):
        """Create node with parallel nodes."""
        node = ParallelNode(
            id="parallel",
            name="Parallel Research",
            parallel_nodes=["research_a", "research_b", "research_c"],
            join_strategy="merge",
        )
        assert len(node.parallel_nodes) == 3
        assert node.join_strategy == "merge"


class TestTransformNode:
    """Test TransformNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = TransformNode(id="transform", name="Transform")
        assert node.id == "transform"
        assert node.node_type == NodeType.TRANSFORM

    def test_with_transform(self):
        """Transform function works."""
        node = TransformNode(
            id="transform",
            name="Transform",
            transform=lambda ctx: {**ctx, "processed": True},
        )
        result = node.transform({"data": "value"})
        assert result["processed"] is True
        assert result["data"] == "value"


class TestWorkflowDefinition:
    """Test WorkflowDefinition class."""

    def test_minimal_definition(self):
        """Create definition with minimal fields."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(id="start", name="Start"),
            },
        )
        assert defn.name == "test"
        assert defn.start_node == "start"

    def test_get_node(self):
        """get_node returns correct node."""
        node = AgentNode(id="test", name="Test")
        defn = WorkflowDefinition(
            name="test",
            nodes={"test": node},
        )
        assert defn.get_node("test") == node
        assert defn.get_node("nonexistent") is None

    def test_validate_empty(self):
        """Validation fails for empty workflow."""
        defn = WorkflowDefinition(name="test")
        errors = defn.validate()
        assert len(errors) > 0
        assert any("at least one node" in e for e in errors)

    def test_validate_missing_start(self):
        """Validation fails for missing start node."""
        defn = WorkflowDefinition(
            name="test",
            nodes={"a": AgentNode(id="a", name="A")},
            start_node="nonexistent",
        )
        errors = defn.validate()
        assert any("not found" in e for e in errors)

    def test_validate_broken_reference(self):
        """Validation fails for broken next_node reference."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A", next_nodes=["nonexistent"]),
            },
        )
        errors = defn.validate()
        assert any("non-existent node" in e for e in errors)

    def test_get_agent_count(self):
        """get_agent_count returns correct count."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A"),
                "b": AgentNode(id="b", name="B"),
                "c": ConditionNode(id="c", name="C"),
            },
        )
        assert defn.get_agent_count() == 2

    def test_get_total_budget(self):
        """get_total_budget sums agent budgets."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A", tool_budget=10),
                "b": AgentNode(id="b", name="B", tool_budget=20),
            },
        )
        assert defn.get_total_budget() == 30

    def test_to_dict(self):
        """to_dict serializes correctly."""
        defn = WorkflowDefinition(
            name="test",
            description="Test workflow",
            nodes={
                "a": AgentNode(id="a", name="A"),
            },
        )
        d = defn.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test workflow"
        assert "a" in d["nodes"]


class TestWorkflowBuilder:
    """Test WorkflowBuilder class."""

    def test_simple_chain(self):
        """Build simple sequential workflow."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("analyze", "researcher", "Analyze code")
            .add_agent("review", "reviewer", "Review findings")
            .add_agent("report", "planner", "Create report")
            .build()
        )

        assert workflow.name == "test"
        assert len(workflow.nodes) == 3
        assert workflow.start_node == "analyze"

        # Check auto-chaining
        assert "review" in workflow.nodes["analyze"].next_nodes
        assert "report" in workflow.nodes["review"].next_nodes

    def test_with_condition(self):
        """Build workflow with condition node."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("analyze", "researcher", "Analyze")
            .add_condition(
                "decide",
                lambda ctx: "fix" if ctx.get("issues") else "done",
                {"fix": "fixer", "done": "report"},
            )
            .add_agent("fixer", "executor", "Fix issues", next_nodes=["report"])
            .add_agent("report", "planner", "Report")
            .build()
        )

        assert len(workflow.nodes) == 4
        assert "decide" in workflow.nodes

    def test_with_parallel(self):
        """Build workflow with parallel node."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("research_a", "researcher", "Research A")
            .add_agent("research_b", "researcher", "Research B")
            .add_parallel(
                "parallel",
                ["research_a", "research_b"],
                next_nodes=["merge"],
            )
            .add_agent("merge", "planner", "Merge results")
            .build()
        )

        parallel = workflow.nodes["parallel"]
        assert parallel.parallel_nodes == ["research_a", "research_b"]

    def test_explicit_chain(self):
        """chain() creates explicit connection."""
        builder = WorkflowBuilder("test")
        builder.add_agent("a", "executor", "A")
        builder.add_agent("b", "executor", "B")
        builder.chain("a", "b")

        workflow = builder.build()
        assert "b" in workflow.nodes["a"].next_nodes

    def test_metadata(self):
        """set_metadata adds metadata."""
        workflow = (
            WorkflowBuilder("test")
            .set_metadata("version", "1.0.0")
            .set_metadata("author", "test")
            .add_agent("a", "executor", "A")
            .build()
        )

        assert workflow.metadata["version"] == "1.0.0"
        assert workflow.metadata["author"] == "test"

    def test_duplicate_node_raises(self):
        """Adding duplicate node ID raises ValueError."""
        builder = WorkflowBuilder("test")
        builder.add_agent("a", "executor", "A")

        with pytest.raises(ValueError, match="already exists"):
            builder.add_agent("a", "executor", "A again")

    def test_chain_unknown_node_raises(self):
        """Chaining unknown node raises ValueError."""
        builder = WorkflowBuilder("test")
        builder.add_agent("a", "executor", "A")

        with pytest.raises(ValueError, match="not found"):
            builder.chain("a", "unknown")


class TestWorkflowDecorator:
    """Test @workflow decorator."""

    def test_decorator_registers_workflow(self):
        """Decorator registers workflow factory."""

        @workflow("test_decorated", "A test workflow")
        def test_workflow():
            return (
                WorkflowBuilder("test_decorated")
                .add_agent("a", "executor", "A")
                .build()
            )

        # Should be in registered workflows
        registered = get_registered_workflows()
        assert "test_decorated" in registered

        # Calling factory should return workflow
        wf = registered["test_decorated"]()
        assert wf.name == "test_decorated"
        assert wf.description == "A test workflow"


class TestWorkflowMetadata:
    """Test WorkflowMetadata class."""

    def test_from_definition(self):
        """from_definition creates correct metadata."""
        defn = WorkflowDefinition(
            name="test",
            description="Test workflow",
            nodes={
                "a": AgentNode(id="a", name="A", tool_budget=10),
                "b": AgentNode(id="b", name="B", tool_budget=20),
            },
            metadata={"tags": ["code", "review"], "version": "2.0.0"},
        )

        meta = WorkflowMetadata.from_definition(defn)
        assert meta.name == "test"
        assert meta.description == "Test workflow"
        assert meta.agent_count == 2
        assert meta.total_budget == 30
        assert "code" in meta.tags
        assert meta.version == "2.0.0"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        meta = WorkflowMetadata(
            name="test",
            description="Test",
            agent_count=2,
            total_budget=30,
        )
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["agent_count"] == 2


class TestWorkflowRegistry:
    """Test WorkflowRegistry class."""

    def test_register_and_get(self):
        """Register and retrieve workflow."""
        registry = WorkflowRegistry()
        workflow = WorkflowDefinition(
            name="test",
            nodes={"a": AgentNode(id="a", name="A")},
        )

        registry.register(workflow)
        retrieved = registry.get("test")

        assert retrieved == workflow

    def test_register_duplicate_raises(self):
        """Registering duplicate raises ValueError."""
        registry = WorkflowRegistry()
        workflow = WorkflowDefinition(
            name="test",
            nodes={"a": AgentNode(id="a", name="A")},
        )

        registry.register(workflow)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(workflow)

    def test_register_with_replace(self):
        """Registering with replace=True overwrites."""
        registry = WorkflowRegistry()
        workflow1 = WorkflowDefinition(
            name="test",
            description="First",
            nodes={"a": AgentNode(id="a", name="A")},
        )
        workflow2 = WorkflowDefinition(
            name="test",
            description="Second",
            nodes={"a": AgentNode(id="a", name="A")},
        )

        registry.register(workflow1)
        registry.register(workflow2, replace=True)

        retrieved = registry.get("test")
        assert retrieved.description == "Second"

    def test_register_factory(self):
        """Register and use factory."""
        registry = WorkflowRegistry()

        def create_workflow():
            return WorkflowDefinition(
                name="lazy",
                nodes={"a": AgentNode(id="a", name="A")},
            )

        registry.register_factory("lazy", create_workflow)

        # Not instantiated yet
        assert "lazy" not in registry._definitions

        # Get triggers instantiation
        workflow = registry.get("lazy")
        assert workflow is not None
        assert workflow.name == "lazy"

        # Now cached
        assert "lazy" in registry._definitions

    def test_list_workflows(self):
        """list_workflows returns all names."""
        registry = WorkflowRegistry()
        registry.register(
            WorkflowDefinition(
                name="a",
                nodes={"x": AgentNode(id="x", name="X")},
            )
        )
        registry.register_factory(
            "b",
            lambda: WorkflowDefinition(
                name="b",
                nodes={"y": AgentNode(id="y", name="Y")},
            ),
        )

        names = registry.list_workflows()
        assert "a" in names
        assert "b" in names

    def test_unregister(self):
        """unregister removes workflow."""
        registry = WorkflowRegistry()
        registry.register(
            WorkflowDefinition(
                name="test",
                nodes={"a": AgentNode(id="a", name="A")},
            )
        )

        assert registry.unregister("test") is True
        assert registry.get("test") is None
        assert registry.unregister("test") is False

    def test_search_by_tags(self):
        """search filters by tags."""
        registry = WorkflowRegistry()
        registry.register(
            WorkflowDefinition(
                name="a",
                nodes={"x": AgentNode(id="x", name="X")},
                metadata={"tags": ["code", "review"]},
            )
        )
        registry.register(
            WorkflowDefinition(
                name="b",
                nodes={"y": AgentNode(id="y", name="Y")},
                metadata={"tags": ["test"]},
            )
        )

        results = registry.search(tags={"code"})
        assert len(results) == 1
        assert results[0].name == "a"


class TestNodeResult:
    """Test NodeResult class."""

    def test_success_result(self):
        """Create successful result."""
        result = NodeResult(
            node_id="test",
            status=NodeStatus.COMPLETED,
            output="Found 5 items",
            tool_calls_used=10,
        )
        assert result.success is True
        assert result.output == "Found 5 items"

    def test_failure_result(self):
        """Create failed result."""
        result = NodeResult(
            node_id="test",
            status=NodeStatus.FAILED,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        result = NodeResult(
            node_id="test",
            status=NodeStatus.COMPLETED,
            output="Done",
        )
        d = result.to_dict()
        assert d["node_id"] == "test"
        assert d["status"] == "completed"


class TestWorkflowContext:
    """Test WorkflowContext class."""

    def test_get_set(self):
        """get and set work correctly."""
        ctx = WorkflowContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("missing", "default") == "default"

    def test_update(self):
        """update adds multiple values."""
        ctx = WorkflowContext()
        ctx.update({"a": 1, "b": 2})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_add_result(self):
        """add_result stores node result."""
        ctx = WorkflowContext()
        result = NodeResult(
            node_id="test",
            status=NodeStatus.COMPLETED,
            output="Done",
        )
        ctx.add_result(result)
        assert ctx.get_result("test") == result

    def test_has_failures(self):
        """has_failures detects failures."""
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="a", status=NodeStatus.COMPLETED)
        )
        assert ctx.has_failures() is False

        ctx.add_result(
            NodeResult(node_id="b", status=NodeStatus.FAILED)
        )
        assert ctx.has_failures() is True

    def test_get_outputs(self):
        """get_outputs returns successful outputs."""
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="a", status=NodeStatus.COMPLETED, output="A output")
        )
        ctx.add_result(
            NodeResult(node_id="b", status=NodeStatus.FAILED)
        )
        ctx.add_result(
            NodeResult(node_id="c", status=NodeStatus.COMPLETED, output="C output")
        )

        outputs = ctx.get_outputs()
        assert outputs == {"a": "A output", "c": "C output"}


class TestWorkflowResult:
    """Test WorkflowResult class."""

    def test_success_result(self):
        """Create successful result."""
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="a", status=NodeStatus.COMPLETED, output="Done")
        )

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            context=ctx,
            total_duration=5.0,
            total_tool_calls=10,
        )

        assert result.success is True
        assert result.get_output("a") == "Done"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        ctx = WorkflowContext()
        result = WorkflowResult(
            workflow_name="test",
            success=True,
            context=ctx,
        )

        d = result.to_dict()
        assert d["workflow_name"] == "test"
        assert d["success"] is True


class TestWorkflowExecutor:
    """Test WorkflowExecutor class."""

    def test_initialization(self):
        """Executor can be initialized."""
        mock_orchestrator = MagicMock()
        executor = WorkflowExecutor(mock_orchestrator)

        assert executor.orchestrator == mock_orchestrator
        assert executor.max_parallel == 4

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self):
        """Execute simple sequential workflow."""
        mock_orchestrator = MagicMock()

        # Mock SubAgentOrchestrator result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Task completed"
        mock_result.error = None
        mock_result.tool_calls_used = 5

        # Create executor and inject mock sub_agents
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_result)
        executor._sub_agents = mock_sub_agents

        workflow = (
            WorkflowBuilder("test")
            .add_agent("analyze", "researcher", "Analyze code")
            .build()
        )

        result = await executor.execute(workflow, {"files": ["main.py"]})

        assert result.success is True
        assert result.total_tool_calls == 5
        mock_sub_agents.spawn.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Execute with timeout."""
        mock_orchestrator = MagicMock()

        async def slow_spawn(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(success=True)

        # Create executor and inject mock sub_agents
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = slow_spawn
        executor._sub_agents = mock_sub_agents

        workflow = (
            WorkflowBuilder("test")
            .add_agent("slow", "executor", "Slow task")
            .build()
        )

        result = await executor.execute(workflow, timeout=0.1)

        assert result.success is False
        assert "timed out" in result.error.lower()


class TestModuleExports:
    """Test module exports are correct."""

    def test_workflows_init_exports(self):
        """Workflows __init__ exports all expected symbols."""
        from victor.workflows import (
            NodeType,
            WorkflowNode,
            AgentNode,
            ConditionNode,
            ParallelNode,
            WorkflowDefinition,
            WorkflowBuilder,
            workflow,
            WorkflowRegistry,
            NodeStatus,
            NodeResult,
            WorkflowContext,
            WorkflowResult,
            WorkflowExecutor,
        )
        # If we get here without ImportError, all exports work
        assert True
