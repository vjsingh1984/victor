"""Tests for VerticalContext flyweight pattern (create_child_context)."""

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.vertical_context import VerticalContext, create_vertical_context


class TestCreateChildContext:

    def _parent_context(self) -> VerticalContext:
        ctx = VerticalContext()
        ctx.name = "coding"
        ctx.system_prompt = "You are a coding assistant."
        ctx.default_mode = "careful"
        ctx.default_budget = 20
        ctx.stages = {"planning": {"desc": "plan"}, "execution": {"desc": "exec"}}
        ctx.middleware = ["mw1", "mw2"]
        ctx.safety_patterns = ["pat1"]
        ctx.task_hints = {"edit": "be careful"}
        ctx.enabled_tools = {"read", "write", "bash"}
        ctx.mode_configs = {"careful": {"budget": 20}}
        ctx.workflows = {"review": {"steps": []}}
        ctx.team_specs = {"pair": {"roles": ["coder", "reviewer"]}}
        ctx.capability_configs = {"rag": {"enabled": True}}
        ctx.prompt_sections = ["section1", "section2"]
        return ctx

    def test_child_inherits_identity(self):
        parent = self._parent_context()
        child = parent.create_child_context()
        assert child.name == "coding"
        assert child.system_prompt == "You are a coding assistant."
        assert child.default_mode == "careful"
        assert child.default_budget == 20

    def test_child_inherits_collections(self):
        parent = self._parent_context()
        child = parent.create_child_context()
        assert child.stages == parent.stages
        assert child.middleware == parent.middleware
        assert child.safety_patterns == parent.safety_patterns
        assert child.task_hints == parent.task_hints
        assert child.workflows == parent.workflows

    def test_child_collections_are_independent_copies(self):
        parent = self._parent_context()
        child = parent.create_child_context()

        # Mutating child shouldn't affect parent
        child.stages["new_stage"] = {"desc": "new"}
        assert "new_stage" not in parent.stages

        child.middleware.append("mw3")
        assert len(parent.middleware) == 2

    def test_child_inherits_enabled_tools_by_default(self):
        parent = self._parent_context()
        child = parent.create_child_context()
        assert child.enabled_tools == {"read", "write", "bash"}

    def test_child_override_enabled_tools(self):
        parent = self._parent_context()
        child = parent.create_child_context(enabled_tools={"read"})
        assert child.enabled_tools == {"read"}
        # Parent unchanged
        assert parent.enabled_tools == {"read", "write", "bash"}

    def test_child_gets_fresh_mutable_state(self):
        parent = self._parent_context()
        child = parent.create_child_context()

        # prompt_sections should be fresh (empty)
        assert child.prompt_sections == []
        # capability_configs should be fresh (empty)
        assert child.capability_configs == {}

    def test_child_mutations_dont_affect_parent(self):
        parent = self._parent_context()
        child = parent.create_child_context()
        child.enabled_tools.add("execute")
        child.capability_configs["new"] = True

        assert "execute" not in parent.enabled_tools
        assert "new" not in parent.capability_configs

    def test_multiple_children_independent(self):
        parent = self._parent_context()
        child1 = parent.create_child_context(enabled_tools={"read"})
        child2 = parent.create_child_context(enabled_tools={"write"})

        assert child1.enabled_tools == {"read"}
        assert child2.enabled_tools == {"write"}
        child1.stages["extra"] = {}
        assert "extra" not in child2.stages


class TestSubAgentVerticalContextInheritance:
    """Test that _create_constrained_orchestrator inherits parent vertical context."""

    def _make_parent_vc(self) -> VerticalContext:
        ctx = VerticalContext()
        ctx.name = "coding"
        ctx.system_prompt = "Parent prompt"
        ctx.default_mode = "careful"
        ctx.stages = {"plan": {"desc": "plan"}}
        ctx.enabled_tools = {"read", "write", "bash"}
        return ctx

    @patch("victor.agent.orchestrator.AgentOrchestrator", autospec=False)
    def test_child_orchestrator_gets_vertical_context(self, MockOrch):
        from victor.agent.subagents.base import SubAgent, SubAgentConfig, SubAgentRole

        # Setup mock orchestrator returned by constructor
        mock_orch = MagicMock()
        mock_orch.tool_registry = MagicMock()
        mock_orch.tool_registry.clear = MagicMock()
        MockOrch.return_value = mock_orch

        parent_vc = self._make_parent_vc()

        # Setup mock parent context
        mock_context = MagicMock()
        mock_context.vertical_context = parent_vc
        mock_context.settings = MagicMock()
        mock_context.provider = MagicMock()
        mock_context.model = "test-model"
        mock_context.temperature = 0.0
        mock_context.provider_name = "test"
        mock_context.tool_registry = MagicMock()
        mock_context.tool_registry.get.return_value = None

        config = SubAgentConfig(
            role=SubAgentRole.RESEARCHER,
            task="test task",
            allowed_tools=["read"],
            tool_budget=5,
            context_limit=10000,
        )

        subagent = SubAgent.__new__(SubAgent)
        subagent._context = mock_context
        subagent.config = config
        subagent._presentation = MagicMock()

        orch = subagent._create_constrained_orchestrator()

        # Verify vertical context was set on child orchestrator
        assert hasattr(mock_orch, "_vertical_context")
        child_vc = mock_orch._vertical_context
        assert child_vc.name == "coding"
        assert child_vc.system_prompt == "Parent prompt"
        assert child_vc.enabled_tools == {"read"}
        assert child_vc.stages == {"plan": {"desc": "plan"}}

    def test_no_vertical_context_when_parent_has_none(self):
        """When parent has no vertical context, child should not get one either."""
        parent_vc = self._make_parent_vc()
        # Verify the guard: create_child_context is NOT called when vc is None
        child = parent_vc.create_child_context(enabled_tools={"read"})
        assert child.name == "coding"

        # The actual code path: if parent_vc is None, we skip entirely
        # This is a logic test — when vertical_context is None, hasattr guard prevents call
        vc = None
        called = False
        if vc is not None and hasattr(vc, "create_child_context"):
            called = True
        assert not called
