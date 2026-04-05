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

"""Tests for ToolFacade domain facade."""

import pytest
from unittest.mock import MagicMock

from victor.agent.facades.tool_facade import ToolFacade
from victor.agent.facades.protocols import ToolFacadeProtocol


class TestToolFacadeInit:
    """Tests for ToolFacade initialization."""

    def test_init_with_all_components(self):
        """ToolFacade initializes with all components provided."""
        tools = MagicMock()
        pipeline = MagicMock()
        executor = MagicMock()
        selector = MagicMock()
        cache = MagicMock()

        facade = ToolFacade(
            tools=tools,
            tool_pipeline=pipeline,
            tool_executor=executor,
            tool_selector=selector,
            tool_cache=cache,
            tool_graph=MagicMock(),
            tool_registrar=MagicMock(),
            tool_budget=100,
            tool_output_formatter=MagicMock(),
            deduplication_tracker=MagicMock(),
            argument_normalizer=MagicMock(),
            parallel_executor=MagicMock(),
            safety_checker=MagicMock(),
            auto_committer=MagicMock(),
            middleware_chain=MagicMock(),
            code_correction_middleware=MagicMock(),
            tool_access_controller=MagicMock(),
            budget_manager=MagicMock(),
            search_router=MagicMock(),
            semantic_selector=MagicMock(),
            task_classifier=MagicMock(),
            sequence_tracker=MagicMock(),
            unified_tracker=MagicMock(),
            plugin_manager=MagicMock(),
        )

        assert facade.tools is tools
        assert facade.tool_pipeline is pipeline
        assert facade.tool_executor is executor
        assert facade.tool_selector is selector
        assert facade.tool_cache is cache
        assert facade.tool_budget == 100

    def test_init_with_minimal_components(self):
        """ToolFacade initializes with only required components."""
        tools = MagicMock()
        pipeline = MagicMock()
        executor = MagicMock()
        selector = MagicMock()

        facade = ToolFacade(
            tools=tools,
            tool_pipeline=pipeline,
            tool_executor=executor,
            tool_selector=selector,
        )

        assert facade.tools is tools
        assert facade.tool_pipeline is pipeline
        assert facade.tool_executor is executor
        assert facade.tool_selector is selector
        assert facade.tool_cache is None
        assert facade.tool_graph is None
        assert facade.tool_registrar is None
        assert facade.tool_budget == 50
        assert facade.tool_output_formatter is None
        assert facade.deduplication_tracker is None
        assert facade.argument_normalizer is None
        assert facade.parallel_executor is None
        assert facade.safety_checker is None
        assert facade.auto_committer is None
        assert facade.middleware_chain is None
        assert facade.code_correction_middleware is None
        assert facade.tool_access_controller is None
        assert facade.budget_manager is None
        assert facade.search_router is None
        assert facade.semantic_selector is None
        assert facade.task_classifier is None
        assert facade.sequence_tracker is None
        assert facade.unified_tracker is None
        assert facade.plugin_manager is None


class TestToolFacadeProperties:
    """Tests for ToolFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create a ToolFacade with mock components."""
        return ToolFacade(
            tools=MagicMock(name="tools"),
            tool_pipeline=MagicMock(name="pipeline"),
            tool_executor=MagicMock(name="executor"),
            tool_selector=MagicMock(name="selector"),
            tool_cache=MagicMock(name="cache"),
            tool_graph=MagicMock(name="graph"),
            tool_registrar=MagicMock(name="registrar"),
            tool_budget=75,
            tool_output_formatter=MagicMock(name="formatter"),
            deduplication_tracker=MagicMock(name="dedup"),
            argument_normalizer=MagicMock(name="normalizer"),
            parallel_executor=MagicMock(name="parallel"),
            safety_checker=MagicMock(name="safety"),
            auto_committer=MagicMock(name="committer"),
            middleware_chain=MagicMock(name="middleware"),
            code_correction_middleware=MagicMock(name="code_correction"),
            tool_access_controller=MagicMock(name="access"),
            budget_manager=MagicMock(name="budget"),
            search_router=MagicMock(name="router"),
            semantic_selector=MagicMock(name="semantic"),
            task_classifier=MagicMock(name="classifier"),
            sequence_tracker=MagicMock(name="tracker"),
            unified_tracker=MagicMock(name="unified"),
            plugin_manager=MagicMock(name="plugins"),
        )

    def test_tools_property(self, facade):
        """Tools property returns the registry."""
        assert facade.tools._mock_name == "tools"

    def test_tool_registry_alias(self, facade):
        """tool_registry property is an alias for tools."""
        assert facade.tool_registry is facade.tools

    def test_tool_pipeline_property(self, facade):
        """ToolPipeline property returns the pipeline."""
        assert facade.tool_pipeline._mock_name == "pipeline"

    def test_tool_executor_property(self, facade):
        """ToolExecutor property returns the executor."""
        assert facade.tool_executor._mock_name == "executor"

    def test_tool_selector_property(self, facade):
        """ToolSelector property returns the selector."""
        assert facade.tool_selector._mock_name == "selector"

    def test_tool_cache_property(self, facade):
        """ToolCache property returns the cache."""
        assert facade.tool_cache._mock_name == "cache"

    def test_tool_budget_property(self, facade):
        """ToolBudget property returns the budget."""
        assert facade.tool_budget == 75

    def test_tool_budget_setter(self, facade):
        """ToolBudget setter updates the budget."""
        facade.tool_budget = 200
        assert facade.tool_budget == 200

    def test_middleware_chain_setter(self, facade):
        """MiddlewareChain setter updates the chain."""
        new_chain = MagicMock(name="new_chain")
        facade.middleware_chain = new_chain
        assert facade.middleware_chain is new_chain

    def test_safety_checker_property(self, facade):
        """SafetyChecker property returns the checker."""
        assert facade.safety_checker._mock_name == "safety"

    def test_search_router_property(self, facade):
        """SearchRouter property returns the router."""
        assert facade.search_router._mock_name == "router"

    def test_semantic_selector_property(self, facade):
        """SemanticSelector property returns the selector."""
        assert facade.semantic_selector._mock_name == "semantic"

    def test_sequence_tracker_property(self, facade):
        """SequenceTracker property returns the tracker."""
        assert facade.sequence_tracker._mock_name == "tracker"

    def test_code_correction_middleware_property(self, facade):
        """CodeCorrectionMiddleware property returns the middleware."""
        assert facade.code_correction_middleware._mock_name == "code_correction"


class TestToolFacadeProtocolConformance:
    """Tests that ToolFacade satisfies ToolFacadeProtocol."""

    def test_satisfies_protocol(self):
        """ToolFacade structurally conforms to ToolFacadeProtocol."""
        facade = ToolFacade(
            tools=MagicMock(),
            tool_pipeline=MagicMock(),
            tool_executor=MagicMock(),
            tool_selector=MagicMock(),
        )
        assert isinstance(facade, ToolFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on ToolFacade."""
        required = [
            "tools",
            "tool_pipeline",
            "tool_executor",
            "tool_selector",
            "tool_cache",
        ]
        facade = ToolFacade(
            tools=MagicMock(),
            tool_pipeline=MagicMock(),
            tool_executor=MagicMock(),
            tool_selector=MagicMock(),
        )
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
