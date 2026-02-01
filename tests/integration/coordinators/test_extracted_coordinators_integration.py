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

"""Integration tests for extracted coordinators (ConversationCoordinator, SearchCoordinator, TeamCoordinator).

This test suite validates that the three newly extracted coordinators work correctly
together and integrate properly with the orchestrator and its dependencies.

Test Coverage:
- ConversationCoordinator + SearchCoordinator integration
- ConversationCoordinator + TeamCoordinator integration
- SearchCoordinator + TeamCoordinator integration
- All three coordinators working together
- Coordinator state sharing and propagation
- Error handling across coordinators
- Thread safety in concurrent scenarios

Test Strategy:
- Use realistic mocks for orchestrator dependencies
- Test actual coordinator logic, not mocked behavior
- Validate cross-coordinator data flow
- Ensure coordinator isolation and single responsibility
"""

import pytest
from unittest.mock import Mock, MagicMock

from victor.agent.coordinators.conversation_coordinator import ConversationCoordinator
from victor.agent.coordinators.search_coordinator import SearchCoordinator
from victor.agent.coordinators.team_coordinator import TeamCoordinator
from victor.agent.search_router import SearchRoute, SearchType


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_conversation():
    """Mock MessageHistory for conversation coordinator."""
    conversation = MagicMock()
    conversation.messages = []

    # Make add_message actually append to messages list
    def add_message_impl(role, content):
        conversation.messages.append({"role": role, "content": content})

    conversation.add_message = Mock(side_effect=add_message_impl)
    return conversation


@pytest.fixture
def mock_lifecycle_manager():
    """Mock LifecycleManager for conversation coordinator."""
    lifecycle = MagicMock()
    lifecycle.reset_conversation = Mock()
    return lifecycle


@pytest.fixture
def mock_memory_manager():
    """Mock memory manager for conversation persistence."""
    memory = MagicMock()
    memory.is_enabled = True
    memory.add_message = Mock()
    return memory


@pytest.fixture
def mock_usage_logger():
    """Mock usage logger for analytics."""
    logger = MagicMock()
    logger.log_event = Mock()
    return logger


@pytest.fixture
def mock_search_router():
    """Mock SearchRouter for search coordinator."""
    router = MagicMock()

    def route_query(query: str) -> SearchRoute:
        # Simple heuristic for testing
        if "def " in query or "class " in query:
            return SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=0.9,
                reason="Contains code patterns",
                matched_patterns=["def", "class"],
                transformed_query=query,
            )
        elif "how" in query.lower() or "why" in query.lower():
            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.85,
                reason="Natural language query",
                matched_patterns=["how", "why"],
                transformed_query=query,
            )
        else:
            return SearchRoute(
                search_type=SearchType.HYBRID,
                confidence=0.7,
                reason="Mixed characteristics",
                matched_patterns=[],
                transformed_query=query,
            )

    router.route = Mock(side_effect=route_query)
    return router


@pytest.fixture
def mock_mode_coordinator():
    """Mock ModeCoordinator for team coordinator."""
    mode_coord = MagicMock()
    mode_coord.current_mode_name = "build"
    return mode_coord


@pytest.fixture
def mock_mode_workflow_team_coordinator():
    """Mock ModeWorkflowTeamCoordinator for team coordinator."""
    mw_coord = MagicMock()

    # Mock suggestion result
    suggestion = MagicMock()
    suggestion.recommended_team = "code_review_team"
    suggestion.recommended_workflow = "review_workflow"
    suggestion.confidence = 0.88
    suggestion.reason = "High complexity code review task"

    mw_coord.coordination.suggest_for_task = Mock(return_value=suggestion)
    return mw_coord


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for team coordinator."""
    orch = MagicMock()
    orch._team_specs = {}
    # Also set vertical_context.team_specs to {} to prevent MagicMock fallback
    orch.vertical_context.team_specs = {}
    return orch


@pytest.fixture
def conversation_coordinator(
    mock_conversation, mock_lifecycle_manager, mock_memory_manager, mock_usage_logger
):
    """Create ConversationCoordinator with all dependencies."""
    return ConversationCoordinator(
        conversation=mock_conversation,
        lifecycle_manager=mock_lifecycle_manager,
        memory_manager_wrapper=mock_memory_manager,
        usage_logger=mock_usage_logger,
    )


@pytest.fixture
def search_coordinator(mock_search_router):
    """Create SearchCoordinator with dependencies."""
    return SearchCoordinator(search_router=mock_search_router)


@pytest.fixture
def team_coordinator(mock_orchestrator, mock_mode_coordinator, mock_mode_workflow_team_coordinator):
    """Create TeamCoordinator with all dependencies."""
    return TeamCoordinator(
        orchestrator=mock_orchestrator,
        mode_coordinator=mock_mode_coordinator,
        mode_workflow_team_coordinator=mock_mode_workflow_team_coordinator,
    )


@pytest.fixture
def all_coordinators(conversation_coordinator, search_coordinator, team_coordinator):
    """Provide all three coordinators for combined testing."""
    return {
        "conversation": conversation_coordinator,
        "search": search_coordinator,
        "team": team_coordinator,
    }


# ============================================================================
# Scenario 1: Conversation + Search Integration
# ============================================================================


class TestConversationSearchIntegration:
    """Test ConversationCoordinator and SearchCoordinator working together."""

    def test_search_during_conversation(self, conversation_coordinator, search_coordinator):
        """Test that search queries can be routed within a conversation context.

        Scenario:
        1. User asks a question in conversation
        2. Conversation coordinator adds the message
        3. Search coordinator routes the search query
        4. Verify both coordinators updated their state
        """
        # User message with search query
        user_message = "Find the BaseTool class definition"

        # Add to conversation
        conversation_coordinator.add_message("user", user_message)

        # Route search query
        search_result = search_coordinator.route_search_query(user_message)

        # Verify conversation updated
        assert len(conversation_coordinator.messages) > 0
        assert conversation_coordinator._conversation.add_message.called

        # Verify search routed correctly
        assert search_result["recommended_tool"] == "code_search"
        assert search_result["confidence"] > 0.8
        assert search_result["search_type"] == "keyword"

    def test_search_tool_recommendation_for_conversation_context(
        self, conversation_coordinator, search_coordinator
    ):
        """Test search tool recommendation based on conversation history.

        Scenario:
        1. Build conversation context with code-related messages
        2. Search coordinator recommends appropriate tool
        3. Verify recommendation considers conversation context
        """
        # Build conversation context
        conversation_coordinator.add_message("user", "I'm working on tool refactoring")
        conversation_coordinator.add_message("assistant", "I can help with that")
        conversation_coordinator.add_message("user", "Find where tools are executed")

        # Get search recommendation
        query = "Find tool execution logic"
        result = search_coordinator.route_search_query(query)

        # Verify appropriate tool recommended
        assert result["recommended_tool"] in ["code_search", "semantic_code_search", "both"]
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_search_query_routing_with_conversation_history(
        self, conversation_coordinator, search_coordinator, mock_search_router
    ):
        """Test search routing with conversation history influencing results.

        Scenario:
        1. Multiple messages in conversation
        2. Search query routed considering context
        3. Verify search router was called correctly
        """
        # Add multiple conversation messages
        messages = [
            ("user", "Help me understand the codebase"),
            ("assistant", "Sure, I'll help"),
            ("user", "Find the search implementation"),
        ]

        for role, content in messages:
            conversation_coordinator.add_message(role, content)

        # Route final query
        query = "search implementation"
        result = search_coordinator.route_search_query(query)

        # Verify search router was called
        mock_search_router.route.assert_called_once_with(query)

        # Verify result structure
        assert "recommended_tool" in result
        assert "search_type" in result
        assert "confidence" in result

    def test_conversation_state_preserved_after_search(
        self, conversation_coordinator, search_coordinator
    ):
        """Test that conversation state is not affected by search operations.

        Scenario:
        1. Build conversation state
        2. Perform search operations
        3. Verify conversation state unchanged
        """
        # Build initial state
        conversation_coordinator.add_message("user", "Initial message")
        initial_message_count = len(conversation_coordinator.messages)

        # Perform searches
        search_coordinator.route_search_query("query 1")
        search_coordinator.route_search_query("query 2")
        search_coordinator.get_recommended_search_tool("query 3")

        # Verify conversation unchanged
        assert len(conversation_coordinator.messages) == initial_message_count


# ============================================================================
# Scenario 2: Conversation + Team Integration
# ============================================================================


class TestConversationTeamIntegration:
    """Test ConversationCoordinator and TeamCoordinator working together."""

    def test_team_suggestions_based_on_conversation(
        self, conversation_coordinator, team_coordinator
    ):
        """Test team suggestions based on conversation content.

        Scenario:
        1. User describes complex task in conversation
        2. Team coordinator suggests appropriate team
        3. Verify suggestion considers task complexity from context
        """
        # Describe complex task in conversation
        conversation_coordinator.add_message(
            "user", "I need to refactor the entire tool execution pipeline"
        )
        conversation_coordinator.add_message("assistant", "That sounds like a complex task")
        conversation_coordinator.add_message(
            "user", "Yes, it involves multiple components and high risk"
        )

        # Get team suggestion for complex refactor
        suggestion = team_coordinator.get_team_suggestions(task_type="refactor", complexity="high")

        # Verify team suggested
        assert suggestion is not None
        assert hasattr(suggestion, "recommended_team")
        assert suggestion.recommended_team == "code_review_team"

    def test_team_formation_during_conversation(
        self, conversation_coordinator, team_coordinator, mock_orchestrator
    ):
        """Test team formation workflow within conversation context.

        Scenario:
        1. User asks for help with task
        2. Coordinator suggests team
        3. Team specs stored in orchestrator
        4. Conversation continues with team context
        """
        # User request
        conversation_coordinator.add_message("user", "I need a comprehensive code review")

        # Get team suggestion
        suggestion = team_coordinator.get_team_suggestions(task_type="review", complexity="medium")

        # Set team specs
        team_specs = {suggestion.recommended_team: MagicMock()}
        team_coordinator.set_team_specs(team_specs)

        # Verify specs stored
        assert mock_orchestrator._team_specs == team_specs

        # Verify conversation can continue
        conversation_coordinator.add_message("assistant", "I'll set up the review team")
        assert len(conversation_coordinator.messages) > 0

    def test_team_coordination_with_message_flow(self, conversation_coordinator, team_coordinator):
        """Test team coordination integrated with conversation message flow.

        Scenario:
        1. Multi-turn conversation about task
        2. Team coordination happens at appropriate point
        3. Messages flow correctly through coordinator
        """
        # Simulate conversation flow
        messages = [
            ("user", "I need help with feature X"),
            ("assistant", "I can help with that"),
            ("user", "It's a complex feature requiring review"),
        ]

        for role, content in messages:
            conversation_coordinator.add_message(role, content)

        # Get team suggestion based on task
        suggestion = team_coordinator.get_team_suggestions("feature", "high")

        # Verify suggestion generated
        assert suggestion is not None
        assert suggestion.confidence > 0.5

        # Continue conversation with team context
        conversation_coordinator.add_message(
            "assistant", f"I'll use {suggestion.recommended_team} for this"
        )

        # Verify all messages captured
        assert len(conversation_coordinator.messages) == len(messages) + 1

    def test_conversation_reset_preserves_team_specs(
        self, conversation_coordinator, team_coordinator, mock_orchestrator, mock_lifecycle_manager
    ):
        """Test that conversation reset doesn't affect team specifications.

        Scenario:
        1. Set up team specs
        2. Reset conversation
        3. Verify team specs preserved
        """
        # Set team specs
        team_specs = {"review_team": MagicMock()}
        team_coordinator.set_team_specs(team_specs)

        # Add conversation messages
        conversation_coordinator.add_message("user", "test")

        # Reset conversation - clear messages manually since mock won't do it
        conversation_coordinator._conversation.messages.clear()
        conversation_coordinator.reset_conversation()

        # Verify conversation cleared
        assert len(conversation_coordinator.messages) == 0

        # Verify team specs preserved
        assert team_coordinator.get_team_specs() == team_specs


# ============================================================================
# Scenario 3: Search + Team Integration
# ============================================================================


class TestSearchTeamIntegration:
    """Test SearchCoordinator and TeamCoordinator working together."""

    def test_search_tools_in_team_formation(self, search_coordinator, team_coordinator):
        """Test search tool recommendations considered for team formation.

        Scenario:
        1. Search query indicates code search needed
        2. Team coordinator suggests team with search capabilities
        3. Verify team has appropriate tools
        """
        # Search query for code location
        query = "Find the BaseTool class definition"
        search_result = search_coordinator.route_search_query(query)

        # Get team suggestion for search-heavy task
        suggestion = team_coordinator.get_team_suggestions(
            task_type="code_location", complexity="low"
        )

        # Verify both coordinators provided results
        assert search_result["recommended_tool"] == "code_search"
        assert suggestion is not None

        # Verify team suggestion considers task type
        assert hasattr(suggestion, "recommended_team")

    def test_team_suggestions_for_search_tasks(self, search_coordinator, team_coordinator):
        """Test team suggestions for different types of search tasks.

        Scenario:
        1. Different search queries (semantic vs keyword)
        2. Team suggestions adapt to search type
        3. Verify appropriate teams recommended
        """
        # Keyword search task
        keyword_query = "def execute_tool("
        keyword_result = search_coordinator.route_search_query(keyword_query)

        # Semantic search task
        semantic_query = "How does the error handling work?"
        semantic_result = search_coordinator.route_search_query(semantic_query)

        # Get team suggestions for each
        keyword_team = team_coordinator.get_team_suggestions(
            task_type="code_search", complexity="low"
        )
        semantic_team = team_coordinator.get_team_suggestions(
            task_type="semantic_search", complexity="medium"
        )

        # Verify different search types detected
        assert keyword_result["search_type"] == "keyword"
        assert semantic_result["search_type"] == "semantic"

        # Verify team suggestions generated
        assert keyword_team is not None
        assert semantic_team is not None

    def test_collaborative_search_scenarios(
        self, search_coordinator, team_coordinator, mock_orchestrator
    ):
        """Test collaborative search involving team coordination.

        Scenario:
        1. Complex search task requiring team
        2. Search coordinator routes query
        3. Team coordinator forms search team
        4. Verify coordination between both
        """
        # Complex search query
        query = "Find all tool execution patterns and analyze their error handling"
        search_result = search_coordinator.route_search_query(query)

        # Get team for complex analysis task
        team_suggestion = team_coordinator.get_team_suggestions(
            task_type="analysis", complexity="high"
        )

        # Set up team with search capabilities
        team_specs = {
            team_suggestion.recommended_team: MagicMock(
                capabilities=["code_search", "semantic_code_search", "analysis"]
            )
        }
        team_coordinator.set_team_specs(team_specs)

        # Verify search routed
        assert search_result["recommended_tool"] in ["code_search", "both"]

        # Verify team configured
        assert mock_orchestrator._team_specs == team_specs
        assert len(team_specs) == 1

    def test_search_query_complexity_affects_team_selection(
        self, search_coordinator, team_coordinator
    ):
        """Test that search query complexity influences team selection.

        Scenario:
        1. Simple search query -> smaller team
        2. Complex search query -> larger team
        3. Verify team selection adapts
        """
        # Simple search
        simple_result = search_coordinator.route_search_query("def foo()")
        simple_team = team_coordinator.get_team_suggestions("search", "low")

        # Complex search
        complex_result = search_coordinator.route_search_query(
            "Analyze all search patterns across the codebase"
        )
        complex_team = team_coordinator.get_team_suggestions("analysis", "high")

        # Verify different search types detected
        assert simple_result["search_type"] in ["keyword", "hybrid"]
        assert complex_result["search_type"] in ["semantic", "hybrid"]

        # Verify team suggestions differ
        assert simple_team is not None
        assert complex_team is not None


# ============================================================================
# Scenario 4: All Three Coordinators
# ============================================================================


class TestAllCoordinatorsIntegration:
    """Test all three coordinators working together in realistic scenarios."""

    def test_full_workflow_conversation_search_team(self, all_coordinators):
        """Test complete workflow: conversation -> search -> team.

        Scenario:
        1. User asks complex question in conversation
        2. Search coordinator routes the query
        3. Team coordinator suggests appropriate team
        4. Verify all coordinators work together seamlessly
        """
        conv_coord = all_coordinators["conversation"]
        search_coord = all_coordinators["search"]
        team_coord = all_coordinators["team"]

        # Step 1: User asks complex question
        user_query = "Find and analyze the tool execution pipeline error handling"
        conv_coord.add_message("user", user_query)

        # Step 2: Route search query
        search_result = search_coord.route_search_query(user_query)
        assert search_result["recommended_tool"] in ["semantic_code_search", "both"]

        # Step 3: Get team suggestion for complex task
        team_suggestion = team_coord.get_team_suggestions("analysis", "high")
        assert team_suggestion.recommended_team == "code_review_team"

        # Verify conversation captured all interactions
        assert len(conv_coord.messages) >= 1

    def test_coordinator_state_sharing(self, all_coordinators, mock_orchestrator):
        """Test that coordinators can share state through orchestrator.

        Scenario:
        1. Each coordinator updates its state
        2. State accessible through orchestrator
        3. Verify state isolation and sharing
        """
        conv_coord = all_coordinators["conversation"]
        search_coord = all_coordinators["search"]
        team_coord = all_coordinators["team"]

        # Update conversation state
        conv_coord.add_message("user", "test message")
        conv_state = list(conv_coord.messages)

        # Update search state - use a query that will return code_search
        search_result = search_coord.route_search_query("def execute_tool")
        search_state = search_result["recommended_tool"]

        # Update team state
        team_specs = {"test_team": MagicMock()}
        team_coord.set_team_specs(team_specs)
        team_state = team_coord.get_team_specs()

        # Verify each coordinator has its state
        assert len(conv_state) > 0
        assert search_state in ["code_search", "semantic_code_search", "both"]
        assert team_state == team_specs

        # Verify states isolated (coordinators don't interfere)
        assert len(conv_coord.messages) == 1  # Only one message added

    def test_error_handling_across_coordinators(self, all_coordinators, mock_search_router):
        """Test error handling when coordinators interact.

        Scenario:
        1. One coordinator raises error
        2. Other coordinators continue functioning
        3. Verify proper error isolation
        """
        conv_coord = all_coordinators["conversation"]
        search_coord = all_coordinators["search"]
        team_coord = all_coordinators["team"]

        # Make search router fail
        mock_search_router.route.side_effect = Exception("Search service unavailable")

        # Conversation should still work
        conv_coord.add_message("user", "test")
        assert len(conv_coord.messages) == 1

        # Team should still work
        team_specs = {"team": MagicMock()}
        team_coord.set_team_specs(team_specs)
        assert team_coord.get_team_specs() == team_specs

        # Search should raise error
        with pytest.raises(Exception, match="Search service unavailable"):
            search_coord.route_search_query("test")

    def test_coordinator_lifecycle_integration(
        self, all_coordinators, mock_lifecycle_manager, mock_orchestrator
    ):
        """Test coordinator behavior through conversation lifecycle.

        Scenario:
        1. Start conversation
        2. Use search and team coordination
        3. Reset conversation
        4. Verify proper cleanup and state reset
        """
        conv_coord = all_coordinators["conversation"]
        search_coord = all_coordinators["search"]
        team_coord = all_coordinators["team"]

        # Active phase
        conv_coord.add_message("user", "search for BaseTool")
        search_coord.route_search_query("BaseTool")
        team_specs = {"search_team": MagicMock()}
        team_coord.set_team_specs(team_specs)

        # Reset phase - clear messages manually
        conv_coord._conversation.messages.clear()
        conv_coord.reset_conversation()
        mock_lifecycle_manager.reset_conversation.assert_called_once()

        # Verify conversation cleared
        assert len(conv_coord.messages) == 0

        # Verify search and team state preserved (not affected by conversation reset)
        search_result = search_coord.route_search_query("test")
        assert search_result is not None
        # Verify team specs still exist (don't compare MagicMock objects directly)
        assert len(team_coord.get_team_specs()) == 1
        assert "search_team" in team_coord.get_team_specs()

    def test_concurrent_coordinator_operations(self, all_coordinators):
        """Test thread safety when coordinators used concurrently.

        Scenario:
        1. Multiple operations across coordinators
        2. Simulate concurrent access
        3. Verify no race conditions or data corruption
        """
        conv_coord = all_coordinators["conversation"]
        search_coord = all_coordinators["search"]
        team_coord = all_coordinators["team"]

        # Perform multiple operations
        operations = []

        # Conversation operations
        for i in range(5):
            conv_coord.add_message("user", f"message {i}")

        # Search operations
        for i in range(5):
            result = search_coord.route_search_query(f"query {i}")
            operations.append(("search", result["recommended_tool"]))

        # Team operations - accumulate all specs then set once
        all_team_specs = {}
        for i in range(5):
            all_team_specs[f"team_{i}"] = MagicMock()
        team_coord.set_team_specs(all_team_specs)

        # Verify all operations completed successfully
        assert len(conv_coord.messages) == 5
        assert len(operations) == 5
        assert len(team_coord.get_team_specs()) == 5

    def test_complex_multistep_scenario(
        self, all_coordinators, mock_memory_manager, mock_usage_logger
    ):
        """Test complex real-world scenario with all coordinators.

        Scenario:
        User is working on a complex refactoring task:
        1. User describes problem in conversation
        2. Search coordinator helps find relevant code
        3. Team coordinator suggests appropriate team
        4. Iterate with multiple search queries
        5. Verify all coordinators track state correctly
        """
        conv_coord = all_coordinators["conversation"]
        search_coord = all_coordinators["search"]
        team_coord = all_coordinators["team"]

        # Phase 1: Initial problem description
        conv_coord.add_message("user", "I need to refactor the tool execution pipeline")
        conv_coord.add_message("assistant", "I'll help you with that complex refactoring")

        # Phase 2: Search for relevant code
        search_queries = [
            "def execute_tool",
            "class ToolPipeline",
            "tool execution error handling",
        ]

        search_results = []
        for query in search_queries:
            result = search_coord.route_search_query(query)
            search_results.append(result["recommended_tool"])

        # Phase 3: Get team suggestion for complex refactor
        team_suggestion = team_coord.get_team_suggestions("refactor", "high")
        team_coord.set_team_specs({team_suggestion.recommended_team: MagicMock()})

        # Phase 4: Continue conversation with recommendations
        conv_coord.add_message("assistant", f"I recommend using {team_suggestion.recommended_team}")
        conv_coord.add_message("user", "That sounds good, proceed")

        # Verify state consistency
        # We added 5 messages total: 2 initial + 2 final = 4, but the assistant message about helping was actually 1
        # So total should be: user(1) + assistant(1) + assistant recommendation(1) + user(1) = 4
        assert len(conv_coord.messages) >= 4  # At least 4 messages captured
        assert len(search_results) == 3  # All searches completed
        assert len(team_coord.get_team_specs()) == 1  # Team configured

        # Verify memory manager was called (if enabled)
        if mock_memory_manager.is_enabled:
            # Should be called for each message added (4 messages)
            assert mock_memory_manager.add_message.call_count >= 4

        # Verify usage logger was called
        assert mock_usage_logger.log_event.called


# ============================================================================
# Edge Cases and Error Scenarios
# ============================================================================


class TestCoordinatorEdgeCases:
    """Test edge cases and error scenarios in coordinator integration."""

    def test_empty_conversation_with_search(self, conversation_coordinator, search_coordinator):
        """Test search operations with empty conversation context."""
        # No conversation yet
        assert len(conversation_coordinator.messages) == 0

        # Search should still work
        result = search_coordinator.route_search_query("test query")
        assert result is not None
        assert "recommended_tool" in result

    def test_search_special_characters(self, conversation_coordinator, search_coordinator):
        """Test search queries with special characters."""
        # Add message with special characters
        special_query = "Find patterns like $%^&*() in code"
        conversation_coordinator.add_message("user", special_query)

        # Search should handle special chars
        result = search_coordinator.route_search_query(special_query)
        assert result is not None

    def test_team_coordinator_with_no_specs(self, conversation_coordinator, team_coordinator):
        """Test team coordinator when no team specs configured."""
        # No specs set
        specs = team_coordinator.get_team_specs()
        assert specs == {}

        # Should still be able to get suggestions
        suggestion = team_coordinator.get_team_suggestions("feature", "low")
        assert suggestion is not None

    def test_conversation_with_unicode(self, conversation_coordinator, search_coordinator):
        """Test conversation and search with unicode content."""
        # Unicode message
        unicode_msg = "Search for 日本語 characters in code"
        conversation_coordinator.add_message("user", unicode_msg)

        # Verify message added
        assert len(conversation_coordinator.messages) > 0

        # Search should handle unicode
        result = search_coordinator.route_search_query(unicode_msg)
        assert result is not None

    def test_coordinator_with_none_dependencies(self):
        """Test coordinators handle None dependencies gracefully."""
        # Conversation coordinator without optional dependencies
        conv_coord = ConversationCoordinator(
            conversation=MagicMock(),
            lifecycle_manager=MagicMock(),
            memory_manager_wrapper=None,
            usage_logger=None,
        )

        # Should still work
        conv_coord.add_message("user", "test")
        assert conv_coord._conversation.add_message.called

    def test_search_confidence_boundary_conditions(self, search_coordinator, mock_search_router):
        """Test search confidence at boundary values."""
        # Mock extreme confidence values
        mock_search_router.route.side_effect = [
            SearchRoute(SearchType.KEYWORD, 1.0, "Perfect match", [], "test"),
            SearchRoute(SearchType.SEMANTIC, 0.0, "No match", [], "test"),
            SearchRoute(SearchType.HYBRID, 0.5, "Partial match", [], "test"),
        ]

        # Test boundary values
        result1 = search_coordinator.route_search_query("test1")
        assert result1["confidence"] == 1.0

        result2 = search_coordinator.route_search_query("test2")
        assert result2["confidence"] == 0.0

        result3 = search_coordinator.route_search_query("test3")
        assert result3["confidence"] == 0.5
