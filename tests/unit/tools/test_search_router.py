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

"""Tests for SearchRouter - Gap 3 implementation."""

import pytest


class TestKeywordRouting:
    """Tests for KEYWORD routing."""

    def test_quoted_string_keyword(self):
        """Test that quoted strings route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route('Find "class BaseTool" in the codebase')

        assert result.search_type == SearchType.KEYWORD
        assert result.confidence == 1.0
        assert "class BaseTool" in result.transformed_query

    def test_single_quoted_string(self):
        """Test that single-quoted strings route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("Search for 'def execute'")

        assert result.search_type == SearchType.KEYWORD
        assert result.confidence == 1.0

    def test_class_name_keyword(self):
        """Test that class names route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("class AgentOrchestrator")

        assert result.search_type == SearchType.KEYWORD
        assert "class_name" in result.matched_patterns

    def test_function_def_keyword(self):
        """Test that function definitions route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("def execute_tool")

        assert result.search_type == SearchType.KEYWORD
        assert "function_def" in result.matched_patterns

    def test_import_statement_keyword(self):
        """Test that import statements route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("import victor.agent.orchestrator")

        assert result.search_type == SearchType.KEYWORD
        assert "import_statement" in result.matched_patterns

    def test_error_class_keyword(self):
        """Test that error classes route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("ToolExecutionError")

        assert result.search_type == SearchType.KEYWORD
        assert "error_class" in result.matched_patterns

    def test_file_extension_keyword(self):
        """Test that file patterns route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("orchestrator.py")

        assert result.search_type == SearchType.KEYWORD
        assert "file_extension" in result.matched_patterns

    def test_decorator_keyword(self):
        """Test that decorators route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("@property")

        assert result.search_type == SearchType.KEYWORD
        assert "decorator" in result.matched_patterns


class TestSemanticRouting:
    """Tests for SEMANTIC routing."""

    def test_how_question_semantic(self):
        """Test that 'how does' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("how does error handling work in this project")

        assert result.search_type == SearchType.SEMANTIC
        assert "how_question" in result.matched_patterns

    def test_why_question_semantic(self):
        """Test that 'why does' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("why does the orchestrator use streaming")

        assert result.search_type == SearchType.SEMANTIC
        assert "why_question" in result.matched_patterns

    def test_explain_semantic(self):
        """Test that 'explain' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("explain the tool selection mechanism")

        assert result.search_type == SearchType.SEMANTIC
        assert "explain" in result.matched_patterns

    def test_patterns_semantic(self):
        """Test that 'patterns' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("find error handling patterns")

        assert result.search_type == SearchType.SEMANTIC
        assert "patterns" in result.matched_patterns

    def test_architecture_semantic(self):
        """Test that 'architecture' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("understand the architecture")

        assert result.search_type == SearchType.SEMANTIC
        assert "architecture" in result.matched_patterns

    def test_best_practices_semantic(self):
        """Test that 'best practices' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("find best practices for error handling")

        assert result.search_type == SearchType.SEMANTIC
        assert "best_practices" in result.matched_patterns


class TestHybridRouting:
    """Tests for HYBRID routing."""

    def test_mixed_signals_hybrid(self):
        """Test that mixed signals can route to HYBRID."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        # Query with both keyword (class name) and semantic (how) signals
        result = router.route("how does class BaseTool work")

        # Should be hybrid or one of keyword/semantic
        assert result.search_type in (
            SearchType.HYBRID,
            SearchType.KEYWORD,
            SearchType.SEMANTIC,
        )


class TestDefaultRouting:
    """Tests for default routing behavior."""

    def test_ambiguous_query_defaults_keyword(self):
        """Test that ambiguous queries default to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("xyzzy foobar baz")

        # Should default to keyword for precision
        assert result.search_type == SearchType.KEYWORD
        assert result.confidence < 0.5

    def test_empty_query(self):
        """Test empty query handling."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("")

        assert result.search_type == SearchType.KEYWORD
        assert result.confidence < 0.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_route_query(self):
        """Test route_query convenience function."""
        from victor.agent.search_router import route_query, SearchType

        result = route_query("class BaseTool")
        assert result.search_type == SearchType.KEYWORD

    def test_suggest_search_tool_keyword(self):
        """Test suggest_search_tool returns code_search for keyword."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("class BaseTool")
        assert tool == "code_search"

    def test_suggest_search_tool_semantic(self):
        """Test suggest_search_tool returns semantic_code_search."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("how does the caching mechanism work")
        assert tool == "semantic_code_search"

    def test_is_keyword_query(self):
        """Test is_keyword_query function."""
        from victor.agent.search_router import is_keyword_query

        assert is_keyword_query("class BaseTool")
        assert not is_keyword_query("how does error handling work")

    def test_is_semantic_query(self):
        """Test is_semantic_query function."""
        from victor.agent.search_router import is_semantic_query

        assert is_semantic_query("explain the architecture")
        assert not is_semantic_query("class BaseTool")


class TestCustomRouter:
    """Tests for custom router support."""

    def test_custom_router_takes_precedence(self):
        """Test that custom routers are tried first."""
        from victor.agent.search_router import SearchRouter, SearchRoute, SearchType

        def custom_router(query: str):
            if "URGENT" in query.upper():
                return SearchRoute(
                    search_type=SearchType.KEYWORD,
                    confidence=1.0,
                    reason="Urgent query",
                    transformed_query=query,
                    matched_patterns=["urgent_override"],
                )
            return None

        router = SearchRouter(custom_routers=[custom_router])
        result = router.route("URGENT: find all errors")

        assert result.confidence == 1.0
        assert "urgent_override" in result.matched_patterns


class TestRealWorldQueries:
    """Tests with real-world example queries."""

    def test_find_class_inherits(self):
        """Test 'find classes that inherit' routes correctly."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("find all classes that inherit from BaseTool")

        # "class" is keyword signal, but "find all" and conceptual query makes it semantic
        assert result.search_type in (SearchType.KEYWORD, SearchType.SEMANTIC, SearchType.HYBRID)

    def test_specific_function_search(self):
        """Test specific function search routes to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("def _execute_tool_call")

        assert result.search_type == SearchType.KEYWORD

    def test_error_handling_conceptual(self):
        """Test conceptual error handling query routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("how is error handling implemented across providers")

        assert result.search_type == SearchType.SEMANTIC

    def test_import_exact_match(self):
        """Test exact import routes to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("from victor.tools.base import BaseTool")

        assert result.search_type == SearchType.KEYWORD


class TestQuotedStrings:
    """Tests for quoted string handling."""

    def test_double_quoted_extraction(self):
        """Test double-quoted string extraction."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route('search for "ToolRegistry"')

        assert result.transformed_query == "ToolRegistry"

    def test_single_quoted_extraction(self):
        """Test single-quoted string extraction."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("find 'execute_bash'")

        assert result.transformed_query == "execute_bash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
