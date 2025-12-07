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

"""Tests for the unified TaskAnalyzer."""


from victor.agent.task_analyzer import (
    TaskAnalyzer,
    TaskAnalysis,
    get_task_analyzer,
    reset_task_analyzer,
)
from victor.agent.complexity_classifier import TaskComplexity
from victor.agent.action_authorizer import ActionIntent


class TestTaskAnalysis:
    """Tests for TaskAnalysis dataclass."""

    def test_is_simple(self):
        """Test is_simple property."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=2,
            complexity_confidence=0.9,
        )
        assert analysis.is_simple
        assert not analysis.is_complex

    def test_is_complex(self):
        """Test is_complex property."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=20,
            complexity_confidence=0.9,
        )
        assert analysis.is_complex
        assert not analysis.is_simple

    def test_is_generation(self):
        """Test is_generation property."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.GENERATION,
            tool_budget=5,
            complexity_confidence=0.9,
        )
        assert analysis.is_generation
        assert not analysis.is_simple

    def test_should_force_completion(self):
        """Test should_force_completion method."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=3,
            complexity_confidence=0.9,
        )
        assert not analysis.should_force_completion(2)
        assert analysis.should_force_completion(3)
        assert analysis.should_force_completion(5)


class TestTaskAnalyzer:
    """Tests for TaskAnalyzer class."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_task_analyzer()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_task_analyzer()

    def test_analyze_simple_query(self):
        """Test analyzing a simple query."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("What files are in this project?")

        assert isinstance(result, TaskAnalysis)
        assert result.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]
        assert result.tool_budget > 0

    def test_analyze_complex_query(self):
        """Test analyzing a complex query."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Refactor the authentication system to use JWT tokens")

        assert isinstance(result, TaskAnalysis)
        # Complex tasks should have higher budget
        assert result.tool_budget >= 5

    def test_analyze_write_request(self):
        """Test analyzing a write request."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Create a new file called hello.py")

        assert isinstance(result, TaskAnalysis)
        assert result.action_intent in [ActionIntent.WRITE_ALLOWED, ActionIntent.AMBIGUOUS]

    def test_analyze_display_request(self):
        """Test analyzing a display-only request."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Show me the contents of README.md")

        assert isinstance(result, TaskAnalysis)
        # Display requests typically don't authorize writes
        assert result.action_intent in [
            ActionIntent.DISPLAY_ONLY,
            ActionIntent.READ_ONLY,
            ActionIntent.AMBIGUOUS,
        ]

    def test_classify_complexity(self):
        """Test quick complexity classification."""
        analyzer = TaskAnalyzer()
        result = analyzer.classify_complexity("List all Python files")

        assert result.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]

    def test_check_write_authorization(self):
        """Test quick write authorization check."""
        analyzer = TaskAnalyzer()

        # Explicit write request with file path
        result = analyzer.check_write_authorization("Create and save a new file at hello.py")
        assert isinstance(result, bool)

        # Display request - should not authorize writes
        result2 = analyzer.check_write_authorization("Show me the contents of README.md")
        assert isinstance(result2, bool)

    def test_is_simple_query(self):
        """Test is_simple_query helper."""
        analyzer = TaskAnalyzer()

        # Returns boolean
        result = analyzer.is_simple_query("What is X?")
        assert isinstance(result, bool)

        # Complex query should not be simple
        result2 = analyzer.is_simple_query(
            "Analyze the entire codebase and refactor all authentication modules"
        )
        assert isinstance(result2, bool)

    def test_get_tool_budget(self):
        """Test get_tool_budget helper."""
        analyzer = TaskAnalyzer()

        budget = analyzer.get_tool_budget("Analyze the entire codebase")
        assert isinstance(budget, int)
        assert budget > 0

    def test_lazy_loading(self):
        """Test that classifiers are lazily loaded."""
        analyzer = TaskAnalyzer()

        # Initially none are loaded
        assert analyzer._complexity_classifier is None
        assert analyzer._action_authorizer is None

        # Access triggers loading
        _ = analyzer.complexity_classifier
        assert analyzer._complexity_classifier is not None

        _ = analyzer.action_authorizer
        assert analyzer._action_authorizer is not None


class TestGlobalAnalyzer:
    """Tests for global analyzer management."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_task_analyzer()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_task_analyzer()

    def test_get_task_analyzer(self):
        """Test getting global analyzer."""
        analyzer = get_task_analyzer()
        assert isinstance(analyzer, TaskAnalyzer)

    def test_get_task_analyzer_same_instance(self):
        """Test that get_task_analyzer returns same instance."""
        analyzer1 = get_task_analyzer()
        analyzer2 = get_task_analyzer()
        assert analyzer1 is analyzer2

    def test_reset_task_analyzer(self):
        """Test resetting global analyzer."""
        analyzer1 = get_task_analyzer()
        reset_task_analyzer()
        analyzer2 = get_task_analyzer()

        assert analyzer1 is not analyzer2
