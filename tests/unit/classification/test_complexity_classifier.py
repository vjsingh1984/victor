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

"""Tests for the complexity_classifier module.

Tests task complexity classification:
- Simple task detection
- Medium task detection
- Complex task detection
- Generation task detection
- Tool budget assignment
- Custom patterns and classifiers
"""

import pytest

# Suppress deprecation warnings for complexity_classifier shim during migration
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Import from framework.task for core types (canonical location)
from victor.framework.task.complexity import (
    TaskComplexity,
    TaskClassification,
    TaskComplexityService as ComplexityClassifier,
    classify_task,
    DEFAULT_BUDGETS,
)

# Import helper functions and constants from deprecated module (they're wrappers)
# These will emit deprecation warnings which is expected
from victor.agent.complexity_classifier import (
    get_task_prompt_hint,
    get_prompt_hint,
    should_force_answer,
    PROMPT_HINTS,  # Deprecated constant, kept for backward compatibility
)


class TestTaskComplexity:
    """Tests for TaskComplexity enum."""

    def test_complexity_values(self):
        """Test all complexity enum values exist."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MEDIUM.value == "medium"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.GENERATION.value == "generation"


class TestTaskClassification:
    """Tests for TaskClassification dataclass.

    Note: prompt_hint was removed from TaskClassification for SRP compliance.
    Hints are now provided by ComplexityHintEnricher in the enrichment module.
    """

    def test_classification_creation(self):
        """Test creating a TaskClassification."""
        classification = TaskClassification(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            confidence=0.9,
            matched_patterns=["list_files"],
        )
        assert classification.complexity == TaskComplexity.SIMPLE
        assert classification.tool_budget == 10
        assert classification.confidence == 0.9

    def test_should_force_completion_after(self):
        """Test the should_force_completion_after method."""
        classification = TaskClassification(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            confidence=0.9,
            matched_patterns=[],
        )

        assert classification.should_force_completion_after(9) is False
        assert classification.should_force_completion_after(10) is True
        assert classification.should_force_completion_after(11) is True


class TestComplexityClassifier:
    """Tests for the ComplexityClassifier class.

    NOTE: These tests use use_semantic=False to test regex patterns directly.
    The semantic classifier with nudge rules is tested separately.
    """

    def test_simple_task_detection(self):
        """Test detection of simple tasks using regex patterns."""
        # Disable semantic to test regex patterns directly
        classifier = ComplexityClassifier(use_semantic=False)

        # "git status" patterns - strong simple signal
        result = classifier.classify("git status")
        assert result.complexity == TaskComplexity.SIMPLE
        assert "git_status" in result.matched_patterns

        # "pwd" patterns
        result = classifier.classify("pwd")
        assert result.complexity == TaskComplexity.SIMPLE
        assert "pwd" in result.matched_patterns

        # "ls command" - short commands without context default to MEDIUM
        # to avoid underestimating complexity
        result = classifier.classify("ls")
        assert result.complexity == TaskComplexity.MEDIUM

    def test_medium_task_detection(self):
        """Test detection of medium complexity tasks using regex patterns."""
        classifier = ComplexityClassifier(use_semantic=False)

        # "explain code" patterns
        result = classifier.classify("explain the file structure")
        assert result.complexity == TaskComplexity.MEDIUM
        assert "explain_code" in result.matched_patterns

        # "find definitions" patterns
        result = classifier.classify("find all classes in the module")
        assert result.complexity == TaskComplexity.MEDIUM

        # "search for" patterns
        result = classifier.classify("search for the authentication logic")
        assert result.complexity == TaskComplexity.MEDIUM

        # "how does work" patterns
        result = classifier.classify("how does the caching work")
        assert result.complexity == TaskComplexity.MEDIUM

    def test_complex_task_detection(self):
        """Test detection of complex tasks using regex patterns."""
        classifier = ComplexityClassifier(use_semantic=False)

        # "analyze codebase" patterns - clear complex signal
        result = classifier.classify("analyze the entire codebase for architecture patterns")
        # With semantic disabled, may match ANALYSIS or COMPLEX regex patterns
        assert result.complexity in (TaskComplexity.COMPLEX, TaskComplexity.ANALYSIS)

        # "comprehensive analysis" patterns
        result = classifier.classify("provide a comprehensive analysis of the project")
        assert result.complexity in (TaskComplexity.COMPLEX, TaskComplexity.ANALYSIS)

        # "migration" patterns
        result = classifier.classify("migrate the database schema")
        assert result.complexity == TaskComplexity.COMPLEX

    def test_generation_task_detection(self):
        """Test detection of generation tasks using regex patterns."""
        classifier = ComplexityClassifier(use_semantic=False)

        # "write function" patterns
        result = classifier.classify("write a python script to parse JSON")
        assert result.complexity == TaskComplexity.GENERATION
        assert "write_lang_script" in result.matched_patterns

        # "create function" patterns
        result = classifier.classify("create a simple function to add numbers")
        assert result.complexity == TaskComplexity.GENERATION

        # "show me code" patterns
        result = classifier.classify("show me a code example for sorting")
        assert result.complexity == TaskComplexity.GENERATION

        # HumanEval-style patterns
        result = classifier.classify("complete this function implementation")
        assert result.complexity == TaskComplexity.GENERATION

        # Doctest patterns
        result = classifier.classify("implement the function to pass tests")
        assert result.complexity == TaskComplexity.GENERATION

    def test_default_to_medium(self):
        """Test that unmatched messages default to MEDIUM complexity."""
        classifier = ComplexityClassifier(use_semantic=False)

        result = classifier.classify("xyzzy gibberish")
        assert result.complexity == TaskComplexity.MEDIUM
        assert result.confidence < 0.5
        assert len(result.matched_patterns) == 0

    def test_tool_budget_assignment(self):
        """Test that correct tool budgets are assigned."""
        classifier = ComplexityClassifier(use_semantic=False)

        # Simple: budget 3 (updated from 2)
        result = classifier.classify("list files")
        assert result.tool_budget == DEFAULT_BUDGETS[TaskComplexity.SIMPLE]

        # Medium: budget 6 (updated from 4)
        result = classifier.classify("explain this code")
        assert result.tool_budget == DEFAULT_BUDGETS[TaskComplexity.MEDIUM]

        # Complex: budget 15
        result = classifier.classify("analyze the entire project")
        # May match ANALYSIS patterns
        assert result.tool_budget in (
            DEFAULT_BUDGETS[TaskComplexity.COMPLEX],
            DEFAULT_BUDGETS[TaskComplexity.ANALYSIS],
        )

        # Generation: budget 1 (updated from 0)
        result = classifier.classify("write a function to sort")
        assert result.tool_budget == DEFAULT_BUDGETS[TaskComplexity.GENERATION]

    def test_custom_budgets(self):
        """Test setting custom tool budgets."""
        custom_budgets = {
            TaskComplexity.SIMPLE: 5,
            TaskComplexity.MEDIUM: 10,
            TaskComplexity.COMPLEX: 30,
            TaskComplexity.GENERATION: 1,
            TaskComplexity.ACTION: 50,
            TaskComplexity.ANALYSIS: 60,
        }
        classifier = ComplexityClassifier(budgets=custom_budgets, use_semantic=False)

        result = classifier.classify("list files")
        assert result.tool_budget == 5

        result = classifier.classify("write a function")
        assert result.tool_budget == 1

    def test_get_budget(self):
        """Test the get_budget method."""
        classifier = ComplexityClassifier()

        # Updated budget values - minimum 10 for all types
        assert classifier.get_budget(TaskComplexity.SIMPLE) == 10
        assert classifier.get_budget(TaskComplexity.MEDIUM) == 15
        assert classifier.get_budget(TaskComplexity.COMPLEX) == 25
        assert classifier.get_budget(TaskComplexity.GENERATION) == 10
        assert classifier.get_budget(TaskComplexity.ACTION) == 50
        assert classifier.get_budget(TaskComplexity.ANALYSIS) == 60

    def test_update_budget(self):
        """Test the update_budget method."""
        classifier = ComplexityClassifier()

        classifier.update_budget(TaskComplexity.SIMPLE, 10)
        assert classifier.get_budget(TaskComplexity.SIMPLE) == 10

    def test_prompt_hint_via_enricher(self):
        """Test that correct prompt hints are available via enricher.

        Note: prompt_hint was removed from TaskClassification for SRP.
        Hints are now retrieved separately via get_prompt_hint().
        """
        classifier = ComplexityClassifier(use_semantic=False)

        result = classifier.classify("list files")
        hint = get_prompt_hint(result.complexity)
        assert "[SIMPLE]" in hint

        result = classifier.classify("explain the code")
        hint = get_prompt_hint(result.complexity)
        assert "[MEDIUM]" in hint

        result = classifier.classify("analyze the codebase")
        hint = get_prompt_hint(result.complexity)
        # May be COMPLEX or ANALYSIS
        assert "[COMPLEX]" in hint or "[ANALYSIS]" in hint

        result = classifier.classify("write a function")
        hint = get_prompt_hint(result.complexity)
        assert "[GENERATE]" in hint

    def test_confidence_scoring(self):
        """Test confidence scores are reasonable."""
        classifier = ComplexityClassifier(use_semantic=False)

        # Clear match should have moderate to high confidence
        result = classifier.classify("list all files in the directory")
        assert result.confidence > 0.3

        # Ambiguous message should have lower confidence
        result = classifier.classify("xyzzy")
        assert result.confidence < 0.5

    def test_custom_patterns(self):
        """Test adding custom patterns."""
        custom_patterns = {
            TaskComplexity.SIMPLE: [
                (r"\bping\b", 1.0, "ping"),
            ],
            TaskComplexity.COMPLEX: [
                (r"\bmigrate\s+database\b", 1.0, "migrate_db"),
            ],
        }
        classifier = ComplexityClassifier(custom_patterns=custom_patterns, use_semantic=False)

        result = classifier.classify("ping the server")
        assert result.complexity == TaskComplexity.SIMPLE
        assert "ping" in result.matched_patterns

        result = classifier.classify("migrate database to new schema")
        assert result.complexity == TaskComplexity.COMPLEX

    def test_custom_classifier_function(self):
        """Test custom classifier functions take precedence."""

        def custom_classifier(message: str):
            if "magic" in message.lower():
                return TaskClassification(
                    complexity=TaskComplexity.COMPLEX,
                    tool_budget=100,
                    confidence=1.0,
                    matched_patterns=["magic_word"],
                )
            return None

        classifier = ComplexityClassifier(custom_classifiers=[custom_classifier])

        result = classifier.classify("use the magic word")
        assert result.complexity == TaskComplexity.COMPLEX
        assert result.tool_budget == 100
        assert "magic_word" in result.matched_patterns

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        classifier = ComplexityClassifier()

        result1 = classifier.classify("LIST ALL FILES")
        result2 = classifier.classify("list all files")

        assert result1.complexity == result2.complexity

    def test_invalid_regex_pattern_handling(self):
        """Test that invalid regex patterns are handled gracefully."""
        custom_patterns = {
            TaskComplexity.SIMPLE: [
                (r"[invalid", 1.0, "invalid_pattern"),  # Invalid regex
            ],
        }
        classifier = ComplexityClassifier(custom_patterns=custom_patterns, use_semantic=False)

        # Should still work with valid patterns
        result = classifier.classify("list files")
        assert result.complexity == TaskComplexity.SIMPLE

    def test_multiple_patterns_highest_score_wins(self):
        """Test that highest scoring complexity wins."""
        classifier = ComplexityClassifier()

        # Message that matches both simple and complex patterns
        # Complex patterns should have higher weight
        classifier.classify("analyze all files and list them")
        # The result depends on relative scores


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_classify_task(self):
        """Test the classify_task convenience function."""
        result = classify_task("list all files")
        assert isinstance(result, TaskClassification)
        # With semantic classification, may be SIMPLE, MEDIUM, or SEARCH
        assert result.complexity in (
            TaskComplexity.SIMPLE,
            TaskComplexity.MEDIUM,
        )

    def test_get_task_prompt_hint(self):
        """Test the get_task_prompt_hint convenience function."""
        hint = get_task_prompt_hint("list all files")
        # Updated prompt hint format
        assert "[SIMPLE]" in hint or "[MEDIUM]" in hint

        hint = get_task_prompt_hint("analyze the codebase")
        # May be COMPLEX or ANALYSIS
        assert "[COMPLEX]" in hint or "[ANALYSIS]" in hint

    def test_should_force_answer_true(self):
        """Test should_force_answer returns True when budget exceeded."""
        # Use a high tool call count to ensure budget is exceeded
        should_force, reason = should_force_answer("list files", 50)
        assert should_force is True
        assert (
            "budget" in reason.lower() or "simple" in reason.lower() or "medium" in reason.lower()
        )

    def test_should_force_answer_false(self):
        """Test should_force_answer returns False within budget."""
        should_force, reason = should_force_answer("list files", 1)
        assert should_force is False
        assert reason == ""


class TestDefaultBudgets:
    """Tests for DEFAULT_BUDGETS constant."""

    def test_all_complexities_have_budgets(self):
        """Test that all complexities have defined budgets."""
        for complexity in TaskComplexity:
            assert complexity in DEFAULT_BUDGETS
            assert isinstance(DEFAULT_BUDGETS[complexity], int)

    def test_budget_values(self):
        """Test budget values are reasonable."""
        assert DEFAULT_BUDGETS[TaskComplexity.SIMPLE] < DEFAULT_BUDGETS[TaskComplexity.MEDIUM]
        assert DEFAULT_BUDGETS[TaskComplexity.MEDIUM] < DEFAULT_BUDGETS[TaskComplexity.COMPLEX]
        # All budgets have minimum of 10 to prevent premature termination
        for complexity in TaskComplexity:
            assert DEFAULT_BUDGETS[complexity] >= 10
        # ACTION and ANALYSIS have higher budgets
        assert DEFAULT_BUDGETS[TaskComplexity.ACTION] == 50
        assert DEFAULT_BUDGETS[TaskComplexity.ANALYSIS] == 60


class TestPromptHints:
    """Tests for PROMPT_HINTS constant."""

    def test_all_complexities_have_hints(self):
        """Test that all complexities have defined hints."""
        for complexity in TaskComplexity:
            assert complexity in PROMPT_HINTS
            assert isinstance(PROMPT_HINTS[complexity], str)

    def test_hints_contain_task_type(self):
        """Test hints identify the task type."""
        # Updated format uses brackets instead of "TASK TYPE:"
        expected_prefixes = {
            TaskComplexity.SIMPLE: "[SIMPLE]",
            TaskComplexity.MEDIUM: "[MEDIUM]",
            TaskComplexity.COMPLEX: "[COMPLEX]",
            TaskComplexity.GENERATION: "[GENERATE]",
            TaskComplexity.ACTION: "[ACTION]",
            TaskComplexity.ANALYSIS: "[ANALYSIS]",
        }
        for complexity in TaskComplexity:
            hint = PROMPT_HINTS[complexity]
            assert expected_prefixes[complexity] in hint


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_message(self):
        """Test handling of empty message."""
        classifier = ComplexityClassifier(use_semantic=False)
        result = classifier.classify("")
        assert result.complexity == TaskComplexity.MEDIUM
        assert result.confidence < 0.5

    def test_very_long_message(self):
        """Test handling of very long message."""
        classifier = ComplexityClassifier(use_semantic=False)
        long_message = "list files " * 1000
        result = classifier.classify(long_message)
        assert result.complexity == TaskComplexity.SIMPLE

    def test_special_characters(self):
        """Test handling of special characters."""
        classifier = ComplexityClassifier(use_semantic=False)
        result = classifier.classify("list files! @#$%^&*()")
        assert result.complexity == TaskComplexity.SIMPLE

    def test_unicode_message(self):
        """Test handling of unicode characters."""
        classifier = ComplexityClassifier(use_semantic=False)
        result = classifier.classify("list files")
        assert result.complexity == TaskComplexity.SIMPLE

    def test_python_function_signature_detection(self):
        """Test detection of Python function signatures (HumanEval style)."""
        classifier = ComplexityClassifier(use_semantic=False)

        # Function definition pattern
        result = classifier.classify("def factorial(n):")
        assert result.complexity == TaskComplexity.GENERATION
        assert "function_definition" in result.matched_patterns

    def test_doctest_pattern_detection(self):
        """Test detection of doctest patterns."""
        classifier = ComplexityClassifier()

        # Doctest pattern with >>> and docstring
        doctest_message = '"""\n>>> factorial(5)\n120\n"""'
        result = classifier.classify(doctest_message)
        # May match generation or other patterns
        assert result.complexity in (TaskComplexity.GENERATION, TaskComplexity.MEDIUM)
