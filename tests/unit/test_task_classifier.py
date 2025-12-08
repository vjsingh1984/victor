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

"""Tests for ComplexityClassifier - Gap 1 implementation."""

import pytest


class TestTaskClassifierSimple:
    """Tests for SIMPLE task classification."""

    def test_list_files_simple(self):
        """Test that 'list files' is classified as SIMPLE."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("List all files in the directory")

        assert result.complexity == TaskComplexity.SIMPLE
        assert result.tool_budget == 2

    def test_show_files_simple(self):
        """Test that 'show files' is classified as SIMPLE."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Show me the Python files in src/")

        assert result.complexity == TaskComplexity.SIMPLE
        assert result.tool_budget == 2

    def test_git_status_simple(self):
        """Test that 'git status' is classified as SIMPLE."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Show me the current git status")

        assert result.complexity == TaskComplexity.SIMPLE

    def test_what_files_simple(self):
        """Test that 'what files are in' is classified as SIMPLE."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("What files are in the victor/agent directory?")

        assert result.complexity == TaskComplexity.SIMPLE


class TestTaskClassifierMedium:
    """Tests for MEDIUM task classification."""

    def test_explain_file_medium(self):
        """Test that 'explain file' is classified as MEDIUM."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Explain the file victor/agent/orchestrator.py")

        assert result.complexity == TaskComplexity.MEDIUM
        assert result.tool_budget == 4

    def test_find_classes_medium(self):
        """Test that 'find classes' is classified as MEDIUM."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Find all classes that inherit from BaseTool")

        assert result.complexity == TaskComplexity.MEDIUM

    def test_where_is_medium(self):
        """Test that 'where is' is classified as MEDIUM."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Where is the error handling implemented?")

        assert result.complexity == TaskComplexity.MEDIUM

    def test_how_does_work_medium(self):
        """Test that 'how does X work' is classified as MEDIUM."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("How does the caching mechanism work?")

        assert result.complexity == TaskComplexity.MEDIUM


class TestTaskClassifierComplex:
    """Tests for COMPLEX task classification."""

    def test_analyze_codebase_complex(self):
        """Test that 'analyze codebase' is classified as COMPLEX."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Analyze the entire codebase for improvements")

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.tool_budget == 15

    def test_refactor_complex(self):
        """Test that 'refactor' is classified as COMPLEX."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Refactor the authentication module")

        assert result.complexity == TaskComplexity.COMPLEX

    def test_comprehensive_analysis_complex(self):
        """Test that 'comprehensive analysis' is classified as COMPLEX."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Provide a comprehensive analysis of the project architecture")

        assert result.complexity == TaskComplexity.COMPLEX


class TestTaskClassifierGeneration:
    """Tests for GENERATION task classification."""

    def test_create_function_generation(self):
        """Test that 'create a function' is classified as GENERATION."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Create a simple function that calculates factorial")

        assert result.complexity == TaskComplexity.GENERATION
        # GENERATION tasks have 0 tool budget - no exploration needed for code generation
        assert result.tool_budget == 0

    def test_write_code_generation(self):
        """Test that 'write code' is classified as GENERATION."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Write a Python script that parses JSON")

        assert result.complexity == TaskComplexity.GENERATION

    def test_show_me_code_generation(self):
        """Test that 'show me code' is classified as GENERATION."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("Show me code for a binary search algorithm")

        assert result.complexity == TaskComplexity.GENERATION


class TestTaskClassificationBehavior:
    """Tests for TaskClassification behavior."""

    def test_should_force_completion(self):
        """Test should_force_completion_after method."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("List files in the directory")

        assert result.complexity == TaskComplexity.SIMPLE
        assert result.tool_budget == 2

        # Should not force at 1 call
        assert not result.should_force_completion_after(1)

        # Should force at 2 calls
        assert result.should_force_completion_after(2)

        # Should force at 3+ calls
        assert result.should_force_completion_after(3)

    def test_confidence_score(self):
        """Test that confidence score is reasonable."""
        from victor.agent.complexity_classifier import ComplexityClassifier

        classifier = ComplexityClassifier()
        result = classifier.classify("List all Python files")

        assert 0.0 <= result.confidence <= 1.0

    def test_matched_patterns_populated(self):
        """Test that matched patterns are captured."""
        from victor.agent.complexity_classifier import ComplexityClassifier

        classifier = ComplexityClassifier()
        result = classifier.classify("git status")

        assert len(result.matched_patterns) > 0
        assert "git_status" in result.matched_patterns


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_classify_task(self):
        """Test classify_task convenience function."""
        from victor.agent.complexity_classifier import classify_task, TaskComplexity

        result = classify_task("Show git status")
        assert result.complexity == TaskComplexity.SIMPLE

    def test_get_task_prompt_hint(self):
        """Test get_task_prompt_hint convenience function."""
        from victor.agent.complexity_classifier import get_task_prompt_hint

        hint = get_task_prompt_hint("List files")
        assert "Simple Query" in hint
        assert "1-2 tool calls" in hint

    def test_should_force_answer(self):
        """Test should_force_answer convenience function."""
        from victor.agent.complexity_classifier import should_force_answer

        should_force, reason = should_force_answer("List files", 1)
        assert not should_force

        should_force, reason = should_force_answer("List files", 2)
        assert should_force
        assert "simple" in reason.lower()


class TestCustomClassifier:
    """Tests for custom classifier support."""

    def test_custom_budgets(self):
        """Test custom tool budgets."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        custom_budgets = {TaskComplexity.SIMPLE: 5, TaskComplexity.MEDIUM: 10}
        classifier = ComplexityClassifier(budgets=custom_budgets)

        result = classifier.classify("List files")
        assert result.tool_budget == 5

    def test_custom_classifier_function(self):
        """Test custom classifier function takes precedence."""
        from victor.agent.complexity_classifier import (
            ComplexityClassifier,
            TaskClassification,
            TaskComplexity,
        )

        def custom_classifier(message: str):
            if "urgent" in message.lower():
                return TaskClassification(
                    complexity=TaskComplexity.SIMPLE,
                    tool_budget=1,
                    prompt_hint="URGENT: Answer immediately!",
                    confidence=1.0,
                    matched_patterns=["urgent_override"],
                )
            return None

        classifier = ComplexityClassifier(custom_classifiers=[custom_classifier])
        result = classifier.classify("URGENT: List files now!")

        assert result.complexity == TaskComplexity.SIMPLE
        assert result.tool_budget == 1
        assert "urgent_override" in result.matched_patterns


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_message(self):
        """Test classification of empty message."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("")

        # Should default to MEDIUM
        assert result.complexity == TaskComplexity.MEDIUM

    def test_unrecognized_message(self):
        """Test classification of unrecognized message."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity

        classifier = ComplexityClassifier()
        result = classifier.classify("xyzzy foobar baz")

        # Should default to MEDIUM
        assert result.complexity == TaskComplexity.MEDIUM
        assert result.confidence < 0.5

    def test_mixed_signals(self):
        """Test message with multiple complexity signals."""
        from victor.agent.complexity_classifier import ComplexityClassifier

        classifier = ComplexityClassifier()
        # "list files" is SIMPLE, "analyze" is COMPLEX
        result = classifier.classify("List files and analyze the codebase")

        # Should pick the stronger signal (COMPLEX has higher weight in this case)
        assert result.complexity is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
