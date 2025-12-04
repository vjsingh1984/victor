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

"""Task complexity classification for intelligent tool budgeting.

This module classifies user queries by complexity to determine:
- Appropriate tool call budgets
- When to force completion
- Prompt hints for the LLM

Design Principles:
- Extensible: Easy to add new task patterns
- Configurable: Thresholds can be tuned
- Testable: Pure functions with clear inputs/outputs
- Future-proof: Supports custom classifiers and ML-based detection
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels with associated tool budgets."""

    # Simple tasks: list, show, status - answer after 1-2 tool calls
    SIMPLE = "simple"

    # Medium tasks: explain, find, search - answer after 2-4 tool calls
    MEDIUM = "medium"

    # Complex tasks: analyze, refactor, implement - use full budget
    COMPLEX = "complex"

    # Generation tasks: create, write, generate - minimal exploration
    GENERATION = "generation"


@dataclass
class TaskClassification:
    """Result of task classification.

    Attributes:
        complexity: The detected task complexity level
        tool_budget: Maximum tool calls before forcing completion
        prompt_hint: Hint to inject into system prompt
        confidence: Confidence score (0.0-1.0) for this classification
        matched_patterns: List of patterns that matched
    """

    complexity: TaskComplexity
    tool_budget: int
    prompt_hint: str
    confidence: float
    matched_patterns: List[str]

    def should_force_completion_after(self, tool_calls: int) -> bool:
        """Check if completion should be forced after N tool calls.

        Args:
            tool_calls: Current number of tool calls

        Returns:
            True if completion should be forced
        """
        return tool_calls >= self.tool_budget


# Pattern definitions for each complexity level
# Format: (regex_pattern, weight, description)
SIMPLE_PATTERNS: List[Tuple[str, float, str]] = [
    (r"\b(list|show|display)\s+(files?|directories?|folders?)\b", 1.0, "list_files"),
    (r"\b(list|show)\s+\w+\s+(files?|in)\b", 0.8, "list_specific"),
    (r"\bshow\s+me\s+(the\s+)?\w+\s+files?\b", 1.0, "show_me_files"),
    (r"\bgit\s+(status|log|branch)\b", 1.0, "git_status"),
    (r"\b(show|what('s|s)?|get)\s+(the\s+)?(current\s+)?(git\s+)?status\b", 0.9, "status_query"),
    (r"\b(what|which)\s+files?\s+(are|is)\s+(in|inside)\b", 0.9, "what_files"),
    (r"\bpwd\b|\bcurrent\s+(directory|dir|folder)\b", 1.0, "pwd"),
    (r"\bls\b(?!\s+\|)", 0.9, "ls_command"),
    (r"\bcount\s+(files?|lines?|words?)\b", 0.8, "count_query"),
]

MEDIUM_PATTERNS: List[Tuple[str, float, str]] = [
    (
        r"\b(explain|describe|summarize)\s+(the\s+)?(file|code|function|class)\b",
        0.9,
        "explain_code",
    ),
    (r"\bread\s+(and\s+)?(explain|summarize)\b", 0.9, "read_explain"),
    (r"\bfind\s+(all\s+)?(classes?|functions?|methods?)\b", 0.8, "find_definitions"),
    (r"\bwhere\s+(is|are|does)\b", 0.8, "where_query"),
    (r"\bsearch\s+(for|the)\b", 0.7, "search_query"),
    (r"\bhow\s+(does|do|is)\s+\w+\s+(work|implemented)\b", 0.8, "how_works"),
    (r"\bwhat\s+(does|is)\s+\w+\s+(do|for)\b", 0.7, "what_does"),
]

COMPLEX_PATTERNS: List[Tuple[str, float, str]] = [
    (
        r"\b(analyze|review|audit)\s+(the\s+)?(entire\s+)?(codebase|project|code)\b",
        1.0,
        "analyze_codebase",
    ),
    (r"\brefactor\b", 0.9, "refactor"),
    (
        r"\b(implement|add|create)\s+(a\s+)?(new\s+)?(feature|system|module)\b",
        0.9,
        "implement_feature",
    ),
    (r"\b(fix|resolve|debug)\s+(all|the|every)\b", 0.8, "fix_all"),
    (r"\b(migrate|upgrade|convert)\b", 0.8, "migration"),
    (r"\b(comprehensive|thorough|detailed)\s+(analysis|review)\b", 0.9, "detailed_analysis"),
    (r"\barchitecture\b", 0.7, "architecture"),
]

GENERATION_PATTERNS: List[Tuple[str, float, str]] = [
    (
        r"\b(create|write|generate)\s+(a\s+)?(simple\s+)?(function|script|code)\b",
        1.0,
        "generate_code",
    ),
    (
        r"\bwrite\s+(a\s+)?(python|javascript|bash|ruby|go|rust)\s+(script|program|code)\b",
        1.0,
        "write_lang_script",
    ),
    (r"\b(show\s+me|give\s+me)\s+(a\s+)?(code|example|function)\b", 0.9, "show_code"),
    (r"\bwrite\s+(a\s+)?(\w+\s+)?(that|which|to)\b", 0.8, "write_that"),
    (r"\b(make|build)\s+(me\s+)?(a\s+)?(\w+\s+)?(function|class|script)\b", 0.8, "make_function"),
    (r"\b(implement|code)\s+(a\s+)?(\w+\s+)?(algorithm|solution)\b", 0.7, "implement_algo"),
]

# Default tool budgets per complexity level
DEFAULT_BUDGETS: Dict[TaskComplexity, int] = {
    TaskComplexity.SIMPLE: 2,
    TaskComplexity.MEDIUM: 4,
    TaskComplexity.COMPLEX: 15,
    TaskComplexity.GENERATION: 1,
}

# Prompt hints per complexity level
PROMPT_HINTS: Dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE: """
TASK TYPE: Simple Query
This is a simple query that should be answered quickly:
1. Use at most 1-2 tool calls to gather the needed information.
2. After getting the information, provide your answer IMMEDIATELY.
3. Do NOT explore further or read additional files.
4. Do NOT add explanations unless specifically asked.
""",
    TaskComplexity.MEDIUM: """
TASK TYPE: Medium Complexity
This task requires moderate exploration:
1. Use 2-4 tool calls to gather context.
2. After reading the relevant code, provide your answer.
3. Be concise and focused on the specific question.
""",
    TaskComplexity.COMPLEX: """
TASK TYPE: Complex Analysis
This task requires thorough exploration:
1. Systematically examine the relevant code.
2. Take time to understand the architecture.
3. Provide a comprehensive answer with specific examples.
""",
    TaskComplexity.GENERATION: """
TASK TYPE: Code Generation
This is a code generation task:
1. Generate the requested code DIRECTLY without exploring the codebase.
2. Do NOT read existing files unless the code must integrate with them.
3. After generating the code, you are DONE - do not make additional tool calls.
4. Only DISPLAY the code unless explicitly asked to write to a file.
""",
}


class ComplexityClassifier:
    """Classifies user tasks by complexity to determine appropriate tool budgets.

    This classifier uses pattern matching with configurable rules.
    It can be extended with custom classifiers or ML-based detection.

    Example:
        classifier = ComplexityClassifier()
        result = classifier.classify("List all Python files in the src directory")
        print(result.complexity)  # TaskComplexity.SIMPLE
        print(result.tool_budget)  # 2

    """

    def __init__(
        self,
        budgets: Optional[Dict[TaskComplexity, int]] = None,
        custom_patterns: Optional[Dict[TaskComplexity, List[Tuple[str, float, str]]]] = None,
        custom_classifiers: Optional[List[Callable[[str], Optional[TaskClassification]]]] = None,
    ):
        """Initialize the task classifier.

        Args:
            budgets: Custom tool budgets per complexity level
            custom_patterns: Additional patterns to match
            custom_classifiers: Custom classifier functions to try first
        """
        self.budgets = budgets or DEFAULT_BUDGETS.copy()
        self.custom_classifiers = custom_classifiers or []

        # Build pattern registry
        self._patterns: Dict[TaskComplexity, List[Tuple[re.Pattern, float, str]]] = {
            TaskComplexity.SIMPLE: [],
            TaskComplexity.MEDIUM: [],
            TaskComplexity.COMPLEX: [],
            TaskComplexity.GENERATION: [],
        }

        # Compile default patterns
        self._add_patterns(TaskComplexity.SIMPLE, SIMPLE_PATTERNS)
        self._add_patterns(TaskComplexity.MEDIUM, MEDIUM_PATTERNS)
        self._add_patterns(TaskComplexity.COMPLEX, COMPLEX_PATTERNS)
        self._add_patterns(TaskComplexity.GENERATION, GENERATION_PATTERNS)

        # Add custom patterns
        if custom_patterns:
            for complexity, patterns in custom_patterns.items():
                self._add_patterns(complexity, patterns)

    def _add_patterns(
        self, complexity: TaskComplexity, patterns: List[Tuple[str, float, str]]
    ) -> None:
        """Add compiled patterns for a complexity level.

        Args:
            complexity: The complexity level
            patterns: List of (regex, weight, name) tuples
        """
        for pattern_str, weight, name in patterns:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self._patterns[complexity].append((compiled, weight, name))
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def classify(self, message: str) -> TaskClassification:
        """Classify a user message to determine task complexity.

        Args:
            message: The user's input message

        Returns:
            TaskClassification with complexity, budget, and hints
        """
        # Try custom classifiers first
        for classifier in self.custom_classifiers:
            result = classifier(message)
            if result is not None:
                return result

        # Score each complexity level
        scores: Dict[TaskComplexity, Tuple[float, List[str]]] = {}

        for complexity, patterns in self._patterns.items():
            total_score = 0.0
            matched = []
            for pattern, weight, name in patterns:
                if pattern.search(message):
                    total_score += weight
                    matched.append(name)

            if matched:
                scores[complexity] = (total_score, matched)

        # Determine winner (highest score wins)
        if not scores:
            # Default to MEDIUM if no patterns match
            return TaskClassification(
                complexity=TaskComplexity.MEDIUM,
                tool_budget=self.budgets[TaskComplexity.MEDIUM],
                prompt_hint=PROMPT_HINTS[TaskComplexity.MEDIUM],
                confidence=0.3,
                matched_patterns=[],
            )

        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        winner, (score, matched) = sorted_scores[0]

        # Calculate confidence based on score differential
        if len(sorted_scores) > 1:
            second_score = sorted_scores[1][1][0]
            confidence = min(1.0, (score - second_score) / 2.0 + 0.5)
        else:
            confidence = min(1.0, score / 2.0 + 0.3)

        return TaskClassification(
            complexity=winner,
            tool_budget=self.budgets[winner],
            prompt_hint=PROMPT_HINTS[winner],
            confidence=confidence,
            matched_patterns=matched,
        )

    def get_budget(self, complexity: TaskComplexity) -> int:
        """Get the tool budget for a complexity level.

        Args:
            complexity: The complexity level

        Returns:
            Maximum tool calls for this complexity
        """
        return self.budgets.get(complexity, DEFAULT_BUDGETS[TaskComplexity.MEDIUM])

    def update_budget(self, complexity: TaskComplexity, budget: int) -> None:
        """Update the tool budget for a complexity level.

        Args:
            complexity: The complexity level
            budget: New maximum tool calls
        """
        self.budgets[complexity] = budget


def classify_task(message: str) -> TaskClassification:
    """Convenience function to classify a task.

    Args:
        message: The user's input message

    Returns:
        TaskClassification with complexity, budget, and hints
    """
    classifier = ComplexityClassifier()
    return classifier.classify(message)


def get_task_prompt_hint(message: str) -> str:
    """Get the prompt hint for a user message.

    Args:
        message: The user's input message

    Returns:
        Prompt hint string to inject into system prompt
    """
    classification = classify_task(message)
    return classification.prompt_hint


def should_force_answer(message: str, tool_calls: int) -> Tuple[bool, str]:
    """Check if the LLM should be forced to answer now.

    Args:
        message: The original user message
        tool_calls: Number of tool calls made so far

    Returns:
        Tuple of (should_force, reason)
    """
    classification = classify_task(message)

    if classification.should_force_completion_after(tool_calls):
        return (
            True,
            f"Task classified as {classification.complexity.value} "
            f"(budget: {classification.tool_budget}, calls: {tool_calls})",
        )

    return (False, "")
