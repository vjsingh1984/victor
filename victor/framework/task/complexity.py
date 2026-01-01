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

This module provides framework-level task classification services.
It determines task complexity and appropriate tool budgets.

Design Principles:
- SRP: Only handles classification and budgets (no prompt hints)
- Prompt hints are handled by enrichment/strategies/complexity_hints.py
- Framework-level service usable by all verticals

Usage:
    from victor.framework.task import TaskComplexityService, TaskComplexity

    service = TaskComplexityService()
    result = service.classify("refactor the authentication module")
    print(f"Complexity: {result.complexity}, Budget: {result.tool_budget}")
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

from .protocols import TaskClassification, TaskClassifierProtocol, TaskComplexity

logger = logging.getLogger(__name__)


# Consolidated pattern definitions for classification
PATTERNS: Dict[TaskComplexity, List[Tuple[str, float, str]]] = {
    TaskComplexity.SIMPLE: [
        (r"\b(list|show|display)\s+.*?(files?|directories?|folders?)\b", 1.0, "list_files"),
        (r"\bwhat\s+files\s+(are\s+)?(in|at)\b", 1.0, "what_files"),
        (r"\bgit\s+(status|log|branch)\b", 1.0, "git_status"),
        (r"\b(show|what|get)\s+(the\s+)?(current\s+)?(git\s+)?status\b", 0.9, "status_query"),
        (r"\bpwd\b|\bcurrent\s+(directory|dir|folder)\b", 1.0, "pwd"),
        (r"\bls\b(?!\s+\|)", 0.9, "ls_command"),
        (r"\bcount\s+(files?|lines?|words?)\b", 0.9, "count_query"),
    ],
    TaskComplexity.MEDIUM: [
        (
            r"\b(explain|describe|summarize)\s+(the\s+)?(file|code|function|class)\b",
            0.9,
            "explain_code",
        ),
        (r"\bread\s+(and\s+)?(explain|summarize)\b", 0.9, "read_explain"),
        (r"\bfind\s+(all\s+)?(classes?|functions?|methods?)\b", 0.8, "find_definitions"),
        (r"\bwhere\s+(is|are|does)\b", 0.8, "where_query"),
        (r"\bhow\s+(does|do|is)\s+.+\s+(work|implemented)\b", 0.9, "how_works"),
    ],
    TaskComplexity.COMPLEX: [
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
        (r"\b(migrate|upgrade)\s+", 0.9, "migration"),
        (
            r"\b(restructure|reorganize)\s+(the\s+)?(project|code|module|layout)\b",
            0.9,
            "restructure",
        ),
        (r"\bconvert\s+.+\s+(to|into)\s+", 0.9, "convert_code"),
        (r"\bconsolidate\s+(the\s+)?(code|duplicate|files?)\b", 0.9, "consolidate"),
    ],
    TaskComplexity.GENERATION: [
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
        (r"\bcomplete\s+(this|the)\s+(function|code|implementation)\b", 1.0, "complete_function"),
        (r"\bshow\s+(me\s+)?(a\s+)?code\s+(example|sample)\b", 0.95, "show_code_example"),
        (r"\bshow\s+me\s+code\s+for\b", 0.95, "show_code_for"),
        (r"\bimplement\s+(the\s+)?function\s+to\s+pass\b", 0.95, "implement_function"),
        (r"\bdef\s+\w+\s*\([^)]*\)\s*:", 0.95, "function_definition"),
        (r'"""\s*\n.*?>>>', 0.95, "doctest_pattern"),
    ],
    TaskComplexity.ACTION: [
        (r"\bgit\s+(add|commit|push|pull|merge|rebase|stash)\b", 1.0, "git_command"),
        (r"\b(run|execute)\s+(the\s+)?(tests?|pytest|unittest|jest|mocha)\b", 1.0, "run_tests"),
        (r"\bpytest\b|\bnpm\s+test\b|\bcargo\s+test\b", 1.0, "test_command"),
        (r"\b(build|compile|deploy)\s+(the\s+)?(project|app|code)\b", 0.9, "build_deploy"),
        (r"\b(perform|do|run)\s+(a\s+)?(web\s*search|websearch)\b", 1.0, "web_search_action"),
    ],
    TaskComplexity.ANALYSIS: [
        (
            r"\b(comprehensive|thorough|detailed|full)\s+(analysis|review|audit)\b",
            1.0,
            "detailed_analysis",
        ),
        (r"\barchitecture\s+(review|analysis|overview)\b", 1.0, "architecture_analysis"),
        (
            r"\b(explain|describe)\s+(the\s+)?(entire|whole|full)\s+(codebase|project|system)\b",
            1.0,
            "explain_codebase",
        ),
        (r"\b(security|vulnerability)\s+(audit|scan|review)\b", 0.9, "security_audit"),
    ],
}

# Default tool budgets per complexity level
# Minimum of 10 for all types to prevent premature termination
DEFAULT_BUDGETS: Dict[TaskComplexity, int] = {
    TaskComplexity.SIMPLE: 10,      # Quick queries still need some room
    TaskComplexity.MEDIUM: 15,      # Moderate exploration
    TaskComplexity.COMPLEX: 25,     # Deep work
    TaskComplexity.GENERATION: 10,  # Code generation may need reads
    TaskComplexity.ACTION: 50,      # Multi-step actions
    TaskComplexity.ANALYSIS: 60,    # Thorough exploration
}

# Mapping from task type strings to complexity levels
TASK_TYPE_TO_COMPLEXITY: Dict[str, TaskComplexity] = {
    # Core task types
    "edit": TaskComplexity.MEDIUM,
    "search": TaskComplexity.MEDIUM,
    "create": TaskComplexity.COMPLEX,
    "create_simple": TaskComplexity.GENERATION,
    "analyze": TaskComplexity.MEDIUM,
    "design": TaskComplexity.ANALYSIS,
    "general": TaskComplexity.MEDIUM,
    "action": TaskComplexity.ACTION,
    "analysis_deep": TaskComplexity.COMPLEX,
    # Research vertical task types
    "literature_review": TaskComplexity.COMPLEX,
    "trend_research": TaskComplexity.COMPLEX,
    "technical_research": TaskComplexity.COMPLEX,
    "competitive_analysis": TaskComplexity.COMPLEX,
    "fact_check": TaskComplexity.MEDIUM,
    "general_query": TaskComplexity.MEDIUM,
    # Coding vertical task types
    "code_generation": TaskComplexity.GENERATION,
    "refactor": TaskComplexity.COMPLEX,
    "debug": TaskComplexity.MEDIUM,
    "test": TaskComplexity.MEDIUM,
    # DevOps vertical task types
    "infrastructure": TaskComplexity.COMPLEX,
    "ci_cd": TaskComplexity.COMPLEX,
    "dockerfile": TaskComplexity.MEDIUM,
    "docker_compose": TaskComplexity.MEDIUM,
    "kubernetes": TaskComplexity.COMPLEX,
    "terraform": TaskComplexity.COMPLEX,
    "monitoring": TaskComplexity.COMPLEX,
    # Data Analysis vertical task types
    "data_analysis": TaskComplexity.COMPLEX,
    "data_profiling": TaskComplexity.MEDIUM,
    "statistical_analysis": TaskComplexity.COMPLEX,
    "correlation_analysis": TaskComplexity.MEDIUM,
    "regression": TaskComplexity.COMPLEX,
    "clustering": TaskComplexity.COMPLEX,
    "time_series": TaskComplexity.COMPLEX,
    "visualization": TaskComplexity.MEDIUM,
}


class TaskComplexityService:
    """Framework service for task complexity classification.

    This service classifies user messages into complexity levels and
    provides appropriate tool budgets. It does NOT provide prompt hints -
    those are handled by the enrichment pipeline.

    Example:
        service = TaskComplexityService()
        result = service.classify("refactor authentication")
        print(f"Budget: {result.tool_budget}")
    """

    def __init__(
        self,
        budgets: Optional[Dict[TaskComplexity, int]] = None,
        custom_patterns: Optional[Dict[TaskComplexity, List[Tuple[str, float, str]]]] = None,
        custom_classifiers: Optional[List[Callable[[str], Optional[TaskClassification]]]] = None,
        use_semantic: bool = True,
        semantic_threshold: float = 0.65,
    ) -> None:
        """Initialize the complexity service.

        Args:
            budgets: Custom budget overrides per complexity level
            custom_patterns: Additional regex patterns for classification
            custom_classifiers: Custom classifier functions to try first
            use_semantic: Whether to use semantic classification
            semantic_threshold: Confidence threshold for semantic classification
        """
        self.budgets = budgets or DEFAULT_BUDGETS.copy()
        self.custom_classifiers = custom_classifiers or []
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold
        self._semantic_classifier = None

        # Compile regex patterns
        self._patterns: Dict[TaskComplexity, List[Tuple[re.Pattern, float, str]]] = {
            complexity: [] for complexity in TaskComplexity
        }
        for complexity, patterns in PATTERNS.items():
            self._add_patterns(complexity, patterns)
        if custom_patterns:
            for complexity, patterns in custom_patterns.items():
                self._add_patterns(complexity, patterns)

    def _add_patterns(
        self, complexity: TaskComplexity, patterns: List[Tuple[str, float, str]]
    ) -> None:
        """Add regex patterns for a complexity level."""
        for pattern_str, weight, name in patterns:
            try:
                self._patterns[complexity].append(
                    (re.compile(pattern_str, re.IGNORECASE), weight, name)
                )
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def _get_semantic_classifier(self):
        """Lazy-load the semantic classifier."""
        if self._semantic_classifier is None and self.use_semantic:
            try:
                from victor.embeddings.task_classifier import TaskTypeClassifier

                self._semantic_classifier = TaskTypeClassifier.get_instance(
                    threshold=self.semantic_threshold
                )
            except ImportError:
                logger.warning("TaskTypeClassifier not available, using regex only")
                self.use_semantic = False
        return self._semantic_classifier

    def _classify_semantic(self, message: str) -> Optional[TaskClassification]:
        """Attempt semantic classification using embeddings."""
        classifier = self._get_semantic_classifier()
        if not classifier:
            return None

        try:
            result = classifier.classify_sync(message)
            complexity = TASK_TYPE_TO_COMPLEXITY.get(
                result.task_type.value, TaskComplexity.MEDIUM
            )
            if result.confidence >= self.semantic_threshold:
                return TaskClassification(
                    complexity=complexity,
                    tool_budget=self.budgets[complexity],
                    confidence=result.confidence,
                    matched_patterns=[f"semantic:{result.task_type.value}"]
                    + [m[0] for m in result.top_matches[:3]],
                )
        except Exception as e:
            logger.debug(f"Semantic classification failed: {e}")
        return None

    def classify(self, message: str) -> TaskClassification:
        """Classify a message's task complexity.

        Args:
            message: User message to classify

        Returns:
            TaskClassification with complexity, budget, and metadata
        """
        # Try custom classifiers first
        for classifier in self.custom_classifiers:
            result = classifier(message)
            if result is not None:
                return result

        # Check high-confidence patterns (order matters: specific before general)
        for complexity in [
            TaskComplexity.SIMPLE,
            TaskComplexity.GENERATION,
            TaskComplexity.ACTION,
            TaskComplexity.MEDIUM,
            TaskComplexity.COMPLEX,
            TaskComplexity.ANALYSIS,
        ]:
            for pattern, weight, name in self._patterns[complexity]:
                if weight >= 0.9 and pattern.search(message):
                    return TaskClassification(
                        complexity=complexity,
                        tool_budget=self.budgets[complexity],
                        confidence=0.95,
                        matched_patterns=[name],
                    )

        # Try semantic classification
        if self.use_semantic:
            semantic_result = self._classify_semantic(message)
            if semantic_result:
                return semantic_result

        # Score all patterns
        scores: Dict[TaskComplexity, Tuple[float, List[str]]] = {}
        for complexity, patterns in self._patterns.items():
            total_score = 0.0
            matched: List[str] = []
            for pattern, weight, name in patterns:
                if pattern.search(message):
                    total_score += weight
                    matched.append(name)
            if matched:
                scores[complexity] = (total_score, matched)

        # Default to MEDIUM if no patterns match
        if not scores:
            return TaskClassification(
                complexity=TaskComplexity.MEDIUM,
                tool_budget=self.budgets[TaskComplexity.MEDIUM],
                confidence=0.3,
                matched_patterns=[],
            )

        # Return highest scoring complexity
        winner, (score, matched) = max(scores.items(), key=lambda x: x[1][0])
        return TaskClassification(
            complexity=winner,
            tool_budget=self.budgets[winner],
            confidence=min(1.0, score / 2.0 + 0.3),
            matched_patterns=matched,
        )

    def get_budget(self, complexity: TaskComplexity) -> int:
        """Get tool budget for a complexity level."""
        return self.budgets.get(complexity, DEFAULT_BUDGETS[TaskComplexity.MEDIUM])

    def update_budget(self, complexity: TaskComplexity, budget: int) -> None:
        """Update budget for a complexity level."""
        self.budgets[complexity] = budget


# Convenience functions for common operations
def classify_task(message: str) -> TaskClassification:
    """Classify a task message using default service."""
    return TaskComplexityService().classify(message)


def get_budget_for_task(message: str, base_multiplier: float = 1.0) -> int:
    """Get tool budget for a task message."""
    return int(classify_task(message).tool_budget * base_multiplier)


def is_action_task(message: str) -> bool:
    """Check if a message represents an action task."""
    return classify_task(message).complexity == TaskComplexity.ACTION


def is_analysis_task(message: str) -> bool:
    """Check if a message represents an analysis task."""
    return classify_task(message).complexity == TaskComplexity.ANALYSIS
