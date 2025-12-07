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

    # Medium tasks: explain, find, search - answer after 3-6 tool calls
    MEDIUM = "medium"

    # Complex tasks: analyze, refactor, implement - use full budget
    COMPLEX = "complex"

    # Generation tasks: create, write, generate - minimal exploration
    GENERATION = "generation"

    # Action tasks: git operations, test runs, script execution - high budget
    ACTION = "action"

    # Analysis tasks: codebase analysis, architecture review - extended budget
    ANALYSIS = "analysis"


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
    (r"\b(list|show)\s+\w+\s+(files?|in)\b", 0.9, "list_specific"),  # Increased to 0.9
    (r"\bshow\s+me\s+(the\s+)?\w+\s+files?\b", 1.0, "show_me_files"),
    (r"\bgit\s+(status|log|branch)\b", 1.0, "git_status"),
    (r"\b(show|what('s|s)?|get)\s+(the\s+)?(current\s+)?(git\s+)?status\b", 0.9, "status_query"),
    (r"\b(what|which)\s+files?\s+(are|is)\s+(in|inside)\b", 0.9, "what_files"),
    (r"\bpwd\b|\bcurrent\s+(directory|dir|folder)\b", 1.0, "pwd"),
    (r"\bls\b(?!\s+\|)", 0.9, "ls_command"),
    (r"\bcount\s+(files?|lines?|words?)\b", 0.9, "count_query"),  # Increased to 0.9
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
    # HumanEval-style function completion patterns
    (r"\bcomplete\s+(this|the)\s+(function|code|implementation)\b", 1.0, "complete_function"),
    (r"\bimplement\s+(this|the)\s+function\b", 1.0, "implement_function"),
    (r"\b(finish|complete)\s+(the\s+)?(implementation|code)\b", 0.9, "finish_impl"),
    (r"\bdef\s+\w+\s*\([^)]*\)\s*:", 0.95, "function_definition"),  # Python function signature
    (r'"""\s*\n.*?>>>', 0.95, "doctest_pattern"),  # Docstring with doctest
    (r"\bpass\s+the\s+test\s*cases?\b", 0.9, "pass_tests"),
    (r"\breturn\s+\w+\s*\.\s*\w+", 0.7, "return_pattern"),  # Return statement pattern
]

# Action patterns: git operations, test runs, script execution
ACTION_PATTERNS: List[Tuple[str, float, str]] = [
    # Git operations
    (r"\b(commit|push|pull|merge)\s+(all|the|these|my)?\s*(changes?|files?)?\b", 1.0, "git_commit"),
    (r"\bgit\s+(add|commit|push|pull|merge|rebase|stash)\b", 1.0, "git_command"),
    (r"\b(create|make|open)\s+(a\s+)?(pr|pull\s*request)\b", 1.0, "create_pr"),
    (r"\bgroup(ed)?\s+commits?\b", 1.0, "grouped_commits"),
    (r"\b(stage|unstage)\s+(all|the|these)?\s*(files?|changes?)?\b", 0.9, "git_stage"),
    (r"\b(cherry.?pick|revert)\b", 0.9, "git_advanced"),
    # Test execution
    (r"\b(run|execute)\s+(the\s+)?(tests?|pytest|unittest|jest|mocha)\b", 1.0, "run_tests"),
    (r"\bpytest\b|\bnpm\s+test\b|\bcargo\s+test\b", 1.0, "test_command"),
    (r"\b(run|execute)\s+(all\s+)?(unit|integration|e2e)\s+tests?\b", 1.0, "run_specific_tests"),
    # Script/command execution
    (r"\b(run|execute|start)\s+(the\s+)?(script|command|program)\b", 0.9, "run_script"),
    (r"\b(build|compile|deploy)\s+(the\s+)?(project|app|code)\b", 0.9, "build_deploy"),
    (r"\bnpm\s+(install|run|build)\b|\bpip\s+install\b|\bcargo\s+build\b", 0.9, "package_command"),
    # File operations requiring multiple steps
    (r"\b(move|rename|copy)\s+(all|the|these)\s+files?\b", 0.8, "bulk_file_ops"),
    (r"\b(delete|remove)\s+(all|the)\s+(unused|old|temp)\b", 0.8, "cleanup_ops"),
    # Web search and API research - requires multiple tool calls
    (r"\b(perform|do|run)\s+(a\s+)?(web\s*search|websearch)\b", 1.0, "web_search_action"),
    (r"\bweb\s*search\b", 0.9, "web_search"),
    (r"\bsearch\s+(the\s+)?(web|internet|online)\b", 0.9, "search_web"),
    (
        r"\b(search|look|find)\s+(for\s+)?(api|documentation|docs)\s+(for|on|about)\b",
        0.9,
        "search_api",
    ),
    (r"\bfetch\s+(for\s+)?(api|data|docs|documentation)\b", 0.8, "fetch_api"),
    (
        r"\b(research|investigate|explore)\s+(the\s+)?\w+\s+(api|library|package)\b",
        0.8,
        "research_api",
    ),
    (r"\bfind\s+(how\s+to\s+)?(use|integrate|connect)\s+\w+\s+api\b", 0.8, "find_api_usage"),
]

# Analysis patterns: thorough codebase exploration
ANALYSIS_PATTERNS: List[Tuple[str, float, str]] = [
    (
        r"\b(analyze|review|audit)\s+(the\s+)?(entire\s+)?(codebase|project|repo)\b",
        1.0,
        "full_analysis",
    ),
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
    (r"\b(map|document)\s+(the\s+)?(codebase|dependencies|structure)\b", 0.9, "map_codebase"),
    (r"\b(security|vulnerability)\s+(audit|scan|review)\b", 0.9, "security_audit"),
    (r"\b(performance|optimization)\s+(analysis|review)\b", 0.9, "perf_analysis"),
    (r"\b(code\s+)?quality\s+(review|assessment|report)\b", 0.9, "quality_review"),
    (r"\btechnical\s+debt\s+(analysis|review|assessment)\b", 0.9, "tech_debt"),
    (r"\b(identify|find)\s+(all|every)\s+(issues?|problems?|bugs?)\b", 0.8, "find_all_issues"),
]

# Default tool budgets per complexity level
# These can be overridden via settings or custom classifiers
DEFAULT_BUDGETS: Dict[TaskComplexity, int] = {
    TaskComplexity.SIMPLE: 2,  # Quick queries: status, list, show
    TaskComplexity.MEDIUM: 4,  # Moderate exploration: explain, find, search
    TaskComplexity.COMPLEX: 15,  # Deep work: refactor, implement features
    TaskComplexity.GENERATION: 0,  # Code generation: no exploration needed
    TaskComplexity.ACTION: 50,  # Actions: git ops, test runs, multi-step execution
    TaskComplexity.ANALYSIS: 60,  # Full analysis: codebase review, architecture audit
}

# Prompt hints per complexity level
# Concise, action-oriented hints that don't waste tokens
PROMPT_HINTS: Dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE: "[SIMPLE] Simple Query. 1-2 tool calls max. Answer immediately after.",
    TaskComplexity.MEDIUM: "[MEDIUM] Moderate exploration. 3-6 tool calls. Be concise.",
    TaskComplexity.COMPLEX: "[COMPLEX] Deep work needed. Examine code systematically. Provide detailed answer.",
    TaskComplexity.GENERATION: "[GENERATE] Write code directly. Minimal exploration. Display or save as requested.",
    TaskComplexity.ACTION: "[ACTION] Execute task. Multiple tool calls allowed. Continue until complete.",
    TaskComplexity.ANALYSIS: "[ANALYSIS] Thorough exploration required. Examine all relevant modules. Comprehensive output.",
}

# Extended hints for providers that benefit from more context (e.g., local models)
EXTENDED_PROMPT_HINTS: Dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE: """
[SIMPLE QUERY] Quick information retrieval.
- Use 1-2 tool calls maximum
- Answer immediately after gathering info
- No additional exploration needed
""",
    TaskComplexity.MEDIUM: """
[MEDIUM COMPLEXITY] Moderate code exploration.
- Use 3-6 tool calls to gather context
- Read relevant files before answering
- Be concise and focused
""",
    TaskComplexity.COMPLEX: """
[COMPLEX TASK] Thorough implementation work.
- Systematically examine relevant code
- Understand architecture before changes
- Provide detailed, well-structured output
""",
    TaskComplexity.GENERATION: """
[CODE GENERATION] Direct code creation.
- Generate code without exploring codebase
- Only read files if integration is required
- Display code or save as requested
""",
    TaskComplexity.ACTION: """
[ACTION EXECUTION] Multi-step task execution.
- Execute commands and operations as needed
- Continue until task is complete
- Report progress and results
""",
    TaskComplexity.ANALYSIS: """
[CODEBASE ANALYSIS] Comprehensive exploration.
- Examine all relevant modules systematically
- Take time to understand architecture
- Provide comprehensive, detailed output
""",
}


class ComplexityClassifier:
    """Classifies user tasks by complexity to determine appropriate tool budgets.

    This classifier uses a hybrid approach:
    1. Semantic classification via TaskTypeClassifier (embeddings + nudge rules)
    2. Regex pattern matching as fallback

    The semantic approach handles paraphrasing better while regex provides
    fast, deterministic matching for common patterns.

    Example:
        classifier = ComplexityClassifier()
        result = classifier.classify("List all Python files in the src directory")
        print(result.complexity)  # TaskComplexity.SIMPLE
        print(result.tool_budget)  # 3

    """

    # Mapping from semantic TaskType to TaskComplexity
    # This bridges the two classification systems
    TASK_TYPE_TO_COMPLEXITY: Dict[str, TaskComplexity] = {
        "edit": TaskComplexity.MEDIUM,
        "search": TaskComplexity.MEDIUM,
        "create": TaskComplexity.COMPLEX,
        "create_simple": TaskComplexity.GENERATION,
        "analyze": TaskComplexity.MEDIUM,
        "design": TaskComplexity.MEDIUM,  # Design/conceptual questions need exploration
        "general": TaskComplexity.MEDIUM,
        "action": TaskComplexity.ACTION,
        "analysis_deep": TaskComplexity.COMPLEX,  # Map deep analysis to COMPLEX for backward compat
    }

    def __init__(
        self,
        budgets: Optional[Dict[TaskComplexity, int]] = None,
        custom_patterns: Optional[Dict[TaskComplexity, List[Tuple[str, float, str]]]] = None,
        custom_classifiers: Optional[List[Callable[[str], Optional[TaskClassification]]]] = None,
        use_semantic: bool = True,
        semantic_threshold: float = 0.65,  # Higher threshold to reject low-confidence matches
    ):
        """Initialize the task classifier.

        Args:
            budgets: Custom tool budgets per complexity level
            custom_patterns: Additional patterns to match
            custom_classifiers: Custom classifier functions to try first
            use_semantic: Whether to use semantic classification (recommended)
            semantic_threshold: Confidence threshold for semantic classification
        """
        self.budgets = budgets or DEFAULT_BUDGETS.copy()
        self.custom_classifiers = custom_classifiers or []
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold

        # Lazy-load semantic classifier to avoid import cycle
        self._semantic_classifier = None

        # Build pattern registry (fallback when semantic not available/confident)
        self._patterns: Dict[TaskComplexity, List[Tuple[re.Pattern, float, str]]] = {
            TaskComplexity.SIMPLE: [],
            TaskComplexity.MEDIUM: [],
            TaskComplexity.COMPLEX: [],
            TaskComplexity.GENERATION: [],
            TaskComplexity.ACTION: [],
            TaskComplexity.ANALYSIS: [],
        }

        # Compile default patterns (order matters - more specific first)
        self._add_patterns(TaskComplexity.ACTION, ACTION_PATTERNS)
        self._add_patterns(TaskComplexity.ANALYSIS, ANALYSIS_PATTERNS)
        self._add_patterns(TaskComplexity.GENERATION, GENERATION_PATTERNS)
        self._add_patterns(TaskComplexity.SIMPLE, SIMPLE_PATTERNS)
        self._add_patterns(TaskComplexity.MEDIUM, MEDIUM_PATTERNS)
        self._add_patterns(TaskComplexity.COMPLEX, COMPLEX_PATTERNS)

        # Add custom patterns
        if custom_patterns:
            for complexity, patterns in custom_patterns.items():
                self._add_patterns(complexity, patterns)

    def _get_semantic_classifier(self):
        """Lazy-load the semantic classifier to avoid import cycles."""
        if self._semantic_classifier is None and self.use_semantic:
            try:
                from victor.embeddings.task_classifier import TaskTypeClassifier

                # Use singleton to avoid duplicate initialization and share cache
                self._semantic_classifier = TaskTypeClassifier.get_instance(
                    threshold=self.semantic_threshold
                )
            except ImportError:
                logger.warning("TaskTypeClassifier not available, using regex only")
                self.use_semantic = False
        return self._semantic_classifier

    def _classify_semantic(self, message: str) -> Optional[TaskClassification]:
        """Classify using semantic embeddings.

        Args:
            message: The user's input message

        Returns:
            TaskClassification if confident, None otherwise
        """
        classifier = self._get_semantic_classifier()
        if classifier is None:
            return None

        try:
            result = classifier.classify_sync(message)

            # Map TaskType to TaskComplexity
            complexity = self.TASK_TYPE_TO_COMPLEXITY.get(
                result.task_type.value, TaskComplexity.MEDIUM
            )

            # Only trust semantic if confidence is high enough
            if result.confidence >= self.semantic_threshold:
                return TaskClassification(
                    complexity=complexity,
                    tool_budget=self.budgets[complexity],
                    prompt_hint=PROMPT_HINTS.get(complexity, ""),
                    confidence=result.confidence,
                    matched_patterns=[f"semantic:{result.task_type.value}"]
                    + [m[0] for m in result.top_matches[:3]],
                )
        except Exception as e:
            logger.debug(f"Semantic classification failed: {e}")

        return None

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

        Uses a hybrid approach:
        1. Custom classifiers (if any)
        2. Regex pattern matching for high-confidence patterns (SIMPLE, GENERATION, ACTION)
        3. Semantic classification via embeddings for ambiguous cases
        4. Regex fallback for remaining cases

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

        # Check for high-confidence regex patterns first
        # Only return immediately if we find a VERY specific pattern (weight >= 0.9)
        # Note: ANALYSIS and MEDIUM are not in priority list to allow semantic classification
        priority_patterns = [
            TaskComplexity.SIMPLE,
            TaskComplexity.GENERATION,
            TaskComplexity.ACTION,
            TaskComplexity.COMPLEX,
        ]

        for complexity in priority_patterns:
            patterns = self._patterns[complexity]
            for pattern, weight, name in patterns:
                if pattern.search(message) and weight >= 0.9:
                    # Found a high-confidence pattern, use it immediately
                    matched = [name]
                    return TaskClassification(
                        complexity=complexity,
                        tool_budget=self.budgets[complexity],
                        prompt_hint=PROMPT_HINTS[complexity],
                        confidence=0.95,  # High confidence for explicit patterns
                        matched_patterns=matched,
                    )

        # Try semantic classification (embeddings + nudge rules)
        if self.use_semantic:
            semantic_result = self._classify_semantic(message)
            if semantic_result is not None:
                logger.debug(
                    f"Semantic classification: {semantic_result.complexity.value} "
                    f"(confidence: {semantic_result.confidence:.2f})"
                )
                return semantic_result

        # Fall back to regex pattern matching for MEDIUM/COMPLEX
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


def get_prompt_hint(
    complexity: TaskComplexity,
    extended: bool = False,
    provider: Optional[str] = None,
) -> str:
    """Get prompt hint for a complexity level with provider-aware formatting.

    Args:
        complexity: The task complexity level
        extended: Use extended hints (better for local models)
        provider: Optional provider name for provider-specific hints

    Returns:
        Appropriate prompt hint string
    """
    # Cloud providers work well with concise hints
    cloud_providers = {"anthropic", "openai", "google", "xai"}

    if provider and provider.lower() in cloud_providers:
        return PROMPT_HINTS.get(complexity, "")

    # Local models benefit from extended hints
    if extended or (provider and provider.lower() in {"ollama", "lmstudio", "vllm"}):
        return EXTENDED_PROMPT_HINTS.get(complexity, PROMPT_HINTS.get(complexity, ""))

    return PROMPT_HINTS.get(complexity, "")


def get_budget_for_task(message: str, base_multiplier: float = 1.0) -> int:
    """Get tool budget for a task with optional multiplier.

    Args:
        message: The user's input message
        base_multiplier: Multiplier for budget (e.g., 1.5 for extended tasks)

    Returns:
        Recommended tool budget
    """
    classification = classify_task(message)
    return int(classification.tool_budget * base_multiplier)


def is_action_task(message: str) -> bool:
    """Quick check if a message is an action task (git, test, execute).

    Args:
        message: The user's input message

    Returns:
        True if this is an action-oriented task
    """
    classification = classify_task(message)
    return classification.complexity == TaskComplexity.ACTION


def is_analysis_task(message: str) -> bool:
    """Quick check if a message is an analysis task.

    Args:
        message: The user's input message

    Returns:
        True if this is an analysis task
    """
    classification = classify_task(message)
    return classification.complexity == TaskComplexity.ANALYSIS
