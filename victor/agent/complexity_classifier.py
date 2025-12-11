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

"""Task complexity classification for intelligent tool budgeting."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    GENERATION = "generation"
    ACTION = "action"
    ANALYSIS = "analysis"


@dataclass
class TaskClassification:
    complexity: TaskComplexity
    tool_budget: int
    prompt_hint: str
    confidence: float
    matched_patterns: List[str]

    def should_force_completion_after(self, tool_calls: int) -> bool:
        return tool_calls >= self.tool_budget


# Consolidated pattern definitions
PATTERNS = {
    TaskComplexity.SIMPLE: [
        (r"\b(list|show|display)\s+(files?|directories?|folders?)\b", 1.0, "list_files"),
        (r"\bgit\s+(status|log|branch)\b", 1.0, "git_status"),
        (r"\b(show|what|get)\s+(the\s+)?(current\s+)?(git\s+)?status\b", 0.9, "status_query"),
        (r"\bpwd\b|\bcurrent\s+(directory|dir|folder)\b", 1.0, "pwd"),
        (r"\bls\b(?!\s+\|)", 0.9, "ls_command"),
        (r"\bcount\s+(files?|lines?|words?)\b", 0.9, "count_query"),
    ],
    TaskComplexity.MEDIUM: [
        (r"\b(explain|describe|summarize)\s+(the\s+)?(file|code|function|class)\b", 0.9, "explain_code"),
        (r"\bread\s+(and\s+)?(explain|summarize)\b", 0.9, "read_explain"),
        (r"\bfind\s+(all\s+)?(classes?|functions?|methods?)\b", 0.8, "find_definitions"),
        (r"\bwhere\s+(is|are|does)\b", 0.8, "where_query"),
        (r"\bhow\s+(does|do|is)\s+\w+\s+(work|implemented)\b", 0.8, "how_works"),
    ],
    TaskComplexity.COMPLEX: [
        (r"\b(analyze|review|audit)\s+(the\s+)?(entire\s+)?(codebase|project|code)\b", 1.0, "analyze_codebase"),
        (r"\brefactor\b", 0.9, "refactor"),
        (r"\b(implement|add|create)\s+(a\s+)?(new\s+)?(feature|system|module)\b", 0.9, "implement_feature"),
        (r"\b(migrate|upgrade|convert)\b", 0.8, "migration"),
    ],
    TaskComplexity.GENERATION: [
        (r"\b(create|write|generate)\s+(a\s+)?(simple\s+)?(function|script|code)\b", 1.0, "generate_code"),
        (r"\bwrite\s+(a\s+)?(python|javascript|bash|ruby|go|rust)\s+(script|program|code)\b", 1.0, "write_lang_script"),
        (r"\bcomplete\s+(this|the)\s+(function|code|implementation)\b", 1.0, "complete_function"),
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
        (r"\b(comprehensive|thorough|detailed|full)\s+(analysis|review|audit)\b", 1.0, "detailed_analysis"),
        (r"\barchitecture\s+(review|analysis|overview)\b", 1.0, "architecture_analysis"),
        (r"\b(explain|describe)\s+(the\s+)?(entire|whole|full)\s+(codebase|project|system)\b", 1.0, "explain_codebase"),
        (r"\b(security|vulnerability)\s+(audit|scan|review)\b", 0.9, "security_audit"),
    ]
}

# Configuration constants
DEFAULT_BUDGETS = {
    TaskComplexity.SIMPLE: 2, TaskComplexity.MEDIUM: 4, TaskComplexity.COMPLEX: 15,
    TaskComplexity.GENERATION: 0, TaskComplexity.ACTION: 50, TaskComplexity.ANALYSIS: 60
}

PROMPT_HINTS = {
    TaskComplexity.SIMPLE: "[SIMPLE] Simple Query. 1-2 tool calls max. Answer immediately after.",
    TaskComplexity.MEDIUM: "[MEDIUM] Moderate exploration. 3-6 tool calls. Be concise.",
    TaskComplexity.COMPLEX: "[COMPLEX] Deep work needed. Examine code systematically. Provide detailed answer.",
    TaskComplexity.GENERATION: "[GENERATE] Write code directly. Minimal exploration. Display or save as requested.",
    TaskComplexity.ACTION: "[ACTION] Execute task. Multiple tool calls allowed. Continue until complete.",
    TaskComplexity.ANALYSIS: "[ANALYSIS] Thorough exploration required. Examine all relevant modules. Comprehensive output."
}

EXTENDED_PROMPT_HINTS = {
    TaskComplexity.SIMPLE: "[SIMPLE QUERY] Quick info retrieval. Use 1-2 tool calls max.",
    TaskComplexity.MEDIUM: "[MEDIUM] Moderate exploration. Use 3-6 tool calls. Be focused.",
    TaskComplexity.COMPLEX: "[COMPLEX] Thorough work. Examine code systematically.",
    TaskComplexity.GENERATION: "[GENERATION] Direct code creation. Minimal exploration.",
    TaskComplexity.ACTION: "[ACTION] Multi-step execution. Continue until complete.",
    TaskComplexity.ANALYSIS: "[ANALYSIS] Comprehensive exploration. Examine all modules."
}


class ComplexityClassifier:

    TASK_TYPE_TO_COMPLEXITY = {
        "edit": TaskComplexity.MEDIUM, "search": TaskComplexity.MEDIUM,
        "create": TaskComplexity.COMPLEX, "create_simple": TaskComplexity.GENERATION,
        "analyze": TaskComplexity.MEDIUM, "design": TaskComplexity.ANALYSIS,
        "general": TaskComplexity.MEDIUM, "action": TaskComplexity.ACTION,
        "analysis_deep": TaskComplexity.COMPLEX
    }

    def __init__(self, budgets=None, custom_patterns=None, custom_classifiers=None, use_semantic=True, semantic_threshold=0.65):
        self.budgets = budgets or DEFAULT_BUDGETS.copy()
        self.custom_classifiers = custom_classifiers or []
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold
        self._semantic_classifier = None
        
        self._patterns = {complexity: [] for complexity in TaskComplexity}
        for complexity, patterns in PATTERNS.items():
            self._add_patterns(complexity, patterns)
        if custom_patterns:
            for complexity, patterns in custom_patterns.items():
                self._add_patterns(complexity, patterns)

    def _get_semantic_classifier(self):
        if self._semantic_classifier is None and self.use_semantic:
            try:
                from victor.embeddings.task_classifier import TaskTypeClassifier
                self._semantic_classifier = TaskTypeClassifier.get_instance(threshold=self.semantic_threshold)
            except ImportError:
                logger.warning("TaskTypeClassifier not available, using regex only")
                self.use_semantic = False
        return self._semantic_classifier

    def _classify_semantic(self, message: str) -> Optional[TaskClassification]:
        if not (classifier := self._get_semantic_classifier()):
            return None
        try:
            result = classifier.classify_sync(message)
            complexity = self.TASK_TYPE_TO_COMPLEXITY.get(result.task_type.value, TaskComplexity.MEDIUM)
            if result.confidence >= self.semantic_threshold:
                return TaskClassification(
                    complexity, self.budgets[complexity], PROMPT_HINTS.get(complexity, ""),
                    result.confidence, [f"semantic:{result.task_type.value}"] + [m[0] for m in result.top_matches[:3]]
                )
        except Exception as e:
            logger.debug(f"Semantic classification failed: {e}")
        return None

    def _add_patterns(self, complexity, patterns):
        for pattern_str, weight, name in patterns:
            try:
                self._patterns[complexity].append((re.compile(pattern_str, re.IGNORECASE), weight, name))
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def classify(self, message: str) -> TaskClassification:
        # Try custom classifiers
        for classifier in self.custom_classifiers:
            if result := classifier(message):
                return result

        # Check high-confidence patterns
        for complexity in [TaskComplexity.SIMPLE, TaskComplexity.GENERATION, TaskComplexity.ACTION, TaskComplexity.COMPLEX]:
            for pattern, weight, name in self._patterns[complexity]:
                if weight >= 0.9 and pattern.search(message):
                    return TaskClassification(complexity, self.budgets[complexity], PROMPT_HINTS[complexity], 0.95, [name])

        # Try semantic classification
        if self.use_semantic and (semantic_result := self._classify_semantic(message)):
            return semantic_result

        # Score all patterns
        scores = {}
        for complexity, patterns in self._patterns.items():
            total_score, matched = 0.0, []
            for pattern, weight, name in patterns:
                if pattern.search(message):
                    total_score += weight
                    matched.append(name)
            if matched:
                scores[complexity] = (total_score, matched)
        
        if not scores:
            return TaskClassification(TaskComplexity.MEDIUM, self.budgets[TaskComplexity.MEDIUM], 
                                    PROMPT_HINTS[TaskComplexity.MEDIUM], 0.3, [])
        
        winner, (score, matched) = max(scores.items(), key=lambda x: x[1][0])
        return TaskClassification(winner, self.budgets[winner], PROMPT_HINTS[winner], min(1.0, score / 2.0 + 0.3), matched)

    def get_budget(self, complexity: TaskComplexity) -> int:
        return self.budgets.get(complexity, DEFAULT_BUDGETS[TaskComplexity.MEDIUM])

    def update_budget(self, complexity: TaskComplexity, budget: int) -> None:
        self.budgets[complexity] = budget


def classify_task(message: str) -> TaskClassification:
    return ComplexityClassifier().classify(message)

def get_task_prompt_hint(message: str) -> str:
    return classify_task(message).prompt_hint


def should_force_answer(message: str, tool_calls: int) -> Tuple[bool, str]:
    c = classify_task(message)
    return (True, f"Task classified as {c.complexity.value} (budget: {c.tool_budget}, calls: {tool_calls})") if c.should_force_completion_after(tool_calls) else (False, "")


def get_prompt_hint(complexity: TaskComplexity, extended: bool = False, provider: Optional[str] = None) -> str:
    if provider and provider.lower() in {"anthropic", "openai", "google", "xai"}:
        return PROMPT_HINTS.get(complexity, "")
    if extended or (provider and provider.lower() in {"ollama", "lmstudio", "vllm"}):
        return EXTENDED_PROMPT_HINTS.get(complexity, PROMPT_HINTS.get(complexity, ""))
    return PROMPT_HINTS.get(complexity, "")


def get_budget_for_task(message: str, base_multiplier: float = 1.0) -> int:
    return int(classify_task(message).tool_budget * base_multiplier)

def is_action_task(message: str) -> bool:
    return classify_task(message).complexity == TaskComplexity.ACTION

def is_analysis_task(message: str) -> bool:
    return classify_task(message).complexity == TaskComplexity.ANALYSIS
