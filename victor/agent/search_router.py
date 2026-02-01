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

"""Search query routing for optimal tool selection.

This module analyzes search queries and routes them to the most
appropriate search tool:
- Keyword search: Exact patterns, code identifiers, literal strings
- Semantic search: Conceptual queries, explanations, pattern finding

Design Principles:
- Fast routing decisions (no LLM calls)
- Defaulting to keyword for precision
- Configurable thresholds
- Explainable decisions
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from re import Pattern
from collections.abc import Callable

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Type of search to perform."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"  # Use both and merge results


@dataclass
class SearchRoute:
    """Result of search routing decision.

    Attributes:
        search_type: The recommended search type
        confidence: Confidence in the routing decision (0.0-1.0)
        reason: Human-readable explanation for the decision
        transformed_query: Optionally transformed query for the search
        matched_patterns: List of patterns that influenced the decision
    """

    search_type: SearchType
    confidence: float
    reason: str
    transformed_query: Optional[str]
    matched_patterns: list[str]


# Patterns that indicate KEYWORD search is better
# Format: (regex_pattern, weight, description, case_sensitive)
# Case-sensitive patterns (4th element True) are NOT compiled with IGNORECASE
KEYWORD_SIGNALS: list[tuple[str, float, str, bool]] = [
    # Quoted strings are literal searches
    (r'"[^"]+"', 1.0, "quoted_string", False),
    (r"'[^']+'", 1.0, "single_quoted_string", False),
    # Code identifiers (case-sensitive for accuracy)
    (r"\bclass\s+[A-Z][a-zA-Z0-9_]*\b", 1.0, "class_name", True),
    (r"\bdef\s+[a-z_][a-zA-Z0-9_]*\b", 1.0, "function_def", True),
    (r"\bfunction\s+[a-z_][a-zA-Z0-9_]*\b", 0.9, "function_name", True),
    (r"\bimport\s+[a-z_][a-zA-Z0-9_.]*\b", 1.0, "import_statement", True),
    (r"\bfrom\s+[a-z_][a-zA-Z0-9_.]*\s+import\b", 1.0, "from_import", True),
    # Error patterns (case-sensitive - must be PascalCase)
    (r"\b[A-Z][a-zA-Z]*Error\b", 0.9, "error_class", True),
    (r"\b[A-Z][a-zA-Z]*Exception\b", 0.9, "exception_class", True),
    # Specific method/attribute access
    (r"\.\s*[a-z_][a-zA-Z0-9_]*\s*\(", 0.8, "method_call", True),
    (r"\b[a-z_][a-zA-Z0-9_]*\s*=", 0.7, "variable_assignment", True),
    # File patterns (case-insensitive)
    (r"\b\w+\.(py|js|ts|java|cpp|go|rs)\b", 0.8, "file_extension", False),
    # Decorators (case-sensitive)
    (r"@[a-z_][a-zA-Z0-9_]*", 0.9, "decorator", True),
    # Type hints
    (r":\s*(str|int|float|bool|list|dict|None)\b", 0.7, "type_hint", True),
    (r"->\s*(str|int|float|bool|list|dict|None)\b", 0.7, "return_type", True),
    # Exact identifiers (CamelCase or snake_case) - MUST be case-sensitive
    (r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b", 0.6, "camel_case", True),
    (r"\b[a-z]+_[a-z_]+\b", 0.5, "snake_case", True),
]

# Patterns that indicate SEMANTIC search is better
# All semantic patterns are case-insensitive (4th element False)
SEMANTIC_SIGNALS: list[tuple[str, float, str, bool]] = [
    # Conceptual queries
    (r"\bhow\s+(does|do|is|are|can|to)\b", 1.0, "how_question", False),
    (r"\bwhy\s+(does|do|is|are)\b", 1.0, "why_question", False),
    (r"\bwhat\s+(is|are|does)\b", 0.9, "what_question", False),
    (r"\bexplain\b", 1.0, "explain", False),
    (r"\bdescribe\b", 0.9, "describe", False),
    (r"\bunderstand\b", 0.8, "understand", False),
    # Pattern/concept finding
    (r"\bpattern(s)?\b", 0.9, "patterns", False),
    (r"\bapproach(es)?\b", 0.8, "approaches", False),
    (r"\bstrateg(y|ies)\b", 0.8, "strategies", False),
    (r"\barchitecture\b", 0.9, "architecture", False),
    (r"\bdesign\b", 0.7, "design", False),
    (r"\bworkflow\b", 0.8, "workflow", False),
    # Relationships
    (r"\brelat(ed|ionship|es)\b", 0.9, "relationships", False),
    (r"\bdependen(cy|cies|t)\b", 0.8, "dependencies", False),
    (r"\bconnect(ed|ion|s)?\b", 0.7, "connections", False),
    (r"\binteract(s|ion|ions)?\b", 0.8, "interactions", False),
    # Quality/characteristics
    (r"\bbest\s+practic(e|es)\b", 0.9, "best_practices", False),
    (r"\bissue(s)?\b", 0.7, "issues", False),
    (r"\bproblem(s)?\b", 0.7, "problems", False),
    (r"\bimprove(ment|ments)?\b", 0.8, "improvements", False),
    (r"\boptimiz(e|ation)\b", 0.8, "optimization", False),
    # Abstract concepts
    (r"\bconcept(s)?\b", 0.9, "concepts", False),
    (r"\bpurpose\b", 0.8, "purpose", False),
    (r"\breason(s|ing)?\b", 0.7, "reasoning", False),
    (r"\bgoal(s)?\b", 0.7, "goals", False),
]


class SearchRouter:
    """Routes search queries to appropriate search tools.

    This router analyzes query characteristics to determine whether
    keyword or semantic search would yield better results.

    Example:
        router = SearchRouter()
        route = router.route("class BaseTool")
        print(route.search_type)  # SearchType.KEYWORD

        route = router.route("how does error handling work")
        print(route.search_type)  # SearchType.SEMANTIC
    """

    def __init__(
        self,
        keyword_threshold: float = 0.5,
        semantic_threshold: float = 0.5,
        hybrid_threshold: float = 0.3,
        custom_signals: Optional[dict[str, list[tuple[str, float, str, bool]]]] = None,
        custom_routers: Optional[list[Callable[[str], Optional[SearchRoute]]]] = None,
    ):
        """Initialize the search router.

        Args:
            keyword_threshold: Min score for keyword routing
            semantic_threshold: Min score for semantic routing
            hybrid_threshold: Min scores to trigger hybrid search
            custom_signals: Additional routing signals
            custom_routers: Custom router functions to try first
        """
        self.keyword_threshold = keyword_threshold
        self.semantic_threshold = semantic_threshold
        self.hybrid_threshold = hybrid_threshold
        self.custom_routers = custom_routers or []

        # Compile patterns
        self._keyword_patterns: list[tuple[Pattern[str], float, str]] = []
        self._semantic_patterns: list[tuple[Pattern[str], float, str]] = []

        self._compile_patterns(KEYWORD_SIGNALS, self._keyword_patterns)
        self._compile_patterns(SEMANTIC_SIGNALS, self._semantic_patterns)

        # Add custom signals
        if custom_signals:
            if "keyword" in custom_signals:
                self._compile_patterns(custom_signals["keyword"], self._keyword_patterns)
            if "semantic" in custom_signals:
                self._compile_patterns(custom_signals["semantic"], self._semantic_patterns)

    def _compile_patterns(
        self,
        signals: list[tuple[str, float, str, bool]],
        target: list[tuple[Pattern[str], float, str]],
    ) -> None:
        """Compile regex patterns.

        Args:
            signals: List of (pattern, weight, name, case_sensitive) tuples
            target: Target list to append compiled patterns
        """
        for pattern_str, weight, name, case_sensitive in signals:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern_str, flags)
                target.append((compiled, weight, name))
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def route(self, query: str) -> SearchRoute:
        """Route a search query to appropriate search type.

        Args:
            query: The search query

        Returns:
            SearchRoute with recommended search type and metadata
        """
        # Try custom routers first
        for router in self.custom_routers:
            result = router(query)
            if result is not None:
                return result

        # Check for quoted strings (always keyword)
        if self._has_quoted_string(query):
            return SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=1.0,
                reason="Query contains quoted literal string",
                transformed_query=self._extract_quoted_string(query),
                matched_patterns=["quoted_string"],
            )

        # Score both search types
        keyword_score, keyword_matches = self._score_patterns(query, self._keyword_patterns)
        semantic_score, semantic_matches = self._score_patterns(query, self._semantic_patterns)

        # Normalize scores
        max_score = max(keyword_score + semantic_score, 1.0)
        keyword_norm = keyword_score / max_score if max_score > 0 else 0
        semantic_norm = semantic_score / max_score if max_score > 0 else 0

        # Determine route
        if keyword_score >= self.hybrid_threshold and semantic_score >= self.hybrid_threshold:
            # Both signals present - hybrid search
            return SearchRoute(
                search_type=SearchType.HYBRID,
                confidence=min(keyword_norm, semantic_norm) + 0.3,
                reason=f"Query has both keyword ({keyword_score:.2f}) and semantic ({semantic_score:.2f}) signals",
                transformed_query=None,
                matched_patterns=keyword_matches + semantic_matches,
            )
        elif keyword_score > semantic_score and keyword_score >= self.keyword_threshold:
            return SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=keyword_norm,
                reason=f"Query matches keyword patterns (score: {keyword_score:.2f})",
                transformed_query=None,
                matched_patterns=keyword_matches,
            )
        elif semantic_score > keyword_score and semantic_score >= self.semantic_threshold:
            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=semantic_norm,
                reason=f"Query matches semantic patterns (score: {semantic_score:.2f})",
                transformed_query=None,
                matched_patterns=semantic_matches,
            )
        else:
            # Default to keyword for precision
            return SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=0.3,
                reason="No strong signals detected, defaulting to keyword search",
                transformed_query=None,
                matched_patterns=[],
            )

    def _score_patterns(
        self,
        query: str,
        patterns: list[tuple[re.Pattern[str], float, str]],
    ) -> tuple[float, list[str]]:
        """Score patterns against query."""
        total_score = 0.0
        matched = []
        for pattern, weight, name in patterns:
            if pattern.search(query):
                total_score += weight
                matched.append(name)
        return total_score, matched

    def _has_quoted_string(self, query: str) -> bool:
        """Check if query contains quoted string."""
        return bool(re.search(r'"[^"]+"', query) or re.search(r"'[^']+'", query))

    def _extract_quoted_string(self, query: str) -> str:
        """Extract quoted string from query."""
        match = re.search(r'"([^"]+)"', query)
        if match:
            return match.group(1)
        match = re.search(r"'([^']+)'", query)
        if match:
            return match.group(1)
        return query

    def suggest_tool(self, query: str) -> str:
        """Suggest the best search tool for a query.

        Args:
            query: The search query

        Returns:
            Tool name: "code_search" or "semantic_code_search"
        """
        route = self.route(query)
        if route.search_type == SearchType.SEMANTIC:
            return "semantic_code_search"
        return "code_search"


def route_query(query: str) -> SearchRoute:
    """Convenience function to route a query.

    Args:
        query: Search query

    Returns:
        SearchRoute with recommendation
    """
    router = SearchRouter()
    return router.route(query)


def suggest_search_tool(query: str) -> str:
    """Convenience function to get recommended tool.

    Args:
        query: Search query

    Returns:
        Tool name: "code_search" or "semantic_code_search"
    """
    router = SearchRouter()
    return router.suggest_tool(query)


def is_keyword_query(query: str) -> bool:
    """Quick check if query should use keyword search.

    Args:
        query: Search query

    Returns:
        True if keyword search is recommended
    """
    route = route_query(query)
    return route.search_type in (SearchType.KEYWORD, SearchType.HYBRID)


def is_semantic_query(query: str) -> bool:
    """Quick check if query should use semantic search.

    Args:
        query: Search query

    Returns:
        True if semantic search is recommended
    """
    route = route_query(query)
    return route.search_type in (SearchType.SEMANTIC, SearchType.HYBRID)
