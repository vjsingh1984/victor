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

"""Framework enrichment utilities for prompt enhancement.

This module provides domain-agnostic enrichment utilities that can be used
by any vertical for prompt enhancement:

- Search term extraction with native Rust acceleration
- Web search result formatting
- Tool history context extraction

Example usage:
    from victor.framework.enrichment import (
        extract_search_terms,
        format_web_results,
        extract_tool_context,
    )

    # Extract search terms from a prompt
    terms = extract_search_terms("What is the capital of France?")

    # Format web search results
    formatted = format_web_results(search_results, max_results=3)

    # Extract context from tool history
    context = extract_tool_context(tool_history, tool_names=["web_search"])
"""

# Search term extraction
from victor.framework.enrichment.search_terms import (
    SearchTermExtractor,
    extract_search_terms,
    get_search_term_patterns,
)

# Web search formatting
from victor.framework.enrichment.web_search import (
    WebSearchFormatter,
    format_web_results,
    truncate_snippet,
)

# Tool history extraction
from victor.framework.enrichment.tool_history import (
    ToolHistoryExtractor,
    extract_tool_context,
    get_relevant_tool_results,
)

__all__ = [
    # Search terms
    "SearchTermExtractor",
    "extract_search_terms",
    "get_search_term_patterns",
    # Web search
    "WebSearchFormatter",
    "format_web_results",
    "truncate_snippet",
    # Tool history
    "ToolHistoryExtractor",
    "extract_tool_context",
    "get_relevant_tool_results",
]
