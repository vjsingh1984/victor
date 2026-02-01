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

"""Escape hatches for Research YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_coverage
      type: condition
      condition: "source_coverage_check"  # References escape hatch
      branches:
        "sufficient": synthesize
        "needs_more": search_more
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def source_coverage_check(ctx: dict[str, Any]) -> str:
    """Check if source coverage is sufficient for synthesis.

    Evaluates source count, diversity, and quality.

    Args:
        ctx: Workflow context with keys:
            - sources (list): Collected sources
            - coverage_score (float): Coverage metric (0-1)
            - min_sources (int): Minimum required sources

    Returns:
        "sufficient", "marginal", or "needs_more"
    """
    sources = ctx.get("sources", [])
    coverage_score = ctx.get("coverage_score", 0)
    min_sources = ctx.get("min_sources", 5)

    source_count = len(sources) if isinstance(sources, list) else 0

    if coverage_score >= 0.8 and source_count >= min_sources:
        return "sufficient"

    if coverage_score >= 0.6 or source_count >= min_sources * 0.7:
        return "marginal"

    return "needs_more"


def should_search_more(ctx: dict[str, Any]) -> str:
    """Determine if additional source searching is needed.

    Multi-factor decision based on coverage, iteration count, and gaps.

    Args:
        ctx: Workflow context with keys:
            - sources (list): Current sources
            - coverage_score (float): Coverage metric
            - search_iterations (int): Number of search rounds completed
            - max_iterations (int): Maximum search iterations
            - gaps (list): Identified information gaps

    Returns:
        "search_more" or "proceed"
    """
    coverage_score = ctx.get("coverage_score", 0)
    iterations = ctx.get("search_iterations", 0)
    max_iter = ctx.get("max_iterations", 3)
    gaps = ctx.get("gaps", [])

    # Proceed if good coverage or max iterations reached
    if coverage_score >= 0.85:
        return "proceed"

    if iterations >= max_iter:
        logger.info(f"Max search iterations ({max_iter}) reached, proceeding")
        return "proceed"

    # Search more if significant gaps remain
    if len(gaps) > 2 and coverage_score < 0.6:
        return "search_more"

    return "proceed"


def source_credibility_check(ctx: dict[str, Any]) -> str:
    """Assess overall source credibility.

    Args:
        ctx: Workflow context with keys:
            - validated_sources (list): Sources with credibility scores
            - min_credibility (float): Minimum average credibility

    Returns:
        "high_credibility", "acceptable", or "low_credibility"
    """
    sources = ctx.get("validated_sources", [])

    if not sources:
        return "low_credibility"

    credibility_scores = []
    for source in sources:
        if isinstance(source, dict):
            credibility_scores.append(source.get("credibility", 0.5))

    if not credibility_scores:
        return "acceptable"

    avg_credibility = sum(credibility_scores) / len(credibility_scores)

    if avg_credibility >= 0.8:
        return "high_credibility"
    elif avg_credibility >= 0.5:
        return "acceptable"
    else:
        return "low_credibility"


def fact_verdict(ctx: dict[str, Any]) -> str:
    """Determine fact-check verdict based on evidence.

    Args:
        ctx: Workflow context with keys:
            - supporting_evidence (list): Evidence supporting the claim
            - refuting_evidence (list): Evidence refuting the claim
            - confidence (float): Confidence in verdict

    Returns:
        "true", "mostly_true", "mixed", "mostly_false", "false", or "unverifiable"
    """
    supporting = ctx.get("supporting_evidence", [])
    refuting = ctx.get("refuting_evidence", [])
    confidence = ctx.get("confidence", 0)

    support_count = len(supporting) if isinstance(supporting, list) else 0
    refute_count = len(refuting) if isinstance(refuting, list) else 0

    if support_count == 0 and refute_count == 0:
        return "unverifiable"

    total = support_count + refute_count
    support_ratio = support_count / total if total > 0 else 0

    if confidence < 0.3:
        return "unverifiable"

    if support_ratio >= 0.9:
        return "true"
    elif support_ratio >= 0.7:
        return "mostly_true"
    elif support_ratio >= 0.4:
        return "mixed"
    elif support_ratio >= 0.2:
        return "mostly_false"
    else:
        return "false"


def literature_relevance(ctx: dict[str, Any]) -> str:
    """Assess paper relevance for literature review.

    Args:
        ctx: Workflow context with keys:
            - paper (dict): Paper metadata
            - keywords (list): Search keywords
            - date_threshold (str): Oldest acceptable date

    Returns:
        "highly_relevant", "relevant", "marginal", or "irrelevant"
    """
    paper = ctx.get("paper", {})
    # keywords = ctx.get("keywords", [])  # For future use

    relevance_score = paper.get("relevance_score", 0.5)
    citation_count = paper.get("citation_count", 0)

    # High relevance: high score and well-cited
    if relevance_score >= 0.8 or citation_count >= 50:
        return "highly_relevant"

    if relevance_score >= 0.6:
        return "relevant"

    if relevance_score >= 0.4:
        return "marginal"

    return "irrelevant"


def competitive_threat_level(ctx: dict[str, Any]) -> str:
    """Assess competitive threat level.

    Args:
        ctx: Workflow context with keys:
            - competitor (dict): Competitor data
            - market_overlap (float): Market segment overlap (0-1)
            - growth_rate (float): Competitor growth rate

    Returns:
        "high", "medium", "low", or "emerging"
    """
    competitor = ctx.get("competitor", {})
    market_overlap = ctx.get("market_overlap", 0)
    growth_rate = ctx.get("growth_rate", 0)

    market_share = competitor.get("market_share", 0)
    is_direct = competitor.get("is_direct_competitor", False)

    if is_direct and market_share >= 0.2:
        return "high"

    if market_overlap >= 0.7 and growth_rate >= 0.2:
        return "high"

    if growth_rate >= 0.5:
        return "emerging"

    if market_overlap >= 0.4 or market_share >= 0.1:
        return "medium"

    return "low"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_search_results(ctx: dict[str, Any]) -> dict[str, Any]:
    """Merge results from parallel search operations.

    Deduplicates and ranks by relevance.

    Args:
        ctx: Workflow context with parallel search results

    Returns:
        Merged and deduplicated results
    """
    web_results = ctx.get("web_results", [])
    academic_results = ctx.get("academic_results", [])
    code_results = ctx.get("code_results", [])

    all_results = []
    seen_urls = set()

    for results, source_type in [
        (web_results, "web"),
        (academic_results, "academic"),
        (code_results, "code"),
    ]:
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        result["source_type"] = source_type
                        all_results.append(result)

    return {
        "sources": all_results,
        "source_count": len(all_results),
        "by_type": {
            "web": len([r for r in all_results if r.get("source_type") == "web"]),
            "academic": len([r for r in all_results if r.get("source_type") == "academic"]),
            "code": len([r for r in all_results if r.get("source_type") == "code"]),
        },
    }


def format_bibliography(ctx: dict[str, Any]) -> dict[str, Any]:
    """Format sources as bibliography entries.

    Args:
        ctx: Workflow context with validated_sources

    Returns:
        Formatted bibliography
    """
    sources = ctx.get("validated_sources", [])
    style = ctx.get("citation_style", "apa")

    entries = []
    for source in sources:
        if isinstance(source, dict):
            entry = {
                "title": source.get("title", "Unknown"),
                "authors": source.get("authors", []),
                "year": source.get("year", "n.d."),
                "url": source.get("url", ""),
                "source_type": source.get("source_type", "web"),
            }
            entries.append(entry)

    return {
        "entries": entries,
        "count": len(entries),
        "style": style,
    }


# =============================================================================
# Chat Workflow Escape Hatches
# =============================================================================


def research_type_check(ctx: dict[str, Any]) -> str:
    """Determine research type for workflow routing.

    Args:
        ctx: Workflow context with keys:
            - research_type (str): Type from understand_request
            - user_message (str): Original user message

    Returns:
        "quick", "deep", or "fact_check"
    """
    research_type = ctx.get("research_type", "").lower()

    if research_type in ["quick", "simple", "basic"]:
        return "quick"

    if research_type in ["fact", "verify", "check", "confirm"]:
        return "fact_check"

    return "deep"


def answer_quality_check(ctx: dict[str, Any]) -> str:
    """Check if answer quality is sufficient.

    Args:
        ctx: Workflow context with keys:
            - answer_found (bool): Whether answer was found
            - confidence_level (str): Confidence in answer
            - search_results (list): Search results

    Returns:
        "sufficient" or "insufficient"
    """
    answer_found = ctx.get("answer_found", False)
    confidence = ctx.get("confidence_level", "").lower()

    if answer_found and confidence in ["high", "very_high"]:
        return "sufficient"

    return "insufficient"


def source_coverage_chat_check(ctx: dict[str, Any]) -> str:
    """Check if source coverage is adequate for chat workflow.

    Args:
        ctx: Workflow context with keys:
            - validated_sources (list): Validated sources
            - quality_score (float): Average quality score

    Returns:
        "adequate" or "inadequate"
    """
    sources = ctx.get("validated_sources", [])
    quality_score = ctx.get("quality_score", 0)

    if len(sources) >= 3 and quality_score >= 0.6:
        return "adequate"

    return "inadequate"


def can_continue_research_iteration(ctx: dict[str, Any]) -> str:
    """Check if workflow can continue with another iteration.

    Args:
        ctx: Workflow context with keys:
            - iteration_count (int): Current iteration count
            - max_iterations (int): Maximum allowed iterations

    Returns:
        "continue" or "max_reached"
    """
    iteration = ctx.get("iteration_count", 0)
    max_iter = ctx.get("max_iterations", 50)

    if iteration < max_iter:
        return "continue"
    return "max_reached"


def has_pending_tool_calls(ctx: dict[str, Any]) -> str:
    """Check if there are pending tool calls to execute.

    Args:
        ctx: Workflow context with keys:
            - tool_calls (list): Tool calls from agent response
            - pending_tool_calls (list): Pending tool calls

    Returns:
        "has_tools" or "no_tools"
    """
    tool_calls = ctx.get("tool_calls", [])
    pending = ctx.get("pending_tool_calls", [])

    if tool_calls or pending:
        return "has_tools"
    return "no_tools"


def update_research_conversation(ctx: dict[str, Any]) -> dict[str, Any]:
    """Update conversation with tool results from execution.

    Args:
        ctx: Workflow context with tool execution results

    Returns:
        Updated conversation state
    """
    conversation_history = ctx.get("conversation_history", [])
    content = ctx.get("content", "")
    tool_calls = ctx.get("tool_calls", [])
    tool_results = ctx.get("tool_results", [])

    # Add assistant message with tool calls
    if tool_calls:
        conversation_history.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        )

        # Add tool results
        for result in tool_results:
            conversation_history.append(
                {
                    "role": "tool",
                    "content": str(result),
                }
            )

    # Increment iteration count
    iteration_count = ctx.get("iteration_count", 0) + 1

    return {
        "conversation_history": conversation_history,
        "iteration_count": iteration_count,
        "last_content": content,
    }


def finalize_research_chat(ctx: dict[str, Any]) -> dict[str, Any]:
    """Format final research chat response for user.

    Args:
        ctx: Workflow context with all conversation data

    Returns:
        Formatted final response
    """
    content = ctx.get("content", "")
    iteration_count = ctx.get("iteration_count", 0)
    synthesis = ctx.get("synthesis", "")
    key_points = ctx.get("key_points", [])
    citations = ctx.get("citations", [])

    final_response = content
    if not final_response:
        final_response = synthesis if synthesis else "Research completed."

    return {
        "final_response": final_response,
        "status": "completed",
        "iterations": iteration_count,
        "research_summary": {
            "synthesis": synthesis,
            "key_points": key_points,
            "sources_count": len(citations),
        },
        "sources": citations,
        "citations": citations,
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    # Original conditions
    "source_coverage_check": source_coverage_check,
    "should_search_more": should_search_more,
    "source_credibility_check": source_credibility_check,
    "fact_verdict": fact_verdict,
    "literature_relevance": literature_relevance,
    "competitive_threat_level": competitive_threat_level,
    # Chat workflow conditions
    "research_type_check": research_type_check,
    "answer_quality_check": answer_quality_check,
    "source_coverage_chat_check": source_coverage_chat_check,
    "can_continue_research_iteration": can_continue_research_iteration,
    "has_pending_tool_calls": has_pending_tool_calls,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    # Original transforms
    "merge_search_results": merge_search_results,
    "format_bibliography": format_bibliography,
    # Chat workflow transforms
    "update_research_conversation": update_research_conversation,
    "finalize_research_chat": finalize_research_chat,
}

__all__ = [
    # Original Conditions
    "source_coverage_check",
    "should_search_more",
    "source_credibility_check",
    "fact_verdict",
    "literature_relevance",
    "competitive_threat_level",
    # Chat workflow conditions
    "research_type_check",
    "answer_quality_check",
    "source_coverage_chat_check",
    "can_continue_research_iteration",
    "has_pending_tool_calls",
    # Original Transforms
    "merge_search_results",
    "format_bibliography",
    # Chat workflow transforms
    "update_research_conversation",
    "finalize_research_chat",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
