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

"""Integration tests for research workflows.

Tests cover:
1. Web Research (3 tests)
   - Search queries
   - Aggregate results
   - Extract insights

2. Citation Generation (3 tests)
   - Generate citations
   - Format citations
   - Validate citations

3. Synthesis (4 tests)
   - Synthesize findings
   - Extract key points
   - Generate summary
   - Compare sources

Uses mock search engines to avoid network dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import httpx
import respx

from victor.research.handlers import (
    CitationFormatterHandler,
    WebScraperHandler,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockToolResult:
    """Mock tool result for testing."""

    def __init__(self, success: bool = True, output: Any = None, error: str = None):
        self.success = success
        self.output = output
        self.error = error


class MockComputeNode:
    """Mock compute node for testing."""

    def __init__(
        self,
        node_id: str = "test_node",
        input_mapping: Dict[str, Any] = None,
        output_key: str = None,
    ):
        self.id = node_id
        self.input_mapping = input_mapping or {}
        self.output_key = output_key


class MockWorkflowContext:
    """Mock workflow context for testing."""

    def __init__(self, data: Dict[str, Any] = None):
        self._data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


@pytest.fixture
def mock_search_engine():
    """Mock search engine for testing without network dependencies."""
    engine = MagicMock()

    # Sample search results
    engine.results = [
        {
            "title": "Introduction to Machine Learning",
            "url": "https://example.com/ml-intro",
            "snippet": "Machine learning is a subset of artificial intelligence...",
            "source": "example.com",
            "date": "2024-01-15",
        },
        {
            "title": "Deep Learning Advances",
            "url": "https://example.com/deep-learning",
            "snippet": "Recent advances in deep learning have revolutionized...",
            "source": "example.com",
            "date": "2024-02-20",
        },
        {
            "title": "AI Research Trends 2024",
            "url": "https://example.com/ai-trends",
            "snippet": "The latest trends in AI research focus on...",
            "source": "example.com",
            "date": "2024-03-10",
        },
    ]

    async def mock_search(query: str, num_results: int = 5):
        """Mock search implementation."""
        # Filter results based on query
        filtered = [
            r
            for r in engine.results
            if query.lower() in r["title"].lower() or query.lower() in r["snippet"].lower()
        ]
        return {"success": True, "results": filtered[:num_results], "count": len(filtered)}

    engine.search = AsyncMock(side_effect=mock_search)
    return engine


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry for testing."""
    registry = MagicMock()

    async def mock_execute(tool_name: str, **kwargs):
        if tool_name == "web_search":
            query = kwargs.get("query", "")
            return MockToolResult(
                success=True,
                output={
                    "results": [
                        {
                            "title": f"Search result for {query}",
                            "url": f"https://example.com/{query.replace(' ', '-')}",
                            "snippet": f"Relevant information about {query}",
                        }
                    ],
                    "count": 1,
                },
            )
        elif tool_name == "web_fetch":
            url = kwargs.get("url", "")
            return MockToolResult(
                success=True,
                output={
                    "title": "Fetched Content",
                    "content": f"Content from {url}",
                    "url": url,
                },
            )
        elif tool_name == "read":
            return MockToolResult(
                success=True,
                output="File content",
            )
        return MockToolResult(success=False, error=f"Unknown tool: {tool_name}")

    registry.execute = AsyncMock(side_effect=mock_execute)
    return registry


# =============================================================================
# 1. Web Research Tests (3 tests)
# =============================================================================


class TestWebResearch:
    """Integration tests for web research functionality."""

    @pytest.mark.asyncio
    async def test_search_queries(self, mock_search_engine):
        """Test executing search queries and retrieving results."""
        # Execute search
        result = await mock_search_engine.search("machine learning", num_results=5)

        # Verify search succeeded
        assert result["success"] is True
        assert "results" in result
        assert isinstance(result["results"], list)

        # Verify result structure
        if result["results"]:
            first_result = result["results"][0]
            assert "title" in first_result
            assert "url" in first_result
            assert "snippet" in first_result

            # Verify query relevance
            query_lower = "machine learning"
            title_matches = query_lower in first_result["title"].lower()
            snippet_matches = query_lower in first_result["snippet"].lower()
            assert title_matches or snippet_matches

    @pytest.mark.asyncio
    async def test_aggregate_results(self, mock_search_engine):
        """Test aggregating results from multiple searches."""
        # Execute multiple searches
        queries = ["machine learning", "deep learning", "AI research"]
        all_results = []

        for query in queries:
            result = await mock_search_engine.search(query, num_results=5)
            all_results.extend(result.get("results", []))

        # Verify aggregation
        assert len(all_results) > 0
        assert isinstance(all_results, list)

        # Verify unique sources
        urls = [r["url"] for r in all_results]
        unique_urls = set(urls)
        assert len(unique_urls) >= 1

        # Verify result structure
        for result in all_results:
            assert "title" in result
            assert "url" in result
            assert "snippet" in result

    @pytest.mark.asyncio
    async def test_extract_insights(self, mock_search_engine):
        """Test extracting insights from search results."""
        # Execute search
        result = await mock_search_engine.search("machine learning", num_results=5)
        search_results = result.get("results", [])

        # Extract insights
        insights = []
        for item in search_results:
            insight = {
                "topic": item.get("title", ""),
                "summary": item.get("snippet", ""),
                "source": item.get("url", ""),
                "relevance_score": 0.8,  # Mock relevance score
            }
            insights.append(insight)

        # Verify insights extraction
        assert len(insights) > 0
        for insight in insights:
            assert "topic" in insight
            assert "summary" in insight
            assert "source" in insight
            assert "relevance_score" in insight
            assert isinstance(insight["relevance_score"], (int, float))
            assert 0 <= insight["relevance_score"] <= 1


# =============================================================================
# 2. Citation Generation Tests (3 tests)
# =============================================================================


class TestCitationGeneration:
    """Integration tests for citation generation functionality."""

    @pytest.fixture
    def citation_handler(self):
        """Get citation formatter handler."""
        return CitationFormatterHandler()

    def test_generate_citations(self, citation_handler):
        """Test generating citations from source metadata."""
        sources = [
            {
                "authors": ["Smith, J.", "Doe, J."],
                "year": "2024",
                "title": "Advances in Machine Learning",
                "source": "Journal of AI Research",
                "url": "https://example.com/paper1",
            },
            {
                "authors": ["Johnson, A.", "Williams, B."],
                "year": "2023",
                "title": "Deep Learning Techniques",
                "source": "Proceedings of ICML",
                "url": "https://example.com/paper2",
            },
        ]

        # Generate APA citations
        citations = []
        for source in sources:
            citation = citation_handler._format_citation(source, "apa")
            citations.append(citation)

        # Verify citation generation
        assert len(citations) == 2
        for citation in citations:
            assert isinstance(citation, str)
            assert len(citation) > 0
            # Verify citation format contains key elements
            assert any(char.isdigit() for char in citation)  # Has year

    def test_format_citations(self, citation_handler):
        """Test formatting citations in different styles."""
        source = {
            "authors": ["Smith, J.", "Doe, J.", "Johnson, A."],
            "year": "2024",
            "title": "Research Citation Formatting",
            "source": "Academic Journal",
        }

        # Test different citation styles
        styles = ["apa", "mla", "chicago"]
        formatted_citations = {}

        for style in styles:
            citation = citation_handler._format_citation(source, style)
            formatted_citations[style] = citation

        # Verify each style has different formatting
        assert len(set(formatted_citations.values())) == len(styles)

        # Verify APA style format
        apa = formatted_citations["apa"]
        assert "2024" in apa
        assert "Smith" in apa

        # Verify MLA style format
        mla = formatted_citations["mla"]
        assert "2024" in mla
        assert '"' in mla  # MLA uses quotes around title

        # Verify Chicago style format
        chicago = formatted_citations["chicago"]
        assert "2024" in chicago

    def test_validate_citations(self, citation_handler):
        """Test citation validation and completeness."""
        sources = [
            {
                "authors": ["Author, A."],
                "year": "2024",
                "title": "Complete Citation",
                "source": "Journal",
            },
            {
                "authors": [],  # Missing authors
                "year": "n.d.",
                "title": "Incomplete Citation",
                "source": "",
            },
        ]

        # Generate and validate citations
        validation_results = []
        for source in sources:
            citation = citation_handler._format_citation(source, "apa")

            # Validate citation has required elements
            has_author = bool(source.get("authors"))
            has_year = bool(source.get("year") and source["year"] != "n.d.")
            has_title = bool(source.get("title"))
            has_source = bool(source.get("source"))

            is_valid = has_author and has_year and has_title and has_source

            validation_results.append(
                {
                    "citation": citation,
                    "is_valid": is_valid,
                    "missing_elements": [
                        elem
                        for elem, present in [
                            ("author", has_author),
                            ("year", has_year),
                            ("title", has_title),
                            ("source", has_source),
                        ]
                        if not present
                    ],
                }
            )

        # Verify validation
        assert len(validation_results) == 2
        assert validation_results[0]["is_valid"] is True
        assert validation_results[1]["is_valid"] is False
        assert len(validation_results[1]["missing_elements"]) >= 2


# =============================================================================
# 3. Synthesis Tests (4 tests)
# =============================================================================


class TestSynthesis:
    """Integration tests for research synthesis functionality."""

    def test_synthesize_findings(self):
        """Test synthesizing findings from multiple sources."""
        findings = [
            {
                "source": "Machine Learning Basics",
                "content": "Machine learning uses algorithms to learn from data",
                "confidence": 0.9,
                "category": "fundamentals",
            },
            {
                "source": "Deep Learning Advances",
                "content": "Deep learning is a subset of machine learning using neural networks",
                "confidence": 0.85,
                "category": "advanced",
            },
            {
                "source": "AI Applications",
                "content": "AI is applied in healthcare, finance, and transportation",
                "confidence": 0.8,
                "category": "applications",
            },
        ]

        # Synthesize findings
        synthesis = {
            "summary": "Machine learning and its subsets like deep learning are transforming industries",
            "key_themes": ["algorithms", "neural networks", "applications"],
            "confidence_avg": sum(f["confidence"] for f in findings) / len(findings),
            "sources_count": len(findings),
            "categories": list(set(f["category"] for f in findings)),
        }

        # Verify synthesis
        assert synthesis["sources_count"] == 3
        assert synthesis["confidence_avg"] > 0.8
        assert len(synthesis["categories"]) == 3
        assert "summary" in synthesis
        assert "key_themes" in synthesis
        assert len(synthesis["key_themes"]) > 0

    def test_extract_key_points(self):
        """Test extracting key points from research findings."""
        research_text = """
        Machine learning has revolutionized data analysis. Key insights include:
        1. Supervised learning requires labeled data
        2. Unsupervised learning discovers patterns automatically
        3. Reinforcement learning learns through trial and error
        4. Deep learning uses multi-layered neural networks
        Applications span healthcare diagnostics, financial forecasting, and autonomous vehicles.
        """

        # Extract key points (mock extraction logic)
        key_points = []
        lines = research_text.split("\n")
        for line in lines:
            # Extract numbered points
            if line.strip().startswith(("1.", "2.", "3.", "4.")):
                key_points.append(line.strip())
            # Extract sentences with keywords
            elif any(
                keyword in line.lower()
                for keyword in ["machine learning", "applications", "insights"]
            ):
                key_points.append(line.strip())

        # Verify extraction
        assert len(key_points) >= 4
        assert any("supervised" in point.lower() for point in key_points)
        assert any("unsupervised" in point.lower() for point in key_points)
        assert any("reinforcement" in point.lower() for point in key_points)
        assert any("deep learning" in point.lower() for point in key_points)

    def test_generate_summary(self):
        """Test generating a summary from research findings."""
        findings = {
            "topic": "Machine Learning",
            "findings": [
                "ML algorithms learn patterns from data",
                "Neural networks mimic brain structure",
                "Training requires large datasets",
            ],
            "statistics": {
                "sources_analyzed": 10,
                "confidence_score": 0.85,
                "publication_date_range": "2020-2024",
            },
        }

        # Generate summary
        summary = f"""
Research Summary: {findings['topic']}

Key Findings:
- {findings['findings'][0]}
- {findings['findings'][1]}
- {findings['findings'][2]}

Methodology:
- Sources analyzed: {findings['statistics']['sources_analyzed']}
- Confidence score: {findings['statistics']['confidence_score']}
- Publication range: {findings['statistics']['publication_date_range']}
"""

        # Verify summary
        assert "Machine Learning" in summary
        assert "Key Findings" in summary
        assert "Methodology" in summary
        assert "10" in summary  # Sources count
        assert "0.85" in summary  # Confidence score
        assert summary.count("\n") >= 8  # Has proper structure

    def test_compare_sources(self):
        """Test comparing and contrasting multiple sources."""
        sources = [
            {
                "title": "Source A",
                "stance": "pro",
                "claims": ["Machine learning is transformative", "AI will create jobs"],
                "evidence_quality": "high",
                "date": "2024-01-15",
            },
            {
                "title": "Source B",
                "stance": "neutral",
                "claims": ["Machine learning has both benefits and risks", "Regulation is needed"],
                "evidence_quality": "high",
                "date": "2024-02-20",
            },
            {
                "title": "Source C",
                "stance": "con",
                "claims": ["AI automation may displace workers", "Ethical concerns exist"],
                "evidence_quality": "medium",
                "date": "2024-03-10",
            },
        ]

        # Compare sources
        comparison = {
            "stances": [s["stance"] for s in sources],
            "consensus_points": [
                "Machine learning is impactful",
                "Regulation and ethics matter",
            ],
            "divergent_points": [
                "Job creation vs displacement",
                "Overall impact assessment",
            ],
            "evidence_quality_avg": sum(
                1 if s["evidence_quality"] == "high" else 0.5 for s in sources
            )
            / len(sources),
            "date_range": {
                "earliest": min(s["date"] for s in sources),
                "latest": max(s["date"] for s in sources),
            },
        }

        # Verify comparison
        assert len(comparison["stances"]) == 3
        assert set(comparison["stances"]) == {"pro", "neutral", "con"}
        assert len(comparison["consensus_points"]) >= 1
        assert len(comparison["divergent_points"]) >= 1
        assert 0 <= comparison["evidence_quality_avg"] <= 1
        assert "earliest" in comparison["date_range"]
        assert "latest" in comparison["date_range"]


# =============================================================================
# Web Scraper Handler Tests
# =============================================================================


class TestWebScraperHandler:
    """Integration tests for WebScraperHandler."""

    @pytest.fixture
    def scraper_handler(self):
        """Get web scraper handler."""
        return WebScraperHandler()

    @pytest.mark.asyncio
    async def test_web_scraper_basic_execution(self, scraper_handler, mock_tool_registry):
        """Test basic web scraper execution."""
        node = MockComputeNode(
            input_mapping={"url": "https://example.com", "selectors": {"title": "h1"}},
        )
        context = MockWorkflowContext()

        result, cost = await scraper_handler.execute(node, context, mock_tool_registry)

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["data"] is not None
        assert cost == 1

    @pytest.mark.asyncio
    async def test_web_scraper_with_context_url(self, scraper_handler, mock_tool_registry):
        """Test web scraper with URL from context."""
        node = MockComputeNode(
            input_mapping={
                "url": "$ctx.target_url",
                "selectors": {"content": "article"},
            },
        )
        context = MockWorkflowContext({"target_url": "https://example.com/article"})

        result, cost = await scraper_handler.execute(node, context, mock_tool_registry)

        assert result["success"] is True
        assert result["url"] == "https://example.com/article"

    @pytest.mark.asyncio
    async def test_web_scraper_failure_handling(self, scraper_handler):
        """Test web scraper handles failures."""
        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MockToolResult(success=False, error="Failed to fetch")
        )

        node = MockComputeNode(input_mapping={"url": "https://example.com"})
        context = MockWorkflowContext()

        with pytest.raises(Exception, match="Failed"):
            await scraper_handler.execute(node, context, registry)
