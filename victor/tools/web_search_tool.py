"""Web search tool for accessing online information.

This tool provides:
1. Web search using DuckDuckGo (no API key required)
2. Result parsing and summarization
3. Content extraction from URLs
4. Source citations
5. Context injection for LLM
"""

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from victor.tools.base import BaseTool, ToolParameter, ToolResult


class WebSearchTool(BaseTool):
    """Web search with result extraction and summarization."""

    def __init__(self, provider=None, model: Optional[str] = None, max_results: int = 5):
        """Initialize web search tool.

        Args:
            provider: LLM provider for summarization
            model: Model to use for summarization
            max_results: Maximum number of results to return
        """
        super().__init__()
        self.provider = provider
        self.model = model
        self.max_results = max_results
        self.user_agent = "Mozilla/5.0 (compatible; Victor/1.0; +https://github.com/vijaykumar/victor)"

    @property
    def name(self) -> str:
        """Get tool name."""
        return "web_search"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Search the web and extract information.

Provides web search capabilities using DuckDuckGo:
- No API key required
- Privacy-focused search
- Result extraction and parsing
- Content fetching from URLs
- Optional AI summarization
- Source citations

Operations:
- search: Search the web for a query
- fetch: Fetch and extract content from a URL
- summarize: Search and summarize results with AI

Example workflows:
1. Quick search:
   web_search(operation="search", query="Python async programming")

2. Fetch specific URL:
   web_search(operation="fetch", url="https://example.com/article")

3. AI-summarized search:
   web_search(operation="summarize", query="latest in AI", max_results=3)
"""

    @property
    def parameters(self) -> List[ToolParameter]:
        """Get tool parameters."""
        return [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: search, fetch, summarize",
                required=True
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Search query (for search and summarize operations)",
                required=False
            ),
            ToolParameter(
                name="url",
                type="string",
                description="URL to fetch (for fetch operation)",
                required=False
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results (default: 5)",
                required=False
            ),
            ToolParameter(
                name="region",
                type="string",
                description="Region for search results (e.g., 'us-en', 'uk-en')",
                required=False
            ),
            ToolParameter(
                name="safe_search",
                type="string",
                description="Safe search level: 'on', 'moderate', 'off' (default: 'moderate')",
                required=False
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute web search operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with search results or error
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation"
            )

        try:
            if operation == "search":
                return await self._search(kwargs)
            elif operation == "fetch":
                return await self._fetch(kwargs)
            elif operation == "summarize":
                return await self._summarize(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Web search error: {str(e)}"
            )

    async def _search(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Search the web using DuckDuckGo.

        Args:
            kwargs: Search parameters

        Returns:
            Tool result with search results
        """
        query = kwargs.get("query")
        if not query:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: query"
            )

        max_results = kwargs.get("max_results", self.max_results)
        region = kwargs.get("region", "wt-wt")  # Worldwide
        safe_search = kwargs.get("safe_search", "moderate")

        # Map safe search to DuckDuckGo values
        safe_map = {"on": "1", "moderate": "-1", "off": "-2"}
        safe_value = safe_map.get(safe_search, "-1")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # DuckDuckGo HTML search
                search_url = f"https://html.duckduckgo.com/html/"

                data = {
                    "q": query,
                    "kl": region,
                    "p": safe_value
                }

                response = await client.post(
                    search_url,
                    data=data,
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True
                )

                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Search failed with status {response.status_code}"
                    )

                # Parse results
                results = self._parse_ddg_results(response.text, max_results)

                if not results:
                    return ToolResult(
                        success=True,
                        output="No results found",
                        error=""
                    )

                # Format results
                output = self._format_results(query, results)

                return ToolResult(
                    success=True,
                    output=output,
                    error=""
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error="Search request timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {str(e)}"
            )

    def _parse_ddg_results(self, html: str, max_results: int) -> List[Dict[str, str]]:
        """Parse DuckDuckGo HTML results.

        Args:
            html: HTML response
            max_results: Maximum number of results

        Returns:
            List of result dictionaries
        """
        soup = BeautifulSoup(html, 'html.parser')
        results = []

        # Find result divs
        for result in soup.find_all('div', class_='result', limit=max_results):
            try:
                # Extract title
                title_elem = result.find('a', class_='result__a')
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')

                # Extract snippet
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet
                    })

            except Exception:
                continue

        return results

    def _format_results(self, query: str, results: List[Dict[str, str]]) -> str:
        """Format search results as text.

        Args:
            query: Search query
            results: List of results

        Returns:
            Formatted results string
        """
        output = [f"Search results for: {query}"]
        output.append("=" * 70)

        for i, result in enumerate(results, 1):
            output.append(f"\n{i}. {result['title']}")
            output.append(f"   URL: {result['url']}")
            if result['snippet']:
                output.append(f"   {result['snippet']}")

        output.append(f"\nFound {len(results)} result(s)")

        return "\n".join(output)

    async def _fetch(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Fetch and extract content from a URL.

        Args:
            kwargs: Fetch parameters

        Returns:
            Tool result with extracted content
        """
        url = kwargs.get("url")
        if not url:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: url"
            )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True
                )

                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Failed to fetch URL (status {response.status_code})"
                    )

                # Extract text content
                content = self._extract_content(response.text)

                if not content:
                    return ToolResult(
                        success=False,
                        output="",
                        error="No content could be extracted from URL"
                    )

                return ToolResult(
                    success=True,
                    output=f"Content from {url}:\n\n{content}",
                    error=""
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error="Request timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to fetch URL: {str(e)}"
            )

    def _extract_content(self, html: str, max_length: int = 5000) -> str:
        """Extract main content from HTML.

        Args:
            html: HTML content
            max_length: Maximum content length

        Returns:
            Extracted text content
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Try to find main content area
        content_areas = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', class_=re.compile('content|main|article', re.I)),
            soup.find('body')
        ]

        for area in content_areas:
            if area:
                text = area.get_text(separator='\n', strip=True)
                # Clean up whitespace
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r' +', ' ', text)

                if len(text) > 100:  # Minimum content length
                    return text[:max_length]

        return ""

    async def _summarize(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Search and summarize results with AI.

        Args:
            kwargs: Search parameters

        Returns:
            Tool result with summarized information
        """
        if not self.provider:
            return ToolResult(
                success=False,
                output="",
                error="No LLM provider available for summarization"
            )

        # First, perform search
        search_result = await self._search(kwargs)

        if not search_result.success:
            return search_result

        # Parse results
        query = kwargs.get("query")
        results_text = search_result.output

        # Prepare prompt for summarization
        prompt = f"""Analyze these web search results and provide a comprehensive summary.

Query: {query}

Search Results:
{results_text}

Provide:
1. A clear summary of the key information
2. Important findings or facts
3. Any conflicting information
4. Sources used (include URLs)

Format the response clearly with sections."""

        try:
            from victor.providers.base import Message

            response = await self.provider.complete(
                model=self.model or "default",
                messages=[Message(role="user", content=prompt)],
                temperature=0.5,
                max_tokens=1000
            )

            summary = response.content.strip()

            return ToolResult(
                success=True,
                output=f"AI Summary for: {query}\n{'=' * 70}\n\n{summary}\n\n{'=' * 70}\n\nOriginal Results:\n{results_text}",
                error=""
            )

        except Exception as e:
            # Fallback to just search results
            return ToolResult(
                success=True,
                output=f"Search results (AI summarization failed: {e}):\n\n{results_text}",
                error=""
            )
