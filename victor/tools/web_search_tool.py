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

"""Web search tool for accessing online information.

This tool provides:
1. Web search using DuckDuckGo (no API key required)
2. Result parsing and summarization
3. Content extraction from URLs
4. Source citations
5. Context injection for LLM
"""

import re
import logging
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool

# Global provider and model for AI summarization (set by orchestrator)
_provider = None
_model: Optional[str] = None
_user_agent = "Mozilla/5.0 (compatible; Victor/1.0; +https://github.com/vijaykumar/victor)"
_config = {
    "fetch_top": None,
    "fetch_pool": None,
    "max_content_length": 5000,
}


def set_web_tool_defaults(
    fetch_top: Optional[int] = None,
    fetch_pool: Optional[int] = None,
    max_content_length: Optional[int] = None,
) -> None:
    """Set default behaviors for web_summarize."""
    if fetch_top is not None:
        _config["fetch_top"] = fetch_top
    if fetch_pool is not None:
        _config["fetch_pool"] = fetch_pool
    if max_content_length is not None:
        _config["max_content_length"] = max_content_length


def set_web_search_provider(provider, model: Optional[str] = None) -> None:
    """Set the global provider and model for web search AI summarization.

    Args:
        provider: LLM provider instance for summarization
        model: Model identifier to use for summarization
    """
    global _provider, _model
    _provider = provider
    _model = model


def _parse_ddg_results(html: str, max_results: int) -> List[Dict[str, str]]:
    """Parse DuckDuckGo HTML results.

    Args:
        html: HTML response
        max_results: Maximum number of results

    Returns:
        List of result dictionaries
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Find result divs
    for result in soup.find_all("div", class_="result", limit=max_results):
        try:
            # Extract title
            title_elem = result.find("a", class_="result__a")
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)
            url = title_elem.get("href", "")

            # Extract snippet
            snippet_elem = result.find("a", class_="result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})

        except Exception:
            continue

    return results


def _format_results(query: str, results: List[Dict[str, str]]) -> str:
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
        if result["snippet"]:
            output.append(f"   {result['snippet']}")

    output.append(f"\nFound {len(results)} result(s)")

    return "\n".join(output)


def _extract_content(html: str, max_length: int = 5000) -> str:
    """Extract main content from HTML.

    Args:
        html: HTML content
        max_length: Maximum content length

    Returns:
        Extracted text content
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Try to find main content area
    content_areas = [
        soup.find("main"),
        soup.find("article"),
        soup.find("div", class_=re.compile("content|main|article", re.I)),
        soup.find("body"),
    ]

    for area in content_areas:
        if area:
            text = area.get_text(separator="\n", strip=True)
            # Clean up whitespace
            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = re.sub(r" +", " ", text)

            if len(text) > 100:  # Minimum content length
                return text[:max_length]

    return ""


@tool(
    cost_tier=CostTier.MEDIUM,
    category="web",
    priority=Priority.CONTEXTUAL,  # Only used when web search is requested
    access_mode=AccessMode.NETWORK,  # Makes external HTTP requests
    danger_level=DangerLevel.SAFE,  # No local side effects
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["query"],  # Different queries indicate progress, not loops
    stages=["planning", "initial"],  # Conversation stages where relevant
    task_types=["research", "analysis"],  # Task types for classification-aware selection
    execution_category="network",  # Can run in parallel with read-only ops
    keywords=[
        "search",
        "web",
        "internet",
        "lookup",
        "find online",
        "google",
        "duckduckgo",
        "summarize",
    ],
    mandatory_keywords=["search web", "search online", "look up online"],  # Force inclusion
)
async def web(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safe_search: str = "moderate",
    ai_summarize: bool = False,
    fetch_top: Optional[int] = None,
    fetch_pool: Optional[int] = None,
    max_content_length: int = 5000,
) -> Dict[str, Any]:
    """Search the web using DuckDuckGo. Optionally summarize with AI.

    Purpose:
    - Find links, docs, references on the public web.
    - Return titles, URLs, and snippets for relevance checking.
    - Optionally use AI to summarize search results.
    - Ideal when the user says "search the web", "find online", "lookup", "docs", "articles".

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5).
        region: Region for search results, e.g., 'us-en', 'uk-en', 'wt-wt' for worldwide (default: 'wt-wt').
        safe_search: Safe search level - 'on', 'moderate', 'off' (default: 'moderate').
        ai_summarize: If True, use AI to summarize results (default: False).
        fetch_top: Number of URLs to fetch for deeper summary (only with ai_summarize=True).
        fetch_pool: Pool of URLs to try fetching from (only with ai_summarize=True).
        max_content_length: Max content length to extract per URL (only with ai_summarize=True).

    Returns:
        Dictionary containing:
        - success: Whether search succeeded
        - results: Formatted search results text
        - result_count: Number of results found
        - summary: AI summary (only if ai_summarize=True)
        - error: Error message if failed
    """
    # Route to summarize implementation if ai_summarize is True
    if ai_summarize:
        return await _summarize_search(
            query, max_results, region, safe_search, fetch_top, fetch_pool, max_content_length
        )
    if not query:
        return {"success": False, "error": "Missing required parameter: query"}

    # Map safe search to DuckDuckGo values
    safe_map = {"on": "1", "moderate": "-1", "off": "-2"}
    safe_value = safe_map.get(safe_search, "-1")
    logger = logging.getLogger(__name__)
    logger = logging.getLogger(__name__)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # DuckDuckGo HTML search
            search_url = "https://html.duckduckgo.com/html/"

            data = {"q": query, "kl": region, "p": safe_value}

            response = await client.post(
                search_url, data=data, headers={"User-Agent": _user_agent}, follow_redirects=True
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Search failed with status {response.status_code}",
                }

            # Parse results
            results = _parse_ddg_results(response.text, max_results)
            logger.info(
                f"[web_search] query='{query}', max_results={max_results}, parsed_results={len(results)}"
            )
            if results:
                sample_urls = [r.get("url", "") for r in results[:5]]
                logger.info(f"[web_search] top URLs: {sample_urls}")

            if not results:
                return {"success": True, "results": "No results found", "result_count": 0}

            # Format results
            output = _format_results(query, results)

            return {"success": True, "results": output, "result_count": len(results)}

    except httpx.TimeoutException:
        return {"success": False, "error": "Search request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}


@tool(
    cost_tier=CostTier.MEDIUM,
    category="web",
    priority=Priority.MEDIUM,  # Used when fetching specific URLs
    access_mode=AccessMode.NETWORK,  # Makes external HTTP requests
    danger_level=DangerLevel.SAFE,  # No local side effects
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["url"],  # Different URLs indicate progress, not loops
    stages=["planning", "initial"],  # Conversation stages where relevant
    task_types=["research", "analysis"],  # Task types for classification-aware selection
    execution_category="network",  # Can run in parallel with read-only ops
    keywords=["fetch", "url", "webpage", "download", "http", "content"],
)
async def fetch(url: str) -> Dict[str, Any]:
    """Fetch and extract main text content from a URL.

    Args:
        url: URL to fetch content from
    """
    if not url:
        return {"success": False, "error": "Missing required parameter: url"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                url, headers={"User-Agent": _user_agent}, follow_redirects=True
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to fetch URL (status {response.status_code})",
                }

            # Extract text content
            content = _extract_content(response.text)

            if not content:
                return {"success": False, "error": "No content could be extracted from URL"}

            return {"success": True, "content": content, "url": url}

    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch URL: {str(e)}"}


async def _summarize_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safe_search: str = "moderate",
    fetch_top: Optional[int] = None,
    fetch_pool: Optional[int] = None,
    max_content_length: int = 5000,
) -> Dict[str, Any]:
    """Internal implementation for web search with AI summarization."""
    if not _provider:
        return {"success": False, "error": "No LLM provider available for summarization"}

    if not query:
        return {"success": False, "error": "Missing required parameter: query"}

    # Map safe search to DuckDuckGo values
    safe_map = {"on": "1", "moderate": "-1", "off": "-2"}
    safe_value = safe_map.get(safe_search, "-1")
    logger = logging.getLogger(__name__)

    default_fetch_top = _config.get("fetch_top")
    default_fetch_pool = _config.get("fetch_pool")
    default_max_len = _config.get("max_content_length", 5000)

    fetch_top = (
        fetch_top
        if fetch_top is not None
        else (default_fetch_top if default_fetch_top is not None else max_results)
    )
    fetch_pool = (
        fetch_pool
        if fetch_pool is not None
        else (
            default_fetch_pool
            if default_fetch_pool is not None
            else max(fetch_top + 2, max_results)
        )
    )
    max_content_length = max_content_length if max_content_length is not None else default_max_len

    fetch_top = max(0, min(fetch_top, 10))
    fetch_pool = max(fetch_top, min(fetch_pool, 12))
    max_content_length = max(500, min(max_content_length, 20000))

    try:
        # First, perform search
        async with httpx.AsyncClient(timeout=15.0) as client:
            search_url = "https://html.duckduckgo.com/html/"

            data = {"q": query, "kl": region, "p": safe_value}

            response = await client.post(
                search_url, data=data, headers={"User-Agent": _user_agent}, follow_redirects=True
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Search failed with status {response.status_code}",
                }

            # Parse results
            results = _parse_ddg_results(response.text, fetch_pool)
            logger.info(f"[web_summarize] search query='{query}', parsed_results={len(results)}")
            if results:
                logger.info(f"[web_summarize] top URLs: {[r.get('url','') for r in results[:5]]}")

            if not results:
                return {
                    "success": True,
                    "summary": "No results found to summarize",
                    "original_results": "",
                }

            # Format results
            results_text = _format_results(query, results)

        # Fetch top content for deeper summary (best-effort)
        fetched_contents = []
        for result in results[:fetch_pool]:
            if len(fetched_contents) >= fetch_top:
                break
            url = result.get("url")
            if not url:
                continue
            fetch_res = await fetch(url=url)
            if fetch_res.get("success") and fetch_res.get("content"):
                content = fetch_res["content"][:max_content_length]
                fetched_contents.append({"url": url, "content": content})

        logger.info(
            f"[web_summarize] fetch_top={fetch_top}, attempted_pool={fetch_pool}, fetched_ok={len(fetched_contents)}"
        )

        # Prepare prompt for summarization
        fetch_section = ""
        if fetched_contents:
            fetch_section = "\n\nFetched content excerpts:\n"
            for i, item in enumerate(fetched_contents, 1):
                fetch_section += (
                    f"\n{i}. URL: {item['url']}\nContent (excerpt):\n{item['content']}\n"
                )

        prompt = f"""Analyze these web search results and provide a comprehensive summary.

Query: {query}

Search Results:
{results_text}

Use fetched page excerpts if provided to improve accuracy:
{fetch_section}

Provide:
1. A clear summary of the key information
2. Important findings or facts
3. Any conflicting information
4. Sources used (include URLs)

Format the response clearly with sections."""

        logger.info(
            f"[web_summarize] building summary prompt for query='{query}', "
            f"results_count={len(results)}, prompt_chars={len(prompt)}"
        )

        try:
            from victor.providers.base import Message

            response = await _provider.complete(
                model=_model or "default",
                messages=[Message(role="user", content=prompt)],
                temperature=0.5,
                max_tokens=1000,
            )

            summary = response.content.strip()
            logger.info(
                f"[web_summarize] summarization model='{_model}', summary_len={len(summary)}"
            )

            return {"success": True, "summary": summary, "original_results": results_text}

        except Exception as e:
            # Fallback to just search results
            return {
                "success": True,
                "summary": f"AI summarization failed: {e}",
                "original_results": results_text,
            }

    except httpx.TimeoutException:
        return {"success": False, "error": "Search request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}
