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

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from victor.tools.decorators import tool

# Global provider and model for AI summarization (set by orchestrator)
_provider = None
_model: Optional[str] = None
_user_agent = "Mozilla/5.0 (compatible; Victor/1.0; +https://github.com/vijaykumar/victor)"


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
        if result['snippet']:
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


@tool
async def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safe_search: str = "moderate"
) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.

    Provides privacy-focused web search without requiring API keys.
    Returns formatted search results with titles, URLs, and snippets.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5).
        region: Region for search results, e.g., 'us-en', 'uk-en', 'wt-wt' for worldwide (default: 'wt-wt').
        safe_search: Safe search level - 'on', 'moderate', 'off' (default: 'moderate').

    Returns:
        Dictionary containing:
        - success: Whether search succeeded
        - results: Formatted search results text
        - result_count: Number of results found
        - error: Error message if failed
    """
    if not query:
        return {
            "success": False,
            "error": "Missing required parameter: query"
        }

    # Map safe search to DuckDuckGo values
    safe_map = {"on": "1", "moderate": "-1", "off": "-2"}
    safe_value = safe_map.get(safe_search, "-1")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # DuckDuckGo HTML search
            search_url = "https://html.duckduckgo.com/html/"

            data = {
                "q": query,
                "kl": region,
                "p": safe_value
            }

            response = await client.post(
                search_url,
                data=data,
                headers={"User-Agent": _user_agent},
                follow_redirects=True
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Search failed with status {response.status_code}"
                }

            # Parse results
            results = _parse_ddg_results(response.text, max_results)

            if not results:
                return {
                    "success": True,
                    "results": "No results found",
                    "result_count": 0
                }

            # Format results
            output = _format_results(query, results)

            return {
                "success": True,
                "results": output,
                "result_count": len(results)
            }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Search request timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}"
        }


@tool
async def web_fetch(url: str) -> Dict[str, Any]:
    """
    Fetch and extract content from a URL.

    Downloads a web page and extracts the main text content,
    removing scripts, styles, navigation, and other non-content elements.

    Args:
        url: URL to fetch and extract content from.

    Returns:
        Dictionary containing:
        - success: Whether fetch succeeded
        - content: Extracted text content
        - url: URL that was fetched
        - error: Error message if failed
    """
    if not url:
        return {
            "success": False,
            "error": "Missing required parameter: url"
        }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                url,
                headers={"User-Agent": _user_agent},
                follow_redirects=True
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to fetch URL (status {response.status_code})"
                }

            # Extract text content
            content = _extract_content(response.text)

            if not content:
                return {
                    "success": False,
                    "error": "No content could be extracted from URL"
                }

            return {
                "success": True,
                "content": content,
                "url": url
            }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Request timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch URL: {str(e)}"
        }


@tool
async def web_summarize(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safe_search: str = "moderate"
) -> Dict[str, Any]:
    """
    Search the web and summarize results with AI.

    Performs a web search and uses an LLM to summarize the results,
    providing key information, findings, and source citations.

    Args:
        query: Search query string.
        max_results: Maximum number of results to include in summary (default: 5).
        region: Region for search results, e.g., 'us-en', 'uk-en', 'wt-wt' for worldwide (default: 'wt-wt').
        safe_search: Safe search level - 'on', 'moderate', 'off' (default: 'moderate').

    Returns:
        Dictionary containing:
        - success: Whether summarization succeeded
        - summary: AI-generated summary of search results
        - original_results: Raw search results
        - error: Error message if failed
    """
    if not _provider:
        return {
            "success": False,
            "error": "No LLM provider available for summarization"
        }

    if not query:
        return {
            "success": False,
            "error": "Missing required parameter: query"
        }

    # Map safe search to DuckDuckGo values
    safe_map = {"on": "1", "moderate": "-1", "off": "-2"}
    safe_value = safe_map.get(safe_search, "-1")

    try:
        # First, perform search
        async with httpx.AsyncClient(timeout=15.0) as client:
            search_url = "https://html.duckduckgo.com/html/"

            data = {
                "q": query,
                "kl": region,
                "p": safe_value
            }

            response = await client.post(
                search_url,
                data=data,
                headers={"User-Agent": _user_agent},
                follow_redirects=True
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Search failed with status {response.status_code}"
                }

            # Parse results
            results = _parse_ddg_results(response.text, max_results)

            if not results:
                return {
                    "success": True,
                    "summary": "No results found to summarize",
                    "original_results": ""
                }

            # Format results
            results_text = _format_results(query, results)

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

            response = await _provider.complete(
                model=_model or "default",
                messages=[Message(role="user", content=prompt)],
                temperature=0.5,
                max_tokens=1000
            )

            summary = response.content.strip()

            return {
                "success": True,
                "summary": summary,
                "original_results": results_text
            }

        except Exception as e:
            # Fallback to just search results
            return {
                "success": True,
                "summary": f"AI summarization failed: {e}",
                "original_results": results_text
            }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Search request timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}"
        }
