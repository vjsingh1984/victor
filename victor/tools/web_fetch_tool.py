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

"""Web content fetching with HTML-to-markdown conversion.

Fetches URLs and converts HTML to clean readable text/markdown.
Supports configurable content limits and domain restrictions.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from victor.tools.decorators import tool
from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; Victor/1.0; +https://github.com/vijaykumar/victor)"


@dataclass
class WebFetchConfig:
    """Configuration for a web fetch request.

    Args:
        url: The URL to fetch.
        max_content_length: Maximum character length of the returned content.
        allowed_domains: If set, only these domains are permitted.
        blocked_domains: If set, these domains are rejected.
        timeout_seconds: HTTP request timeout in seconds.
    """

    url: str
    max_content_length: int = 5000
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    timeout_seconds: int = 30


@dataclass
class WebFetchResult:
    """Result of a web fetch operation.

    Args:
        url: The requested URL.
        status_code: HTTP status code (0 if the request failed).
        content: Extracted text content.
        content_type: The Content-Type header value.
        bytes_fetched: Raw response body size in bytes.
        duration_ms: Wall-clock time for the request in milliseconds.
        error: Error description if the request failed.
    """

    url: str
    status_code: int = 0
    content: str = ""
    content_type: str = ""
    bytes_fetched: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None


def _check_domain_restrictions(
    url: str,
    allowed_domains: Optional[List[str]],
    blocked_domains: Optional[List[str]],
) -> Optional[str]:
    """Validate the URL against domain allow/block lists.

    Returns an error message string if the domain is restricted, or
    ``None`` if the request is permitted.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if allowed_domains is not None:
        if not any(hostname == d or hostname.endswith("." + d) for d in allowed_domains):
            return f"Domain '{hostname}' is not in the allowed domains list"

    if blocked_domains is not None:
        if any(hostname == d or hostname.endswith("." + d) for d in blocked_domains):
            return f"Domain '{hostname}' is blocked"

    return None


# ---------------------------------------------------------------------------
# HTML to readable text / markdown conversion
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(
    r"<\s*(script|style|noscript)[^>]*>.*?</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

_HEADER_RE = re.compile(
    r"<\s*(h[1-6])[^>]*>(.*?)</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

_LINK_RE = re.compile(
    r'<\s*a\s[^>]*href\s*=\s*["\']([^"\']*)["\'][^>]*>(.*?)</\s*a\s*>',
    re.DOTALL | re.IGNORECASE,
)

_LIST_ITEM_RE = re.compile(
    r"<\s*li[^>]*>(.*?)</\s*li\s*>",
    re.DOTALL | re.IGNORECASE,
)

_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)

_BLOCK_TAG_RE = re.compile(
    r"<\s*/?\s*(?:div|p|section|article|header|footer|nav|main|aside|blockquote|pre|table|tr|td|th|ul|ol|dl|dt|dd|figure|figcaption|form|fieldset)\b[^>]*>",
    re.IGNORECASE,
)

_ALL_TAGS_RE = re.compile(r"<[^>]+>")

_WHITESPACE_COLLAPSE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _html_to_readable_text(html: str, max_length: int) -> str:
    """Convert raw HTML to clean readable markdown-ish text.

    The conversion is intentionally lightweight (no external dependency on
    html2text or similar). It handles the most common structural elements
    to produce output that is useful as LLM context.

    Args:
        html: Raw HTML string.
        max_length: Maximum character length of the returned text.

    Returns:
        Cleaned plain-text / light-markdown string.
    """
    text = html

    # 1. Remove script, style, noscript blocks.
    text = _SCRIPT_STYLE_RE.sub("", text)

    # 2. Remove HTML comments.
    text = _COMMENT_RE.sub("", text)

    # 3. Convert headers to markdown.
    def _header_repl(m: re.Match[str]) -> str:
        level = int(m.group(1)[1])  # h1 -> 1, h6 -> 6
        inner = _ALL_TAGS_RE.sub("", m.group(2)).strip()
        return "\n\n" + "#" * level + " " + inner + "\n\n"

    text = _HEADER_RE.sub(_header_repl, text)

    # 4. Convert links to markdown format.
    def _link_repl(m: re.Match[str]) -> str:
        href = m.group(1).strip()
        label = _ALL_TAGS_RE.sub("", m.group(2)).strip()
        if not label:
            return href
        return f"[{label}]({href})"

    text = _LINK_RE.sub(_link_repl, text)

    # 5. Convert list items to markdown bullets.
    def _li_repl(m: re.Match[str]) -> str:
        inner = _ALL_TAGS_RE.sub("", m.group(1)).strip()
        return "\n- " + inner

    text = _LIST_ITEM_RE.sub(_li_repl, text)

    # 6. Replace <br> with newlines.
    text = _BR_RE.sub("\n", text)

    # 7. Replace block-level tags with newlines for visual separation.
    text = _BLOCK_TAG_RE.sub("\n", text)

    # 8. Strip remaining HTML tags.
    text = _ALL_TAGS_RE.sub("", text)

    # 9. Decode common HTML entities.
    entity_map = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
        "&nbsp;": " ",
        "&mdash;": "--",
        "&ndash;": "-",
        "&hellip;": "...",
        "&laquo;": "<<",
        "&raquo;": ">>",
    }
    for entity, char in entity_map.items():
        text = text.replace(entity, char)

    # 10. Collapse whitespace.
    text = _WHITESPACE_COLLAPSE_RE.sub(" ", text)
    # Normalise line breaks: strip trailing spaces per line, collapse
    # excessive blank lines.
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = text.strip()

    # 11. Truncate to max length.
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0] + "\n\n[Content truncated]"

    return text


@tool(
    category="web",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.SAFE,
    cost_tier=CostTier.LOW,
    execution_category=ExecutionCategory.NETWORK,
    stages=["exploration", "execution"],
    task_types=["analysis", "research"],
    keywords=["fetch", "url", "web", "html", "download", "page", "website"],
)
async def web_fetch(config: WebFetchConfig) -> WebFetchResult:
    """Fetch a URL and return its content as readable text.

    For HTML responses the content is converted to lightweight markdown.
    For other content types the raw text body is returned (truncated to
    ``max_content_length``).

    Args:
        config: Fetch parameters including URL, limits, and domain rules.

    Returns:
        A :class:`WebFetchResult` with the fetched content or an error.
    """
    # --- Domain restriction check ---
    domain_error = _check_domain_restrictions(
        config.url, config.allowed_domains, config.blocked_domains
    )
    if domain_error is not None:
        logger.warning("Domain restriction: %s", domain_error)
        return WebFetchResult(url=config.url, error=domain_error)

    start = time.monotonic()

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(config.url)
    except httpx.TimeoutException:
        duration = (time.monotonic() - start) * 1000
        logger.error("Timeout fetching %s", config.url)
        return WebFetchResult(
            url=config.url,
            duration_ms=duration,
            error=f"Request timed out after {config.timeout_seconds}s",
        )
    except httpx.HTTPError as exc:
        duration = (time.monotonic() - start) * 1000
        logger.error("HTTP error fetching %s: %s", config.url, exc)
        return WebFetchResult(
            url=config.url,
            duration_ms=duration,
            error=str(exc),
        )

    duration = (time.monotonic() - start) * 1000
    raw_bytes = len(response.content)
    content_type = response.headers.get("content-type", "")

    # Determine if the response is HTML.
    is_html = "text/html" in content_type

    body_text = response.text
    if is_html:
        content = _html_to_readable_text(body_text, config.max_content_length)
    else:
        content = body_text[: config.max_content_length]

    logger.info(
        "Fetched %s (%d bytes, %s) in %.0f ms",
        config.url,
        raw_bytes,
        content_type.split(";")[0],
        duration,
    )

    return WebFetchResult(
        url=config.url,
        status_code=response.status_code,
        content=content,
        content_type=content_type,
        bytes_fetched=raw_bytes,
        duration_ms=duration,
    )
