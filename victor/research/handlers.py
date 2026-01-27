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

"""Research vertical compute handlers.

Domain-specific handlers for research workflows:
- web_scraper: Structured web content extraction
- citation_formatter: Reference formatting

Usage:
    # Handlers are auto-registered via @handler_decorator
    # Just import this module to register them

    # In YAML workflow:
    - id: scrape_page
      type: compute
      handler: web_scraper
      inputs:
        url: $ctx.target_url
        selectors:
          title: h1
          content: article
      output: scraped_data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from victor.framework.workflows.base_handler import BaseHandler
from victor.framework.handler_registry import handler_decorator

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import WorkflowContext

logger = logging.getLogger(__name__)


@handler_decorator("web_scraper", description="Structured web content extraction")
@dataclass
class WebScraperHandler(BaseHandler):
    """Structured web content extraction.

    Fetches and parses web content.

    Example YAML:
        - id: scrape_page
          type: compute
          handler: web_scraper
          inputs:
            url: $ctx.target_url
            selectors:
              title: h1
              content: article
          output: scraped_data
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute web scraping."""
        url = node.input_mapping.get("url", "")
        if isinstance(url, str) and url.startswith("$ctx."):
            url = context.get(url[5:]) or url

        selectors_input: Any = node.input_mapping.get("selectors", {})
        selectors: Dict[str, Any] = selectors_input if isinstance(selectors_input, dict) else {}

        result = await tool_registry.execute(
            "web_fetch",
            url=url,
            selectors=selectors,
            _exec_ctx=context,
        )

        # Raise exception if tool execution failed
        if not result.success:
            raise Exception(result.error or "Web fetch failed")

        output = {
            "url": url,
            "success": result.success,
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }

        return output, 1


@handler_decorator("citation_formatter", description="Format references and citations")
@dataclass
class CitationFormatterHandler(BaseHandler):
    """Format references and citations.

    Converts references to standard citation formats.

    Example YAML:
        - id: format_refs
          type: compute
          handler: citation_formatter
          inputs:
            references: $ctx.raw_references
            style: apa
          output: formatted_citations
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute citation formatting."""
        refs_key = node.input_mapping.get("references")
        references = context.get(refs_key) if refs_key else []
        style = node.input_mapping.get("style", "apa")

        ref_list = references if isinstance(references, list) else [references]
        formatted = [self._format_citation(ref, style) for ref in ref_list]

        output = {
            "style": style,
            "count": len(formatted),
            "citations": formatted,
        }

        return output, 0

    def _format_citation(self, ref: Dict[str, Any], style: str) -> str:
        """Format a single citation."""
        if not isinstance(ref, dict):
            return str(ref)

        authors = ref.get("authors", ["Unknown"])
        year = ref.get("year", "n.d.")
        title = ref.get("title", "Untitled")
        source = ref.get("source", "")

        if style == "apa":
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            return f"{author_str} ({year}). {title}. {source}"
        elif style == "mla":
            author_str = ", ".join(authors[:3])
            return f'{author_str}. "{title}." {source}, {year}.'
        elif style == "chicago":
            author_str = ", ".join(authors)
            return f"{author_str}. {title}. {source}, {year}."
        else:
            return f"{authors[0] if authors else 'Unknown'} ({year}). {title}"


__all__ = [
    "WebScraperHandler",
    "CitationFormatterHandler",
    "HANDLERS",
    "register_handlers",
]

# Handler registry for tests and programmatic access
HANDLERS = {
    "web_scraper": WebScraperHandler,
    "citation_formatter": CitationFormatterHandler,
}


def register_handlers() -> None:
    """Register Research handlers with the step handler registry.

    This function is called automatically when the Research vertical
    is loaded. It can also be called manually for testing purposes.

    Example:
        from victor.research.handlers import register_handlers
        register_handlers()
    """
    # Handlers are auto-registered via StepHandlerRegistry
    # This function exists for API compatibility and testing
    pass
