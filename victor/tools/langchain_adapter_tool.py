# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""LangChain tool adapter — projects LangChain tools as native Victor tools.

Adapter Pattern: LangChainAdapterTool wraps a LangChain BaseTool into
Victor's BaseTool interface. The LLM sees native names and JSON Schema
parameters — indistinguishable from built-in tools.

Factory Pattern: LangChainToolProjector batch-creates adapters from
a list of LangChain tools with name collision handling.

Requires: pip install victor-ai[langchain]

Usage:
    from langchain_community.tools import DuckDuckGoSearchRun
    from victor.tools.langchain_adapter_tool import LangChainAdapterTool

    search = DuckDuckGoSearchRun()
    adapter = LangChainAdapterTool(search)
    tool_registry.register(adapter)

    # Or batch:
    from victor.tools.langchain_adapter_tool import LangChainToolProjector
    adapters = LangChainToolProjector.project([tool1, tool2, tool3])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.tools.base import BaseTool, CostTier, ToolResult

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool as LCBaseTool

logger = logging.getLogger(__name__)


def _pydantic_to_json_schema(args_schema: Any) -> Dict[str, Any]:
    """Convert LangChain args_schema to JSON Schema dict.

    Handles:
    - Pydantic v2 models (.model_json_schema())
    - Pydantic v1 models (.schema())
    - Already-converted dicts (passthrough)
    - None (empty schema)
    """
    if args_schema is None:
        return {"type": "object", "properties": {}}

    if isinstance(args_schema, dict):
        return args_schema

    # Pydantic v2
    if hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:
            pass

    # Pydantic v1
    if hasattr(args_schema, "schema"):
        try:
            return args_schema.schema()
        except Exception:
            pass

    return {"type": "object", "properties": {}}


class LangChainAdapterTool(BaseTool):
    """Adapts a LangChain BaseTool to Victor's BaseTool interface.

    Makes LangChain tools first-class citizens in Victor's tool ecosystem.
    The LLM sees native names, descriptions, and JSON Schema parameters.
    Execution routes through LangChain's ainvoke() for async compatibility.
    """

    def __init__(
        self,
        langchain_tool: "LCBaseTool",
        name_prefix: str = "",
        source: str = "langchain",
    ):
        self._lc_tool = langchain_tool
        self._name_prefix = name_prefix
        self._source = source
        self._json_schema = _pydantic_to_json_schema(getattr(langchain_tool, "args_schema", None))

    @property
    def name(self) -> str:
        base_name = self._lc_tool.name
        if self._name_prefix:
            return f"{self._name_prefix}_{base_name}"
        return base_name

    @property
    def description(self) -> str:
        desc = self._lc_tool.description or f"LangChain tool: {self._lc_tool.name}"
        return f"{desc} (via LangChain)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._json_schema

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.MEDIUM  # External execution

    @property
    def is_idempotent(self) -> bool:
        return False  # Cannot guarantee for LangChain tools

    @property
    def default_schema_level(self) -> str:
        """LangChain tools default to STUB schema for token efficiency."""
        return "stub"

    @property
    def langchain_tool_name(self) -> str:
        """The original LangChain tool name (without prefix)."""
        return self._lc_tool.name

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute by routing through LangChain's ainvoke().

        ainvoke() handles async/sync bridging internally — if the tool
        only implements _run(), LangChain runs it in a thread pool.
        """
        try:
            # LangChain ainvoke accepts dict input
            tool_input = kwargs if kwargs else {}
            result = await self._lc_tool.ainvoke(tool_input)

            return ToolResult(
                success=True,
                output=result,
                error=None,
                metadata={
                    "source": self._source,
                    "langchain_tool": self._lc_tool.name,
                },
            )
        except Exception as e:
            logger.warning("LangChain tool %s failed: %s", self.name, e)
            return ToolResult(
                success=False,
                output="",
                error=f"LangChain tool execution failed: {e}",
                metadata={"source": self._source},
            )


class LangChainToolProjector:
    """Factory for batch-adapting LangChain tools with collision handling.

    Same pattern as MCPToolProjector — iterates tools, creates adapters,
    handles name collisions.
    """

    @staticmethod
    def project(
        tools: List["LCBaseTool"],
        prefix: str = "",
        conflict_strategy: str = "prefix_source",
    ) -> List[LangChainAdapterTool]:
        """Create adapter tools for a list of LangChain tools.

        Args:
            tools: List of LangChain BaseTool instances
            prefix: Optional prefix for all tool names
            conflict_strategy: How to handle name collisions:
                "prefix_source" — prepend source index (default)
                "skip" — skip duplicates, keep first

        Returns:
            List of LangChainAdapterTool instances ready for registration
        """
        adapted: List[LangChainAdapterTool] = []
        seen_names: Dict[str, int] = {}

        for tool in tools:
            adapter = LangChainAdapterTool(tool, name_prefix=prefix)
            tool_name = adapter.name

            if tool_name in seen_names:
                if conflict_strategy == "skip":
                    logger.debug("Skipping duplicate LangChain tool: %s", tool_name)
                    continue
                # prefix_source: add index suffix
                idx = seen_names[tool_name]
                adapter = LangChainAdapterTool(
                    tool,
                    name_prefix=f"{prefix}_{idx}" if prefix else f"lc_{idx}",
                )
                tool_name = adapter.name

            seen_names[tool_name] = seen_names.get(tool_name, 0) + 1
            adapted.append(adapter)

        logger.info("Projected %d LangChain tools", len(adapted))
        return adapted


__all__ = [
    "LangChainAdapterTool",
    "LangChainToolProjector",
]
