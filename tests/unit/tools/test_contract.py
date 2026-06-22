# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""tool-supply P6 — resolve_contract is the single tool-metadata fusion authority."""

from __future__ import annotations

from typing import Any, Dict

from victor.tools.base import BaseTool, ToolResult
from victor.tools.contract import resolve_contract
from victor.tools.enums import CostTier
from victor.tools.metadata import ToolMetadata


class _ExplicitTool:
    name = "x"
    description = "d"
    parameters: Dict[str, Any] = {}
    cost_tier = CostTier.FREE
    metadata = ToolMetadata(category="search", keywords=["k"])


class _AutogenTool:
    name = "grepper"
    description = "search files by pattern"
    parameters: Dict[str, Any] = {"type": "object"}
    cost_tier = CostTier.LOW
    metadata = None


def test_explicit_metadata_returned_as_is():
    t = _ExplicitTool()
    assert resolve_contract(t) is t.metadata


def test_autogen_matches_generate_from_tool():
    t = _AutogenTool()
    expected = ToolMetadata.generate_from_tool(
        name=t.name, description=t.description, parameters=t.parameters, cost_tier=t.cost_tier
    )
    assert resolve_contract(t).to_dict() == expected.to_dict()


def test_autogen_result_is_cached_per_instance():
    t = _AutogenTool()
    first = resolve_contract(t)
    assert resolve_contract(t) is first  # cached, not recomputed


def test_non_weakreferenceable_tool_still_resolves():
    class _Slotted:
        __slots__ = ("name", "description", "parameters", "cost_tier", "metadata")

        def __init__(self):
            self.name = "s"
            self.description = "d"
            self.parameters = {}
            self.cost_tier = CostTier.FREE
            self.metadata = None

    # __slots__ without __weakref__ -> not weak-referenceable; must not raise.
    result = resolve_contract(_Slotted())
    assert isinstance(result, ToolMetadata)


class _RealTool(BaseTool):
    @property
    def name(self) -> str:
        return "real_read"

    @property
    def description(self) -> str:
        return "read a file from disk"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self, _exec_ctx: Dict[str, Any], **kwargs: Any
    ) -> ToolResult:  # pragma: no cover
        return ToolResult(success=True, output="")


def test_base_tool_get_metadata_delegates_to_resolve_contract():
    t = _RealTool()
    # get_metadata now delegates; result must equal resolve_contract for the same tool.
    assert t.get_metadata().to_dict() == resolve_contract(t).to_dict()
