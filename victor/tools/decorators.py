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


import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from docstring_parser import parse

from victor.tools.base import BaseTool, CostTier, ToolMetadata, ToolResult


def tool(
    func: Optional[Callable] = None,
    *,
    cost_tier: CostTier = CostTier.FREE,
    category: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    use_cases: Optional[List[str]] = None,
    examples: Optional[List[str]] = None,
    priority_hints: Optional[List[str]] = None,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    A decorator that converts a Python function into a Victor tool.

    This decorator automatically extracts the tool's name, description,
    and parameters from the function's signature and docstring. It also
    supports optional semantic metadata for dynamic tool selection.

    Args:
        func: The function to wrap (optional, for @tool without parentheses)
        cost_tier: Cost tier for the tool (FREE, LOW, MEDIUM, HIGH)
        category: Tool category for semantic grouping (e.g., 'git', 'filesystem')
        keywords: Keywords that trigger this tool in user requests
        use_cases: High-level use cases for semantic matching
        examples: Example requests that should match this tool
        priority_hints: Usage hints for tool selection

    The docstring should follow the Google Python Style Guide format.

    Example with auto-generated metadata:
        @tool
        def my_tool(param1: str, param2: int = 5):
            '''This is the tool's description.

            Args:
                param1: Description of the first parameter.
                param2: Description of the second parameter.
            '''
            # ... tool logic ...

    Example with explicit metadata:
        @tool(
            cost_tier=CostTier.MEDIUM,
            category="web",
            keywords=["search", "google", "find online"],
            use_cases=["searching the web", "finding information online"],
            examples=["search for python tutorials", "find documentation"],
        )
        def web_search_tool(query: str):
            '''Search the web - makes external API calls.'''
            # ... tool logic ...
    """
    # Capture metadata parameters
    metadata_params = {
        "category": category,
        "keywords": keywords,
        "use_cases": use_cases,
        "examples": examples,
        "priority_hints": priority_hints,
    }

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            # This wrapper is what gets called if the decorated function is called directly
            return fn(*args, **kwargs)

        # Mark as tool for dynamic discovery
        wrapper._is_tool = True  # type: ignore[attr-defined]
        # We will attach a class to the wrapper that is the actual tool
        wrapper.Tool = _create_tool_class(fn, cost_tier=cost_tier, metadata_params=metadata_params)

        return wrapper

    # Support both @tool and @tool(cost_tier=...) syntax
    if func is not None:
        # Called without parentheses: @tool
        return decorator(func)
    else:
        # Called with parentheses: @tool(cost_tier=...)
        return decorator


def _create_tool_class(
    func: Callable,
    cost_tier: CostTier = CostTier.FREE,
    metadata_params: Optional[Dict[str, Any]] = None,
) -> type:
    """Dynamically creates a class that wraps the given function to act as a BaseTool.

    Args:
        func: The function to wrap
        cost_tier: The cost tier for this tool
        metadata_params: Optional dict with category, keywords, use_cases, examples, priority_hints
    """
    metadata_params = metadata_params or {}

    docstring = parse(func.__doc__ or "")
    tool_description = docstring.short_description or "No description provided."
    if docstring.long_description:
        tool_description += "\n\n" + docstring.long_description

    # Create the JSON schema for the parameters
    sig = inspect.signature(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        param_type = "string"  # Default type
        if param.annotation != inspect.Parameter.empty:
            if param.annotation in (int, float):
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"

        properties[name] = {
            "type": param_type,
            "description": param_docs.get(name, "No description."),
        }

        if param.default == inspect.Parameter.empty:
            required.append(name)

    tool_params_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # Capture closures
    _cost_tier = cost_tier
    _metadata_params = metadata_params

    # Build explicit metadata if any params were provided
    _explicit_metadata: Optional[ToolMetadata] = None
    has_explicit = any(v is not None for v in _metadata_params.values())
    if has_explicit:
        _explicit_metadata = ToolMetadata(
            category=_metadata_params.get("category") or "",
            keywords=_metadata_params.get("keywords") or [],
            use_cases=_metadata_params.get("use_cases") or [],
            examples=_metadata_params.get("examples") or [],
            priority_hints=_metadata_params.get("priority_hints") or [],
        )

    # Dynamically create the tool class
    class FunctionTool(BaseTool):
        def __init__(self, fn: Callable):
            self._fn = fn
            self._name = fn.__name__
            self._description = tool_description
            self._parameters = tool_params_schema
            self._cost_tier = _cost_tier
            self._explicit_metadata = _explicit_metadata

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return self._description

        @property
        def parameters(self) -> Dict[str, Any]:
            return self._parameters

        @property
        def cost_tier(self) -> CostTier:
            return self._cost_tier

        @property
        def metadata(self) -> Optional[ToolMetadata]:
            """Return explicit metadata if provided via decorator."""
            return self._explicit_metadata

        def get_metadata(self) -> ToolMetadata:
            """Get semantic metadata (ToolMetadataProvider contract).

            Returns explicit metadata if provided via @tool decorator,
            otherwise auto-generates from tool properties.
            """
            if self._explicit_metadata is not None:
                return self._explicit_metadata

            # Auto-generate metadata from tool properties
            return ToolMetadata.generate_from_tool(
                name=self._name,
                description=self._description,
                parameters=self._parameters,
                cost_tier=self._cost_tier,
            )

        async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
            try:
                # Check if the target function wants the context
                sig = inspect.signature(self._fn)
                if "context" in sig.parameters:
                    kwargs["context"] = context

                if inspect.iscoroutinefunction(self._fn):
                    result = await self._fn(**kwargs)
                else:
                    result = self._fn(**kwargs)
                return ToolResult(success=True, output=result)
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=str(e),
                    metadata={"exception": type(e).__name__},
                )

    return FunctionTool(func)
