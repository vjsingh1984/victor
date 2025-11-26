
import inspect
from functools import wraps
from typing import Any, Callable, Dict

from docstring_parser import parse

from victor.tools.base import BaseTool, ToolResult


def tool(func: Callable) -> Callable:
    """
    A decorator that converts a Python function into a Victor tool.

    This decorator automatically extracts the tool's name, description,
    and parameters from the function's signature and docstring.

    The docstring should follow the Google Python Style Guide format.
    Example:
        @tool
        def my_tool(param1: str, param2: int = 5):
            '''This is the tool's description.

            Args:
                param1: Description of the first parameter.
                param2: Description of the second parameter.
            '''
            # ... tool logic ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # This wrapper is what gets called if the decorated function is called directly
        return func(*args, **kwargs)

    # We will attach a class to the wrapper that is the actual tool
    wrapper.Tool = _create_tool_class(func)

    return wrapper


def _create_tool_class(func: Callable) -> type:
    """Dynamically creates a class that wraps the given function to act as a BaseTool."""

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
            elif param.annotation == bool:
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

    # Dynamically create the tool class
    class FunctionTool(BaseTool):
        def __init__(self, fn: Callable):
            self._fn = fn
            self._name = fn.__name__
            self._description = tool_description
            self._parameters = tool_params_schema

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return self._description

        @property
        def parameters(self) -> Dict[str, Any]:
            return self._parameters

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
