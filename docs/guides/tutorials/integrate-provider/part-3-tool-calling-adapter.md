# Tutorial: Integrating a New LLM Provider - Part 3

**Part 3 of 5:** Tool Calling Adapter (Optional)

---

## Navigation

- [Part 1: Architecture & Steps 1-2](part-1-provider-architecture.md)
- [Part 2: Streaming & Error Handling](part-2-streaming-error-handling.md)
- **[Part 3: Tool Calling Adapter](#)** (Current)
- [Part 4: Registration & Testing](part-4-registration-testing.md)
- [Part 5: Best Practices & Examples](part-5-best-practices-examples.md)
- [**Complete Guide**](integrate-provider.md)

---

## 3. Tool Calling Adapter (Optional)

If your provider does not support native tool calling, or you need custom parsing logic, create a tool calling adapter.

### When to Create an Adapter

- The provider returns tool calls in a non-standard format
- You need to parse tool calls from text content (fallback parsing)
- The provider has model-specific tool calling quirks

### Creating a Tool Calling Adapter

Create `victor/agent/tool_calling/custom_adapter.py`:

```python
"""Tool calling adapter for CustomLLM provider."""

from typing import Any, Dict, List, Optional

from victor.agent.tool_calling.base import (
    BaseToolCallingAdapter,
    FallbackParsingMixin,
    ToolCallingCapabilities,
    ToolCallFormat,
    ToolCallParseResult,
)
from victor.providers.base import ToolDefinition


class CustomLLMAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Tool calling adapter for CustomLLM.

    Handles tool call parsing and conversion for CustomLLM's format.
    """

    @property
    def provider_name(self) -> str:
        return "custom_llm"

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Return tool calling capabilities for this provider/model."""
        return ToolCallingCapabilities(
            native_tool_calls=True,
            streaming_tool_calls=True,
            parallel_tool_calls=True,
            tool_choice_param=True,
            tool_call_format=ToolCallFormat.OPENAI,
            argument_format="json",
            recommended_max_tools=30,
            recommended_tool_budget=15,
        )

    def convert_tools(
        self, tools: List[ToolDefinition]
    ) -> List[Dict[str, Any]]:
        """Convert tools to provider format.

        Args:
            tools: List of standard ToolDefinition objects

        Returns:
            Provider-formatted tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse tool calls from response.

        Args:
            content: Response content text
            raw_tool_calls: Native tool_calls from provider

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        # Try native tool calls first
        if raw_tool_calls:
            result = self.parse_native_tool_calls(
                raw_tool_calls,
                validate_name_fn=self.is_valid_tool_name,
            )
            if result.tool_calls:
                return result

        # Fall back to content parsing if native parsing failed
        return self.parse_from_content(
            content,
            validate_name_fn=self.is_valid_tool_name,
        )
```

---

**Continue to [Part 4: Registration & Testing](part-4-registration-testing.md)**

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 1 min
**Last Updated:** February 01, 2026
**Part 3 of 5**
