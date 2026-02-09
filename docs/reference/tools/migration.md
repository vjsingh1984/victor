# Tool Implementation Patterns

Victor supports two tool styles:

- Class‑based tools (`BaseTool`)
- Function‑based tools using the `@tool` decorator

## Recommendation
- Use `@tool` for new tools.
- Migrate class‑based tools when you’re already touching them.

## Minimal @tool Example

```python
from victor.tools.decorators import tool

@tool
async def my_tool(param: str) -> dict:
    """Short description of the tool."""
    return {"success": True, "result": param}
```text

## Notes
- Both styles are supported during migration.
- Keep tool docstrings concise; they become tool descriptions.
- Regenerate the catalog after changes: `python scripts/generate_tool_catalog.py`

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
