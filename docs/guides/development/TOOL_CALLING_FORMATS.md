# Tool Calling Formats (Local Models)

Local models vary in how they emit tool calls. Victor supports multiple formats to improve compatibility in offline and
  air-gapped setups.

## Supported Formats
- Native tool call structures when the provider exposes them
- JSON-in-content responses (parsed as a fallback)
- Streaming responses where the tool call appears after the stream completes

## Configuration Notes
- Prefer models with native tool calling if available.
- If tool calls are returned as JSON text, ensure the fallback parser is enabled.
- Some models behave better with non-streaming responses; try both.

## Troubleshooting
- Tools not triggering: switch to keyword tool selection or use a smaller tool set.
- JSON tool calls in content: check logs for parse warnings and retry in non-streaming mode.
- Air-gapped use: confirm web tools are disabled and the model is local.

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
