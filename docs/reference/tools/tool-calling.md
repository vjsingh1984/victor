# Tool Calling

Tool calling depends on the provider and model. Some models return structured tool calls; others return JSON in text
  that requires a fallback parser.

## If Tool Calls Don’t Trigger
- Try a different model or provider.
- Switch to non‑streaming responses.
- Reduce the tool list (keyword or hybrid selection).
- Verify the tool schema matches the model’s expectations.

## Configuration Touchpoints
- Profile provider/model settings in `~/.victor/profiles.yaml`
- Tool selection strategy: `tool_selection_strategy` (or legacy `use_semantic_tool_selection`)
- Air‑gapped mode: `airgapped_mode: true`

## References
- `../embeddings/TOOL_CALLING_FORMATS.md`
- `victor/config/tool_calling_models.yaml` (optional reference; may be outdated)

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
