# Tool Selection Decomposition Plan

Date: 2026-04-25

## Goal

Reduce the size and branching pressure of `victor/agent/tool_selection.py`
without changing the public `ToolSelector` entrypoint.

Target state:
- `ToolSelector` orchestrates selection flow
- pure state-passed policy modules own stage pruning and fallback assembly
- post-selection transforms become separately testable units

## Phase Plan

### Phase 1
- Extract stage-aware pruning and semantic fallback assembly into a canonical
  tool-selection policy module.
- Keep `ToolSelector` as the runtime facade for current callers.

### Phase 2
- Extract post-selection transforms:
  - edge-model filtering
  - MCP capping
  - schema promotion
  - tool-schema token budgeting

### Phase 3
- Split semantic-selection orchestration concerns:
  - cache lookup / serialization
  - keyword blending
  - final metrics recording

## Current Batch

- Phase 1 completed:
  - added `victor.agent.tool_selection_policy` as the canonical pure policy
    module for stage pruning and semantic fallback assembly
  - migrated `ToolSelector.prioritize_by_stage(...)` and
    `ToolSelector._get_fallback_tools(...)` onto that policy while preserving
    the existing selector API
  - added focused TDD for the extracted policy and retained surrounding
    regression on the existing selector-facing tests
- Phase 2 completed:
  - added `victor.agent.tool_selection_postprocessor` as the canonical
    post-selection transform chain for semantic tool selection
  - migrated `ToolSelector.select_semantic(...)` to delegate edge filtering,
    MCP capping, schema promotion, and token-budget enforcement through that
    postprocessor
  - added focused TDD for transform ordering and optional-step gating while
    retaining surrounding regression on the existing optimization and selector
    tests
- Phase 3 started:
  - added `victor.agent.tool_selection_cache_key` for canonical semantic cache
    key construction
  - added `victor.agent.tool_selection_recorder` for canonical selection
    result recording and callback dispatch
  - added `victor.agent.tool_selection_cache` for semantic selection cache
    payload restore/serialize behavior
  - added `victor.agent.tool_selection_assembler` for semantic + keyword +
    explicit web-tool assembly and stable deduplication
  - migrated `ToolSelector.select_semantic(...)` off inline cache payload
    reconstruction/serialization and off inline semantic-keyword-web assembly
  - moved semantic cache read/write control flow onto
    `SemanticToolSelectionCacheAdapter`, so `ToolSelector` no longer performs
    direct cache payload `get` / `set` choreography
  - migrated `ToolSelector.select_semantic(...)` off inline cache-key
    construction and off inline semantic-vs-fallback result recording
  - added focused TDD for cache payload normalization and bounded keyword/web
    assembly while retaining surrounding selector/runtime regression

## Next Resume Point

- Phase 3 is complete. The next optional seam is Phase 4:
  - decide whether `select_semantic(...)` should keep orchestration ownership
    or whether cache initialization, semantic selection execution, and final
    enabled-tool filtering should move into a dedicated runtime coordinator
  - only take that step if the extra indirection is justified by new behavior,
    because the remaining body is now materially smaller and mostly orchestration
