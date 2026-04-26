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

## Next Resume Point

- Phase 2: extract post-selection transforms out of `ToolSelector.select_semantic(...)`
  so the selection flow becomes:
  1. select
  2. prune / fallback
  3. decorate / budget
  4. cache / record
