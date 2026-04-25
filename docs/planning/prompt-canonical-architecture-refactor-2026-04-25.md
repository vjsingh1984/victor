# Prompt Canonical Architecture Refactor

Date: 2026-04-25

Purpose:
- Canonicalize the prompt subsystem before continuing the broader agentic optimization roadmap.
- Preserve the prior roadmap checkpoint so implementation can resume at the right place afterward.

Resume point for the main roadmap:
- Continue from `docs/planning/agentic-optimization-implementation-plan-2026-04-25.md`
- Next pending slice remains `Phase 3.3 Safe default-on preview pruning for read-only tools`

## Design Goal

Replace scattered prompt assembly responsibilities with a shared prompt model:
- `PromptDocument`: canonical collection of prompt blocks
- `PromptBlock`: canonical unit of rendered prompt content
- Shared processors for document-level deduplication and priority trimming
- Lossless dictionary compression applied as a post-processing stage where useful

## Migrated Components

| Component | Migration |
|---|---|
| `victor.framework.prompt_builder.PromptBuilder` | Now builds via `PromptDocument` and exposes `build_document()` |
| `victor.agent.prompt_builder.SystemPromptBuilder` | Now builds via `PromptDocument` and exposes `build_document()` |
| `victor.agent.system_prompt_policy.SystemPromptPolicy` | Deduplication now delegates to the canonical builder/document processor path |
| `victor.agent.prompt_normalizer.PromptNormalizer` | Section deduplication now delegates to the canonical document processor path |
| `victor.agent.prompt_pipeline.UnifiedPromptPipeline` | Turn-prefix assembly now uses canonical prompt blocks and post-process compression |
| `victor.framework.prompt_dictionary_compressor` | Serves as the canonical lossless compression utility for repeated prompt boilerplate |

## TDD Coverage

- `tests/unit/framework/test_prompt_document.py`
- `tests/unit/framework/test_framework_prompt_builder.py`
- `tests/unit/agent/test_agent_prompt_builder.py`
- `tests/unit/agent/test_system_prompt_policy.py`
- `tests/unit/agent/test_prompt_normalizer.py`
- `tests/unit/agent/test_prompt_pipeline.py`
- `tests/unit/agent/test_prompt_pipeline_canonicalization.py`
- `tests/unit/agent/test_prompt_coordinator.py`
- `tests/unit/agent/coordinators/test_system_prompt_coordinator.py`

## Remaining Cleanup

These are not blockers for resuming the roadmap, but remain candidates for later cleanup:
- Narrow and modernize prompt-related protocols so they match the canonical runtime shape.
- Rename or retire older specialized builder abstractions that still use the overloaded `PromptBuilder` name in non-canonical modules.
- Convert more prompt post-processing steps into explicit processor objects when they accumulate enough shared behavior.
