# Research-Validated Tech Debt: Memory, Retrieval, and Context Gaps

**Created**: 2026-05-05  
**Status**: Active gap tracker derived from local corpus review  
**Companion roadmap**: [research-validated-memory-context-roadmap-2026-05-05.md](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md)

## Purpose

This is the short list of architecture and documentation debt exposed by the manual arXiv review.

These are not generic wishlist items. Each gap below is tied to:

- a verified repo state
- a paper-backed mechanism or evaluation pattern
- a concrete Victor landing zone

## Active Gaps

| ID | Gap | Current state | Why it matters | Suggested landing zone |
| --- | --- | --- | --- | --- |
| RG-1 | Research docs overstated missing LanceDB and vector retrieval work. | Victor already has `ConversationEmbeddingStore`, LanceDB-backed semantic retrieval, and unified SQLite + LanceDB storage. | False greenfield assumptions distort planning and waste effort. | Documentation cleanup completed; keep validating against `CURRENT_STATE.md` |
| RG-2 | `arxive` search work is not tracked as a reproducible in-repo artifact. | Search phrases, ranks, and rationale lived in terminal output and ad hoc notes. | The research process is hard to repeat and easy to misstate. | Research docs or evaluation utilities |
| RG-3 | Retrieval quality is not benchmarked well enough for multi-turn memory and context-overflow scenarios. | Existing retrieval works, but evaluation coverage is weaker than the memory literature expects. | New storage or compaction work should not ship without quality baselines. | `victor/evaluation/`, `tests/integration/` |
| RG-4 | Context assembly is still similarity-heavy and not explicit enough about store selection. | Victor can retrieve semantically relevant messages, but the decision of which store or lane to query is under-modeled. | `2603.15658` and `2603.16496` suggest routing is a first-class design axis. | `ContextService`, `IntelligentPromptBuilder`, memory coordinator |
| RG-5 | Typed memory categories are not first-class enough at the conversation-retrieval layer. | Unified memory types exist, but the reviewed papers point toward finer-grained memory classes for retrieval policy. | `Memanto` and `ENGRAM` imply better precision and controllability. | `victor/storage/memory/`, `ConversationStore`, unified memory adapters |
| RG-6 | Provenance, conflict handling, and rollback semantics for memory are under-specified. | Retrieved messages carry metadata, but research-backed ground-truth preservation and conflict resolution are not a clear planning axis. | `MemMachine` makes memory integrity a feature, not an afterthought. | storage metadata, adapters, memory results |
| RG-7 | Compaction and retrieval are not evaluated together under fixed token budgets. | Victor has compaction and retrieval, but the interaction between them is not benchmarked as a combined system. | `HiGMem`, `Adaptive Context Compression`, and `Hybrid Graph Priors` all target this seam. | `ContextCompactor`, `conversation/assembler`, evaluation tests |
| RG-8 | Prompt optimization strategy coverage is narrower than current research suggests. | GEPA and MIPRO are present, but preference-based and retrieval-backed strategies are not first-class options. | `PrefPO`, `RASPRef`, and `AIR` are credible additions to the strategy registry. | `victor/framework/rl/learners/` |
| RG-9 | Team-formation defaults are not benchmarked enough against self-organizing protocols. | Victor supports multiple formations, including hierarchical, but the default-choice logic is not strongly evidence-backed. | `2603.28990`, `2604.09459`, and `2604.00722` suggest evaluation-first work. | `victor/teams/unified_coordinator.py`, `victor/teams/mixins/`, team tests |

## Priority Order

1. `RG-3` retrieval and context evaluation
2. `RG-4` store routing
3. `RG-5` typed memory lanes
4. `RG-7` compaction plus retrieval joint benchmarking
5. `RG-8` prompt strategy expansion
6. `RG-6` provenance and conflict handling
7. `RG-9` coordination benchmarking
8. `RG-2` research reproducibility hygiene

## Done in This Documentation Pass

- corrected the research docs to reflect existing Victor storage and retrieval surfaces
- separated validated recommendations from speculative transcript output
- added category-level paper synthesis and a roadmap grounded in current ownership boundaries

## Related Documents

- [Category Review](../architecture/ARXIV_CATEGORY_REVIEW_2026-05-05.md)
- [Research Validation](../architecture/ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
- [Research-Validated Roadmap](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md)
