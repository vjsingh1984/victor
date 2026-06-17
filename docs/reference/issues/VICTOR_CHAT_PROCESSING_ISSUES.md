# Victor Chat Processing and victor-coding Issues

Evidence source: `victor chat -p zai-coding` transcript for `/Users/vijaysingh/code/proximaDB`, mapped to:
- `~/.victor/logs/victor.log`
- `~/.victor/logs/usage.jsonl`
- local plugin repo `/Users/vijaysingh/code/victor-coding`

## High-Priority Victor Chat UX Issues

1. Broad graph calls stall the session for minutes.
   - Transcript: `graph mode='overview'` failed after `169036ms`.
   - Logs: `2026-05-10 04:56:52` `graph mode 'overview' exceeded 90s budget`.
   - `usage.jsonl`: `2026-05-10T09:57:03Z`, `tool_name=graph`, `duration_ms=169036.2`.
   - Fix: make `graph(overview)` degrade automatically for large graphs using `top_k`, `files_only/modules_only`, or a cheap summary path before entering expensive ranking.

2. Timeout/budget errors are routed to the wrong recovery handler.
   - Logs: `PermissionErrorHandler - WARNING - No handler could process error for graph: graph mode 'overview' exceeded 90s budget`.
   - Fix: classify `"exceeded ... budget"` as timeout/resource-budget, not permission.

3. Timeout display is inconsistent.
   - Same broad graph operation reports a 90s graph budget, ~169s actual wall time, and another graph call reports pipeline timeout at 180s.
   - Fix: use a single effective timeout source in UI, graph tool, and pipeline.

4. Graph analytics unexpectedly triggers semantic indexing.
   - Transcript: `graph mode='module_pagerank'` started, then embedding/indexing output appeared, then graph timed out after 180s.
   - Logs: graph call at `05:02:21`; indexing `1532 files`, `38091 symbols`; timeout at `05:05:21`.
   - Fix: graph analytics should use the project graph database only, or explicitly announce/ask before expensive semantic rebuilds.

5. Unknown tool calls after a complete answer cause unnecessary continuation.
   - Transcript: after a full answer, model emitted `setGlobalAxisManager`.
   - Logs: `[pipeline] All 1 tool calls in batch were skipped: Unknown or disabled tool: set_global_axis_manager`.
   - Fix: when content has a high-confidence completion marker and only unknown/skipped tools remain, complete instead of retrying/grounding again.

6. Tool naming normalization is incomplete.
   - Transcript uses camelCase `setGlobalAxisManager`; runtime reports snake_case `set_global_axis_manager`, but no alias/tool exists.
   - Fix: normalize camelCase/provider-invented tool names before validation and add “nearest valid tool” guidance only when useful.

7. Context repair is too destructive/noisy.
   - Logs repeatedly show `[fix_orphaned_tool_messages] Stripped tool_calls... Removed ... orphaned tool response messages`, including 20, 32, and 24 removals.
   - Fix: preserve assistant tool-call and tool-response pairs during compaction; log removals with correlation IDs and reason.

8. Compaction repeatedly runs but frees nothing.
   - Logs: `Compaction triggered ... Compaction complete: 0 messages removed, 0 tool outputs truncated, ~0 tokens freed`.
   - Fix: skip compaction when no eligible payload exists, or lower log level and improve pruning eligibility.

9. Verification loop ran after sufficient evidence and consumed user time.
   - The assistant produced a full answer, then entered multiple verification rounds because grounding feedback complained about ambiguous paths and a bogus symbol `"information"`.
   - Fix: grounding feedback should distinguish actionable verification failures from false positives and should not force broad loops after a complete answer.

## Axis and HGMI Consolidation Direction

The legacy “axis” behavior visible in the transcript is a provider-emitted,
tool-shaped UI/control hint (`setGlobalAxisManager`). It is imperative, global,
and not a Victor tool. Treating it as a real runtime tool creates noisy skipped
tool loops and makes the UX depend on provider-specific vocabulary.

The new axis/HGMI direction should be framework-owned interaction state instead:
structured dimensions that help the runtime decide how much autonomy, evidence,
planning, risk handling, and UI guidance a turn needs. These axes may influence
tool selection, prompting, verification, or display, but they should not be
exposed as provider-callable tools.

Consolidation recommendation:
- Canonicalize provider names at ingress, before validation, so camelCase and
  mixed-case emissions fail or execute consistently under one snake_case name.
- Do not add a `set_global_axis_manager` compatibility tool. Unknown axis calls
  should become terminal, non-retryable skipped calls, and a completed answer
  should finish instead of continuing hidden recovery loops.
- If HGMI axes become product behavior, define one framework-level axis model
  and make chat/CLI/UI surfaces compose that model. Avoid separate “legacy axis”
  and “HGMI axis” code paths.
- Since backward compatibility is not required here, prefer deleting or
  preventing provider-specific axis shims over preserving aliases that imply a
  callable tool exists.

## victor-coding Plugin Issues

1. Plugin emits stdout directly into the chat UI.
   - Transcript shows repeated:
     - `Registered embedding provider: chromadb/lancedb/proximadb/victor_structural_bridge`
     - `✓ Embeddings enabled...`
     - `🗑️ Cleared index`
   - Source evidence:
     - `../victor-coding/victor_coding/codebase/embeddings/registry.py` prints provider registration.
     - `../victor-coding/victor_coding/codebase/indexer.py` prints embeddings enabled.
     - `../victor-coding/victor_coding/codebase/embeddings/lancedb_provider.py` prints clear-index status.
   - Fix: replace runtime `print()` calls with logger calls; route progress through Victor tool progress events only when user-facing.

2. Path-scoped searches create nested `.victor/embeddings` directories in source subtrees.
   - Transcript shows storage under:
     - `/Users/vijaysingh/code/proximaDB/src/storage/engines/.victor/embeddings`
     - `/Users/vijaysingh/code/proximaDB/src/utils/.victor/embeddings`
     - `/Users/vijaysingh/code/proximaDB/src/.victor/embeddings`
   - Fix: semantic indexes should live under the project `.victor` root keyed by searched subpath, not inside arbitrary searched directories.

3. Code search clears/rebuilds indexes too eagerly.
   - Transcript shows repeated “Cleared index” during analysis-only search.
   - Logs show fresh indexing for scoped paths even when a project index exists.
   - Fix: reuse compatible persisted indexes; do not clear on ordinary search unless manifest/schema mismatch or explicit `reindex=True`.

4. Filename search mishandles compound queries.
   - Query `.bak .bak2 .disabled` returned incomplete/noisy results; later `.bak` alone found 10 files.
   - Fix: either split filename queries on whitespace as OR patterns or reject compound filename queries with a clear error.

5. Semantic fallback after literal no-result may be surprising.
   - Logs: literal query `fn deserialize bytes Box dyn Any` returned 0, auto-escalated to semantic and returned approximate matches.
   - Fix: preserve mode distinction in UI: “literal found 0; semantic fallback found approximate matches.” For verification, exact search should not silently become approximate evidence.

6. File watcher proliferation.
   - Logs show watchers created for project root, `src`, and `src/storage/engines`.
   - Fix: consolidate watchers under project root where possible, and subscribe subpath indexes to the root watcher.

## Suggested Fix Order

1. Stop plugin stdout leakage by replacing `print()` in victor-coding embedding/index paths with logging.
2. Add robust timeout/budget classification in Victor error recovery.
3. Make graph overview/module analytics cheap or explicitly bounded on large graphs.
4. Stop nested `.victor/embeddings` creation for subpath searches.
5. Fix unknown-tool completion behavior for completed answers.
6. Improve context compaction/tool-message pairing so orphan cleanup is rare and explainable.
