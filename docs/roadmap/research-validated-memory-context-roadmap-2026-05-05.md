# Research-Validated Roadmap for Memory, Retrieval, and Prompt Architecture

**Created**: 2026-05-05  
**Last Updated**: 2026-05-07 — appended Phase 4 (filtered-ANN routing + hybrid gateway) and Phase 5 (PageIndex/NaviRAG-style hierarchical retrieval) based on `ARXIV_CATEGORY_REVIEW_2026-05-05.md` Categories 5–6  
**Status**: Proposed next-version roadmap  
**Inputs**: local corpus review in [ARXIV_CATEGORY_REVIEW_2026-05-05.md](../architecture/ARXIV_CATEGORY_REVIEW_2026-05-05.md) and [ARXIV_RESEARCH_VALIDATION_2026-05-05.md](../architecture/ARXIV_RESEARCH_VALIDATION_2026-05-05.md)

## Premise

Victor already has:

- LanceDB-backed conversation retrieval
- unified SQLite + LanceDB local storage
- prompt optimization strategies and registries
- context compaction
- team formations and credit-assignment mixins

So this roadmap is about **measured extensions to current seams**, not new parallel subsystems.

## Where to Restart

If implementation resumes later, start here:

1. Phase 0 only.
2. Establish baseline retrieval and compaction metrics.
3. Create or curate memory-grounded evaluation fixtures.
4. Do not start with storage rewrites, new coordinator layers, or large memory refactors.

The first branch should be a benchmark or evaluation branch, not a greenfield feature branch.

## Phase 0: Benchmark and Reproducibility

**Target window**: 1-2 weeks

### Goals

- make retrieval and compaction quality measurable before redesign
- capture corpus-search inputs and selected papers as reproducible artifacts
- establish current baseline performance for context assembly

### Tasks

1. Add multi-turn retrieval and context-quality benchmarks
   - Papers: `2604.16310` RAG-DIVE, `2603.23160` UniDial-EvalKit
   - Landing zones: `victor/evaluation/`, `tests/integration/`, `victor/agent/conversation/`

2. Generate or curate memory-grounded test conversations
   - Paper: `2604.12179` AgenticAI-DialogGen
   - Landing zones: `tests/fixtures/`, `tests/integration/`, evaluation utilities

3. Benchmark current conversation retrieval and compaction baselines
   - Landing zones: `victor/agent/conversation/store.py`, `victor/agent/conversation_embedding_store.py`, `victor/agent/context_compactor.py`

### Suggested Branches

- `research/eval-retrieval-baseline`
- `research/memory-fixtures`
- `research/context-compaction-bench`

### Exit Criteria

- baseline retrieval precision and recall recorded
- token-budget and answer-quality tradeoffs recorded
- benchmark harness runs in CI or as a repeatable local command

## Phase 1: Retrieval Planning and Typed Memory

**Target window**: 2-4 weeks

### Goals

- retrieve from the right store before retrieving from every store
- make memory types explicit enough to support targeted retrieval
- keep all work inside current memory and session ownership boundaries

### Tasks

1. Prototype typed memory lanes
   - Papers: `2604.22085` Memanto, `2511.12960` ENGRAM
   - Likely shapes: role/type tags, explicit memory categories, selective search
   - Landing zones: `victor/storage/memory/`, `victor/agent/conversation/store.py`, `victor/storage/memory/unified.py`

2. Prototype cost-sensitive store routing
   - Papers: `2603.15658`, `2603.16496`
   - Landing zones: `victor/agent/services/context_service.py`, `victor/agent/intelligent_prompt_builder.py`, `victor/storage/memory/unified.py`

3. Add provenance and conflict handling for memory writes
   - Paper: `2604.04853` MemMachine
   - Landing zones: conversation storage metadata, memory adapters, retrieval result metadata

### Exit Criteria

- queries can selectively target memory lanes or stores
- routing policies are benchmarked against the Phase 0 baseline
- provenance metadata is available in retrieved memory results

## Phase 2: Hybrid Retrieval and Compaction

**Target window**: 2-4 weeks

### Goals

- improve retrieval quality without replacing the local storage architecture
- reduce answer-stage context size while preserving evidence quality

### Tasks

1. Add hybrid ranking and reranking experiments
   - Papers: `2604.16394`, `2603.22587`
   - Techniques: lexical + vector fusion, filtered retrieval, query-time modulation, auditable reranking
   - Landing zones: `victor/agent/conversation_embedding_store.py`, `victor/agent/intelligent_prompt_builder.py`, retrieval adapters

2. Add adaptive compaction strategy
   - Paper: `2603.29193`
   - Landing zone: `victor/agent/context_compactor.py`

3. Add hierarchical evidence selection strategy
   - Papers: `2604.18349`, `2604.23277`
   - Landing zones: `victor/agent/context_compactor.py`, `victor/agent/conversation/assembler.py`

### Exit Criteria

- token usage decreases without degrading benchmarked answer quality
- hierarchical or adaptive strategies beat the current baseline on at least one benchmark family

## Phase 3: Prompt and Coordination Experiments

**Target window**: 2-3 weeks

### Goals

- broaden prompt optimization without displacing the current default path
- improve team behavior by measuring formation and credit strategies

### Tasks

1. Add optional prompt-optimization strategies
   - Papers: `2603.19311`, `2603.27008`, `2604.09418`
   - Landing zones: `victor/framework/rl/learners/prompt_optimizer.py`, strategy registry, prompt-optimization config

2. Add ambiguity-normalization experiment
   - Paper: `2604.23263`
   - Landing zones: prompt assembly and pre-query processing seams

3. Benchmark coordination protocols and credit heuristics
   - Papers: `2604.09459`, `2603.28990`, `2604.00722`
   - Landing zones: `victor/teams/mixins/`, `victor/teams/unified_coordinator.py`, team evaluation tests

### Exit Criteria

- at least one non-default prompt strategy shows measurable value on a bounded task family
- formation defaults are benchmarked instead of chosen by intuition alone

## Phase 4: Filtered-ANN Routing and Hybrid Retrieval Gateway (added 2026-05-07)

**Target window**: 2-3 weeks  
**Driver**: ARXIV_CATEGORY_REVIEW_2026-05-05.md Category 5 — vector-storage literature consensus is "keep the index, route the query plan."

### Goals

- pick the right filtering strategy per query rather than hardcoding pre- or post-filtering
- replace per-callsite ad-hoc combinations of BM25, vector, and metadata filters with a single auditable gateway
- preserve LanceDB / ChromaDB / ProximaDB / SQLite-FTS5 as backends — only the layer above them changes

### Tasks

1. Add a filtered-ANN query planner above the existing vector providers
   - Papers: `2602.17914` Filtered-ANN Query Planning, `2603.23710` Filter-Agnostic Vector Search on PostgreSQL, `2510.27141` Compass, `2602.11443` Filtered ANN System Design (GLS metric)
   - Likely shape: a `FilterStrategyRouter` that estimates selectivity from chunk-level statistics and chooses pre-, inline-, or post-filtering per query
   - Landing zones: `victor/storage/vector_stores/base.py`, `victor/storage/vector_stores/lancedb_provider.py`, `victor/storage/vector_stores/chromadb_provider.py`, `victor/storage/vector_stores/registry.py`
   - Acceptance: documented win on a workload that mixes high-selectivity (rare predicate) and low-selectivity (common predicate) filters versus a single fixed strategy

2. Build a unified hybrid retrieval gateway
   - Papers: `2604.16394` Hybrid Retrieval Reference Architecture, `2603.22587` flexvec, `2603.21018` DSL-R1, `2604.14222` Adaptive Query Routing
   - Likely shape: one entry point that runs FTS5 (already on `ConversationStore`), dense ANN (existing providers), and structured filters, then fuses with reciprocal rank fusion + optional reranker
   - Landing zones: new module e.g. `victor/storage/retrieval/gateway.py`, callers in `victor/agent/conversation/store.py`, `victor/agent/conversation_embedding_store.py`, `victor/core/graph_rag/retrieval.py`
   - Acceptance: the gateway replaces at least the conversation-store and embedding-store retrieval paths without regressing answer quality on the Phase 0 benchmarks

3. Audit existing vector-store providers for selectivity hooks
   - Today `lancedb_provider`, `chromadb_provider`, and `proximadb_provider` already exist; only `proximadb_multi.py` exposes `hybrid_search`. Decide whether the gateway lives above the providers (recommended) or whether each provider grows a uniform interface
   - Landing zones: `victor/storage/vector_stores/`

### Exit Criteria

- `FilterStrategyRouter` is selected by at least one production retrieval path
- the hybrid gateway is the canonical seam for new retrieval callers
- BM25 + dense + filter fusion is no longer reimplemented per call site

### Suggested Branches

- `research/filter-strategy-router`
- `research/hybrid-retrieval-gateway`

## Phase 5: PageIndex / Hierarchical Retrieval Strategy (added 2026-05-07)

**Target window**: 3-4 weeks  
**Driver**: ARXIV_CATEGORY_REVIEW_2026-05-05.md Category 6 — academic NaviRAG/FlexStructRAG line is the validated form of the PageIndex idea.

### Goals

- add a third retrieval strategy alongside vector ANN and BM25 for long structured documents
- exploit Victor's existing code symbol graph as the navigable structure for code retrieval
- gate hierarchical navigation on query type and budget — never as the unconditional default

### Tasks

1. Build a hierarchical-summary head over long documents and large source modules
   - Papers: `2604.12766` NaviRAG, `2604.16312` FlexStructRAG, `2604.16350` LiteSemRAG (cost contrast)
   - Likely shape: a per-document outline (TOC tree of summaries) with pointers to raw chunks, persisted alongside embeddings
   - Landing zones: new helper next to `victor/core/graph_rag/indexing.py`, plus a writer in `victor/agent/conversation_embedding_store.py` for non-code text

2. Add a `NavigationRetriever` strategy that walks the outline / symbol graph
   - For code: reuse `victor/core/graph_rag/` symbol graph — it is already the right tree
   - For docs: walk the new outline-tree heads
   - Landing zones: `victor/core/graph_rag/retrieval.py`, new `NavigationRetriever` exposed through the Phase 4 hybrid gateway

3. Wire navigation as a gated strategy under the hybrid gateway
   - Decision criteria: long-document QA, multi-hop reasoning queries, large symbol-graph traversals
   - Cost guard: token budget aware of LLM-in-the-loop cost; fall back to flat ANN when the budget is tight
   - Landing zones: hybrid gateway from Phase 4, `RuntimeIntelligenceService`, retrieval-decision telemetry

4. Add an evaluation harness comparing flat ANN, hybrid, and navigation strategies
   - Papers: `2604.09666` Do We Still Need GraphRAG?, `2603.12180` Strategic Navigation or Stochastic Search?
   - Landing zones: `victor/evaluation/`, retrieval-strategy A/B harness (reuse Phase 0)

### Exit Criteria

- at least one workload (long-document QA, multi-hop code question) is measurably better with navigation than flat ANN
- navigation is never invoked when the cost guard would be violated
- the symbol graph is reused as the navigable structure for code rather than a parallel TOC tree

### Suggested Branches

- `research/hierarchical-outline-head`
- `research/navigation-retriever`

## Deferred Research

- `2604.22446` OMC / talent-market runtime
- `2604.20714` self-evolving default runtime
- `2604.04942` TDA-RC
- `2605.02289` EngiAgent-style coordinator redesign
- `2601.09985` FaTRQ tiered residual quantization (no scale need yet)
- `2603.03065` V3DB ZK proofs for verifiable vector search (no compliance need yet)
- `2506.23397` NaviX graph-DBMS-native vector indices (would require swapping graph store)

## Suggested Metrics

- retrieval precision@k and answer F1 on multi-turn memory tasks
- context tokens per answerable query
- compaction token reduction versus answer-quality delta
- prompt-strategy win rate versus current GEPA/MIPRO baseline
- team success rate, latency overhead, and token overhead by formation
- (Phase 4) filtered-ANN recall and latency at varying selectivity vs single-strategy baseline
- (Phase 4) hybrid gateway answer F1 vs single-modality (BM25-only or ANN-only) baseline
- (Phase 5) navigation strategy answer F1 and token cost vs flat ANN on long-document QA
- (Phase 5) navigation invocation rate gated by cost guard, ensuring it stays a strategy not a default

## Related Documents

- [Category Review](../architecture/ARXIV_CATEGORY_REVIEW_2026-05-05.md)
- [Research Validation](../architecture/ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
- [Research-Validated Tech Debt](../tech-debt/research-validated-memory-context-gaps-2026-05-05.md)
