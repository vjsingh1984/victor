# Research-Validated Roadmap for Memory, Retrieval, and Prompt Architecture

**Created**: 2026-05-05  
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

## Deferred Research

- `2604.22446` OMC / talent-market runtime
- `2604.20714` self-evolving default runtime
- `2604.04942` TDA-RC
- `2605.02289` EngiAgent-style coordinator redesign

## Suggested Metrics

- retrieval precision@k and answer F1 on multi-turn memory tasks
- context tokens per answerable query
- compaction token reduction versus answer-quality delta
- prompt-strategy win rate versus current GEPA/MIPRO baseline
- team success rate, latency overhead, and token overhead by formation

## Related Documents

- [Category Review](../architecture/ARXIV_CATEGORY_REVIEW_2026-05-05.md)
- [Research Validation](../architecture/ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
- [Research-Validated Tech Debt](../tech-debt/research-validated-memory-context-gaps-2026-05-05.md)
