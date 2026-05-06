# arXiv Category Review and Architecture Synthesis

**Reviewed On**: 2026-05-05  
**Method**: local `../arxive` semantic search + local PDF extraction via `pdftotext`  
**Companion docs**: [Research Validation](ARXIV_RESEARCH_VALIDATION_2026-05-05.md), [Quick Reference](ARXIV_RESEARCH_QUICK_REFERENCE.md)

## Purpose

This document is the expanded paper-by-paper pass behind the validated roadmap.

Unlike the original transcript-derived suite, this review:

- uses the local `../arxive` corpus already present on disk
- works from merged multi-query search results per category
- verifies each result against the extracted PDF text
- maps each paper onto existing Victor architecture instead of assuming a greenfield system

## Query Sets Used

### Prompt Optimization

- `prompt engineering optimization LLM`
- `automated instruction revision prompt optimization`
- `reasoning chain optimization prompt adaptation`
- `few-shot prompt optimization agent`

### Agentic AI

- `multi-agent coordination self-improving agents`
- `agent planning coordination workflow autonomous coding`
- `multi-agent credit assignment language agents`
- `heterogeneous agent organization talent market`

### Hybrid Storage and Retrieval

- `SQL vector retrieval embedding modulation SQLite`
- `hybrid vector relational retrieval local agent memory`
- `message storage embeddings LanceDB SQLite agent`
- `filtered vector search structured metadata retrieval`

### Context and Memory Management

- `context compression long conversation memory retrieval`
- `multi-turn dialogue memory agentic memory context overflow`
- `episodic semantic procedural memory conversational agents`
- `multi-turn RAG evaluation conversation retrieval`

## Cross-Category Conclusions

- The strongest new signals are not "build LanceDB" or "replace the runtime". They are typed memory, store routing, hybrid reranking, adaptive compression, hierarchical evidence selection, and better evaluation.
- Victor already has the main storage, retrieval, compaction, and team surfaces. Most useful work fits as extensions to `ConversationStore`, `ConversationEmbeddingStore`, `ContextCompactor`, `UnifiedMemoryCoordinator`, `UnifiedTeamCoordinator`, and the existing prompt-optimization strategy registry.
- The most actionable prompt papers are `2603.19311` PrefPO, `2603.27008` RASPRef, and `2604.09418` AIR from the earlier core validation pass. `2604.04942` TDA-RC remains too research-heavy for vNext.
- The most actionable memory papers are `2604.22085` Memanto, `2603.15658` store routing, `2604.04853` MemMachine, `2603.16496` AdaMem, and `2604.18349` HiGMem.
- The most actionable evaluation papers are `2604.16310` RAG-DIVE from the earlier core pass, `2603.23160` UniDial-EvalKit, and `2604.12179` AgenticAI-DialogGen.

## Category 1: Prompt Optimization

| Rank | Paper | Verified contribution | Victor fit | Recommendation |
| --- | --- | --- | --- | --- |
| 1 | `2604.20140` HiPO | Segment-aware preference optimization for structured reasoning outputs. | Useful as a design reference for section-aware prompt evaluation, but closer to training/alignment than runtime prompt assembly. | Reference only |
| 2 | `2509.03117` PromptCOS | Copyright and provenance auditing for system prompts. | Relevant to governance and IP hygiene, not to prompt quality or context management. | Backlog / governance only |
| 3 | `2604.18612` Agent-GWO | Search-based joint optimization of prompts and decoding settings with collaborating agents. | Fits only as an offline optimizer experiment; too heavy for default runtime behavior. | Prototype offline only |
| 4 | `2604.21510` OptiVerse | Benchmark for hard optimization problems. | Can help benchmark planning-heavy agents, but it does not imply a direct architecture change. | Reference / benchmark only |
| 5 | `2603.19311` PrefPO | Pairwise preference prompt optimization using only a starting prompt and natural-language criteria. | Strong fit for Victor's existing `PromptOptimizationStrategy` seam. | Prototype |
| 6 | `2604.04942` TDA-RC | Persistent-homology-based repair for reasoning chains. | Still too research-heavy relative to likely runtime payoff. | Defer |
| 7 | `2604.23263` SLM ambiguity resolver | Uses a small model before the main model to resolve prompt ambiguity. | Good fit as a narrow pre-processing step for ambiguous user turns. | Prototype |
| 8 | `2603.27008` RASPRef | Retrieval-backed self-supervised prompt refinement from prior reasoning traces. | Strong fit for offline prompt evolution using Victor execution traces and retrieval history. | Prototype |
| 9 | `2604.03677` Prompt infilling for diffusion LMs | Unlocks prompt infilling in diffusion language models. | Not relevant to Victor's current provider/runtime direction. | Reject for vNext |
| 10 | `2604.21765` PrismaDV | Task-aware data unit test generation. | Potentially useful for data tooling, but not part of the prompt architecture problem. | Reference only |

### Prompt Synthesis

- The practical prompt work is strategy-level, not subsystem-level.
- `PrefPO`, `RASPRef`, and `AIR` are the best fits for Victor because they can plug into the current optimization pipeline.
- A lightweight ambiguity-normalization pass is more plausible than heavy reasoning-chain repair.

## Category 2: Agentic AI

| Rank | Paper | Verified contribution | Victor fit | Recommendation |
| --- | --- | --- | --- | --- |
| 1 | `2604.09459` Credit Assignment survey | Taxonomy, reporting checklist, and benchmark protocol for credit assignment in reasoning and agentic RL. | High fit for evaluating and refining current credit-assignment mixins. | Adopt for evaluation |
| 2 | `2508.11126` AI Agentic Programming | Broad survey of agentic programming patterns. | Useful as a landscape reference, not as a direct implementation plan. | Reference only |
| 3 | `2603.28990` Drop the Hierarchy and Roles | Large comparison showing self-organizing sequential/hybrid protocols can beat rigid hierarchies. | Strong fit for benchmarking formation defaults inside `UnifiedTeamCoordinator`. | Prototype / evaluate |
| 4 | `2604.22446` Skills to Talent / OMC | Company-like organization layer with talent market and typed interfaces. | Interesting long-range idea, but too large and invasive for vNext. | Defer |
| 5 | `2604.00722` LangMARL | Natural-language multi-agent RL centered on agent-specific credit assignment. | Good fit for offline heuristic and evaluation work on existing team surfaces. | Prototype |
| 6 | `2603.25681` Self-Improvement overview | Closed-loop self-improvement taxonomy for LLM systems. | Useful background for experimentation, but not a scoped feature. | Reference only |
| 7 | `2603.27703` KAT-Coder-V2 report | Specialized agentic coding model with domain specialization. | Good benchmark/reference for coding-agent capability, not an architecture guide. | Reference only |
| 8 | `2603.24324` Incentive-aware reward design | LLM-assisted reward-shaping design for cooperative MARL. | Niche fit for team reward experiments; not a near-term runtime change. | Defer |
| 9 | `2604.03515` Inside the Scaffold | Source-code taxonomy of coding-agent scaffolds across control, tools, and resource management. | Strong fit for auditing Victor's scaffold decisions and avoiding accidental architecture sprawl. | Adopt as review lens |
| 10 | `2603.05344` Terminal coding agents | Harness, context-engineering, and workflow lessons for terminal-native coding agents. | Strong fit for Victor's CLI-oriented agent workflows and context engineering. | Adopt / reference |

### Agentic Synthesis

- The best next work is measurement and protocol benchmarking, not a new team layer.
- `UnifiedTeamCoordinator` should remain the main landing zone.
- The strongest questions are: which formation defaults work best, how should credit be tracked, and when should self-organization beat a fixed hierarchy.

## Category 3: Hybrid Storage and Retrieval

| Rank | Paper | Verified contribution | Victor fit | Recommendation |
| --- | --- | --- | --- | --- |
| 1 | `2604.22085` Memanto | Typed semantic memory with predefined categories, conflict handling, and information-theoretic retrieval. | Excellent fit for typed message or memory lanes on top of current storage. | Prototype |
| 2 | `2603.02240` SuperLocalMemory | Local-first multi-agent memory with poisoning defense and adaptive reranking. | Good fit for memory-poisoning defenses and retrieval reranking, but narrower than the core storage gap. | Prototype narrow |
| 3 | `2604.16394` Hybrid retrieval reference architecture | BM25 + dense retrieval + RRF + auditability + offline pseudo-query augmentation. | Strong fit for retrieval assembly and auditable ranking on current stores. | Prototype |
| 4 | `2604.04853` MemMachine | Ground-truth-preserving personalized memory with profile plus episodic layers. | Strong fit for provenance, rollback, and conflict-aware memory handling. | Prototype |
| 5 | `2603.22587` flexvec | SQL pre-filtering plus query-time embedding modulation. | Good fit for reranking and modulation without changing persistence architecture. | Prototype |
| 6 | `2604.14004` Memory Transfer Learning | Memory reuse across coding domains. | Medium fit for later cross-project or cross-session transfer work. | Later prototype |
| 7 | `2504.02441` CognitiveMemory | Survey of memory mechanisms in LLMs and agents. | Useful terminology and design-space reference. | Reference only |
| 8 | `2604.01707` Memory in the LLM Era | Unified framework and evaluation of memory architectures. | Strong as a conceptual and evaluation reference, not a direct feature. | Reference only |
| 9 | `2603.15658` Did You Check the Right Pocket? | Cost-sensitive routing among specialized memory stores before retrieval. | One of the best fits in the whole review for Victor's context assembly problem. | Prototype high priority |
| 10 | `2604.18271` EmbodiedLGR | Lightweight graph representation and retrieval for robotic memory. | Mostly domain-specific; only limited graph-memory inspiration transfers. | Reference only |

### Storage and Retrieval Synthesis

- Typed memory and store routing outrank a storage rewrite.
- Hybrid ranking should combine structured filters, lexical retrieval, vector similarity, and reranking instead of treating embedding search as the only semantic path.
- Provenance and conflict handling are under-modeled today relative to the memory literature.

## Category 4: Context and Memory Management

| Rank | Paper | Verified contribution | Victor fit | Recommendation |
| --- | --- | --- | --- | --- |
| 1 | `2603.29193` Adaptive context compression | Importance-aware, coherence-sensitive compression with dynamic budget allocation. | Strong fit as an alternative `ContextCompactor` strategy. | Prototype |
| 2 | `2601.09113` AI Hippocampus | Survey and taxonomy of memory in LLM and agent systems. | Useful terminology and planning aid, not an implementation blueprint. | Reference only |
| 3 | `2404.00573` Human-like recall and consolidation | Dynamic recall plus consolidation for dialogue agents. | Useful for heuristics around recall and consolidation, but not a direct system spec. | Prototype narrow |
| 4 | `2603.16496` AdaMem | Working, episodic, persona, and graph memories with question-conditioned retrieval planning. | Extremely strong fit for query-aware context assembly and participant-aware retrieval. | Prototype high priority |
| 5 | `2508.14048` RAG-Boost | ASR-specific retrieval augmentation. | Not applicable to Victor's current problem. | Reject for vNext |
| 6 | `2604.12179` AgenticAI-DialogGen | Synthetic topic-guided conversation and QA generation for short/long-term memory evaluation. | Excellent fit for evaluation and synthetic benchmark generation. | Adopt for evaluation |
| 7 | `2603.23160` UniDial-EvalKit | Unified evaluation toolkit for multi-turn conversational abilities. | Strong fit for test harnesses around context, memory, and retrieval quality. | Adopt for evaluation |
| 8 | `2603.19595` All-Mem | Lifelong memory with topology evolution and online/offline consolidation. | Promising, but heavier than the next iteration likely needs. | Later prototype |
| 9 | `2604.18349` HiGMem | Hierarchical event-turn memory with LLM-guided evidence selection. | Strong fit for hierarchical evidence selection before answer generation. | Prototype |
| 10 | `2603.11123` Uni-ASR | Unified ASR architecture. | Not part of Victor's core context-management problem. | Reject for vNext |

### Context and Memory Synthesis

- The main gap is not "add embeddings". It is deciding which memory store, which granularity, and which evidence slice should be used for a turn.
- `AdaMem`, `HiGMem`, `2603.29193`, and earlier validated `2604.23277` point toward better retrieval planning and compaction strategies.
- Evaluation is an equal priority with implementation here; `UniDial-EvalKit`, `AgenticAI-DialogGen`, and `RAG-DIVE` are the right support papers.

## Synthesized Feature Set for the Next Version

### Highest-Value Features

1. **Typed memory lanes**
   - Primary papers: `2604.22085`, `2511.12960`
   - Victor mapping: typed retrieval over existing message and memory records

2. **Store routing before retrieval**
   - Primary papers: `2603.15658`, `2603.16496`
   - Victor mapping: choose working, episodic, persona, graph, or vector-backed retrieval paths before pulling context

3. **Hybrid retrieval and reranking**
   - Primary papers: `2604.16394`, `2603.22587`
   - Victor mapping: structured filters + lexical search + vector search + fusion/reranking

4. **Adaptive and hierarchical compaction**
   - Primary papers: `2603.29193`, `2604.18349`, `2604.23277`
   - Victor mapping: alternative compaction strategies under `ContextCompactor`

5. **Memory evaluation harness**
   - Primary papers: `2604.16310`, `2603.23160`, `2604.12179`
   - Victor mapping: retrieval, context, and answer-quality benchmarks before architecture expansion

6. **Prompt strategy extensions**
   - Primary papers: `2603.19311`, `2603.27008`, `2604.09418`, `2604.23263`
   - Victor mapping: optional strategies inside the current prompt optimizer

7. **Coordination benchmarking**
   - Primary papers: `2604.09459`, `2603.28990`, `2604.00722`
   - Victor mapping: better formation defaults and credit metrics, not a second team abstraction

## Strong Deferrals

- `2604.22446` OMC / talent-market architecture
- `2604.04942` TDA-RC reasoning-chain repair
- `2604.20714` TPGO / self-evolving default runtime
- `2605.02289` EngiAgent-style coordinator redesign
- diffusion-LM prompt-infilling work
- ASR- or robotics-specific memory architectures

## Related Documents

- [Research Validation](ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
- [Research-Validated Roadmap](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md)
- [Research-Validated Tech Debt](../tech-debt/research-validated-memory-context-gaps-2026-05-05.md)
