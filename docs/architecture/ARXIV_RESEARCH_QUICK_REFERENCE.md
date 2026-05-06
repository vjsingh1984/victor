# Victor Architecture Research: Quick Reference

**Analysis Date**: 2026-05-05  
**Status**: Validated shortlist for next-version planning  
**Primary companion**: [Validation and vNext Planning](ARXIV_RESEARCH_VALIDATION_2026-05-05.md)

Use this page for the fast version. The detailed audit lives in
[ARXIV_RESEARCH_VALIDATION_2026-05-05.md](ARXIV_RESEARCH_VALIDATION_2026-05-05.md).

## Resume Here

If work is paused and you need to restart later, use this sequence:

1. Read [ARXIV_RESEARCH_QUICK_REFERENCE.md](ARXIV_RESEARCH_QUICK_REFERENCE.md).
2. Read [ARXIV_RESEARCH_VALIDATION_2026-05-05.md](ARXIV_RESEARCH_VALIDATION_2026-05-05.md).
3. Read [research-validated-memory-context-roadmap-2026-05-05.md](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md).
4. Pick one P0 item only:
   - evaluation harness
   - retrieval benchmark
   - typed-memory plus store-routing prototype
5. Re-check current implementation before coding:
   - `victor/agent/conversation/store.py`
   - `victor/agent/conversation_embedding_store.py`
   - `victor/agent/context_compactor.py`
   - `victor/storage/memory/unified.py`
   - `victor/framework/rl/learners/prompt_optimizer.py`

Do not restart from the old greenfield plan in the raw AsciiDoc or appendix analysis.

## Read These First

- [ARXIV_RESEARCH_VALIDATION_2026-05-05.md](ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
- [ARXIV_CATEGORY_REVIEW_2026-05-05.md](ARXIV_CATEGORY_REVIEW_2026-05-05.md)
- [Current Runtime State](CURRENT_STATE.md)
- [Architecture Overview](overview.md)
- [Full Analysis](ARXIV_RESEARCH_ANALYSIS_2026-05-05.md)

## What Is Already True in Victor

The research suite originally treated several areas as missing. They are not missing:

- LanceDB-backed retrieval already exists for conversations and unified storage.
- Prompt optimization already exists through the current GEPA and MIPRO pipeline.
- Context compaction already exists.
- Team coordination already exists around `UnifiedTeamCoordinator`.
- The runtime is already service-first and should stay that way.

That means the right next step is **evaluation and bounded experiments**, not greenfield architecture.

## Validated Next-Version Candidates

These are the highest-confidence candidates after reviewing the repo and core papers.

1. **Evaluation harness first**
   - Why: `2604.16310`, `2603.23160`, and `2604.12179` make stronger retrieval and memory evaluation the safest first move.
   - Landing zone: `victor/evaluation/`, `tests/integration/`.

2. **Audit and benchmark current LanceDB-backed retrieval**
   - Why: the docs overstated missing functionality; measurement should come before storage redesign.
   - Landing zone: `victor/agent/conversation/`, `victor/agent/services/session_service.py`.

3. **Typed-memory plus store-routing prototype**
   - Why: `2604.22085`, `2511.12960`, `2603.15658`, and `2603.16496` all point toward explicit memory lanes and better routing.
   - Landing zone: `victor/storage/memory/`, `victor/agent/intelligent_prompt_builder.py`, context services.

4. **Hybrid retrieval and reranking experiments**
   - Why: `2604.16394` and `2603.22587` fit Victor's current local storage model without requiring a rewrite.
   - Landing zone: retrieval and reranking seams, not persistence architecture.

5. **Adaptive and hierarchical compaction**
   - Why: `2603.29193`, `2604.18349`, and `2604.23277` suggest better evidence selection under token budgets.
   - Landing zone: `victor/agent/context_compactor.py`, conversation assembly.

6. **Prompt strategy expansion**
   - Why: `2603.19311`, `2603.27008`, `2604.09418`, and `2604.23263` fit the current strategy-based prompt optimizer.
   - Landing zone: `victor/framework/rl/learners/prompt_optimizer.py`.

7. **Team coordination evaluation**
   - Why: `2604.09459`, `2603.28990`, and `2604.00722` support benchmarking formations and credit heuristics rather than adding new abstractions.
   - Landing zone: `victor/teams/mixins/`, `victor/teams/unified_coordinator.py`.

## Defer from Next Version

These ideas are not justified for the next release:

- OMC / talent-market architecture from `2604.22446`
- TPGO self-improving multi-agent evolution from `2604.20714`
- TDA-RC topological reasoning repair from `2604.04942`
- EngiAgent-style fully connected coordinator redesign from `2605.02289`

## Commands

```bash
git fetch origin
git worktree add ../victor-research-<topic> -b research/<topic>
cd ../victor-research-<topic>
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test-quick
make lint
python scripts/ci/repo_hygiene_check.py
```

## Suggested First Build Thread

If we resume implementation later, the safest first thread is:

1. Add retrieval and compaction evaluation fixtures.
2. Benchmark the current LanceDB-backed conversation retrieval path.
3. Only then prototype typed memory or store routing.

That preserves the validated rule from this review: **measurement before architecture**.

## Rules

- Search the repo before proposing a new subsystem.
- Extend current ownership boundaries instead of adding parallel abstractions.
- Use GitHub Issues, PRs, Discussions, or a FEP for coordination.
- Treat the long analysis document as a research appendix, not an implementation spec.

## Related Documents

- [Validation and vNext Planning](ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
- [Category Review](ARXIV_CATEGORY_REVIEW_2026-05-05.md)
- [Research-Validated Roadmap](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md)
- [Research-Validated Tech Debt](../tech-debt/research-validated-memory-context-gaps-2026-05-05.md)
- [Getting Started with Research](GETTING_STARTED_WITH_RESEARCH.md)
- [Full Analysis](ARXIV_RESEARCH_ANALYSIS_2026-05-05.md)
- [Current Runtime State](CURRENT_STATE.md)
