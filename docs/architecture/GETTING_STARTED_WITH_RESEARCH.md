# Getting Started with Architecture Research

**Created**: 2026-05-05  
**Status**: Research guide; validate every task against current Victor runtime

This guide explains how to use the arXiv research suite without drifting away from the
actual state of this repository.

## Documentation Order

Read the suite in this order:

1. [Quick Reference](ARXIV_RESEARCH_QUICK_REFERENCE.md)
2. [Validation and vNext Planning](ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
3. [Category Review](ARXIV_CATEGORY_REVIEW_2026-05-05.md)
4. [Current Runtime State](CURRENT_STATE.md)
5. [Architecture Overview](overview.md)
6. [Full Analysis](ARXIV_RESEARCH_ANALYSIS_2026-05-05.md)

Use the quick reference to orient yourself, then use the current-state and overview docs to
verify whether a proposed feature already exists in some form. Then use the validation doc to
see which paper-backed ideas survived repo review and transcript review. Only then use the long
analysis as a source of ideas, tradeoffs, and papers.

## Resume Path

If you are coming back to this work after a pause, do this first:

1. Read [Quick Reference](ARXIV_RESEARCH_QUICK_REFERENCE.md).
2. Read [Validation and vNext Planning](ARXIV_RESEARCH_VALIDATION_2026-05-05.md).
3. Read [Research-Validated Roadmap](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md).
4. Open the current landing-zone files before writing code:
   - `victor/agent/conversation/store.py`
   - `victor/agent/conversation_embedding_store.py`
   - `victor/agent/context_compactor.py`
   - `victor/storage/memory/unified.py`
   - `victor/framework/rl/learners/prompt_optimizer.py`

The intended restart point is the validated roadmap, not the older speculative week-by-week plan.

## What the Research Is and Is Not

The research suite is useful for:

- prioritizing candidate areas to investigate
- finding papers and design patterns worth adapting
- sketching roadmaps and experiments
- coordinating larger multi-commit threads

It is not a safe source of truth for:

- exact file paths
- current module ownership
- team names or chat channels
- copy-paste-ready implementation patches

Several code blocks and worktree names in the full analysis are intentionally illustrative.

## Quick Start

### 1. Read the quick reference

Use the Markdown page for repo work:

```bash
less docs/architecture/ARXIV_RESEARCH_QUICK_REFERENCE.md
```

The raw handoff artifact is still available at:

```text
docs/architecture/ARXIV_RESEARCH_QUICK_REFERENCE.adoc
```

### 2. Read the validation pass

```bash
less docs/architecture/ARXIV_RESEARCH_VALIDATION_2026-05-05.md
```

This is the document that separates validated next-version candidates from appendix-only
research notes.

### 3. Read the category review if you need paper-level backing

```bash
less docs/architecture/ARXIV_CATEGORY_REVIEW_2026-05-05.md
```

Use this when you need the actual top-10-per-category paper tables, search phrases, and the
paper-to-feature mapping that led to the updated roadmap.

### 4. Pick a bounded task

Good first tasks are small, testable, and clearly owned by an existing subsystem.
Examples:

- add evaluation coverage for retrieval, compaction, or multi-turn memory
- improve conversation retrieval quality in `victor/agent/conversation/`
- extend prompt optimization in `victor/framework/rl/learners/`
- tighten context compaction in `victor/agent/context_compactor.py`
- improve team coordination behavior in `victor/teams/`

Avoid starting with a large greenfield feature branch unless you have already verified that
Victor does not contain an overlapping implementation.

### 5. Confirm the landing zone

Before you open a branch, search the repo:

```bash
rg -n "keyword" victor docs tests
rg --files victor | rg "conversation|memory|prompt|team|workflow|storage"
```

Then cross-check these docs:

- [Current Runtime State](CURRENT_STATE.md)
- [Migration Guide](migration.md)
- [State-Passed Architecture](state-passed-architecture.md)
- [Validation and vNext Planning](ARXIV_RESEARCH_VALIDATION_2026-05-05.md)

### 6. Create a worktree only if the thread is substantial

```bash
git fetch origin
git worktree add ../victor-research-<topic> -b research/<topic>
cd ../victor-research-<topic>
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test-quick
```

If the task is a small documentation or focused code change, working in the main checkout is fine.

## How to Pick the Right Kind of Task

### Good tasks

- extend an existing service, store, workflow executor, or prompt pipeline
- add tests that expose a current gap in behavior
- document how a research idea maps onto an existing Victor subsystem
- prototype behind a flag or opt-in setting

### Risky tasks

- adding a second orchestration layer
- creating a new multi-agent graph abstraction
- bypassing current prompt pipeline ownership
- adding greenfield storage layers without auditing current LanceDB-backed flows
- changing public framework protocols without a design discussion

If the work touches `victor/framework/`, protocol surfaces, workflow DSL structure, or major
runtime architecture, expect to pair the code change with a FEP or design discussion.

## Development Workflow

### Daily flow

```bash
git fetch origin
git rebase origin/main
make test-quick
```

During implementation:

- keep the branch scope narrow
- add tests near the changed behavior
- update docs when user-facing behavior changes
- preserve service-first ownership in `victor/agent`

Before a handoff or PR:

```bash
make lint
python scripts/ci/repo_hygiene_check.py
```

Run narrower `pytest` commands for the affected area as needed.

## Multi-Agent Coordination

Use git worktrees when multiple contributors or agents are working in parallel.
Split ownership by subtree, not by vague feature label.

Recommended coordination protocol:

1. One issue per workstream.
2. Record the intended landing zone before coding.
3. Record touched files and blockers in the issue or PR description.
4. Hand off by linking the branch or PR plus the tests you ran.
5. Use GitHub Discussions or a FEP for architecture decisions that cross subsystem boundaries.

Avoid concurrent edits to the same files unless the split is explicit and coordinated.

## Paper Reading Guide

If the sibling `../arxive` checkout exists, you can use the research commands directly:

```bash
find ../arxive/corpus -name "2601.09113.pdf"
pdftotext ../arxive/corpus/cs/AI/2026/01/2601.09113/2601.09113.pdf - | less
```

If that checkout is not present, treat the paper paths in the research suite as optional local notes.

Suggested reading order:

1. Read the abstract, introduction, and conclusion.
2. Extract the concrete mechanism the paper proposes.
3. Identify the closest Victor subsystem that already exists.
4. Write down one change you could test in this repo.
5. Ignore any paper idea that requires duplicating an existing Victor abstraction.

## Current Repo Anchors

These paths are the fastest way to ground the research suite:

- `victor/core/database.py`
- `victor/agent/services/session_service.py`
- `victor/agent/services/context_service.py`
- `victor/agent/conversation/store.py`
- `victor/agent/conversation_embedding_store.py`
- `victor/storage/unified/sqlite_lancedb.py`
- `victor/storage/memory/`
- `victor/framework/rl/learners/prompt_optimizer.py`
- `victor/teams/unified_coordinator.py`
- `victor/workflows/`

## Pre-Implementation Checklist

- [ ] I have read the quick reference Markdown page.
- [ ] I have verified the current implementation in the repo.
- [ ] I know the exact modules I plan to change.
- [ ] I know which tests I will run.
- [ ] I understand whether the work changes a public API or architecture surface.
- [ ] I have opened or identified the tracking issue if the task is substantial.
- [ ] I am extending an existing subsystem rather than duplicating it.

## Related Documents

- [Quick Reference](ARXIV_RESEARCH_QUICK_REFERENCE.md)
- [Validation and vNext Planning](ARXIV_RESEARCH_VALIDATION_2026-05-05.md)
- [Category Review](ARXIV_CATEGORY_REVIEW_2026-05-05.md)
- [Research-Validated Roadmap](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md)
- [Research-Validated Tech Debt](../tech-debt/research-validated-memory-context-gaps-2026-05-05.md)
- [Full Analysis](ARXIV_RESEARCH_ANALYSIS_2026-05-05.md)
- [Architecture Overview](overview.md)
- [Current Runtime State](CURRENT_STATE.md)
- [Migration Guide](migration.md)
