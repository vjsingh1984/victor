# Competitive Benchmarks

**Suite Version**: 1.0
**Last Updated**: 2026-03-10

## Overview

This directory contains the competitive benchmark suite for comparing Victor against other agentic AI frameworks.

## Contents

- `competitive-benchmark-rubric.md` - Main rubric document with competitor matrix and task categories
- `scoring-methodology.md` - Detailed scoring methodology and execution protocol
- `tasks/` - Individual task definitions (C1-C5, R1-R4, T1-T5, A1-A4, W1-W4)
- `results/` - Benchmark results (to be populated in M2)

## Quick Start

### Run a Single Task

```bash
# Export your LLM API key
export OPENAI_API_KEY=sk-...

# Run task C1 with Victor
python -m victor.benchmarks.run --task C1 --framework victor

# Run task C1 with LangGraph
python -m victor.benchmarks.run --task C1 --framework langgraph
```

### Run Full Suite

```bash
# Run all 20 tasks for Victor
python -m victor.benchmarks.run_suite --framework victor

# Run all 20 tasks for all frameworks
python -m victor.benchmarks.run_suite --all-frameworks
```

### Generate Report

```bash
python -m victor.benchmarks.report --format markdown
```

## Task Categories

| Category | ID Range | Description | # Tasks |
|----------|----------|-------------|---------|
| Code Generation | C1-C5 | Generate, edit, refactor code | 5 |
| Multi-Step Reasoning | R1-R4 | Research, design, planning, debugging | 4 |
| Tool Usage | T1-T5 | File ops, git, web, database, shell | 5 |
| Analysis | A1-A4 | Security, performance, dependencies, coverage | 4 |
| Workflow | W1-W4 | Sequential, parallel, HITL, error recovery | 4 |

**Total**: 22 tasks (exceeds M1 target of 20)

## Supported Frameworks

| Framework | Status | Notes |
|-----------|--------|-------|
| **Victor** | ✅ Native | Primary framework, full support |
| LangGraph | 🔄 Planned | Popular workflow framework |
| CrewAI | 🔄 Planned | Role-based agents |
| AutoGPT | 🔄 Planned | Autonomous agent pioneer |
| OpenAI Swarm | 🔄 Planned | Lightweight multi-agent |
| Semantic Kernel | 🔄 Planned | Enterprise-focused |

## Task Status

| Task | Definition | Victor Test | LangGraph Test | CrewAI Test |
|------|------------|-------------|----------------|-------------|
| C1 | ✅ Complete | ✅ Tested | 🔄 Pending | 🔄 Pending |
| C2 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| C3 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| C4 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| C5 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| R1 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| R2 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| R3 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| R4 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| T1 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| T2 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| T3 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| T4 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| T5 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| A1 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| A2 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| A3 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| A4 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| W1 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| W2 | ✅ Complete | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| W3 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |
| W4 | 🔄 Planned | 🔄 Pending | 🔄 Pending | 🔄 Pending |

**Legend**: ✅ Complete | 🔄 In Progress | ⏳ Planned

## Milestones

### M1: Benchmark Rubric Frozen (Current)
- [x] Rubric document created
- [x] Task categories defined (22 tasks)
- [x] Scoring methodology documented
- [x] Competitor matrix complete
- [x] 3 example task definitions created (C1, R2, W2)

### M2: Benchmark Execution
- [ ] Victor benchmarked on all 22 tasks
- [ ] 2+ competitors benchmarked on all 22 tasks
- [ ] Results compiled into comparison matrix
- [ ] Statistical significance analysis performed

### M3: Report and Action Items
- [ ] Comprehensive benchmark report published
- [ ] Action items identified for Victor improvements
- [ ] Next-quarter priorities driven by benchmark gaps
- [ ] Competitive advantages documented

## Adding New Tasks

1. Create task definition in `tasks/{ID}_{name}.md`
2. Follow the task template from rubric
3. Add to appropriate category in rubric
4. Update task count in README
5. Create test case in `magenta/benchmarks/tests/`

## Contributing

When adding new benchmark tasks:
1. Ensure tasks are reproducible
2. Define clear success criteria
3. Specify scoring rubric
4. Include validation steps
5. Test on at least Victor before submitting

## References

- Main rubric: `competitive-benchmark-rubric.md`
- Scoring methodology: `scoring-methodology.md`
- Task definitions: `tasks/`
- Results: `results/` (M2)

## Contact

For questions about benchmarks, contact the Platform Lead or open an issue.
