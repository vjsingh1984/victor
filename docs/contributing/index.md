# Contributing

This is the canonical contributor hub for Victor. Use it instead of older, duplicated guides.

**New contributor?** Follow our [Contributor Workflow](../diagrams/user-journeys/contributor-workflow.mmd) for a step-by-step guide from setup to merged PR.

## Quick Start

```bash
# Clone and install
git clone https://github.com/vjsingh1984/victor.git
cd victor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
make test
pytest tests/unit -v
```

## Core Contributor Docs

| Document | Purpose |
|----------|---------|
| [Setup](./setup.md) | Development environment setup |
| [Code Style](./code-style.md) | Formatting and linting rules |
| [Testing](./testing.md) | Test strategy and commands |
| [FEP Process](./FEP_PROCESS.md) | Feature proposals and approvals |
| [Pre-Commit Hooks](./PRE_COMMIT_HOOKS.md) | Local checks before commit |
| [Documentation Guide](./documentation-guide.md) | Docs structure and workflow |

## Project Orientation

- [Architecture Overview](../architecture/overview.md)
- [Reference Index](../reference/index.md)
- [Guides Index](../guides/index.md)

## Repo-Level Guide

For contribution workflow, PR expectations, and governance, see the root guide:

- Root CONTRIBUTING: `CONTRIBUTING.md`

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
