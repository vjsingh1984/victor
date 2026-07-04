# Victor Documentation

Build and serve the documentation site:

```bash
pip install -e ./victor-contracts -e ".[docs]"
mkdocs serve     # Dev server with live reload at http://localhost:8000
mkdocs build     # Static site → ./site/
mkdocs build --strict  # Fail on broken links
```

> **Do not hand-edit `site/`** — it is generated from the `docs/` source.

## Document Hierarchy

```
docs/
├── index.md                     ← Landing page (documentation map)
├── architecture.md               ← System architecture (single source of truth)
├── roadmap.md                    ← 90-day priorities, tech debt register
├── features.md                   ← Feature catalog grounded in implementation
├── tech-stack.md                 ← Technology choices and dependency versions
├── user-guide/                   ← End-user guides
├── development/                  ← Contributor/developer guides
├── api-reference/                ← API documentation
├── reference/                    ← Reference tables and lookup docs
├── diagrams/                     ← Diagram index (consolidated from scattered files)
├── architecture/                 ← Detailed architecture docs (ADR, decomposition)
├── feps/                         ← Framework Enhancement Proposals
└── README.md                     ← This file
```

## Canonical Files

| Purpose | File | Notes |
|---------|------|-------|
| Landing | `docs/index.md` | Documentation map with links to all sections |
| Architecture | `docs/architecture.md` | Supersedes root `ARCHITECTURE.md` and `docs/architecture/overview.md` |
| Roadmap | `docs/roadmap.md` | Supersedes root `roadmap.md` |
| Features | `docs/features.md` | Grounded feature catalog |
| Diagrams | `docs/diagrams/index.md` | Consolidated diagram registry |

## Formatting Standards

- **Format**: Markdown (`.md`) only — no `.adoc` or `.rst` for new docs
- **Diagrams**: Mermaid (preferred) or PlantUML — both render in GitHub and MkDocs
- **Line length**: 100 characters max
- **Styling**: Professional, concise, implementation-grounded
