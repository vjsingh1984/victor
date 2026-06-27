# Verticals (in-repo, monorepo)

First-party Victor verticals, folded into the monorepo from their former
standalone repos (history preserved via `git subtree`). Each remains a **distinct
installable package** with its own `pyproject.toml` and entry points — mono repo,
poly release.

| Path | Package (import) | Dist |
|------|------------------|------|
| `verticals/victor-coding` | `victor_coding` | `victor-coding` |
| `verticals/victor-devops` | `victor_devops` | `victor-devops` |
| `verticals/victor-research` | `victor_research` | `victor-research` |
| `verticals/victor-rag` | `victor_rag` | `victor-rag` |
| `verticals/victor-dataanalysis` | `victor_dataanalysis` | `victor-dataanalysis` |

(The vertical **marketplace/registry** (`victor-registry`) stays a **standalone
repo** — it is a discovery service, not a vertical, so it is not folded here.)

## The contract boundary (why this stays decoupled)

Folding the verticals in does **not** re-couple them to framework internals. They
import **only** `victor_contracts` — never `victor.*` framework internals. This is
enforced by a blocking CI audit:

```bash
make check-vertical-boundaries   # scripts/ci/check_extracted_vertical_boundaries.py
```

This lint — not the old repo walls — is what preserves the decoupling. Third-party
verticals still install the published `victor-contracts` + register via the
`victor.plugins` entry point exactly as before.

## Dev workflow

```bash
make install-verticals    # contracts + framework + all verticals (editable)
make test-verticals       # each vertical's own test suite
make check-vertical-boundaries
```

## Notes

- `verticals/` and `registry/` are excluded from the `victor-ai` dist
  (`[tool.setuptools.packages.find] exclude`) — they ship as their own packages.
- One documented boundary exception exists today
  (`victor-coding/.../plugin.py` probes a framework-internal protocol); see the
  `[tool.victor.contract_audit]` note in that package's `pyproject.toml`. The
  proper fix is to promote that protocol into `victor_contracts`.
