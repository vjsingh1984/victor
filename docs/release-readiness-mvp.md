# Victor MVP Release Readiness

Date: 2026-06-23
Scope: repo-local review of architecture docs, packaging metadata, CI/release workflows, core runtime
layout, and focused guardrail tests.

## Decision

Status: not ready to cut an MVP release tag yet.

The core Python framework is close: version files are aligned, canonical docs pass the drift guard,
service-layer guardrails pass, contract-boundary tests pass, and import smoke works under Python 3.12.
The remaining MVP work is mostly release discipline: clean build verification, distribution checklist
fixes, CI gates that currently allow important failures, and deciding which advertised optional surfaces
must be release-supported versus documented as experimental.

## Evidence Collected

Commands run from the repository root:

| Check | Result |
| --- | --- |
| `python scripts/check_version_sync.py` | Passed. `victor-ai` is `0.7.1`; `victor-contracts` is `0.7.0`; dependency range is compatible. |
| `python scripts/ci/check_docs_drift.py` | Passed. Docs align to `version=0.7.1`, `providers=24`, `tool_modules=34`, `verticals=9`. |
| `python scripts/ci/repo_hygiene_check.py` | Passed. |
| Python import smoke | Passed under Python 3.12 after ignored local egg metadata was refreshed: `victor.__version__ == 0.7.1`, `Agent` imports, `StateGraph` imports. |
| `pytest tests/unit/contracts -q` | Passed: 94 passed, 6 skipped, 1 deprecation warning. |
| `pytest tests/unit/agent/services/test_service_layer_validation.py -q` | Passed: 13 passed. |
| selected import-boundary guard tests | Passed: 4 passed. |
| `python -m build --wheel --no-isolation` | Failed locally because `wheel` is not installed in this Python environment. CI isolated builds should install build-system requirements, but a clean release build still needs to be rerun. |
| `make check-dist` | Failed: reports Homebrew formula missing. The file exists at `scripts/homebrew/victor.rb`, while the target checks `Formula/victor.rb`. |
| `victor --version` in current shell | Failed because `/opt/homebrew/bin/victor` is bound to Python 3.9.25. This is a local environment/install issue, not source importability under supported Python. |

Full unit, integration, security, Docker, VS Code, native Rust, and release workflows were not run locally.

## Design Document Reconciliation

| Document | Current Status | Release Interpretation |
| --- | --- | --- |
| `VISION.md` | Current and edited locally. Adds durable code memory / ProximaDB as a near-term product bet. | OK as strategic direction, but not an MVP promise. |
| `docs/architecture.md` | Canonical architecture. Service-first runtime, 6 canonical services, provider/tool/storage layers. Adds ProximaDB direction. | Mostly current. ProximaDB section must remain clearly "planned direction." |
| `docs/tech-stack.md` | Canonical stack and debt register. TD-11, TD-12, TD-13 added for ProximaDB CCG work. | Current if those items stay Planned. Good place to track release debt. |
| `docs/roadmap.md` | Canonical roadmap. Q2 items mostly complete; EVR Q3 backlog added. | Needs date/status refresh before release because "Current Priorities (2026Q2)" is stale as of 2026-06-23. |
| `docs/features.md` | Canonical feature catalog. Claims 24 providers, 34 tools, 9 verticals, Chainlit UI, policy engine, sandbox. | Needs support-level tagging for optional/experimental features before release. |
| `docs/architecture/proximadb-codegraph-backend.md` | New untracked design-intent doc. | Good design record. Not part of MVP unless implemented behind `GraphStoreProtocol`. |
| FEP/ADR EVR docs | Evaluation-centric runtime design exists. | Q3 backlog, not MVP release criteria unless used as a release gate. |
| `docs/development/releasing/publishing.md` | Release process exists but includes old example versions and manual version-edit guidance. | Needs update to current process: `VERSION` + `sync_version`, clean build, twine check, CLI smoke. |

## Implemented MVP Surface

- Public Python API: `victor.framework.Agent`, `StateGraph`, `WorkflowEngine`, tools, events.
- Service-first runtime: `ChatService`, `ToolService`, `SessionService`, `ContextService`,
  `ProviderService`, `RecoveryService` are present and guard-tested.
- Provider layer: 24 provider adapter files are present and docs drift check derives that count.
- Tool layer: documented 34 tool-module canon is enforced by docs drift.
- Contract boundary: `victor-contracts` exists as an independently versioned package and contract tests pass.
- Packaging metadata: `pyproject.toml`, root `VERSION`, and `victor-contracts/VERSION` are internally aligned.
- Release automation: tag-triggered release workflow exists for PyPI, native wheels, binaries, Docker,
  checksums, and GitHub Release.

## MVP Release Blockers

1. Fix distribution checklist drift.
   `make check-dist` currently fails because it checks `Formula/victor.rb`, but the repo has
   `scripts/homebrew/victor.rb`.

2. Prove clean packaging from a clean environment.
   Run `python -m build`, `twine check dist/*`, install the built wheel with the built
   `victor-contracts` wheel, and run import plus CLI smoke under Python 3.10, 3.11, and 3.12.

3. Make release-critical CI gates blocking.
   `packages.yml` allows package tests and CLI smoke to fail. `ci-integration.yml` allows integration
   suite failures. External vertical import/loading checks are also advisory. For MVP, either make the
   advertised surfaces blocking or explicitly mark them experimental/unsupported in release notes.

4. Refresh roadmap dates/status.
   `docs/roadmap.md` still frames priorities as 2026Q2 while this review is dated 2026-06-23. Before
   release, convert completed Q2 work into delivered status and move active items into an MVP/Q3 section.

5. Finalize release notes.
   `CHANGELOG.md` already has a `0.7.1` section dated 2026-06-21, but the MVP cut still needs a final
   support-level pass: core runtime, contracts, CLI/API/MCP, optional chat UI, external verticals,
   native extensions, Docker/VS Code.

6. Decide external vertical support level.
   Docs advertise external packages (`victor-coding`, `victor-devops`, `victor-rag`,
   `victor-dataanalysis`, `victor-research`), but CI treats their import/loading failures as allowed.
   MVP needs either blocking compatibility tests or docs/release notes that say these are preview.

7. Verify optional surfaces that are advertised as MVP.
   At minimum: `victor --help`, `victor --version`, one no-network chat/help path, API server import,
   MCP server import, `victor ui --help` with `chat-ui` extra installed, and Docker image smoke.

## Not MVP

- ProximaDB CCG backend implementation (`ProximaGraphStore`, one-`oid` graph/vector/relational record).
- EVR Q3 evaluation-centric runtime backlog.
- Publishing benchmark superiority claims beyond whatever has a reproducible artifact.
- Full observability productization unless the release explicitly supports the dashboard/API surface.
- Workspace-isolation rename completion if current behavior is stable and documented.

## Recommended MVP Cut Checklist

1. Clean tree except intentional release docs.
2. Update `CHANGELOG.md` and roadmap status.
3. Fix `make check-dist`.
4. Run `make check-version`, `make check-repo-hygiene`, docs drift, format/lint/type gates.
5. Run focused guardrail tests plus full unit suite.
6. Run package build/install smoke from wheel on Python 3.10, 3.11, 3.12.
7. Run release workflow dry run or TestPyPI publish.
8. Decide and document support level for external verticals, Chainlit UI, Docker, native Rust wheels,
   VS Code extension, and observability.
9. Tag only after release artifacts and smoke tests are green in CI.
