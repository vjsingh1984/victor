# Vertical Platform Convergence Plan

Last updated: 2026-03-10
Owner: Architecture/Foundation
Status: Active
Decision status: Waiting on ADR-007 acceptance

## Goal

Converge Victor onto a single, internally consistent vertical architecture and
preserve implementation state across solution sessions.

This file is the persistent execution tracker for the vertical architecture work.
Future sessions should update this file first, continue from the highest-priority
open item, and append to the session log before stopping.

## How To Use This File

- Treat this document as the execution backlog, decision record, and handoff note
  for the vertical-architecture workstream.
- Update the task checkboxes and feature/epic status before ending a session.
- If code realities change, update the **Verified Context Snapshot** and
  **Measured Baseline** before moving the backlog forward.
- Do not start breaking packaging or resolver changes while ADR-007 remains
  `Proposed` unless the user explicitly authorizes that work.

## Working Assumption

Until maintainers explicitly accept or reject ADR-007, planning assumes an
**SDK-first extracted vertical end state**:

- `victor-sdk` owns definition contracts
- `victor-ai` owns runtime orchestration
- each vertical has one authoritative implementation source
- bundled contrib modules are temporary compatibility artifacts, not the target
  source-of-truth model

If ADR-007 is rejected, this document must be rewritten before further execution.

## Verified Context Snapshot

These facts were re-verified against the codebase on 2026-03-10 and should be
treated as the baseline for future sessions:

1. `victor-ai` currently bundles contrib verticals and publishes them in
   `pyproject.toml` through `victor.verticals` entry points.
2. The import resolver still prefers external wheel namespaces first, then legacy
   `victor.<vertical>`, then `victor.verticals.contrib.<vertical>`.
3. `victor-sdk` already owns `StageDefinition`, `Tier`, and `VerticalConfig`, but
   it does not yet own canonical tool identifiers currently exposed via
   `victor.tools.tool_names`.
4. `victor.core.verticals.base.VerticalBase` still imports `victor.framework.Agent`
   inside `create_agent()`.
5. Discovery is lazy and cached through `VerticalLoader`, `EntryPointCache`, and
   `entry_point_loader` caches; startup scanning is not a raw uncached hot path.
6. Current contrib verticals and the example external vertical remain framework/core
   coupled, so the SDK-only authoring story is not yet true in practice.
7. `victor vertical install` already exists, so install UX is not missing; it only
   needs alignment with the accepted packaging model.
8. `StageDefinition`, `Tier`, and `VerticalConfig` already live in `victor-sdk`, so
   future SDK work should not duplicate those moves.

## Source-of-Truth Evidence

- Bundled contrib vertical packaging:
  - `pyproject.toml`
- Runtime-coupled base class:
  - `victor/core/verticals/base.py`
- Hybrid import fallback:
  - `victor/core/verticals/import_resolver.py`
- SDK contract baseline:
  - `victor-sdk/victor_sdk/core/types.py`
  - `victor-sdk/README.md`
- Loader/service behavior:
  - `victor/core/verticals/__init__.py`
  - `victor/core/verticals/vertical_loader.py`
  - `victor/framework/entry_point_loader.py`
  - `victor/framework/vertical_service.py`
- Framework-coupled examples/verticals:
  - `examples/external_vertical/src/victor_security/assistant.py`
  - `examples/external_vertical/README.md`
  - `docs/development/extending/verticals.md`
  - `victor/verticals/contrib/coding/assistant.py`

## Tracking Conventions

- `Not Started`: no execution has begun beyond backlog definition
- `In Progress`: currently active or partially complete
- `Blocked`: cannot proceed until dependency or decision clears
- `Completed`: acceptance criteria met and recorded here

Priority scale:

- `P0`: must settle before architecture-changing execution
- `P1`: foundational implementation work
- `P2`: convergence, packaging, and quality hardening
- `P3`: optimization or follow-on cleanup

## Priority Order

1. Architecture decision and migration governance
2. SDK contract completion
3. Runtime boundary cleanup
4. Definition-layer decoupling
5. Packaging/source-of-truth convergence
6. Hardening, CI guardrails, and compatibility verification

## Cross-Epic Dependency Map

| Epic | Depends On | Why |
|---|---|---|
| VPC-E0 | None | Establishes target architecture, rules, and migration guardrails |
| VPC-E1 | VPC-E0 | SDK contract should reflect the accepted direction |
| VPC-E2 | VPC-E1 | Runtime factory/adapter depends on the SDK definition contract |
| VPC-E3 | VPC-E1, VPC-E2 | Verticals can only migrate cleanly after the contract and runtime boundary exist |
| VPC-E4 | VPC-E3 | Packaging/source-of-truth changes should follow actual vertical decoupling |
| VPC-E5 | VPC-E1 through VPC-E4 | Guardrails and benchmarks should validate the finished shape, not the transient one |

## Epic Overview

| Epic ID | Priority | Epic | Status | Exit Criteria |
|---|---|---|---|---|
| VPC-E0 | P0 | Architecture decision and transition governance | In Progress | ADR accepted/rejected, compatibility rules documented, source-of-truth policy explicit |
| VPC-E1 | P1 | SDK contract completion | In Progress | SDK exports all definition-layer contracts required for SDK-only vertical authoring |
| VPC-E2 | P1 | Runtime boundary cleanup | Complete | Vertical definition layer no longer depends on host runtime creation |
| VPC-E3 | P1 | Definition-layer decoupling | Complete | Vertical definition modules use SDK-only imports and pass guardrail checks |
| VPC-E4 | P2 | Packaging and source-of-truth convergence | In Progress | One authoritative implementation source per vertical |
| VPC-E5 | P2 | Hardening and CI guardrails | In Progress | Import-boundary checks, packaging smoke tests, cache invalidation tests, startup benchmarks |

## Decision Gate

- Decision maker: Vijaykumar Singh
- Acceptance event: ADR-007 status changes from `Proposed` to `Accepted` or
  `Rejected`, and the session log records the decision date.
- Allowed while ADR-007 is still `Proposed`:
  - inventories and baselines
  - SDK contract design work that is additive and non-breaking
  - documentation/template preparation
  - tests/guardrails that observe current state
- Blocked while ADR-007 is still `Proposed`:
  - deleting bundled contrib verticals
  - removing resolver fallback branches
  - publishing docs that claim extraction is already complete

## Compatibility Policy

This is the working migration policy until ADR-007 is formally accepted or changed:

1. Bundled contrib modules remain allowed only as temporary compatibility shims
   once a vertical flips to an external authoritative package.
2. After source-of-truth flips, no new feature work should land in the bundled
   copy; only forwarding behavior, warnings, and compatibility fixes are allowed.
3. Compatibility shims must not silently diverge from the authoritative external
   package.
4. Removal of a shim requires:
   - the external package to be released and installable
   - the smoke-test matrix to pass from a clean environment
   - docs and install guidance to be updated
5. Planned support window for bundled shims:
   - at least one minor release after the authoritative external package goes GA
   - earlier removal only with explicit maintainer approval and documented release
     notes

## Measured Baseline

### Coupling Baseline

Measured on 2026-03-10 using `rg "victor\\.(framework|core|tools)"`
against `victor/verticals/contrib`.

| Vertical | Files with framework/core/tool imports | Import-line matches |
|---|---|---|
| coding | 23 | 70 |
| rag | 16 | 39 |
| devops | 14 | 37 |
| dataanalysis | 14 | 36 |
| research | 12 | 34 |
| Total | 79 | 216 |

Interpretation:

- The decoupling problem is broad, not isolated to a single vertical.
- `coding` is the highest-coupling migration target and should be the first
  implementation proving ground.

### Documentation And Example Drift Baseline

Measured on 2026-03-10 from `docs/`, `examples/external_vertical/`, and SDK docs.

| Area | Current pattern | Why it conflicts with the target model |
|---|---|---|
| `docs/development/extending/verticals.md` | Teaches `victor.core.verticals.VerticalBase`, `StageDefinition`, `VerticalRegistry`, and `victor.tools.tool_names` | Encourages framework-coupled vertical authoring |
| `docs/verticals/coding.md` and peers | Describe tool usage through `victor.tools.tool_names` | Canonical tool identifiers are not yet in SDK |
| `examples/external_vertical/README.md` | Shows imports from `victor.core.verticals` and protocol modules in core | External example does not follow SDK-only definition guidance |
| `examples/external_vertical/src/victor_security/*` | Uses `VerticalBase`, `StageDefinition`, and extension protocols from core | Example package is runtime-coupled |

### Discovery And Cache Baseline

- Discovery is currently lazy and cached; it is not a pure startup-time linear
  rescan of every installed vertical on each command.
- Current performance work should focus on measurement and invalidation
  correctness before structural loader changes.
- Cache staleness remains a real risk because `VerticalBase` configuration caching
  is TTL-based rather than package-version or install-event aware.

## Import Boundary Matrix

| Layer | Allowed Dependencies | Forbidden Dependencies | Notes |
|---|---|---|---|
| SDK definition layer | `victor_sdk`, stdlib, third-party libs owned by the vertical | `victor.framework`, `victor.core.verticals`, `victor.tools.tool_names`, runtime registries | This is the contract future external verticals must follow |
| Runtime extension layer | `victor_sdk`, `victor-ai` runtime APIs, extension protocols, third-party libs | Direct dependency on bundled contrib implementations as authoritative sources | Runtime-heavy helpers belong here, not in definition modules |
| Core/framework runtime | `victor_sdk`, runtime internals, loader/registry/injection services | Importing contrib vertical internals as the only supported path | Host owns orchestration, capability injection, and compatibility shims |
| Compatibility shims | Thin forwarding code, warnings, metadata | Business logic forks or silent behavior divergence | Shims must stay minimal and temporary |

## Source-Of-Truth Registry

| Vertical | Current implementation source | Current fallback/runtime path | Target authoritative package | Migration note |
|---|---|---|---|---|
| coding | Bundled in `victor.verticals.contrib.coding` | Resolver also checks `victor_coding` and `victor.coding` | `victor-coding` | First proving-ground migration target |
| rag | Bundled in `victor.verticals.contrib.rag` | Resolver also checks `victor_rag` and `victor.rag` | `victor-rag` | High integration surface with retrieval/runtime add-ons |
| devops | Bundled in `victor.verticals.contrib.devops` | Resolver also checks `victor_devops` and `victor.devops` | `victor-devops` | Likely capability-heavy migration |
| dataanalysis | Bundled in `victor.verticals.contrib.dataanalysis` | Resolver also checks `victor_dataanalysis` and `victor.dataanalysis` | `victor-dataanalysis` | Uses package-name override already |
| research | Bundled in `victor.verticals.contrib.research` | Resolver also checks `victor_research` and `victor.research` | `victor-research` | Web/search capability boundary needs definition |

Rule:

- Until extraction is complete, the bundled contrib copy is the current runtime
  source of truth because it is what `pyproject.toml` publishes today.
- After extraction, each row above should flip to its external package and the
  bundled copy should become a shim or be removed.

## SDK-Only Vertical Acceptance Checklist

A vertical is considered migrated only when all of the following are true:

- The definition-layer module imports `victor_sdk` contracts only.
- No definition-layer file imports `victor.framework`,
  `victor.core.verticals`, or `victor.tools.tool_names`.
- Tool and capability references use SDK-owned identifiers.
- The vertical exposes a serializable definition/manifest surface.
- Runtime-only integrations are moved behind host-owned adapters or explicit
  extension points.
- The package advertises the correct `victor.verticals` entry point.
- Clean-environment smoke tests pass for install, discovery, and activation.

## Initial CI Guardrail Candidates

- Forbidden-import check for definition-layer modules
- SDK-only example build/install/import smoke test
- External-package discovery smoke test
- Cache invalidation test after install/upgrade/uninstall
- Resolver-order and missing-package error-path tests
- Startup/discovery benchmark trend report

## Detailed Backlog

### VPC-E0: Architecture Decision And Transition Governance

Objective:

- Lock the target architecture, document the migration rules, and preserve enough
  measured context that future sessions do not need to rediscover the problem.

Feature summary:

| Feature ID | Feature | Status | Dependencies | Acceptance Summary |
|---|---|---|---|---|
| VPC-F0.1 | Architecture decision and compatibility policy | Completed | None | ADR, decision gate, migration policy, and decision follow-through are documented |
| VPC-F0.2 | Baseline drift inventory | Completed | None | Coupling baseline, doc drift, package mapping, and cache baseline are recorded |
| VPC-F0.3 | Boundary matrix and source-of-truth registry | Completed | None | Allowed imports, source-of-truth rules, and SDK-only checklist are written down |

#### VPC-F0.1: Architecture Decision And Compatibility Policy

Context:

- This feature turns the analysis into enforceable rules. Without it, later work
  risks mixing bundled and extracted assumptions.

Acceptance criteria:

- ADR-007 exists and captures the target state and alternatives.
- The deciding owner and acceptance gate are explicit.
- Compatibility policy for bundled contrib modules is documented.
- The decision announcement package and follow-on documentation work are explicit.
- Remaining governance tasks are visible and sequenced.

Likely touchpoints:

- `docs/architecture/adr/007-vertical-distribution-and-sdk-boundary.md`
- `docs/architecture/adr/README.md`
- `docs/roadmap/vertical-platform-convergence-plan.md`

Tasks:

- [x] VPC-T0.1 Create persistent cross-session tracker with resume protocol.
- [x] VPC-T0.2 Draft ADR-007 for vertical distribution model and SDK boundary.
- [x] VPC-T0.3 Define acceptance gate for ADR-007 and name the deciding owner/group.
- [x] VPC-T0.4 Document compatibility policy for bundled contrib modules during migration.
- [x] VPC-T0.5 Define support-window/semver expectations for compatibility shims.
- [x] VPC-T0.6 Record the decision announcement package and required follow-on doc updates once ADR-007 is accepted or rejected.

Decision announcement package:

- Update ADR-007 status and record the decision date plus any rationale changes.
- Update this roadmap:
  - decision status
  - current tranche
  - next executable epic/task
  - session log entry recording the decision
- Publish a concise maintainer-facing summary covering:
  - chosen target architecture
  - what remains blocked vs unblocked
  - compatibility policy for bundled contrib shims
  - release/semver expectations

Required follow-on documentation updates after ADR outcome:

- If ADR-007 is accepted:
  - update `docs/architecture/overview.md` to describe the accepted vertical boundary
  - update `docs/development/extending/verticals.md` to stop teaching core/framework-coupled definition authoring
  - update `victor-sdk/README.md`, `victor-sdk/SDK_GUIDE.md`, and `victor-sdk/VERTICAL_DEVELOPMENT.md`
    to reflect the accepted SDK contract
  - update `examples/external_vertical/README.md` and source templates under `victor/templates/vertical/`
    once the supported authoring contract is in place
  - update release notes and migration docs to explain shim behavior and package topology
- If ADR-007 is rejected:
  - rewrite this roadmap's working assumption, epic dependencies, and source-of-truth target table
  - update ADR-007 alternatives/consequences to reflect the selected non-extraction direction
  - align `docs/development/extending/verticals.md` and example docs with the retained bundled/hybrid model
  - close or re-scope E1-E5 tasks that only make sense under the extracted SDK-first target

#### VPC-F0.2: Baseline Drift Inventory

Context:

- Later migrations need a before-state. This feature records measurable coupling
  and the current contradictions between docs, packaging, and code.

Acceptance criteria:

- A measurable import-coupling baseline exists for contrib verticals.
- Contradictions between SDK docs, example docs, packaging, and runtime code are
  enumerated.
- Current discovery/cache behavior is written down so optimization work is not
  based on a false premise.

Likely touchpoints:

- `docs/roadmap/vertical-platform-convergence-plan.md`
- `docs/development/extending/verticals.md`
- `examples/external_vertical/README.md`
- `examples/external_vertical/src/victor_security/assistant.py`

Tasks:

- [x] VPC-T0.7 Capture current framework/core import counts for contrib verticals.
- [x] VPC-T0.8 Capture doc/example contradictions against the target model.
- [x] VPC-T0.9 Record current package/entry-point/import-resolution topology.
- [x] VPC-T0.10 Record current discovery and cache baseline.

#### VPC-F0.3: Boundary Matrix And Source-Of-Truth Registry

Context:

- The project needs a crisp answer for what may live in SDK definitions versus
  runtime adapters versus compatibility shims.

Acceptance criteria:

- An import-boundary matrix exists for definition, runtime, core, and shim layers.
- Each current vertical has a named source-of-truth rule.
- A reusable checklist exists for deciding whether a vertical is actually migrated.
- Candidate CI guardrails are enumerated.

Likely touchpoints:

- `docs/roadmap/vertical-platform-convergence-plan.md`

Tasks:

- [x] VPC-T0.11 Create import-boundary matrix for definition/runtime/core/shim layers.
- [x] VPC-T0.12 Record authoritative source-of-truth rule for each current vertical.
- [x] VPC-T0.13 Define acceptance checklist for an SDK-only vertical.
- [x] VPC-T0.14 Define the first CI guardrail candidates to add after contract work lands.

### VPC-E1: SDK Contract Completion

Objective:

- Finish the SDK surface so a vertical definition can be authored without importing
  runtime internals.

Feature summary:

| Feature ID | Feature | Status | Dependencies | Acceptance Summary |
|---|---|---|---|---|
| VPC-F1.1 | Canonical SDK identifiers and shared constants | Completed | VPC-E0 | Tool and capability identifiers live in SDK with compatibility guidance |
| VPC-F1.2 | Serializable vertical definition contract | Complete | VPC-E0 | SDK exposes a versioned definition/manifest surface |
| VPC-F1.3 | SDK authoring materials and migration kit | Complete | VPC-F1.1, VPC-F1.2 | Examples, templates, and docs teach the supported contract |

#### VPC-F1.1: Canonical SDK Identifiers And Shared Constants

Context:

- Today, verticals still need `victor.tools.tool_names`, which keeps definitions
  coupled to the core repo.

Acceptance criteria:

- SDK exports canonical tool identifiers.
- SDK exports typed capability identifiers or an equivalent stable requirement surface.
- Core tool-name usage has a documented compatibility path.

Likely touchpoints:

- `victor-sdk/victor_sdk/`
- `victor-sdk/victor_sdk/constants/`
- `victor/tools/tool_names.py`
- `victor-sdk/README.md`

Implementation note:

- Canonical naming surface selected: `victor_sdk.constants.ToolNames`
- Supported convenience exports: `victor_sdk.ToolNames`
- Backward-compatibility bridge: `victor.tools.tool_names` now re-exports the
  SDK-owned registry
- Runtime helper module `victor.framework.tool_naming` now imports the registry
  from the SDK instead of the legacy core-owned module
- Capability naming surface selected: `victor_sdk.constants.CapabilityIds`
- Structured capability requirement surface added:
  - `victor_sdk.CapabilityRequirement`
  - `victor_sdk.normalize_capability_requirement()`
  - `victor_sdk.normalize_capability_requirements()`
- `victor_sdk.verticals.metadata.VerticalMetadata` can now carry structured
  capability requirements while preserving legacy string requirements

Tasks:

- [x] VPC-T1.1 Choose the canonical naming surface for SDK tool identifiers (`ToolNames`, `ToolIds`, or alias strategy).
- [x] VPC-T1.2 Add canonical tool identifiers to `victor-sdk`.
- [x] VPC-T1.3 Add typed capability identifiers or requirement literals to `victor-sdk`.
- [x] VPC-T1.4 Add compatibility aliases or deprecation wrappers for `victor.tools.tool_names`.
- [x] VPC-T1.5 Add SDK tests for identifier stability and import compatibility.

#### VPC-F1.2: Serializable Vertical Definition Contract

Context:

- The target architecture requires definition-layer code to return declarative
  configuration rather than host runtime objects.

Acceptance criteria:

- SDK exposes a versioned `VerticalDefinition` or equivalent manifest structure.
- Tool and capability requirements are serializable.
- Stage/workflow/prompt references needed by the definition layer are represented.
- Validation rules and compatibility/version checks exist.

Likely touchpoints:

- `victor-sdk/victor_sdk/core/types.py`
- `victor-sdk/victor_sdk/verticals/`
- `victor/core/verticals/base.py`

Implementation note:

- `victor_sdk.VerticalDefinition` now exists as the serializable definition-layer
  contract and supports:
  - schema version field (`definition_version`)
  - serializable stage payloads
  - typed capability requirements
  - `to_config()` / `from_config()` bridging for compatibility
- `victor_sdk.ToolRequirement` now exists as the serializable tool requirement
  contract and supports:
  - legacy string normalization
  - explicit required vs optional tool declarations
  - serializable purpose/metadata payloads
- Prompt/workflow definition metadata now exists and supports:
  - `PromptMetadata`, `PromptTemplateDefinition`, and `TaskTypeHintDefinition`
  - `WorkflowMetadata` for stage order, initial stage, provider hints, and
    evaluation criteria
  - compatibility normalization from existing dict-based payloads and richer
    victor-ai stage objects
- Definition validation and version checks now exist and support:
  - constructor-time normalization of typed and serialized payloads
  - schema compatibility checks via `definition_version`
  - explicit `VerticalConfigurationError` failures for malformed or
    inconsistent manifest data
  - `VerticalDefinition.from_dict()` round-tripping from serialized payloads
- `victor_sdk.verticals.protocols.base.VerticalBase` now provides a default
  `get_definition()` and routes `get_config()` through the definition contract,
  wrapping malformed hook output as `VerticalConfigurationError`

Tasks:

- [x] VPC-T1.6 Define the `VerticalDefinition` data model and minimum required fields.
- [x] VPC-T1.7 Define serializable tool and capability requirement types.
- [x] VPC-T1.8 Define how prompts, stages, and workflow metadata are represented in the definition contract.
- [x] VPC-T1.9 Add API versioning/validation rules for the new definition surface.
- [x] VPC-T1.10 Add serialization and compatibility tests for the definition contract.

#### VPC-F1.3: SDK Authoring Materials And Migration Kit

Context:

- The code and the docs must teach the same contract. Right now they do not.

Acceptance criteria:

- SDK docs explain the supported definition-layer contract.
- Example external vertical and templates follow the SDK-only path.
- Maintainers have a migration guide from legacy `VerticalBase` authoring.

Likely touchpoints:

- `victor-sdk/README.md`
- `examples/external_vertical/`
- `docs/development/extending/verticals.md`

Tasks:

- [x] VPC-T1.11 Update SDK documentation to describe canonical tool/capability identifiers and the definition contract.
- [x] VPC-T1.12 Rewrite the external vertical example to match SDK-only definition authoring.
- [x] VPC-T1.13 Write a migration guide from current `VerticalBase` authoring to the SDK-first contract.
- [x] VPC-T1.14 Add or update scaffolding/templates for new vertical packages.

### VPC-E2: Runtime Boundary Cleanup

Objective:

- Move runtime instantiation and capability injection out of the definition layer
  and into host-owned services.

Feature summary:

| Feature ID | Feature | Status | Dependencies | Acceptance Summary |
|---|---|---|---|---|
| VPC-F2.1 | Host-owned runtime factory | Complete | VPC-E1 | Runtime creation no longer lives in the definition-facing base class |
| VPC-F2.2 | Capability injection boundary | Complete | VPC-F2.1 | Runtime resolves SDK-declared capability needs without definition-layer imports |
| VPC-F2.3 | Deprecation and compatibility path | Complete | VPC-F2.1, VPC-F2.2 | Legacy verticals continue to run during migration with warnings/tests |

#### VPC-F2.1: Host-Owned Runtime Factory

Context:

- `VerticalBase.create_agent()` currently imports `victor.framework.Agent`, which
  is the most direct runtime-coupling point.

Acceptance criteria:

- Runtime creation moves behind a host-owned factory or adapter.
- Definition-layer vertical classes no longer instantiate runtime agents directly.
- Vertical application remains a runtime concern owned by core/framework services.

Likely touchpoints:

- `victor/core/verticals/base.py`
- `victor/framework/vertical_service.py`
- `victor/framework/agent.py`

Tasks:

- [x] VPC-T2.1 Design `AgentFactory` / `VerticalRuntimeAdapter` interfaces.
- [x] VPC-T2.2 Implement host-owned translation from vertical definition to runtime agent configuration.
- [x] VPC-T2.3 Route vertical application through the new factory/adapter path.
- [x] VPC-T2.4 Remove or isolate direct `Agent` imports from the definition-facing base class.

#### VPC-F2.2: Capability Injection Boundary

Context:

- Current verticals import capability implementations directly from the framework.

Acceptance criteria:

- Runtime resolves capability requirements declared through the SDK.
- Missing capability handling is explicit and testable.
- Definition modules no longer import capability implementation classes.

Likely touchpoints:

- `victor/framework/capabilities/`
- `victor/core/bootstrap.py`
- `victor/core/capability_registry.py`

Tasks:

- [x] VPC-T2.5 Introduce a runtime capability registry/resolver for SDK-declared capability IDs.
- [x] VPC-T2.6 Map SDK capability identifiers to runtime implementations.
- [x] VPC-T2.7 Define error paths and diagnostics for unavailable capabilities.
- [x] VPC-T2.8 Add tests covering capability injection and failure handling.

#### VPC-F2.3: Deprecation And Compatibility Path

Context:

- Migration will be gradual, so the runtime needs a safe bridge for old-style
  verticals while new-style definitions are adopted.

Acceptance criteria:

- `VerticalBase.create_agent()` has a documented deprecation path.
- Legacy verticals can still execute during the transition.
- Compatibility behavior is covered by tests and release notes.

Likely touchpoints:

- `victor/core/verticals/base.py`
- `docs/development/deprecation-inventory-2026-03-03.md`
- `tests/`

Tasks:

- [x] VPC-T2.9 Deprecate `VerticalBase.create_agent()` with clear warnings and migration guidance.
- [x] VPC-T2.10 Add compatibility shims or adapters for legacy vertical implementations.
- [x] VPC-T2.11 Document the removal milestone and release-note requirements.
- [x] VPC-T2.12 Add backward-compatibility tests for legacy vertical activation.

### VPC-E3: Definition-Layer Decoupling

Objective:

- Migrate vertical definition modules so they depend on SDK contracts only, while
  runtime-heavy helpers move behind adapters or extension points.

Feature summary:

| Feature ID | Feature | Status | Dependencies | Acceptance Summary |
|---|---|---|---|---|
| VPC-F3.1 | Shared migration scaffolding | Completed | VPC-E1, VPC-E2 | Common split pattern exists for definition vs runtime modules |
| VPC-F3.2 | Coding vertical migration | Completed | VPC-F3.1 | `coding` becomes the first SDK-only definition migration |
| VPC-F3.3 | RAG vertical migration | Completed | VPC-F3.1 | `rag` definition layer follows SDK-only pattern and parity coverage is in place |
| VPC-F3.4 | DevOps vertical migration | Completed | VPC-F3.1 | `devops` definition layer follows SDK-only pattern and parity coverage is in place |
| VPC-F3.5 | Data Analysis vertical migration | Completed | VPC-F3.1 | `dataanalysis` definition layer follows SDK-only pattern |
| VPC-F3.6 | Research vertical migration | Completed | VPC-F3.1 | `research` definition layer follows SDK-only pattern |
| VPC-F3.7 | External example and template migration | Completed | VPC-F3.1 | Example packages teach the supported contract |
| VPC-F3.8 | Extension declaration cleanup | Completed | VPC-F3.2 through VPC-F3.7 | Runtime add-ons are declared cleanly and consistently |

#### VPC-F3.1: Shared Migration Scaffolding

Context:

- Every vertical currently mixes definition concerns with runtime-heavy helpers.
  The project needs a repeatable split pattern before migrating verticals one by one.

Acceptance criteria:

- A target file/package layout exists for definition vs runtime layers.
- Shared helpers are classified as SDK contract, runtime adapter, or shim.
- Temporary mixed-mode adapters are defined for use during the migration.

Likely touchpoints:

- `victor/verticals/contrib/`
- `victor/core/verticals/`
- `docs/development/extending/verticals.md`

Tasks:

- [x] VPC-T3.1 Classify existing vertical modules into definition-layer, runtime-layer, and shim candidates.
- [x] VPC-T3.2 Define the target package layout for migrated verticals.
- [x] VPC-T3.3 Extract common runtime-only helpers out of definition modules.
- [x] VPC-T3.4 Create temporary mixed-mode adapter patterns for verticals not yet fully migrated.

#### VPC-F3.2: Coding Vertical Migration

Context:

- `coding` has the highest measured coupling and is the best proving ground for the
  new SDK contract and runtime adapter model.

Acceptance criteria:

- `coding` definition modules use SDK-only imports.
- Direct capability implementation imports are removed from the definition layer.
- Runtime-specific behavior is moved behind adapters or extension hooks.
- Migration tests cover discovery, activation, and behavior parity.

Likely touchpoints:

- `victor/verticals/contrib/coding/`
- `tests/verticals/`
- `tests/integration/`

Tasks:

- [x] VPC-T3.5 Inventory and classify `coding` imports by definition/runtime/shim layer.
- [x] VPC-T3.6 Replace `victor.tools.tool_names` usage with SDK identifiers in `coding`.
- [x] VPC-T3.7 Replace direct framework capability imports with SDK-declared capability requirements in `coding`.
- [x] VPC-T3.8 Move runtime-specific middleware, workflows, and helpers out of `coding` definition modules.
- [x] VPC-T3.9 Add coding vertical migration tests and parity checks.

#### VPC-F3.3: RAG Vertical Migration

Context:

- `rag` likely exercises retrieval, vector, and document integrations, making it a
  strong test of the capability-requirement contract.

Acceptance criteria:

- `rag` definition modules use SDK-only imports.
- Retrieval/vector/document capability needs are declared via SDK identifiers.
- Runtime integrations are isolated from the definition layer.

Likely touchpoints:

- `victor/verticals/contrib/rag/`
- `tests/verticals/`

Tasks:

- [x] VPC-T3.10 Inventory and classify `rag` imports by layer.
- [x] VPC-T3.11 Replace definition-layer imports with SDK contracts in `rag`.
- [x] VPC-T3.12 Express retrieval/vector/document needs through SDK capability identifiers in `rag`.
- [x] VPC-T3.13 Move runtime integrations out of `rag` definition modules.
- [x] VPC-T3.14 Add RAG migration tests and parity checks.

#### VPC-F3.4: DevOps Vertical Migration

Context:

- `devops` is likely to depend on shell, git, and operational tooling, which makes
  it important for capability injection and safety boundaries.

Acceptance criteria:

- `devops` definition modules use SDK-only imports.
- Shell/git/infra capability needs are declarative.
- Runtime-specific integrations are separated from the definition layer.

Likely touchpoints:

- `victor/verticals/contrib/devops/`
- `tests/verticals/`

Tasks:

- [x] VPC-T3.15 Inventory and classify `devops` imports by layer.
- [x] VPC-T3.16 Replace definition-layer imports with SDK contracts in `devops`.
- [x] VPC-T3.17 Express shell/git/infra needs through SDK capability identifiers in `devops`.
- [x] VPC-T3.18 Move runtime integrations out of `devops` definition modules.
- [x] VPC-T3.19 Add DevOps migration tests and parity checks.

#### VPC-F3.5: Data Analysis Vertical Migration

Context:

- `dataanalysis` already has special package-name handling in the resolver, so its
  migration needs both contract cleanup and packaging awareness.

Acceptance criteria:

- `dataanalysis` definition modules use SDK-only imports.
- Data/file/notebook capabilities are declarative.
- Resolver/package naming stays correct during migration.

Likely touchpoints:

- `victor/verticals/contrib/dataanalysis/`
- `victor/core/verticals/import_resolver.py`
- `tests/verticals/`

Tasks:

- [x] VPC-T3.20 Inventory and classify `dataanalysis` imports by layer.
- [x] VPC-T3.21 Replace definition-layer imports with SDK contracts in `dataanalysis`.
- [x] VPC-T3.22 Express data/file/notebook needs through SDK capability identifiers in `dataanalysis`.
- [x] VPC-T3.23 Move runtime integrations out of `dataanalysis` definition modules.
- [x] VPC-T3.24 Add Data Analysis migration tests and parity checks.

#### VPC-F3.6: Research Vertical Migration

Context:

- `research` will stress the capability boundary around web/search/document access.

Acceptance criteria:

- `research` definition modules use SDK-only imports.
- Web/search/document capabilities are declarative.
- Runtime-specific integrations are isolated from the definition layer.

Likely touchpoints:

- `victor/verticals/contrib/research/`
- `tests/verticals/`

Tasks:

- [x] VPC-T3.25 Inventory and classify `research` imports by layer.
- [x] VPC-T3.26 Replace definition-layer imports with SDK contracts in `research`.
- [x] VPC-T3.27 Express web/search/document needs through SDK capability identifiers in `research`.
- [x] VPC-T3.28 Move runtime integrations out of `research` definition modules.
- [x] VPC-T3.29 Add Research migration tests and parity checks.

#### VPC-F3.7: External Example And Template Migration

Context:

- The external example is part of the public contract. If it stays wrong, users
  will keep building against the wrong boundary.

Acceptance criteria:

- Example vertical source and README use the supported SDK-only definition pattern.
- Example build/install/discovery succeeds in a clean environment.
- Templates and contributor docs match the same pattern.

Likely touchpoints:

- `examples/external_vertical/`
- `victor-sdk/README.md`
- `docs/development/extending/verticals.md`

Tasks:

- [x] VPC-T3.30 Rewrite `examples/external_vertical` to use SDK-only definition imports.
- [x] VPC-T3.31 Validate clean-environment install/discovery for the external example.
- [x] VPC-T3.32 Update scaffolding/templates used for new vertical packages.
- [x] VPC-T3.33 Update contributor documentation to reference the new example.

#### VPC-F3.8: Extension Declaration Cleanup

Context:

- Prompts, stages, tools, teams, safety rules, and other extensions are currently
  declared through a mixture of runtime-facing patterns.

Acceptance criteria:

- The project has one documented declaration pattern for definition-layer metadata.
- Runtime-only add-ons are attached through explicit adapters or entry points.
- Definition-layer modules no longer reach into framework registries directly.

Likely touchpoints:

- `victor/core/verticals/protocols/`
- `victor/framework/entry_point_loader.py`
- `victor/framework/vertical_service.py`

Tasks:

- [x] VPC-T3.34 Standardize how prompts, stages, tools, and teams are declared in the definition contract.
- [x] VPC-T3.35 Move runtime-only add-ons behind adapters or extension entry points.
- [x] VPC-T3.36 Remove direct framework-registry access from migrated definition modules.
- [x] VPC-T3.37 Add import-boundary verification for all migrated verticals.

### VPC-E4: Packaging And Source-Of-Truth Convergence

Objective:

- Remove the split-brain packaging model so each vertical has one authoritative
  implementation path.

Feature summary:

| Feature ID | Feature | Status | Dependencies | Acceptance Summary |
|---|---|---|---|---|
| VPC-F4.1 | Release topology and extraction sequence | Completed | VPC-E3 | Maintainers know which vertical moves when and under what release policy |
| VPC-F4.2 | Entry-point and package metadata alignment | Not Started | VPC-F4.1 | Packaging metadata matches the accepted source-of-truth model |
| VPC-F4.3 | Bundled contrib shims and removals | Not Started | VPC-F4.2 | Bundled copies are reduced to shims or removed safely |
| VPC-F4.4 | Import resolver convergence | Not Started | VPC-F4.3 | Resolver behavior matches the final architecture |
| VPC-F4.5 | Operational docs and install UX alignment | Not Started | VPC-F4.2 | User-facing install/upgrade flows match the real packaging topology |

#### VPC-F4.1: Release Topology And Extraction Sequence

Context:

- The architecture can only converge cleanly if the release order and ownership
  model are explicit.

Acceptance criteria:

- Extraction order is defined per vertical.
- Package naming, ownership, and versioning rules are explicit.
- Compatibility-shim retirement milestones are documented.

Likely touchpoints:

- `pyproject.toml`
- release tooling under `scripts/`
- `docs/roadmap/vertical-platform-convergence-plan.md`
- `docs/development/vertical-extraction-order-and-dependency-graph-2026-03-12.md`

Tasks:

- [x] VPC-T4.1 Define the per-vertical extraction order and dependency graph.
- [x] VPC-T4.2 Define package naming, ownership, and versioning policy for extracted verticals.
- [x] VPC-T4.3 Define bundled-shim retirement milestones and release criteria.
- [x] VPC-T4.4 Update release checklists for the extracted-vertical model.

#### VPC-F4.2: Entry-Point And Package Metadata Alignment

Context:

- Today the main package advertises bundled contrib verticals directly. That has
  to change once a vertical has a different authoritative package.

Acceptance criteria:

- Entry points and package metadata reflect the authoritative implementation source.
- Core package extras/default installs are aligned with the chosen packaging model.
- CLI install/list flows reflect real package behavior.

Likely touchpoints:

- `pyproject.toml`
- extracted vertical package `pyproject.toml` files
- `victor/ui/commands/vertical.py`

Tasks:

- [ ] VPC-T4.5 Update entry points for extracted vertical packages.
- [ ] VPC-T4.6 Stop advertising bundled contrib implementations as authoritative once a vertical flips.
- [ ] VPC-T4.7 Align extras/default install behavior with the accepted packaging strategy.
- [ ] VPC-T4.8 Verify `victor vertical install` and related CLI output against the new package topology.

#### VPC-F4.3: Bundled Contrib Shims And Removals

Context:

- The main repo currently contains bundled copies. Extraction only becomes real
  when those copies stop being parallel authorities.

Acceptance criteria:

- Bundled copies either become thin shims or are removed after the support window.
- No functional divergence remains between bundled and authoritative packages.
- Tests and imports are updated to stop depending on duplicate implementations.

Likely touchpoints:

- `victor/verticals/contrib/`
- `tests/`

Tasks:

- [ ] VPC-T4.9 Convert bundled contrib implementations to forwarding shims or migration adapters.
- [ ] VPC-T4.10 Remove duplicated business logic from the core repo after source-of-truth flips.
- [ ] VPC-T4.11 Update tests and imports that still rely on bundled contrib implementations.
- [ ] VPC-T4.12 Delete compatibility shims after the documented support window ends.

#### VPC-F4.4: Import Resolver Convergence

Context:

- Resolver fallback order is a compatibility asset today but becomes technical debt
  once packaging converges.

Acceptance criteria:

- Resolver ordering matches the accepted final architecture.
- Dead fallback branches are removed.
- Missing-package errors stay explicit and actionable.

Likely touchpoints:

- `victor/core/verticals/import_resolver.py`
- `tests/unit/`

Tasks:

- [ ] VPC-T4.13 Simplify import resolver ordering to match the accepted source-of-truth model.
- [ ] VPC-T4.14 Remove obsolete fallback branches once no longer needed.
- [ ] VPC-T4.15 Improve missing-package diagnostics for install/upgrade scenarios.
- [ ] VPC-T4.16 Add resolver tests for mixed-version, upgrade, and downgrade environments.

#### VPC-F4.5: Operational Docs And Install UX Alignment

Context:

- Even after the code is fixed, user confusion will persist if install and upgrade
  docs still describe the wrong package layout.

Acceptance criteria:

- Install/upgrade/pin flows are documented for the chosen topology.
- Contributor docs explain how to add a new vertical without violating the boundary.
- Source-of-truth registry is reflected in user-facing and contributor-facing docs.

Likely touchpoints:

- `docs/verticals/`
- `docs/development/extending/verticals.md`
- `victor/ui/commands/vertical.py`

Tasks:

- [ ] VPC-T4.17 Publish install/upgrade/pinning guidance for extracted verticals.
- [ ] VPC-T4.18 Update contributor docs for adding new verticals under the accepted architecture.
- [ ] VPC-T4.19 Keep the source-of-truth registry synchronized with the live package topology.

### VPC-E5: Hardening And CI Guardrails

Objective:

- Make the target architecture hard to regress by adding automated checks,
  smoke tests, invalidation tests, and benchmarks.

Feature summary:

| Feature ID | Feature | Status | Dependencies | Acceptance Summary |
|---|---|---|---|---|
| VPC-F5.1 | Forbidden-import CI guardrails | Completed | VPC-E3 | Definition-layer boundary violations fail automatically |
| VPC-F5.2 | SDK-only example and packaging smoke tests | Completed | VPC-E3, VPC-E4 | Public authoring/install path is continuously verified |
| VPC-F5.3 | Cache invalidation and plugin refresh hardening | Completed | VPC-E2, VPC-E4 | Refresh/install/update flows invalidate stale state correctly |
| VPC-F5.4 | Startup and discovery telemetry | In Progress | VPC-E4 | Performance work is measurement-driven and regression visible |
| VPC-F5.5 | Release readiness gates | Not Started | VPC-E1 through VPC-E4 | Bundled-shim removals and releases have an explicit go/no-go checklist |

#### VPC-F5.1: Forbidden-Import CI Guardrails

Context:

- Without automated import-boundary checks, the codebase will drift back to
  framework-coupled authoring.

Acceptance criteria:

- Definition-layer modules fail CI when they import forbidden runtime modules.
- Exception handling, if any, is explicit and documented.
- Guardrails are easy for maintainers to run locally.

Likely touchpoints:

- `scripts/`
- `tests/`
- CI configuration

Tasks:

- [x] VPC-T5.1 Implement a definition-layer forbidden-import check.
- [x] VPC-T5.2 Wire the check into CI and local developer workflows.
- [x] VPC-T5.3 Document the exception process, if any, for temporary migration waivers.
- [x] VPC-T5.4 Fail on `victor.framework`, `victor.core.verticals`, and core tool-registry imports from definition modules.

#### VPC-F5.2: SDK-Only Example And Packaging Smoke Tests

Context:

- The architecture only becomes trustworthy when the documented external authoring
  flow is tested continuously.

Acceptance criteria:

- The example external vertical builds, installs, discovers, and activates in CI.
- Packaging smoke tests cover SDK plus at least one extracted vertical.
- Docs/examples are kept in sync by testable checks.

Likely touchpoints:

- `examples/external_vertical/`
- `tests/integration/`
- package build scripts

Tasks:

- [x] VPC-T5.5 Add a smoke test that builds and installs the SDK-only external vertical example.
- [x] VPC-T5.6 Add an entry-point discovery and activation smoke test for an external vertical package.
- [x] VPC-T5.7 Add wheel/sdist verification for `victor-sdk` and at least one vertical package.
- [x] VPC-T5.8 Add a docs/example integrity check so examples cannot drift silently.

#### VPC-F5.3: Cache Invalidation And Plugin Refresh Hardening

Context:

- Current caches are helpful for performance but can hide changes after package
  installs or upgrades.

Acceptance criteria:

- The runtime invalidates stale vertical metadata after install/upgrade/uninstall.
- Cache refresh behavior is explicit and tested.
- CLI flows that change vertical packages trigger the right invalidation path.

Likely touchpoints:

- `victor/core/verticals/base.py`
- `victor/core/verticals/vertical_loader.py`
- `victor/framework/entry_point_loader.py`
- `victor/ui/commands/vertical.py`

Tasks:

- [x] VPC-T5.9 Define cache invalidation triggers for install, uninstall, upgrade, and reload.
- [x] VPC-T5.10 Implement invalidation hooks or version-aware refresh behavior.
- [x] VPC-T5.11 Add tests for TTL override, explicit refresh, and post-install/post-upgrade behavior.
- [x] VPC-T5.12 Add CLI-level tests for `victor vertical install` and refresh flows.

#### VPC-F5.4: Startup And Discovery Telemetry

Context:

- Loader optimization was overstated in the external analysis. The right response
  is measurement first, not speculative rewrites.

Acceptance criteria:

- Startup/discovery benchmarks exist.
- Discovery-path telemetry or logging makes regressions visible.
- A regression threshold is defined for future changes.

Likely touchpoints:

- benchmark scripts under `scripts/` or `tests/benchmark/`
- `victor/core/verticals/vertical_loader.py`
- `victor/framework/entry_point_loader.py`

Tasks:

- [x] VPC-T5.13 Add a startup/discovery benchmark script or test.
- [x] VPC-T5.14 Capture baseline metrics before and after resolver/discovery changes.
- [ ] VPC-T5.15 Add telemetry or structured logging around discovery/cache behavior.
- [ ] VPC-T5.16 Define regression thresholds and reporting rules.

#### VPC-F5.5: Release Readiness Gates

Context:

- Shim removals and source-of-truth flips should be governed by an explicit release
  checklist rather than by ad hoc confidence.

Acceptance criteria:

- The project has a go/no-go checklist for removing bundled copies.
- Compatibility matrices exist for supported upgrade paths.
- Maintainers can run a final migration dry run before release.

Likely touchpoints:

- release docs
- `docs/roadmap/vertical-platform-convergence-plan.md`
- `scripts/`

Tasks:

- [ ] VPC-T5.17 Define the release go/no-go checklist for removing bundled vertical copies.
- [ ] VPC-T5.18 Add compatibility matrices covering supported upgrade and downgrade paths.
- [ ] VPC-T5.19 Add a final migration dry-run procedure for clean environments.

## Current Tranche

Scope:

- Continue additive non-breaking platform convergence work
- Preserve architecture context across sessions in one place
- Keep downstream epics execution-ready but do not start breaking packaging work
  until ADR-007 is explicitly accepted
- Prefer planning, guardrails, smoke tests, and release-ops work while
  packaging flips remain gated

Immediate next tasks:

1. VPC-T5.15 Add telemetry or structured logging around discovery/cache behavior.
2. VPC-T5.16 Define regression thresholds and reporting rules.
3. VPC-T4.5 Update entry points for extracted vertical packages (gated on ADR-007 acceptance).

Likely touchpoints:

- `docs/architecture/adr/007-vertical-distribution-and-sdk-boundary.md`
- `docs/architecture/adr/README.md`
- `docs/roadmap/vertical-platform-convergence-plan.md`
- `docs/development/vertical-module-layer-classification-2026-03-10.md`
- `docs/development/vertical-package-layout-target-2026-03-10.md`
- `docs/development/vertical-extraction-order-and-dependency-graph-2026-03-12.md`
- `docs/development/vertical-package-naming-ownership-versioning-policy-2026-03-12.md`
- `docs/development/bundled-vertical-shim-retirement-milestones-2026-03-12.md`
- `docs/development/definition-import-boundary-waivers-2026-03-12.md`

## Session Log

### 2026-03-10 (Session A)

- Verified current code paths before starting execution:
  - bundled contrib verticals are published from `victor-ai`
  - import resolution still prefers external wheel namespaces
  - SDK already contains `StageDefinition`, `Tier`, and `VerticalConfig`
  - `VerticalBase.create_agent()` still imports `victor.framework.Agent`
  - discovery/startup entry-point scanning is lazy and cached
  - current contrib verticals and example external vertical remain framework-coupled
- Created ADR-007 as the decision gate for the vertical distribution model.
- Created this persistent execution tracker for future sessions.
- Next:
  - complete VPC-T0.3 through VPC-T0.6
  - define the compatibility policy and import-boundary matrix
  - do not begin extraction or packaging removal until ADR-007 is explicitly accepted

### 2026-03-10 (Session B)

- Expanded this tracker from a seed roadmap into a reusable backlog artifact.
- Added the decision gate, compatibility policy, import-boundary matrix,
  source-of-truth registry, and SDK-only acceptance checklist.
- Captured measured baseline data:
  - 79 contrib files currently import `victor.framework`, `victor.core`, or
    `victor.tools`
  - import-line match counts by vertical: coding 70, rag 39, devops 37,
    dataanalysis 36, research 34
- Recorded the primary doc/example contradictions:
  - `docs/development/extending/verticals.md` teaches core/framework-coupled authoring
  - `docs/verticals/*` still reference `victor.tools.tool_names`
  - `examples/external_vertical/*` still uses `victor.core.verticals` and core protocols
- Expanded all identified epics, features, and tasks with context, dependencies,
  acceptance criteria, and likely touchpoints for future sessions.
- Remaining immediate item:
  - VPC-T0.6 decision announcement package and follow-on doc update list

### 2026-03-10 (Session C)

- Started additive non-breaking VPC-E1 work while ADR-007 remains `Proposed`.
- Implemented the first SDK-owned shared-constants layer:
  - added `victor_sdk.constants.ToolNames` and alias helpers
  - re-exported the canonical registry from top-level `victor_sdk`
  - converted `victor.tools.tool_names` into a compatibility wrapper
  - pointed `victor.framework.tool_naming` at the SDK-owned registry
- Marked VPC-T1.1, VPC-T1.2, VPC-T1.4, and VPC-T1.5 complete.
- Added compatibility tests for:
  - SDK exports and alias resolution
  - legacy `victor.tools.tool_names` import compatibility
  - existing framework tool-naming behavior
- Verification:
  - `../.venv/bin/pytest -q victor-sdk/tests/unit/test_tool_names.py tests/unit/tools/test_tool_names_sdk_compat.py tests/unit/framework/test_tool_naming.py`
  - result: 118 passed
- Next recommended implementation layer:
  - VPC-T1.3 typed capability identifiers/requirement literals in `victor-sdk`

### 2026-03-10 (Session D)

- Implemented the second additive SDK contract layer for capabilities:
  - added `victor_sdk.constants.CapabilityIds`
  - added `CapabilityRequirement` plus normalization helpers in `victor_sdk.core.types`
  - updated capability protocols to accept typed requirements while preserving legacy strings
  - extended `VerticalMetadata` to carry structured capability requirements
- Marked VPC-T1.3 complete and VPC-F1.1 complete.
- Completed the remaining P0 governance item VPC-T0.6 by recording the ADR
  decision announcement package and the required follow-on documentation updates.
- Verification:
  - `../.venv/bin/pytest -q victor-sdk/tests/unit/test_capability_requirements.py victor-sdk/tests/unit/test_protocols.py victor-sdk/tests/unit/test_tool_names.py tests/integration/test_sdk_integration.py tests/unit/tools/test_tool_names_sdk_compat.py`
  - result: 42 passed
- Next recommended implementation layer:
  - VPC-T1.6 define the `VerticalDefinition` data model and minimum required fields

### 2026-03-10 (Session E)

- Implemented the next additive SDK definition-contract layer:
  - added `victor_sdk.VerticalDefinition`
  - added `StageDefinition.to_dict()` for nested serialization
  - added `VerticalDefinition.to_dict()`, `to_config()`, and `from_config()`
  - added default `VerticalBase.get_definition()` and routed SDK `get_config()`
    through the definition contract
- Marked VPC-T1.6 complete and moved VPC-F1.2 to `In Progress`.
- Added tests covering:
  - definition serialization and config round-tripping
  - default SDK `get_definition()` generation
  - victor-ai compatibility for inherited `get_definition()`
- Verification:
  - `../.venv/bin/pytest -q victor-sdk/tests/unit/test_vertical_definition.py victor-sdk/tests/unit/test_protocols.py victor-sdk/tests/unit/test_capability_requirements.py victor-sdk/tests/unit/test_tool_names.py tests/integration/test_sdk_integration.py tests/unit/tools/test_tool_names_sdk_compat.py`
  - result: 46 passed
- Next recommended implementation layer:
  - VPC-T1.7 define serializable tool and capability requirement types

### 2026-03-10 (Session F)

- Implemented the typed tool-requirement layer for the SDK definition contract:
  - added `victor_sdk.ToolRequirement`
  - added normalization helpers for legacy string tool declarations
  - extended `VerticalDefinition` to carry serialized `tool_requirements`
  - added default `VerticalBase.get_tool_requirements()` with backward-compatible
    fallback to `get_tools()`
- Marked VPC-T1.7 complete.
- Added tests covering:
  - tool requirement normalization
  - `VerticalDefinition` round-tripping with typed tool requirements
  - inherited victor-ai `get_definition()` compatibility with tool requirements
- Verification:
  - `../.venv/bin/pytest -q victor-sdk/tests/unit/test_tool_requirements.py victor-sdk/tests/unit/test_vertical_definition.py victor-sdk/tests/unit/test_protocols.py victor-sdk/tests/unit/test_capability_requirements.py victor-sdk/tests/unit/test_tool_names.py tests/integration/test_sdk_integration.py tests/unit/tools/test_tool_names_sdk_compat.py`
  - result: 49 passed
- Next recommended implementation layer:
  - VPC-T1.8 define how prompts, stages, and workflow metadata are represented in the definition contract

### 2026-03-10 (Session G)

- Completed the remaining `VPC-F1.2` manifest-contract work:
  - closed the prompt/workflow metadata gap with constructor-time normalization
  - added schema-version compatibility helpers for `definition_version`
  - added validation for malformed tool, stage, and workflow references
  - added `VerticalDefinition.from_dict()` for serialized manifest round-tripping
  - preserved victor-ai compatibility by normalizing richer runtime stage objects
- Marked VPC-T1.8, VPC-T1.9, and VPC-T1.10 complete.
- Added tests covering:
  - serialized tool and capability requirement normalization
  - definition version helper behavior
  - invalid workflow/stage/tool-reference validation failures
  - `VerticalDefinition.from_dict()` round-tripping
  - victor-ai inherited `get_definition()` compatibility after validation
- Verification:
  - `../.venv/bin/pytest -q victor-sdk/tests/unit/test_vertical_definition.py victor-sdk/tests/unit/test_prompt_workflow_metadata.py victor-sdk/tests/unit/test_protocols.py victor-sdk/tests/unit/test_tool_requirements.py victor-sdk/tests/unit/test_capability_requirements.py tests/integration/test_sdk_integration.py tests/unit/tools/test_tool_names_sdk_compat.py tests/unit/framework/test_tool_naming.py`
  - result: 169 passed
- Next recommended implementation layer:
  - VPC-T1.11 update SDK documentation to describe the supported definition contract

### 2026-03-10 (Session H)

- Completed `VPC-T1.11` and moved `VPC-F1.3` to `In Progress`.
- Updated the SDK-facing documentation to teach the current supported contract:
  - rewrote `victor-sdk/README.md` around SDK-owned identifiers and the
    manifest-first authoring model
  - updated `victor-sdk/VERTICAL_DEVELOPMENT.md` to document `ToolNames`,
    `CapabilityIds`, typed requirements, and `get_definition()`
  - added a front-loaded SDK-first authoring note to
    `docs/development/extending/verticals.md` and corrected the canonical
    tool/capability reference section
- Validation:
  - verified the edited docs no longer reference `victor.tools.tool_names`,
    `SDK_GUIDE.md`, or `MIGRATION_GUIDE.md`
- Next recommended implementation layer:
  - VPC-T1.12 rewrite the external vertical example to match SDK-only definition authoring

### 2026-03-10 (Session I)

- Completed `VPC-T1.12` by rewriting the external vertical example around the
  SDK-only definition contract:
  - replaced the example `SecurityAssistant` with a manifest-first
    `victor_sdk` implementation
  - switched the example package dependency from `victor-ai` to `victor-sdk`
    with `victor-ai` moved to an optional `runtime` extra
  - simplified the package exports to the SDK-only assistant entry point
  - removed the runtime-only prompt/safety modules that made the example
    framework-coupled
  - rewrote the example README to explain SDK authoring versus runtime usage
- Validation:
  - confirmed `examples/external_vertical` no longer imports `victor.core`,
    `victor.framework`, or `victor.tools.tool_names`
  - smoke-tested `SecurityAssistant.get_definition()` via:
    `python -c "import sys; sys.path.insert(0, 'victor-sdk'); sys.path.insert(0, 'examples/external_vertical/src'); from victor_security import SecurityAssistant; definition = SecurityAssistant.get_definition(); ..."`
- Next recommended implementation layer:
  - VPC-T1.13 write a migration guide from current `VerticalBase` authoring to the SDK-first contract

### 2026-03-10 (Session J)

- Completed `VPC-T1.13` by adding a dedicated SDK migration guide:
  - added `victor-sdk/MIGRATION_GUIDE.md`
  - documented import replacement mapping, package dependency migration,
    before/after class migration, runtime-boundary rules, and verification steps
  - linked the guide from `victor-sdk/README.md`,
    `victor-sdk/VERTICAL_DEVELOPMENT.md`, and
    `docs/development/extending/verticals.md`
- Completed `VPC-T1.14` by updating the vertical scaffold/template set:
  - rewrote `assistant.py.j2` to generate an SDK-first definition layer using
    `ToolRequirement`, `CapabilityRequirement`, and `StageDefinition`
  - simplified `__init__.py.j2` to export only the assistant definition
  - converted `prompts.py.j2`, `safety.py.j2`, and `mode_config.py.j2` into
    optional metadata/runtime placeholders that do not import runtime-only modules
  - updated `victor/ui/commands/scaffold.py` messaging to describe the SDK-first
    scaffold shape
  - updated scaffold tests and CLI docs to match the new output
- Marked `VPC-F1.3` complete.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/commands/test_scaffold.py`
  - result: 11 passed
- Next recommended implementation layer:
  - VPC-T2.1 design `AgentFactory` / `VerticalRuntimeAdapter` interfaces

### 2026-03-10 (Session K)

- Started `VPC-E2` and implemented the first host-owned runtime boundary slice:
  - added `victor/framework/vertical_runtime_adapter.py` with
    `VerticalRuntimeAdapter` and `VerticalRuntimeBinding`
  - defined the additive runtime interface around resolving a vertical source
    into an SDK `VerticalDefinition`, translating that definition into the
    runtime `VerticalConfig`, and creating agents through the host runtime
  - kept compatibility for older vertical shapes by allowing adapter fallback
    from legacy `get_config()` output when needed
- Routed current runtime entrypoints through the new adapter:
  - updated `victor/framework/agent.py` so `Agent.create(vertical=...)` uses the
    adapter instead of calling `vertical.get_config()` directly
  - updated `victor/core/verticals/base.py` so `VerticalBase.create_agent()`
    delegates to the host-owned adapter instead of importing
    `victor.framework.Agent`
- Marked `VPC-T2.1`, `VPC-T2.2`, and `VPC-T2.4` complete.
- Added tests covering:
  - definition-to-runtime translation in the adapter
  - legacy-config fallback resolution
  - adapter-owned agent creation routing
  - `Agent.create(vertical=...)` working without the legacy `get_config()` path
  - `victor-ai` `VerticalBase.create_agent()` delegating to the adapter
- Verification:
  - `../.venv/bin/pytest -q tests/unit/framework/test_vertical_runtime_adapter.py tests/unit/framework/test_agent.py tests/integration/test_sdk_integration.py`
  - result: 112 passed
- Next recommended implementation layer:
  - VPC-T2.3 route vertical application through the new factory/adapter path

### 2026-03-10 (Session L)

- Completed `VPC-T2.3` and closed out `VPC-F2.1`.
- Routed the shared vertical-application path through the same host-owned
  adapter used by `Agent.create()` and `VerticalBase.create_agent()`:
  - updated `victor/framework/vertical_integration.py` so pipeline context
    creation builds runtime config via `VerticalRuntimeAdapter`
  - removed the remaining direct `vertical.get_config()` dependency from the
    shared application path
- Added a regression in `tests/unit/framework/test_vertical_service.py` proving
  a definition-only vertical can be applied even when legacy `get_config()`
  intentionally fails.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/framework/test_vertical_service.py tests/unit/framework/test_framework_internal.py tests/unit/framework/test_vertical_runtime_adapter.py tests/unit/framework/test_agent.py tests/integration/test_sdk_integration.py`
  - result: 144 passed
- Next recommended implementation layer:
  - VPC-T2.5 introduce a runtime capability registry/resolver for SDK-declared capability IDs

### 2026-03-10 (Session M)

- Implemented the runtime capability-resolution layer for SDK-declared capability
  requirements.
- Added `victor/framework/sdk_capability_registry.py` with:
  - a host-owned registry of SDK capability ID bindings
  - explicit mapping categories for orchestrator capabilities, tool bundles,
    core capability-registry providers, and built-in framework providers
  - batch resolution helpers that produce serializable diagnostics
- Added the first concrete runtime mapping set for SDK capability IDs including:
  - `file_ops`, `git`, `lsp`, `web_access`
  - `prompt_contributions`, `privacy`, `secrets_masking`, `audit_logging`
  - `stages`, `grounding_rules`, `validation`, `safety_rules`, `task_hints`,
    and `source_verification`
- Wired vertical application to resolve and record capability requirements:
  - `victor/framework/vertical_integration.py` now stores
    `sdk_capability_resolutions` in `VerticalContext.capability_configs`
  - unresolved requirements now produce explicit integration warnings instead of
    remaining passive metadata
- Marked `VPC-T2.5`, `VPC-T2.6`, `VPC-T2.7`, and `VPC-T2.8` complete.
- Added tests covering:
  - runtime registry bootstrap and known bindings
  - tool-bundle, orchestrator-capability, built-in-provider, and unknown-ID resolution
  - vertical application diagnostics and stored resolution snapshots
- Verification:
  - `../.venv/bin/pytest -q tests/unit/framework/test_sdk_capability_registry.py tests/unit/framework/test_vertical_service.py tests/unit/framework/test_framework_internal.py tests/unit/framework/test_vertical_runtime_adapter.py tests/unit/framework/test_agent.py tests/integration/test_sdk_integration.py`
  - result: 151 passed
- Next recommended implementation layer:
  - VPC-T2.9 deprecate `VerticalBase.create_agent()` with clear warnings and migration guidance

### 2026-03-10 (Session N)

- Completed `VPC-T2.9`, `VPC-T2.10`, and `VPC-T2.12`.
- Added an explicit runtime deprecation warning for
  `victor.core.verticals.VerticalBase.create_agent()`:
  - warns with `DeprecationWarning`
  - points callers to `Agent.create(vertical=MyVertical, ...)`
- Documented the deprecation in the shared inventory and migration guidance:
  - added a tracked inventory entry in
    `docs/development/deprecation-inventory-2026-03-03.md`
  - added migration guidance to `victor-sdk/MIGRATION_GUIDE.md`
- Added a compatibility shim for legacy `get_config()`-only vertical classes:
  - `VerticalRuntimeAdapter.as_runtime_vertical_class(...)` now wraps legacy
    verticals in a runtime-compatible shim class backed by the host-owned
    definition/config translation layer
  - shared vertical application paths now normalize resolved verticals through
    that shim before step-handler execution
- Added/updated tests covering:
  - deprecation warning emission for `VerticalBase.create_agent()`
  - successful activation of legacy config-only verticals through
    `apply_vertical_configuration(...)`
- Verification:
  - `../.venv/bin/pytest -q tests/unit/framework/test_vertical_service.py tests/unit/framework/test_sdk_capability_registry.py tests/unit/framework/test_framework_internal.py tests/unit/framework/test_vertical_runtime_adapter.py tests/unit/framework/test_agent.py tests/integration/test_sdk_integration.py`
  - result: 152 passed
- Next recommended implementation layer:
  - VPC-T2.11 document the removal milestone and release-note requirements

### 2026-03-10 (Session O)

- Completed `VPC-T2.11` and closed `VPC-F2.3` plus `VPC-E2`.
- Documented the removal milestone and release-note requirements for the runtime
  boundary migration in the main deprecation policy:
  - `VerticalBase.create_agent()` remains on the `v0.8.0` / `2026-12-31` removal track
  - release notes must state replacement path, removal version/date, and shim status
- Updated release-facing documentation so the requirement is operational instead of
  implicit:
  - `CHANGELOG.md` now records the `create_agent()` deprecation and legacy shim status
  - `docs/RELEASE_NOTES_TEMPLATE.md` now requires removal date, replacement, and shim status
  - `docs/development/releasing/publishing.md` pre-release checklist now includes
    deprecation inventory and release-note validation
- Updated the shared deprecation inventory to record that the legacy config-only
  vertical shim shares the same provisional removal milestone as
  `VerticalBase.create_agent()`.
- Verification:
  - no code changes in this task; prior runtime-boundary verification remains the
    latest signal:
    `../.venv/bin/pytest -q tests/unit/framework/test_vertical_service.py tests/unit/framework/test_sdk_capability_registry.py tests/unit/framework/test_framework_internal.py tests/unit/framework/test_vertical_runtime_adapter.py tests/unit/framework/test_agent.py tests/integration/test_sdk_integration.py`
  - result: 152 passed
- Next recommended implementation layer:
  - VPC-T3.1 classify existing vertical modules into definition-layer, runtime-layer, and shim candidates

### 2026-03-10 (Session P)

- Completed `VPC-T3.1` and started `VPC-F3.1`.
- Added a reusable module-layer inventory at
  `docs/development/vertical-module-layer-classification-2026-03-10.md`.
- Recorded the first E3 baseline:
  - `coding`: 22 Python files currently import `victor.framework`,
    `victor.core`, or `victor.tools`
  - `rag`: 16
  - `research`: 12
  - `devops`: 13
  - `dataanalysis`: 13
  - current contrib vertical packages contain zero `victor_sdk` Python imports
- Classified cross-cutting module families:
  - `assistant.py` and `prompts.py` are definition-layer targets
  - `__init__.py` and `tool_dependencies.py` are shim candidates
  - workflows, teams, capabilities, mode config, safety, handlers, and deep
    domain engines remain runtime-layer
- This confirms the migration order for `VPC-E3`:
  - split `assistant.py` and `prompts.py` first
  - keep package-root and tool-dependency compatibility shims until `VPC-E4`
  - treat `coding` as the highest-risk proving ground
- Verification:
  - classification task was documentation/inventory only; no tests run
- Next recommended implementation layer:
  - VPC-T3.2 define the target package layout for migrated verticals

### 2026-03-10 (Session Q)

- Completed `VPC-T3.2`.
- Added the package-layout blueprint in
  `docs/development/vertical-package-layout-target-2026-03-10.md`.
- Chose the lowest-churn migration shape:
  - keep `assistant.py` and `prompts.py` at the package root as the SDK-only
    definition layer
  - move runtime-heavy modules under `runtime/`
  - keep `__init__.py` and root `tool_dependencies.py` as narrow compatibility
    shims during transition
- Updated migration-facing docs so the layout is not tracker-only:
  - `victor-sdk/MIGRATION_GUIDE.md`
  - `docs/development/extending/verticals.md`
- Updated the external-package example to use `victor-sdk` as the primary
  dependency and show runtime extras explicitly.
- Verification:
  - documentation/layout task only; no tests run
- Next recommended implementation layer:
  - VPC-T3.3 extract common runtime-only helpers out of definition modules

### 2026-03-10 (Session R)

- Completed `VPC-T3.3`.
- Extracted two shared runtime-only helper patterns out of assistant entrypoints:
  - capability-config loading now lives in `victor.core.verticals.metadata`
    and auto-resolves `get_capability_configs()` from the vertical
    `capabilities` module
  - the shared file-operation tool group now lives in `victor_sdk.ToolNames`
    instead of requiring `FileOperationsCapability` inside definition modules
- Simplified bundled assistant entrypoints to rely on shared defaults:
  - removed assistant-local capability-config wrappers from `coding`, `rag`,
    and `research`
  - removed assistant-local capability-provider wrappers from `devops` and
    `dataanalysis`
  - removed redundant RAG extension-loader wrappers for prompt, safety, RL,
    and team-provider resolution
  - switched vertical tool references in assistant modules to SDK `ToolNames`
- Added regression coverage in
  `tests/unit/core/verticals/test_runtime_helper_defaults.py`.
- Verification:
  - `../.venv/bin/pytest -q victor-sdk/tests/unit/test_tool_names.py tests/unit/tools/test_tool_names_sdk_compat.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py`
  - `32 passed in 7.10s`
- Next recommended implementation layer:
  - VPC-T3.4 create temporary mixed-mode adapter patterns for verticals not yet fully migrated

### 2026-03-10 (Session S)

- Completed `VPC-T3.4` and closed `VPC-F3.1`.
- Added a shared mixed-mode runtime resolution pattern in
  `victor.core.verticals.import_resolver`:
  - runtime-owned module families now resolve `runtime.<module>` before the
    package-root shim within each package namespace
  - definition-layer modules such as `prompts` still use root-only resolution
- Routed shared runtime consumers through the mixed-mode resolver:
  - `victor.core.verticals.metadata` for capability-config loading
  - `victor.core.verticals.extension_loader` for safety/mode/rl/teams/capability
    extension discovery
  - `victor.core.verticals.workflow_provider` for handlers and workflows
  - `victor.framework.escape_hatch_registry` for escape-hatch discovery
- Documented the temporary adapter order in
  `docs/development/vertical-package-layout-target-2026-03-10.md`.
- Added regression coverage for:
  - runtime-first candidate ordering in `test_import_resolver.py`
  - runtime handlers/workflows in `test_workflow_provider_resolution.py`
  - runtime capabilities/safety/escape-hatch discovery in
    `test_mixed_mode_runtime_resolution.py`
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_import_resolver.py tests/unit/core/verticals/test_workflow_provider_resolution.py tests/unit/core/verticals/test_mixed_mode_runtime_resolution.py tests/unit/framework/test_escape_hatch_registry.py`
  - `57 passed, 1 skipped in 4.15s`
- Next recommended implementation layer:
  - VPC-T3.5 inventory and classify `coding` imports by definition/runtime/shim layer

### 2026-03-10 (Session T)

- Completed `VPC-T3.5` and started `VPC-F3.2`.
- Added a dedicated `coding` migration inventory in
  `docs/development/coding-vertical-import-layer-inventory-2026-03-10.md`.
- Recorded the current `coding` boundary state:
  - 21 Python files still import `victor.framework`, `victor.core`, or
    `victor.tools` (down from the original 22-file baseline)
  - 1 Python file currently imports `victor_sdk` (`assistant.py`)
  - both definition targets (`assistant.py`, `prompts.py`) still have runtime/core
    import blockers
  - both shim candidates (`__init__.py`, `tool_dependencies.py`) still depend on
    runtime/core surfaces
- Grouped the runtime-heavy modules into migration buckets for `VPC-T3.6`
  through `VPC-T3.8`:
  - capability/middleware/safety/DI runtime
  - workflow/team/RL runtime
  - enrichment/tool-composition/code-intelligence helpers
- Verification:
  - `rg -l "^(from|import) victor(\\.framework|\\.core|\\.tools)" victor/verticals/contrib/coding -g '*.py' | sort`
  - `rg -l "victor_sdk" victor/verticals/contrib/coding -g '*.py' | sort`
- Next recommended implementation layer:
  - VPC-T3.6 replace remaining framework-owned `ToolNames` usage in `coding`

### 2026-03-10 (Session U)

- Completed `VPC-T3.6`.
- Removed the remaining framework-owned tool-name import from
  `victor/verticals/contrib/coding/rl/config.py` by switching it to
  `victor_sdk.ToolNames`.
- Added regression coverage in
  `tests/unit/core/verticals/test_coding_rl_config_sdk_tool_names.py`.
- Updated the `coding` import inventory document to reflect the new state:
  - 21 Python files still import `victor.framework`, `victor.core`, or
    `victor.tools`
  - 2 Python files now import `victor_sdk`
  - no remaining `victor.framework.tool_naming` or
    `victor.tools.tool_names` imports remain under `coding`
- Verification:
  - `rg -n "victor\\.framework\\.tool_naming|victor\\.tools\\.tool_names" victor/verticals/contrib/coding -g '*.py' -S`
  - no matches
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_coding_rl_config_sdk_tool_names.py tests/unit/core/verticals/test_import_resolver.py tests/unit/core/verticals/test_workflow_provider_resolution.py tests/unit/core/verticals/test_mixed_mode_runtime_resolution.py tests/unit/framework/test_escape_hatch_registry.py tests/unit/tools/test_tool_names_sdk_compat.py`
  - `60 passed, 1 skipped in 7.72s`
- Next recommended implementation layer:
  - VPC-T3.7 replace direct framework capability imports with SDK-declared capability requirements in `coding`

### 2026-03-10 (Session V)

- Completed `VPC-T3.7`.
- Added SDK-declared capability requirements to
  `victor/verticals/contrib/coding/assistant.py`:
  - required `file_ops`
  - optional `git`
  - optional `lsp`
- This makes the `coding` definition contract explicit about host/runtime needs
  without changing current activation behavior, because the core `VerticalBase`
  already inherits the SDK definition hooks.
- Added regression coverage in
  `tests/unit/core/verticals/test_coding_definition_capability_requirements.py`
  to verify:
  - `CodingAssistant.get_definition()` exposes the expected SDK capability IDs
  - the host-owned `VerticalRuntimeAdapter` carries those requirements into
    runtime metadata
- Updated the `coding` import inventory document to record that capability
  requirements are now explicit in the definition contract.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_coding_definition_capability_requirements.py tests/unit/core/verticals/test_coding_rl_config_sdk_tool_names.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/framework/test_vertical_runtime_adapter.py tests/integration/test_sdk_integration.py`
  - `25 passed in 4.70s`
- Next recommended implementation layer:
  - VPC-T3.8 move runtime-specific middleware, workflows, and helpers out of
    `coding` definition modules

### 2026-03-10 (Session W)

- Started `VPC-T3.8` with the first non-breaking runtime-hook extraction slice.
- Moved two runtime-specific hooks out of `victor/verticals/contrib/coding/assistant.py`:
  - removed the assistant-owned `get_middleware()` override
  - removed the assistant-owned `get_service_provider()` override
- Added shared loader support in `victor.core.verticals.extension_loader` so
  verticals can resolve:
  - `middleware.py:get_middleware()`
  - `service_provider.py:{Vertical}ServiceProvider` or `get_service_provider()`
- Added runtime-module factories for `coding` in:
  - `victor/verticals/contrib/coding/middleware.py`
  - `victor/verticals/contrib/coding/service_provider.py`
- Added regression coverage to prove `CodingAssistant` now inherits those hooks
  from shared loader defaults rather than defining them directly.
- Kept `VPC-T3.8` open because `assistant.py` still owns runtime helper methods
  for composed chains and personas.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/core/verticals/test_coding_definition_capability_requirements.py tests/unit/core/verticals/test_coding_rl_config_sdk_tool_names.py tests/unit/core/verticals/test_import_resolver.py tests/unit/core/verticals/test_workflow_provider_resolution.py tests/unit/core/verticals/test_mixed_mode_runtime_resolution.py tests/integration/verticals/test_vertical_independence.py`
  - `51 passed in 8.13s`
- Next recommended implementation layer:
  - continue `VPC-T3.8` by moving `get_composed_chains()` and `get_personas()`
    out of `coding/assistant.py`

### 2026-03-10 (Session X)

- Completed `VPC-T3.8`.
- Finished extracting runtime-specific hooks from
  `victor/verticals/contrib/coding/assistant.py` by removing:
  - `get_middleware()`
  - `get_service_provider()`
  - `get_composed_chains()`
  - `get_personas()`
- Extended the shared runtime loader in
  `victor.core.verticals.extension_loader` so verticals can now resolve:
  - middleware via `middleware.py:get_middleware()`
  - service providers via `service_provider.py:{Vertical}ServiceProvider` or
    `get_service_provider()`
  - composed chains via `composed_chains.py:get_composed_chains()` or
    `<VERTICAL>_CHAINS`
  - personas via `teams:get_personas()` or `<VERTICAL>_PERSONAS`
- Added runtime factories for `coding` in:
  - `victor/verticals/contrib/coding/composed_chains.py`
  - `victor/verticals/contrib/coding/teams/__init__.py`
  - plus the earlier `middleware.py` and `service_provider.py` factories
- Updated the `coding` import inventory document to record that assistant-level
  runtime helper hooks are now gone.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/core/verticals/test_coding_definition_capability_requirements.py tests/unit/core/verticals/test_coding_rl_config_sdk_tool_names.py tests/unit/core/verticals/test_import_resolver.py tests/unit/core/verticals/test_workflow_provider_resolution.py tests/unit/core/verticals/test_mixed_mode_runtime_resolution.py tests/unit/framework/test_framework_step_handler.py tests/integration/verticals/test_vertical_independence.py`
  - `68 passed, 1 skipped in 81.63s`
- Next recommended implementation layer:
  - VPC-T3.9 add coding vertical migration tests and parity checks

### 2026-03-10 (Session Y)

- Completed `VPC-T3.9`.
- Added dedicated coding migration parity coverage in:
  - `tests/integration/verticals/test_coding_migration_parity.py`
- The new parity tests now lock three roadmap guarantees in one place:
  - discovery parity through `VerticalLoader.load("coding")` plus
    `VerticalRuntimeAdapter.build_runtime_binding()`
  - activation parity through `bootstrap_container(..., vertical="coding")`
    with DI-visible `VerticalExtensions` and coding service registrations
  - behavior parity through shared runtime-helper defaults for middleware,
    service provider, composed chains, and personas after those hooks were
    removed from `assistant.py`
- Updated the `coding` inventory document to record `VPC-T3.9` completion and
  the remaining definition-layer blockers.
- Verification:
  - `../.venv/bin/pytest -q tests/integration/verticals/test_coding_migration_parity.py tests/integration/verticals/test_vertical_independence.py tests/integration/test_registry_wiring.py tests/unit/core/verticals/test_coding_definition_capability_requirements.py tests/unit/core/verticals/test_runtime_helper_defaults.py`
  - result: 46 passed in 5.79s
- Next recommended implementation layer:
  - VPC-T3.10 inventory and classify `rag` imports by layer

### 2026-03-10 (Session Z)

- Completed `VPC-T3.10`.
- Added the reusable `rag` inventory artifact:
  - `docs/development/rag-vertical-import-layer-inventory-2026-03-10.md`
- Recorded the measured baseline for the package:
  - 29 Python files total under `victor/verticals/contrib/rag`
  - 15 files currently import `victor.framework`, `victor.core`, or
    `victor.tools`
  - 1 file currently imports `victor_sdk`
- Classified the current blockers:
  - definition layer: `assistant.py`, `prompts.py`
  - shim layer: `__init__.py`, `tool_dependencies.py`
  - runtime groups: capability/safety/enrichment, workflow/team/RL, and tool
    entrypoints
- Marked `VPC-F3.3` as `In Progress` and advanced the current tranche to
  `VPC-T3.11` and `VPC-T3.12`.
- Verification:
  - no tests run; this session added inventory/tracking artifacts only
- Next recommended implementation layer:
  - VPC-T3.11 replace definition-layer imports with SDK contracts in `rag`

### 2026-03-10 (Session AA)

- Started `VPC-T3.11` with the prompt/task-hint metadata slice for `rag`.
- Added `victor/verticals/contrib/rag/prompt_metadata.py` as a data-only prompt
  metadata module shared by the definition contract and runtime prompt adapter.
- Updated `RAGAssistant` to expose prompt templates and task-type hints through
  SDK-facing definition hooks, so `RAGAssistant.get_definition()` now carries
  serializable RAG prompt metadata.
- Reworked `victor/verticals/contrib/rag/prompts.py` into a thin runtime
  adapter over shared metadata instead of a definition module built from
  runtime prompt protocol types.
- Added regression coverage in:
  - `tests/unit/core/verticals/test_rag_definition_prompt_metadata.py`
- Updated the `rag` inventory document to record that `prompts.py` is no longer
  a definition blocker; the remaining `VPC-T3.11` blocker is `assistant.py`
  and its runtime base/stage/tier imports.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_rag_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/test_sdk_integration.py`
  - result: 21 passed in 4.37s
- Next recommended implementation layer:
  - continue `VPC-T3.11` by removing the remaining runtime definition imports
    from `rag/assistant.py`

### 2026-03-10 (Session AB)

- Continued `VPC-T3.11` with the SDK stage/tier contract slice for `rag`.
- Extended the SDK-owned `StageDefinition` contract to carry runtime-compatible
  stage metadata:
  - `keywords`
  - `next_stages`
  - `min_confidence`
  - derived `tools` property for compatibility with runtime stage consumers
- Extended the SDK-owned `TieredToolConfig` contract to carry runtime-compatible
  tier metadata:
  - `mandatory`
  - `vertical_core`
  - `semantic_pool`
  - `stage_tools`
  - `readonly_only_for_analysis`
  - plus compatibility helpers like `get_base_tools()` and
    `get_effective_semantic_pool()`
- Updated `victor/verticals/contrib/rag/assistant.py` to import
  `StageDefinition` and `TieredToolConfig` from `victor_sdk` instead of
  `victor.core`.
- Added regression coverage for the widened SDK contract and the `rag`
  assistant’s stage/tier behavior.
- Updated the `rag` inventory document to record that `assistant.py` is no
  longer blocked on runtime stage/tier types; the remaining `VPC-T3.11`
  blocker is runtime base inheritance plus missing capability requirements.
- Verification:
  - `../.venv/bin/pytest -q victor-sdk/tests/unit/test_protocols.py tests/unit/core/verticals/test_rag_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/test_sdk_integration.py`
  - result: 46 passed in 4.41s
- Next recommended implementation layer:
  - continue `VPC-T3.11` by removing the remaining runtime base dependency from
    `rag/assistant.py`, then proceed to `VPC-T3.12`

### 2026-03-10 (Session AC)

- Completed `VPC-T3.11` by removing the remaining runtime base dependency from
  `rag/assistant.py` while keeping runtime behavior backward compatible.
- Updated the host runtime boundary so SDK-only vertical classes can be loaded
  and activated safely:
  - `victor/core/verticals/base.py` now validates external verticals against
    the SDK `VerticalBase` protocol instead of requiring the runtime subclass
  - `victor/core/verticals/vertical_loader.py` now accepts SDK-only discovered
    classes and normalizes them through `VerticalRuntimeAdapter` at activation
    time
  - `victor/framework/vertical_runtime_adapter.py` now distinguishes real
    runtime subclasses from SDK-only definitions, preserves source class names
    for convention-based extension loading, and forwards optional definition
    hooks like `get_tiered_tool_config()`
- Migrated `victor/verticals/contrib/rag/assistant.py` to the SDK base and moved
  runtime compatibility to the package boundary:
  - `assistant.py` is now definition-layer only
  - `victor/verticals/contrib/rag/__init__.py` exports a runtime-compatible shim
    as `RAGAssistant` and keeps the SDK definition class available as
    `RAGAssistantDefinition`
- Added regression coverage for:
  - SDK-only loader acceptance and runtime activation shims
  - runtime-adapter shim behavior for forwarded hooks
  - SDK-only external plugin discovery validation
  - `rag` package export parity between definition-layer and runtime-layer
    contracts
- Updated the `rag` inventory document to record that definition-layer import
  cleanup is complete and the next blocker is capability declaration.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/framework/test_vertical_runtime_adapter.py tests/unit/core/test_vertical_loader_contract_enforcement.py tests/integration/verticals/test_vertical_plugin_loading.py tests/unit/core/verticals/test_rag_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/test_sdk_integration.py`
  - result: 67 passed in 6.85s
- Next recommended implementation layer:
  - `VPC-T3.12` express retrieval/vector/document needs through SDK capability
    identifiers in `rag`

### 2026-03-10 (Session AD)

- Completed `VPC-T3.12` by declaring RAG runtime needs through SDK capability
  identifiers and wiring them into the host capability registry.
- Extended the SDK capability identifier registry with RAG-relevant canonical
  IDs:
  - `document_ingestion`
  - `retrieval`
  - `vector_indexing`
- Added corresponding runtime bindings in
  `victor/framework/sdk_capability_registry.py` so the current RAG tool bundle
  satisfies those requirements without emitting unknown-capability warnings.
- Updated `victor/verticals/contrib/rag/assistant.py` to declare:
  - required file operations
  - required document ingestion
  - required retrieval
  - required vector indexing
  - optional web access for remote document ingestion
- Added regression coverage for:
  - RAG definition capability requirements
  - runtime-binding round-trip of those requirements
  - runtime registry resolution for the new SDK capability IDs
- Updated the `rag` inventory document to record that the definition contract is
  now complete and that the next work is runtime-module extraction/parity.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_rag_definition_capability_requirements.py tests/unit/core/verticals/test_rag_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/framework/test_sdk_capability_registry.py tests/integration/test_sdk_integration.py`
  - result: 33 passed in 7.01s
- Next recommended implementation layer:
  - `VPC-T3.13` move runtime integrations out of `rag` definition modules

### 2026-03-10 (Session AE)

- Started `VPC-T3.13` with the first concrete `rag/runtime/` extraction slice.
- Added `victor/verticals/contrib/rag/runtime/` and moved these runtime-owned
  modules under it:
  - `runtime/mode_config.py`
  - `runtime/safety_enhanced.py`
- Converted the package-root modules into compatibility shims:
  - `victor/verticals/contrib/rag/mode_config.py`
  - `victor/verticals/contrib/rag/safety_enhanced.py`
- Updated the package export surface in `victor/verticals/contrib/rag/__init__.py`
  to import those symbols from the new runtime package instead of the root
  shims, keeping the package root aligned with the target layout.
- Added regression coverage proving:
  - the root shims re-export the runtime classes
  - `RAGAssistant.get_mode_config_provider()` resolves through the runtime
    module path
  - shared mixed-mode runtime resolution still passes after the package move
- Updated the `rag` inventory document to record the new `runtime/` package and
  the completed first extraction slice.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/core/verticals/test_rag_definition_prompt_metadata.py tests/unit/core/verticals/test_mixed_mode_runtime_resolution.py tests/integration/test_sdk_integration.py`
  - result: 29 passed in 5.02s
- Next recommended implementation layer:
  - continue `VPC-T3.13` by moving `capabilities.py`, `enrichment.py`, or the
    primary `safety.py` module under `rag/runtime/`

### 2026-03-11 (Session AF)

- Continued `VPC-T3.13` with the next runtime extraction slice for `rag`.
- Moved the RAG capability provider/config implementation under
  `victor/verticals/contrib/rag/runtime/capabilities.py`.
- Converted `victor/verticals/contrib/rag/capabilities.py` into a compatibility
  shim that re-exports the runtime-owned provider, config helpers, and loader
  surfaces.
- Updated `victor/verticals/contrib/rag/__init__.py` to import
  `RAGCapabilityProvider` from the runtime package instead of the root shim,
  keeping the package export surface aligned with the target layout.
- Added regression coverage proving:
  - the root `capabilities.py` shim re-exports the runtime provider and config
    helper
  - `RAGAssistant.get_capability_provider()` resolves to the runtime provider
  - direct `rag` capability integration tests still pass through the root import
    path
- Updated the `rag` inventory document to record that the runtime capability
  slice is complete and that the next likely runtime moves are `enrichment.py`
  or the primary `safety.py` module.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/framework/test_vertical_capability_integration.py -k rag`
  - result: 11 passed, 52 deselected in 4.49s
- Next recommended implementation layer:
  - continue `VPC-T3.13` by moving `enrichment.py` or `safety.py` under
    `rag/runtime/`

### 2026-03-11 (Session AG)

- Continued `VPC-T3.13` with the next runtime extraction slice for `rag`.
- Moved the primary RAG safety extension and framework safety-rule factories
  under `victor/verticals/contrib/rag/runtime/safety.py`.
- Converted `victor/verticals/contrib/rag/safety.py` into a compatibility shim
  that re-exports the runtime-owned extension, safety constants, and rule
  factory functions, preserving the public `victor.rag.safety` import surface.
- Updated `victor/verticals/contrib/rag/runtime/__init__.py` to export the
  runtime safety helper alongside the already migrated mode-config,
  capabilities, and enhanced-safety modules.
- Added regression coverage proving:
  - the root `safety.py` shim re-exports the runtime safety extension and rule
    factory
  - `RAGAssistant.get_safety_extension()` resolves to the runtime-owned class
  - the external-first `victor.rag.safety` import path still passes the RAG
    safety integration suite
- Updated the `rag` inventory document to record the new `runtime/safety.py`
  module and the reduced root runtime surface.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/framework/test_safety_integration.py -k rag`
  - result: 2 passed, 1 skipped, 4 deselected in 12.12s
- Next recommended implementation layer:
  - continue `VPC-T3.13` by moving `enrichment.py` under `rag/runtime/`, then
    proceed to grouped runtime moves for `workflows/`, `teams/`, and `rl/`

### 2026-03-11 (Session AH)

- Continued `VPC-T3.13` with the remaining runtime-module extraction slices for
  `rag`.
- Moved `victor/verticals/contrib/rag/enrichment.py` under
  `victor/verticals/contrib/rag/runtime/enrichment.py` and reduced the
  package-root module to a compatibility shim for direct query-tool imports.
- Moved the workflow, team, and RL implementations under:
  - `victor/verticals/contrib/rag/runtime/workflows.py`
  - `victor/verticals/contrib/rag/runtime/teams.py`
  - `victor/verticals/contrib/rag/runtime/rl.py`
- Converted the root `workflows`, `teams`, and `rl` packages into compatibility
  shims and updated `victor/verticals/contrib/rag/runtime/__init__.py` to
  export the runtime-owned provider classes.
- Expanded regression coverage proving:
  - the root `enrichment`, `workflows`, `teams`, and `rl` shims re-export the
    runtime-owned classes/functions
  - `RAGAssistant` runtime helper resolution still returns the workflow, RL, and
    team providers through the shared loader defaults
  - the RAG workflow provider remains importable through the external-first
    integration path
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/framework/test_vertical_capability_integration.py -k rag`
  - result: 12 passed, 52 deselected in 4.72s
- Next recommended implementation layer:
  - close `VPC-T3.13` and add explicit `rag` migration parity coverage

### 2026-03-11 (Session AI)

- Completed `VPC-T3.14` by adding
  `tests/integration/verticals/test_rag_migration_parity.py`.
- The new parity suite covers:
  - discovery and runtime binding through `VerticalLoader` and
    `VerticalRuntimeAdapter`
  - bootstrap activation of RAG prompt, safety, mode-config, workflow, RL, and
    team extensions
  - behavior parity through the runtime wrapper class exported at the RAG
    package root
- Updated the tests to reflect the real migration boundary: `assistant.py`
  remains the definition class while `victor.verticals.contrib.rag` exposes the
  runtime wrapper class for loader-driven helpers.
- Marked `VPC-F3.3` complete and advanced the current tranche to the `devops`
  migration starting at `VPC-T3.15`.
- Verification:
  - `../.venv/bin/pytest -q tests/integration/verticals/test_rag_migration_parity.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/framework/test_vertical_capability_integration.py -k rag`
  - result: 15 passed, 52 deselected in 5.05s
- Next recommended implementation layer:
  - `VPC-T3.15` inventory and classify `devops` imports by layer

### 2026-03-11 (Session AJ)

- Started `VPC-F3.4` by completing `VPC-T3.15` for the `devops` vertical.
- Added `docs/development/devops-vertical-import-layer-inventory-2026-03-11.md`
  with the measured import-boundary baseline:
  - 16 Python files in `victor/verticals/contrib/devops`
  - 12 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
- Identified the current definition-layer blockers:
  - `assistant.py` still uses the core vertical base and stage types and still
    composes middleware directly
  - `prompts.py` still uses core prompt protocol objects instead of serializable
    metadata
- Classified `__init__.py` and `tool_dependencies.py` as shim-layer targets and
  grouped the runtime surface into capability/config/safety, workflow/team/RL,
  and lower-friction runtime helpers.
- Marked `VPC-F3.4` as `In Progress` and advanced the current tranche to
  `VPC-T3.16`.
- No tests were run because this session established the inventory/tracking
  baseline only.
- Next recommended implementation layer:
  - `VPC-T3.16` replace definition-layer imports with SDK contracts in `devops`

### 2026-03-11 (Session AK)

- Completed `VPC-T3.16` for the `devops` vertical.
- Migrated `victor/verticals/contrib/devops/assistant.py` to SDK-owned
  definition contracts:
  - SDK `VerticalBase`
  - SDK `StageDefinition`
  - serializable prompt metadata hooks
- Added `victor/verticals/contrib/devops/prompt_metadata.py` as the shared,
  serializable prompt/task-hint data source for the vertical.
- Converted `victor/verticals/contrib/devops/prompts.py` into a runtime adapter
  over that shared metadata using `PromptContributorAdapter`.
- Updated `victor/verticals/contrib/devops/__init__.py` to export a runtime
  wrapper `DevOpsAssistant` built from the SDK definition class, preserving the
  `victor.devops` package boundary for runtime helper calls like
  `get_extensions()`.
- Added focused regression coverage in
  `tests/unit/core/verticals/test_devops_definition_prompt_metadata.py` and
  updated shared runtime-helper coverage to import the package-root DevOps
  runtime wrapper.
- Verification:
  - `../.venv/bin/pytest -q tests/unit/core/verticals/test_devops_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py -k devops`
  - result: 9 passed, 24 deselected in 14.40s
- Next recommended implementation layer:
  - `VPC-T3.17` express shell/git/infra needs through SDK capability identifiers
    in `devops`

### 2026-03-11 (Session AL)

- Completed `VPC-T3.17` for the `devops` vertical.
- Added SDK-owned DevOps capability requirements in
  `victor/verticals/contrib/devops/assistant.py` for:
  - file operations
  - shell access
  - git access
  - container runtime access
  - validation
  - optional web access
- Extended the SDK capability registry with the new stable IDs:
  - `shell_access`
  - `container_runtime`
- Verified the current DevOps tool bundle satisfies those requirements through
  runtime resolution and added focused regression coverage for both the
  definition contract and the registry bindings.
- Verification:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_devops_definition_capability_requirements.py tests/unit/core/verticals/test_devops_definition_prompt_metadata.py tests/unit/framework/test_sdk_capability_registry.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py -k devops`
  - result: 12 passed, 31 deselected in 3.01s
- Next recommended implementation layer:
  - `VPC-T3.18` move runtime integrations out of `devops` definition modules

### 2026-03-11 (Session AM)

- Completed `VPC-T3.18` and `VPC-T3.19` for the `devops` vertical.
- Extracted assistant-owned runtime middleware into
  `victor/verticals/contrib/devops/middleware.py`, leaving the SDK definition
  class free of runtime helper implementations.
- Added `tests/integration/verticals/test_devops_migration_parity.py` to cover:
  - discovery and runtime binding through `VerticalLoader` and
    `VerticalRuntimeAdapter`
  - bootstrap activation of DevOps middleware, prompt, mode-config, workflow,
    RL, and team extensions
  - runtime-wrapper behavior through the package-root `DevOpsAssistant`
- Marked `VPC-F3.4` complete and advanced the current tranche to
  `dataanalysis` starting at `VPC-T3.20`.
- Verification:
  - `../.venv/bin/pytest -c /dev/null -q tests/integration/verticals/test_devops_migration_parity.py tests/unit/core/verticals/test_devops_definition_capability_requirements.py tests/unit/core/verticals/test_devops_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py -k devops`
  - result: 15 passed, 24 deselected in 3.11s
- Note:
  - the repo-level `pytest` configuration could not be used in this session
    because the existing `pyproject.toml` in the worktree contains an unrelated
    parse error, so focused verification used `-c /dev/null`
- Next recommended implementation layer:
  - `VPC-T3.20` inventory and classify `dataanalysis` imports by layer

### 2026-03-11 (Session AN)

- Started `VPC-F3.5` by completing `VPC-T3.20` for the `dataanalysis`
  vertical.
- Added
  `docs/development/dataanalysis-vertical-import-layer-inventory-2026-03-11.md`
  with the measured import-boundary baseline:
  - 16 Python files in `victor/verticals/contrib/dataanalysis`
  - 12 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
- Identified the current definition-layer blockers:
  - `assistant.py` still uses the core vertical base and stage types
  - `prompts.py` still uses core prompt protocol objects instead of serializable
    metadata
- Classified `__init__.py` and `tool_dependencies.py` as shim-layer targets and
  grouped the runtime surface into capability/config/safety, workflow/team/RL,
  and lower-friction runtime helpers.
- Marked `VPC-F3.5` as `In Progress` and advanced the current tranche to
  `VPC-T3.21`.
- No tests were run because this session established the inventory/tracking
  baseline only.
- Next recommended implementation layer:
  - `VPC-T3.21` replace definition-layer imports with SDK contracts in
    `dataanalysis`

### 2026-03-11 (Session AO)

- Completed `VPC-T3.21` for the `dataanalysis` vertical.
- Migrated `victor/verticals/contrib/dataanalysis/assistant.py` to the SDK
  definition contract:
  - SDK `VerticalBase`
  - SDK `StageDefinition`
  - serializable prompt metadata hooks
- Added
  `victor/verticals/contrib/dataanalysis/prompt_metadata.py` as the shared
  definition-layer source for prompt templates, task hints, grounding rules,
  and prompt priority.
- Converted `victor/verticals/contrib/dataanalysis/prompts.py` into a runtime
  adapter over the shared metadata and updated
  `victor/verticals/contrib/dataanalysis/__init__.py` to export a runtime
  wrapper plus `DataAnalysisAssistantDefinition`.
- Added regression coverage in:
  - `tests/unit/core/verticals/test_dataanalysis_definition_prompt_metadata.py`
  - `tests/unit/core/verticals/test_runtime_helper_defaults.py`
- Updated
  `docs/development/dataanalysis-vertical-import-layer-inventory-2026-03-11.md`
  with the post-migration boundary state:
  - 17 Python files in `victor/verticals/contrib/dataanalysis`
  - 12 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
  - 0 remaining definition-layer blockers in the active entrypoints
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_dataanalysis_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py -k dataanalysis`
    - `4 passed, 30 deselected, 1 warning`
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/core/verticals/test_dataanalysis_definition_prompt_metadata.py`
    - `12 passed, 1 warning`
- Advanced the current tranche to `VPC-T3.22`.
- Next recommended implementation layer:
  - `VPC-T3.22` express data/file/notebook needs through SDK capability
    identifiers in `dataanalysis`

### 2026-03-11 (Session AP)

- Completed `VPC-T3.22` for the `dataanalysis` vertical.
- Added SDK capability requirements to
  `victor/verticals/contrib/dataanalysis/assistant.py`:
  - `CapabilityIds.FILE_OPS`
  - `CapabilityIds.SHELL_ACCESS`
  - `CapabilityIds.VALIDATION`
  - optional `CapabilityIds.WEB_ACCESS`
- Added regression coverage in
  `tests/unit/core/verticals/test_dataanalysis_definition_capability_requirements.py`
  and verified the current Data Analysis tool bundle satisfies the new
  declarations.
- Started `VPC-T3.23` and extracted the first runtime-owned Data Analysis
  modules under `victor/verticals/contrib/dataanalysis/runtime/`:
  - `runtime/mode_config.py`
  - `runtime/safety_enhanced.py`
  - `runtime/__init__.py`
- Reduced the root modules to compatibility shims:
  - `mode_config.py`
  - `safety_enhanced.py`
- Updated the package export surface and regression coverage so root imports and
  runtime helper resolution still point at the runtime-owned implementations:
  - `victor/verticals/contrib/dataanalysis/__init__.py`
  - `tests/unit/core/verticals/test_runtime_helper_defaults.py`
- Updated
  `docs/development/dataanalysis-vertical-import-layer-inventory-2026-03-11.md`
  with the post-`VPC-T3.22` and partial-`VPC-T3.23` state:
  - 20 Python files in `victor/verticals/contrib/dataanalysis`
  - 12 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
  - no remaining definition-layer blockers in the active entrypoints
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_dataanalysis_definition_capability_requirements.py tests/unit/core/verticals/test_dataanalysis_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py -k dataanalysis`
    - `6 passed, 30 deselected, 1 warning`
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/core/verticals/test_dataanalysis_definition_prompt_metadata.py tests/unit/core/verticals/test_dataanalysis_definition_capability_requirements.py tests/integration/verticals/test_vertical_independence.py -k dataanalysis`
    - `7 passed, 30 deselected, 1 warning`
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_runtime_helper_defaults.py`
    - `9 passed, 1 warning`
- Advanced the current tranche to `VPC-T3.23`.
- Next recommended implementation layer:
  - continue `VPC-T3.23` by moving `capabilities.py` and `safety.py` behind
    `dataanalysis/runtime/` shims

### 2026-03-11 (Session AQ)

- Continued `VPC-T3.23` for the `dataanalysis` vertical.
- Moved the primary runtime-owned capability and safety modules under
  `victor/verticals/contrib/dataanalysis/runtime/`:
  - `runtime/capabilities.py`
  - `runtime/safety.py`
- Reduced the root modules to compatibility shims:
  - `capabilities.py`
  - `safety.py`
- Continued the same extraction pattern for additional runtime-owned modules:
  - `runtime/workflows.py`
  - `runtime/rl.py`
  - `runtime/tool_dependencies.py`
- Restored the public import paths with compatibility shims:
  - `workflows/__init__.py`
  - `rl/__init__.py`
  - `tool_dependencies.py`
- Updated the runtime package export surface and package-root exports:
  - `victor/verticals/contrib/dataanalysis/runtime/__init__.py`
  - `victor/verticals/contrib/dataanalysis/__init__.py`
- Expanded `tests/unit/core/verticals/test_runtime_helper_defaults.py` so the
  Data Analysis runtime-shim regression now covers capability, safety, workflow,
  RL, and tool-dependency resolution.
- Updated
  `docs/development/dataanalysis-vertical-import-layer-inventory-2026-03-11.md`
  with the narrower remaining runtime surface:
  - 25 Python files in `victor/verticals/contrib/dataanalysis`
  - 12 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
  - most direct runtime imports now live under `dataanalysis/runtime/`
  - `teams/` is now the largest remaining root runtime package
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/framework/test_vertical_capability_integration.py tests/integration/framework/test_safety_integration.py -k dataanalysis`
    - `11 passed, 1 skipped, 55 deselected, 1 warning`
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/framework/workflows/test_base_yaml_provider.py tests/unit/framework/workflows/test_workflow_hooks.py tests/integration/framework/test_vertical_capability_integration.py -k dataanalysis`
    - `15 passed, 1 skipped, 103 deselected, 7 warnings`
- Kept `VPC-T3.23` as the active tranche because `teams/__init__.py` and
  `teams/personas.py` still need to move behind runtime shims.
- Next recommended implementation layer:
  - continue `VPC-T3.23` by migrating the `dataanalysis/teams` package into
    `dataanalysis/runtime/`

### 2026-03-11 (Session AR)

- Completed the remaining `dataanalysis` migration gates.
- Finished the last `VPC-T3.23` runtime extraction slice by moving the
  `dataanalysis/teams` package under `victor/verticals/contrib/dataanalysis/runtime/`:
  - `runtime/teams.py`
  - `runtime/team_personas.py`
- Restored the public import surface with compatibility shims:
  - `teams/__init__.py`
  - `teams/personas.py`
- Expanded runtime-helper coverage so `dataanalysis` now asserts runtime
  resolution for:
  - capability provider
  - safety extensions
  - mode config
  - workflow provider
  - RL config
  - team spec provider
  - tool dependency provider
- Added the parity gate in
  `tests/integration/verticals/test_dataanalysis_migration_parity.py`.
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/framework/test_team_registry.py tests/integration/framework/test_persona_integration.py -k 'dataanalysis or data_analysis'`
    - `2 passed, 1 skipped, 72 deselected, 10 warnings`
  - `../.venv/bin/pytest -c /dev/null -q tests/integration/verticals/test_dataanalysis_migration_parity.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/framework/test_team_registry.py tests/integration/framework/test_persona_integration.py -k 'dataanalysis or data_analysis'`
    - `5 passed, 1 skipped, 72 deselected, 10 warnings`
- Marked `VPC-T3.23`, `VPC-T3.24`, and `VPC-F3.5` complete.
- Started `VPC-F3.6` by completing `VPC-T3.25` for the `research` vertical.
- Added `docs/development/research-vertical-import-layer-inventory-2026-03-11.md`
  with the measured import-boundary baseline:
  - 15 Python files in `victor/verticals/contrib/research`
  - 11 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
  - `assistant.py` and `prompts.py` are the current definition-layer blockers
- Advanced the current tranche to `VPC-T3.26`.
- Next recommended implementation layer:
  - `VPC-T3.26` replace definition-layer imports with SDK contracts in
    `research`

### 2026-03-11 (Session AS)

- Completed `VPC-T3.26` for the `research` vertical.
- Migrated `victor/verticals/contrib/research/assistant.py` to the SDK
  definition contract:
  - SDK `VerticalBase`
  - SDK `StageDefinition`
  - serializable prompt metadata hooks
- Added `victor/verticals/contrib/research/prompt_metadata.py` as the shared
  definition-layer source for prompt templates, task hints, grounding rules,
  and prompt priority.
- Converted `victor/verticals/contrib/research/prompts.py` into a runtime
  adapter over the shared metadata and updated
  `victor/verticals/contrib/research/__init__.py` to export a runtime wrapper
  plus `ResearchAssistantDefinition`.
- Added the research capability contract for `VPC-T3.27` in
  `victor/verticals/contrib/research/assistant.py`:
  - `CapabilityIds.FILE_OPS`
  - `CapabilityIds.WEB_ACCESS`
  - `CapabilityIds.SOURCE_VERIFICATION`
  - `CapabilityIds.VALIDATION`
- Added regression coverage in:
  - `tests/unit/core/verticals/test_research_definition_prompt_metadata.py`
  - `tests/unit/core/verticals/test_research_definition_capability_requirements.py`
- Updated
  `docs/development/research-vertical-import-layer-inventory-2026-03-11.md`
  with the post-`VPC-T3.26` / `VPC-T3.27` boundary state:
  - 16 Python files in `victor/verticals/contrib/research`
  - 11 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
  - no remaining definition-layer blockers in the active entrypoints
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_research_definition_prompt_metadata.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py -k research`
    - `9 passed, 26 deselected, 1 warning`
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_research_definition_prompt_metadata.py tests/unit/core/verticals/test_research_definition_capability_requirements.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_vertical_independence.py -k research`
    - `11 passed, 26 deselected, 1 warning`
- Marked `VPC-T3.26` and `VPC-T3.27` complete.
- Advanced the current tranche to `VPC-T3.28`.
- Next recommended implementation layer:
  - `VPC-T3.28` move runtime integrations out of `research` definition modules

### 2026-03-11 (Session AT)

- Completed the remaining `research` runtime extraction work for `VPC-T3.28`.
- Moved the remaining team runtime modules under
  `victor/verticals/contrib/research/runtime/`:
  - `runtime/teams.py`
  - `runtime/team_personas.py`
- Restored the public import surface with compatibility shims:
  - `teams/__init__.py`
  - `teams/personas.py`
- Expanded `victor/verticals/contrib/research/runtime/__init__.py` to export
  the runtime team and persona helpers alongside the existing runtime-owned
  modules.
- Expanded `tests/unit/core/verticals/test_runtime_helper_defaults.py` so the
  shared regression now verifies `research` root shims for:
  - team spec provider
  - persona helper imports
  - assistant team-provider resolution
- Added the parity gate in
  `tests/integration/verticals/test_research_migration_parity.py`.
- Updated
  `docs/development/research-vertical-import-layer-inventory-2026-03-11.md`
  with the completed post-migration state:
  - 26 Python files in `victor/verticals/contrib/research`
  - 11 files importing `victor.framework` / `victor.core` / `victor.tools`
  - 1 file importing `victor_sdk`
  - package-root and `teams/` import paths now preserved via runtime shims
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_runtime_helper_defaults.py tests/unit/framework/test_team_registry.py tests/integration/framework/test_persona_integration.py tests/integration/verticals/test_research_migration_parity.py tests/integration/verticals/test_vertical_independence.py -k research`
    - `10 passed, 1 skipped, 90 deselected, 10 warnings`
- Marked `VPC-T3.28`, `VPC-T3.29`, and `VPC-F3.6` complete.
- Advanced the current tranche to `VPC-T3.30`.
- Next recommended implementation layer:
  - `VPC-T3.30` rewrite `examples/external_vertical` to use SDK-only
    definition imports

### 2026-03-11 (Session AU)

- Verified that `examples/external_vertical` already matches the target SDK-only
  authoring contract for `VPC-T3.30`:
  - `src/victor_security/assistant.py` imports only `victor_sdk`
  - `pyproject.toml` depends on `victor-sdk` and keeps `victor-ai` in the
    optional `runtime` extra
  - `README.md` documents SDK-only authoring plus entry-point runtime discovery
- Ran a source-path smoke check:
  - `python -c "import sys; sys.path.insert(0, 'examples/external_vertical/src'); from victor_security import SecurityAssistant; definition = SecurityAssistant.get_definition(); print(definition.name); print([req.tool_name for req in definition.tool_requirements]); print([req.capability_id for req in definition.capability_requirements])"`
    - printed:
      - `security`
      - `['read', 'ls', 'code_search', 'overview', 'shell', 'web_search', 'write']`
      - `['file_ops', 'git', 'web_access']`
- Marked `VPC-T3.30` complete.
- Advanced the current tranche to `VPC-T3.31`.
- Next recommended implementation layer:
  - `VPC-T3.31` validate clean-environment install/discovery for the external
    example

### 2026-03-11 (Session AV)

- Completed `VPC-T3.31` by adding the offline clean-environment regression in
  `tests/integration/verticals/test_external_vertical_install_discovery.py`.
- The new slow integration coverage validates two install paths in throwaway
  venvs:
  - SDK-only install of `victor-sdk` + `examples/external_vertical`
  - runtime install of `victor-sdk` + local `victor-ai` + `examples/external_vertical`
- The regression proves both entry-point exposure and runtime discovery through
  `VerticalLoader`, while forcing a temp cache path so entry-point caching stays
  sandbox-safe.
- Verified `VPC-T3.32` is already satisfied by the SDK-first scaffold/template
  path:
  - `victor/templates/vertical/assistant.py.j2`
  - `victor/ui/commands/scaffold.py`
  - `tests/unit/commands/test_scaffold.py`
- Completed `VPC-T3.33` by updating contributor docs to reference the repo
  external example explicitly:
  - `victor-sdk/README.md`
  - `victor-sdk/VERTICAL_DEVELOPMENT.md`
  - `docs/development/extending/verticals.md`
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/integration/verticals/test_external_vertical_install_discovery.py`
    - `2 passed, 3 warnings`
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/commands/test_scaffold.py`
    - `11 passed, 1 warning`
- Marked `VPC-T3.31`, `VPC-T3.32`, `VPC-T3.33`, and `VPC-F3.7` complete.
- Advanced the current tranche to `VPC-T3.34`.
- Next recommended implementation layer:
  - `VPC-T3.34` standardize how prompts, stages, tools, and teams are declared
    in the definition contract

### 2026-03-11 (Session AW)

- Completed `VPC-T3.34` by extending the SDK definition contract to cover
  declarative team metadata alongside tools, prompts, stages, and workflow
  metadata.
- Added new SDK-owned serializable team types in
  `victor-sdk/victor_sdk/core/types.py`:
  - `TeamMemberDefinition`
  - `TeamDefinition`
  - `TeamMetadata`
  - normalization helpers and validation
- Updated the SDK base protocol in
  `victor-sdk/victor_sdk/verticals/protocols/base.py` with the new hooks:
  - `get_team_declarations()`
  - `get_default_team()`
  - `get_team_metadata()`
- Updated the host-owned runtime bridge so definition-backed teams are usable
  without runtime-side `teams` modules:
  - `victor/framework/vertical_runtime_adapter.py`
  - `victor/core/verticals/extension_loader.py`
- Extended the public external example to exercise the new contract in
  `examples/external_vertical/src/victor_security/assistant.py` by declaring
  a `security_review_team` through SDK hooks only.
- Added and expanded regression coverage:
  - `victor-sdk/tests/unit/test_team_metadata.py`
  - `tests/unit/framework/test_vertical_runtime_adapter.py`
  - `tests/integration/test_sdk_integration.py`
  - `tests/integration/verticals/test_external_vertical_install_discovery.py`
- Updated authoring docs to describe the new team declaration pattern:
  - `victor-sdk/README.md`
  - `victor-sdk/VERTICAL_DEVELOPMENT.md`
  - `docs/development/extending/verticals.md`
  - `examples/external_vertical/README.md`
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q victor-sdk/tests/unit/test_team_metadata.py victor-sdk/tests/unit/test_vertical_definition.py victor-sdk/tests/unit/test_protocols.py tests/unit/framework/test_vertical_runtime_adapter.py tests/integration/test_sdk_integration.py tests/integration/verticals/test_external_vertical_install_discovery.py`
    - `57 passed, 18 warnings`
- Marked `VPC-T3.34` complete and started `VPC-F3.8`.
- Advanced the current tranche to `VPC-T3.35`.
- Next recommended implementation layer:
  - `VPC-T3.35` move runtime-only add-ons behind adapters or extension entry
    points

### 2026-03-11 (Session AX)

- Completed `VPC-T3.35` by moving provider-style runtime add-ons onto explicit
  entry-point resolution paths while keeping existing runtime-module fallbacks:
  - added reusable runtime-extension entry-point loading in
    `victor/framework/entry_point_loader.py`
  - added new explicit entry-point groups for prompt contributors, mode configs,
    workflow providers, team spec providers, capability providers, and service
    providers
  - added `load_rl_config_provider_from_entry_points()` so RL config providers
    can be resolved as runtime objects instead of only raw config dicts
- Routed shared extension defaults through the framework entry-point loader in
  `victor/core/verticals/extension_loader.py`:
  - prompt contributors now prefer `victor.prompt_contributors`
  - mode config providers now prefer `victor.mode_configs`
  - tool dependency providers now route through
    `load_tool_dependency_provider_from_entry_points()`
  - RL config providers now route through
    `load_rl_config_provider_from_entry_points()`
  - workflow, team spec, capability, and service providers now prefer explicit
    provider entry-point groups before package-module fallback
- Added regression coverage for the new boundary:
  - `tests/unit/framework/test_entry_point_loader_runtime_extensions.py`
  - `tests/unit/core/verticals/test_extension_entry_point_resolution.py`
  - exercised existing normalization/stats/tool-dependency/runtime-helper suites
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/framework/test_entry_point_loader_runtime_extensions.py tests/unit/framework/test_entry_point_loader_normalization.py tests/unit/framework/test_entry_point_loader_stats.py tests/unit/framework/test_entry_point_loader_tool_dependencies.py tests/unit/core/verticals/test_extension_entry_point_resolution.py tests/unit/core/verticals/test_mixed_mode_runtime_resolution.py tests/unit/core/verticals/test_runtime_helper_defaults.py`
    - `39 passed, 9 warnings`
- Marked `VPC-T3.35` complete.
- Advanced the current tranche to `VPC-T3.36`.
- Next recommended implementation layer:
  - `VPC-T3.36` remove direct framework-registry access from migrated
    definition modules

### 2026-03-11 (Session AY)

- Completed `VPC-T3.36` by removing the last core/runtime base dependency from
  the `coding` assistant definition layer:
  - migrated `victor/verticals/contrib/coding/assistant.py` to SDK-owned
    `VerticalBase` and `StageDefinition`
  - replaced the runtime `customize_config()` hook with SDK-owned
    `get_metadata()` so coding metadata remains serializable and survives the
    runtime binding
  - converted stage declarations from the old core-only `tools=` shape to the
    SDK stage contract
- Split coding definition/runtime exports the same way the other migrated
  verticals already work:
  - `victor.verticals.contrib.coding.assistant` is now the SDK definition class
  - `victor.verticals.contrib.coding` exports the host-owned runtime wrapper via
    `VerticalRuntimeAdapter.as_runtime_vertical_class(...)`
- Updated coding parity/regression coverage:
  - `tests/integration/verticals/test_coding_migration_parity.py`
  - `tests/unit/core/verticals/test_runtime_helper_defaults.py`
  - `tests/unit/core/verticals/test_coding_definition_capability_requirements.py`
- Refreshed the persisted coding inventory in
  `docs/development/coding-vertical-import-layer-inventory-2026-03-10.md`:
  - runtime/core/tool imports under `coding` dropped from `21` to `20`
  - `assistant.py` is no longer a definition-layer blocker
  - `prompts.py` remains the remaining root-level definition blocker
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_coding_definition_capability_requirements.py tests/unit/core/verticals/test_runtime_helper_defaults.py tests/integration/verticals/test_coding_migration_parity.py`
    - `16 passed, 17 warnings`
- Marked `VPC-T3.36` complete.
- Advanced the current tranche to `VPC-T3.37`.
- Next recommended implementation layer:
  - `VPC-T3.37` add import-boundary verification for all migrated verticals

### 2026-03-12 (Session AZ)

- Completed `VPC-T3.37` by adding a repo-local import-boundary guardrail over all
  migrated definition-layer entrypoints and prompt-metadata modules:
  - added `tests/unit/core/verticals/test_definition_import_boundaries.py`
  - verifies migrated definition files only import `victor_sdk` plus explicitly
    allowed package-local definition helpers
  - verifies public definition entrypoints continue to import `victor_sdk`
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_definition_import_boundaries.py tests/unit/core/verticals/test_coding_definition_capability_requirements.py tests/unit/core/verticals/test_rag_definition_prompt_metadata.py tests/unit/core/verticals/test_devops_definition_prompt_metadata.py tests/unit/core/verticals/test_dataanalysis_definition_prompt_metadata.py tests/unit/core/verticals/test_research_definition_prompt_metadata.py`
    - `35 passed, 15 warnings`
- Marked `VPC-T3.37` complete and closed `VPC-F3.8`, which completes `VPC-E3`.
- Completed `VPC-T4.1` by defining the extraction order and packaging-risk graph
  in `docs/development/vertical-extraction-order-and-dependency-graph-2026-03-12.md`:
  - no vertical has a hard runtime dependency on another vertical
  - extraction order is therefore driven by packaging blast radius, not by
    import coupling between verticals
  - recommended operational sequence:
    `research -> devops -> dataanalysis -> rag -> coding`
  - `coding` remains last because it has the highest measured runtime coupling
    and the widest repo-level external-package assumptions
- Completed the remaining `VPC-F4.1` planning tasks:
  - `VPC-T4.2` in
    `docs/development/vertical-package-naming-ownership-versioning-policy-2026-03-12.md`
    defines canonical package names, import roots, ownership, and lockstep
    suite-versioning for first-party extracted verticals
  - `VPC-T4.3` in
    `docs/development/bundled-vertical-shim-retirement-milestones-2026-03-12.md`
    defines `N`, `N+1`, `N+2` milestone behavior plus the `2` minor release /
    `90` day minimum support window for bundled shims
  - `VPC-T4.4` updates `docs/development/releasing/publishing.md` and
    `docs/RELEASE_NOTES_TEMPLATE.md` so releases must call out authoritative
    packages, shim status, and removal milestones
- No new runtime tests were needed for `VPC-F4.1` because this tranche was
  documentation/release-policy work only.
- Completed `VPC-F5.1` as an explicit local/CI guardrail instead of leaving it
  implicit inside the general unit suite:
  - extended `tests/unit/core/verticals/test_definition_import_boundaries.py`
    so migrated definition modules now fail explicitly on
    `victor.framework`, `victor.core.verticals`,
    `victor.tools.tool_names`, and `victor.framework.tool_naming`
  - added `make test-definition-boundaries` to the local developer workflow
  - added the `Definition Import Boundaries` job to
    `.github/workflows/ci-fast.yml`
  - documented the waiver process in
    `docs/development/definition-import-boundary-waivers-2026-03-12.md`
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/verticals/test_definition_import_boundaries.py`
    - `26 passed, 1 warning`
- Completed the first `VPC-F5.2` packaging-smoke slice by strengthening
  `tests/integration/verticals/test_external_vertical_install_discovery.py`:
  - the SDK-only example is now built into a wheel before install instead of
    being tested only via editable source installs
  - the runtime-discovery smoke path now also installs the built wheel before
    loading the vertical through `VerticalLoader`
  - this closes `VPC-T5.5` and `VPC-T5.6`
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/integration/verticals/test_external_vertical_install_discovery.py`
    - `2 passed, 3 warnings`
- Completed `VPC-T5.8` by adding
  `tests/unit/core/test_external_vertical_example_integrity.py`:
  - validates that `examples/external_vertical/pyproject.toml`,
    `examples/external_vertical/README.md`, and
    `examples/external_vertical/src/victor_security/assistant.py` stay aligned
  - checks package identity, runtime extra, entry-point declaration, install
    snippets, and the example definition metadata contract
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/test_external_vertical_example_integrity.py tests/integration/verticals/test_external_vertical_install_discovery.py`
    - `4 passed, 3 warnings`
- Completed `VPC-T5.7` by adding
  `tests/integration/core/test_sdk_vertical_package_artifacts.py`:
  - builds wheel and sdist artifacts for `victor-sdk`
  - builds wheel and sdist artifacts for the SDK-only external vertical example
  - verifies packaged contract files and entry-point metadata inside the built
    archives instead of relying only on install-time smoke tests
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/integration/core/test_sdk_vertical_package_artifacts.py`
    - `2 passed, 3 warnings`
- Completed `VPC-F5.2`.
- Completed the first `VPC-F5.3` cache-invalidation slice:
  - added the trigger matrix in
    `docs/development/vertical-cache-invalidation-triggers-2026-03-12.md`
    to define install, uninstall, upgrade, and reload invalidation behavior
  - added the host-owned invalidation helper in
    `victor/core/verticals/cache_invalidation.py`
  - wired `VerticalRegistryManager.install(...)` and
    `VerticalRegistryManager.uninstall(...)` to invalidate runtime/discovery
    caches after successful package changes
  - updated `victor/ui/commands/vertical.py` so the CLI now states that the
    current process refreshed package caches and only other Victor sessions
    still need restart
  - added focused coverage in
    `tests/unit/core/test_vertical_cache_invalidation.py`
    and extended `tests/unit/commands/test_vertical_commands.py`
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/test_vertical_cache_invalidation.py tests/unit/commands/test_vertical_commands.py`
    - `53 passed, 1 warning`
- Completed the remaining `VPC-F5.3` verification and CLI coverage:
  - `tests/unit/core/test_vertical_base.py` now proves subclass TTL overrides
    actually change cache-expiry behavior instead of only asserting the class
    attribute exists
  - `tests/unit/core/test_vertical_loader_contract_enforcement.py` now proves
    `discover_verticals(force_refresh=True)` bypasses the loader cache and
    rescans entry points
  - `victor/core/verticals/registry_manager.py` now classifies successful
    package changes as `install` vs `upgrade` before invalidation
  - `tests/unit/commands/test_vertical_commands.py` now covers upgrade-triggered
    invalidation and the CLI messaging that the current process refreshed caches
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/core/test_vertical_base.py tests/unit/core/test_vertical_loader_contract_enforcement.py tests/unit/commands/test_vertical_commands.py`
    - `107 passed, 26 warnings`
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/commands/test_vertical_commands.py`
    - `52 passed, 1 warning`
- Completed `VPC-F5.3`.
- Advanced the current tranche to `VPC-T5.13`, while leaving `VPC-T4.5` and
  `VPC-T4.6` explicitly gated on ADR-007 acceptance.
- Next recommended implementation layers:
  - `VPC-T5.13` add a startup/discovery benchmark script or test
  - `VPC-T5.14` capture baseline metrics before and after resolver/discovery
    changes
  - `VPC-T4.5` update entry points for extracted vertical packages once ADR-007
    is accepted
- Completed `VPC-T5.13` by extending `scripts/benchmark_startup_kpi.py`:
  - added an explicit `--discovery-probe` mode for cold/warm vertical discovery
    timing and cache counters
  - added `--skip-agent-create` so startup/discovery measurements can run
    without a live provider backend
  - added discovery thresholds and minimums for future CI/regression use
  - added focused unit coverage in `tests/unit/test_benchmark_startup_kpi.py`
- Verification used `-c /dev/null` because the current worktree still has an
  unrelated `pyproject.toml` parse error:
  - `../.venv/bin/pytest -c /dev/null -q tests/unit/test_benchmark_startup_kpi.py`
    - `11 passed, 1 warning`
- Completed `VPC-T5.14` by capturing the initial startup/discovery baseline in
  `docs/development/startup-discovery-baseline-2026-03-12.md` using:
  - `../.venv/bin/python scripts/benchmark_startup_kpi.py --json --skip-agent-create --discovery-probe --iterations 5`
  - baseline snapshot:
    - `import_victor.cold_ms`: `1800.67`
    - `discovery_probe.cold_ms`: `127.94`
    - `discovery_probe.warm_mean_ms`: `0.00103`
    - `discovery_probe.discovered_count`: `7`
    - `discovery_probe.cache_hit_total`: `5`
- Advanced the current tranche to `VPC-T5.15`.
- Next recommended implementation layers:
  - `VPC-T5.15` add telemetry or structured logging around discovery/cache
    behavior
  - `VPC-T5.16` define regression thresholds and reporting rules
  - `VPC-T4.5` update entry points for extracted vertical packages once ADR-007
    is accepted

## Resume Protocol

1. Open this file first.
2. Confirm ADR-007 status. If still `Proposed`, continue only non-breaking
   planning, guardrails, smoke tests, documentation, and host-owned adapter work;
   do not flip package authority or remove bundled shims.
3. Continue from the first unchecked P0 task, then the first unchecked P1 task.
4. Update:
   - task checkboxes
   - feature and epic status
   - measured baseline if code realities changed
   - current tranche
   - session log
5. If a task changes the target architecture, update ADR-007 and this file in the
   same session.
