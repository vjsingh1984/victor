# Consolidated Implementation Plan (2026-03-27)

**Status**: Active execution plan
**Purpose**: Turn the tiered assessment and backlog into an implementation-grade plan with architecture patterns, sequencing, rollout strategy, and acceptance criteria.

## Executive Summary

Victor should execute the next tranche of work in this order:

1. Stabilize the planning and security foundation.
2. Remove the highest-risk async/sync compatibility seams.
3. Decompose the largest product-surface modules behind explicit boundaries.
4. Convert technical improvements into product proof: published benchmarks, clearer onboarding, and productionized observability.

The implementation style should be conservative and best-in-class:

- use a strangler-fig approach for large refactors
- introduce ports/adapters before moving behavior
- keep sync entrypoints at the outermost shell only
- favor protocol-based seams and composition roots over wide utility modules
- add contract tests before swapping implementations
- release behind feature flags where behavior changes are non-trivial

## Engineering Principles

### Architectural Principles

1. **Hexagonal boundaries**
   - Domain logic depends on protocols or small service contracts.
   - Infrastructure adapters depend on domain contracts, not the reverse.
2. **Composition-root ownership**
   - Wiring belongs in composition roots, not in domain services.
   - For Victor, that means API bootstrap, orchestrator setup, and vertical activation should compose services rather than own business logic.
3. **Async-first internals**
   - Runtime and service layers should be natively async.
   - Sync wrappers should be limited to CLI/TUI shell edges.
4. **Strangler-fig refactoring**
   - Keep legacy paths working while new seams land.
   - Add shadow execution or delegation shims only temporarily, with explicit removal criteria.
5. **Observability by default**
   - Every new boundary should emit structured metrics, counters, and failure reasons.
6. **Compatibility with intent**
   - Deprecation shims must have owners, removal dates, and tests that prove forward compatibility.

### Delivery Principles

1. Ship vertical slices, not giant rewrites.
2. Add characterization tests before major refactors.
3. Define performance budgets for hot paths.
4. Use ADRs for boundary changes that alter contracts or layering.
5. Keep public API churn lower than internal churn.

## Primary Workstreams

## W0: Planning, Governance, and Delivery Safety Rails

**Goal**: Make the execution model durable before deep refactors begin.

**Patterns**
- single source of truth
- lightweight governance
- issue-driven execution

**Touchpoints**
- `roadmap.md`
- `VISION.md`
- `docs/tech-debt/codebase-assessment-2026-03-26.md`
- `docs/planning/tiered-execution-backlog-2026-03-26.md`
- `docs/planning/RANKED_90_DAY_EXECUTION_PLAN.md`

**Implementation Steps**
1. Keep the roadmap current on a weekly cadence with exact dates.
2. Treat the tiered backlog as the GitHub issue source until remote issue updates are completed.
3. Add a release-review checkpoint that reconciles roadmap, backlog, and benchmark status.
4. Require every large refactor work item to include:
   - migration plan
   - test plan
   - rollback path
   - acceptance metrics

**Acceptance Criteria**
- No stale canonical roadmap dates.
- One current assessment doc and one active backlog doc.
- Every P0/P1 item has owner, milestone, and acceptance criteria.

## W1: Security and Secret Handling Hardening

**Goal**: Normalize secrets handling and codify operator-safe release practices.

**Patterns**
- secure-by-default configuration
- typed secret wrappers
- explicit trust boundaries

**Touchpoints**
- `victor/config/settings.py`
- `victor/config/unified_settings.py`
- `victor/providers/base.py`
- `victor/config/provider_config_registry.py`
- `SECURITY.md`

**Implementation Steps**
1. Introduce a clear split between:
   - user-facing plain config inputs
   - internal secret-bearing settings objects
2. Migrate generic/server-side secrets to `SecretStr` or a dedicated `RedactedSecret` wrapper.
3. Ensure logging and repr paths never expose secret values.
4. Add a central helper for extracting secret values at adapter edges only.
5. Document SBOM consumption, scanner thresholds, and exception handling in `SECURITY.md`.
6. Add code-owner review rules for any new `eval()` or `exec()` call site.

**Tests**
- unit tests for serialization/redaction
- config resolution tests for env, keyring, and file precedence
- regression tests proving secrets do not leak into logs or exception messages

**Acceptance Criteria**
- No plain `str` fields for long-lived API keys or server/session secrets in primary settings models.
- Secret resolution happens in one audited layer.
- Operator documentation matches the enforced CI baseline.

## W2: Async Boundary Cleanup and Orchestrator Hardening

**Goal**: Remove the highest-risk event-loop hazards and simplify runtime behavior.

**Patterns**
- async core / sync shell
- anti-corruption layer for legacy sync APIs
- protocol-driven orchestration

**Touchpoints**
- `victor/agent/orchestrator.py`
- `victor/agent/provider_coordinator.py`
- `victor/agent/conversation_memory.py`
- `victor/core/async_utils.py`
- `victor/framework/workflows/nodes.py`

**Implementation Steps**
1. Inventory every runtime `asyncio.run()` usage and classify it as:
   - acceptable shell edge
   - transitional shim
   - unsafe internal boundary
2. Remove deprecated sync provider switching from the orchestrator.
3. Replace sync memory search bridges with explicit async APIs plus shell-only adapters.
4. Introduce small protocol interfaces for provider switching, memory search, and workflow node execution.
5. Add feature flags only where compatibility needs a staged rollout.
6. Remove temporary shims after the async path becomes the only supported runtime path.

**Tests**
- characterization tests around provider switching, chat flow, and memory search
- async integration tests for nested event loop scenarios
- performance tests for switching/search latency

**Acceptance Criteria**
- No internal runtime path calls `asyncio.run()` from orchestration or memory services.
- Sync support remains only at CLI/TUI/API shell boundaries where explicitly intended.
- Public async APIs are documented and covered by contract tests.

## W3: Workflow Compiler Completion

**Goal**: Replace the compiler migration stub with a real compilation boundary.

**Patterns**
- parser / validator / compiler / executor separation
- immutable intermediate representation
- contract tests around compiled graphs

**Touchpoints**
- `victor/workflows/compiler/unified_compiler.py`
- `victor/workflows/unified_compiler.py`
- `victor/workflows/yaml_loader.py`
- `victor/workflows/validation/`
- `victor/framework/workflows/`

**Target Architecture**
- `WorkflowParser`: YAML to normalized IR
- `WorkflowValidator`: structural and semantic validation
- `WorkflowCompiler`: IR to executable graph
- `WorkflowExecutor`: runtime invocation and orchestration

**Implementation Steps**
1. Define a stable IR for workflow definitions.
2. Move loading and normalization out of the compiler itself.
3. Preserve existing behavior through characterization tests against the legacy compiler.
4. Swap the stubbed compiler to produce graphs from the new IR.
5. Keep the legacy compiler only as a fallback during the transition window.
6. Remove fallback once parity and performance are confirmed.

**Tests**
- golden tests for representative workflow YAML inputs
- graph-shape assertions
- parity tests between legacy and new compiler on supported workflows
- error-message tests for invalid DSL inputs

**Acceptance Criteria**
- The compiler no longer contains TODO-based legacy delegation.
- Parser, validator, compiler, and executor are separable and testable in isolation.
- Workflow compilation latency and failure diagnostics are measured.

## W4: API Server Decomposition

**Goal**: Turn the HTTP API into a maintainable composition root plus focused services.

**Patterns**
- composition root
- service layer
- route thinness
- request-scoped dependency resolution

**Touchpoints**
- `victor/integrations/api/server.py`
- `victor/integrations/api/routes/`
- `victor/integrations/api/event_bridge.py`

**Target Architecture**
- `app_factory.py`: create and wire the application
- `dependencies.py`: request-scoped dependencies
- `services/`: workspace, metrics, orchestration, session, and execution services
- `routes/`: thin transport-only handlers

**Implementation Steps**
1. Move non-routing logic out of `VictorAPIServer`.
2. Extract workspace metrics, session health, and orchestration operations into service objects.
3. Standardize request validation, error mapping, and response shaping.
4. Centralize lifecycle hooks and startup/shutdown behavior.
5. Add API-level metrics for route latency, failure class, and saturation.

**Tests**
- route tests with service-layer fakes
- app bootstrap tests
- lifecycle tests for startup/shutdown hooks
- API contract tests for critical routes

**Acceptance Criteria**
- `server.py` becomes a thin composition module.
- Business logic lives in isolated service modules with unit tests.
- API metrics cover latency, errors, and saturation for top routes.

## W5: Vertical Integration Decomposition

**Goal**: Preserve Victor’s extension advantage while reducing framework/vertical coupling.

**Patterns**
- protocol-based extension contracts
- manifest negotiation
- runtime adapter layer
- anti-corruption layer between core and external verticals

**Touchpoints**
- `victor/framework/vertical_integration.py`
- `victor/core/verticals/base.py`
- `victor/core/verticals/vertical_loader.py`
- `victor/core/verticals/capability_negotiator.py`
- `victor/core/capability_registry.py`

**Target Architecture**
- `manifest_service`
- `capability_resolution_service`
- `vertical_runtime_adapter`
- `vertical_configuration_applier`
- `extension_lifecycle_service`

**Implementation Steps**
1. Split discovery/manifest negotiation from runtime activation.
2. Move configuration application and capability resolution behind smaller service contracts.
3. Preserve entry-point discovery and SDK boundary compatibility throughout.
4. Add consumer-style contract tests for built-in and external verticals.
5. Document the allowed dependency direction explicitly in an ADR update if the boundary changes.

**Tests**
- contract tests for external vertical discovery
- compatibility tests for manifest negotiation
- regression tests for built-in vertical activation

**Acceptance Criteria**
- `vertical_integration.py` no longer centralizes unrelated concerns.
- External vertical compatibility is tested independently from framework internals.
- Manifest negotiation and runtime wiring are traceable and observable.

## W6: ProximaDB Multi-Model Provider Decomposition

**Goal**: Keep the differentiated semantic/graph/metrics capability while making it maintainable and scalable.

**Patterns**
- pipeline decomposition
- CQRS-style query/write split
- extract-transform-load stages
- strategy pattern for index/query policies

**Touchpoints**
- `victor/storage/vector_stores/proximadb_multi.py`
- `victor/verticals/contrib/coding/tools/graph_tool.py`
- `victor/core/search/`

**Target Architecture**
- `symbol_extractor`
- `chunk_builder`
- `record_builder`
- `graph_writer`
- `metrics_writer`
- `hybrid_query_service`
- `bug_context_service`

**Implementation Steps**
1. Separate file analysis from storage writes.
2. Separate write pipelines from query pipelines.
3. Introduce small result models for hybrid search and graph traversal.
4. Add batching and backpressure controls around vector/document/graph writes.
5. Benchmark indexing throughput, query latency, and memory usage before and after decomposition.
6. Keep the current provider class as a facade until the extracted services stabilize.

**Tests**
- deterministic fixture-based indexing tests
- hybrid query ranking tests
- graph traversal tests
- throughput and memory regression tests

**Acceptance Criteria**
- The provider facade delegates to focused services rather than owning every concern.
- Indexing/query performance is at least unchanged, with clear measurements.
- Graph/vector/document/metrics behavior can be tested independently.

## W7: Observability Productization

**Goal**: Decide which observability surfaces are production-grade and make that decision explicit in code and docs.

**Patterns**
- telemetry pipeline separation
- thin UI over stable telemetry contracts
- SLO-driven instrumentation

**Touchpoints**
- `victor/integrations/api/event_bridge.py`
- `victor/integrations/api/metrics_exporter.py`
- `victor/observability/dashboard/app.py`
- `victor/observability/metrics.py`

**Implementation Steps**
1. Decide whether the dashboard is:
   - a supported product surface, or
   - a prototype to archive or rewrite
2. Define a stable telemetry contract shared by exporter, dashboard, and future APIs.
3. Expand metrics beyond EventBridge reliability into:
   - request latency
   - queue depth / saturation
   - tool execution latency
   - provider switch outcomes
4. Keep UI layers dependent on telemetry contracts, not raw runtime objects.

**Tests**
- telemetry schema tests
- exporter correctness tests
- dashboard smoke tests if retained

**Acceptance Criteria**
- Dashboard status is explicit: supported or archived.
- Metrics cover core runtime health, not only the event bridge.
- Observability docs match the implemented surfaces.

## W8: Benchmark Proof and DX

**Goal**: Turn infrastructure into externally credible product proof and simpler onboarding.

**Patterns**
- evidence-based product claims
- golden path onboarding
- release-readiness reviews

**Touchpoints**
- `docs/benchmarking/`
- benchmark scripts
- `README.md`
- `docs/getting-started/`

**Implementation Steps**
1. Execute and publish benchmark runs for Victor plus comparison frameworks.
2. Tie benchmark outcomes to roadmap priorities.
3. Create one “happy path” install and first-run flow.
4. Keep secondary surfaces documented, but reduce the top-level narrative to the most important path.

**Acceptance Criteria**
- Benchmark results are published and reproducible.
- README and quickstart prioritize one path instead of many equal-weight paths.
- Release notes and roadmap use measured outcomes, not only capability lists.

## Phase Plan

## Phase 0: Alignment and Safety Rails

**Focus**
- W0 and W1

**Exit Criteria**
- Canonical docs current
- secret-hardening design approved
- backlog mapped to milestones

## Phase 1: Runtime Boundary Hardening

**Focus**
- W2 and W3

**Exit Criteria**
- deprecated sync runtime seams removed or isolated
- compiler IR and parity tests in place

## Phase 2: Service Decomposition

**Focus**
- W4, W5, and W6

**Exit Criteria**
- API server, vertical integration, and ProximaDB provider all running behind extracted services
- behavior parity proven by contract/integration tests

## Phase 3: Product Proof and Scale Readiness

**Focus**
- W7 and W8

**Exit Criteria**
- benchmark results published
- observability status clarified
- onboarding path simplified

## Cross-Cutting Test Strategy

1. **Characterization tests first**
   - Freeze existing behavior before moving logic.
2. **Contract tests on each new seam**
   - Provider switching, workflow compilation, vertical activation, hybrid query, API routes.
3. **Integration tests at runtime boundaries**
   - Async execution, startup/shutdown, entry-point discovery, benchmark harness execution.
4. **Performance regression tests**
   - workflow compile latency
   - provider switch latency
   - hybrid search latency
   - indexing throughput
5. **Security regression tests**
   - secret redaction
   - config serialization
   - sandboxed eval guardrails

## Rollout and Risk Management

### Release Strategy

- Land refactors behind boundary-preserving facades.
- Use feature flags only when runtime behavior changes materially.
- Remove flags once the new path is stable; do not allow permanent flag debt.

### Risk Controls

- No large module rewrite without characterization tests.
- No new public API without docs and type coverage.
- No migration shim without a removal milestone.
- No performance-sensitive refactor without a before/after benchmark.

## Success Metrics

| Area | Metric |
|---|---|
| Security | no long-lived server/provider secrets stored as plain strings in primary settings models |
| Async boundaries | zero internal runtime `asyncio.run()` shims in orchestrator/memory/compiler paths |
| Workflow compiler | no legacy compiler stub delegation for supported workflows |
| API surface | `server.py` reduced to composition logic only |
| Vertical integration | runtime activation and capability negotiation separated into focused services |
| ProximaDB provider | indexing/query behavior delegated to extracted services with measured parity |
| Product proof | benchmark results published and linked from roadmap/docs |
| DX | one primary quickstart path documented and tested |

## Immediate Next Steps

1. Use this plan as the execution reference for P0 and P1 backlog items.
2. Start with W1 and W2 in parallel only where ownership differs; otherwise finish foundation before deep refactors.
3. For each workstream, open an implementation PR series instead of one large branch:
   - PR1: characterization tests + boundary interfaces
   - PR2: extracted service + delegated facade
   - PR3: flag removal / dead-code cleanup / docs
