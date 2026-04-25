# Runtime Intelligence Consolidation Plan

Date: 2026-04-25

## Problem

The active runtime intelligence path is scattered across multiple modules with overlapping responsibilities:

| Concern | Current modules | Problem |
|---|---|---|
| Turn perception and task intent | `victor/framework/perception_integration.py`, `victor/agent/task_analyzer.py`, `victor/agent/action_authorizer.py` | Perception and task analysis are conceptually one runtime concern, but callers stitch them together ad hoc. |
| Live evaluation and retry policy | `victor/framework/agentic_loop.py`, `victor/framework/enhanced_completion_evaluation.py` | Evaluation policy is split between the loop and the evaluator, which makes retry/clarification logic easy to duplicate or bypass. |
| Decision-service access | `streaming/pipeline.py`, `prompt_builder.py`, `task_completion.py`, `tool_selection.py`, `state_machine.py`, others | Container lookup and budget management are repeated in multiple call sites. |
| Prompt optimization retrieval | `victor/agent/optimization_injector.py`, `victor/agent/prompt_pipeline.py`, orchestrator init | GEPA, MiPRO, CoT distillation, failure hints, and related prompt optimizations are a separate injection path rather than part of a canonical runtime intelligence service. |

This creates SRP and DIP violations:
- Consumers know too much about low-level collaborators.
- Runtime policy changes must be repeated in multiple modules.
- New agentic optimizations get added as sidecars instead of joining a typed subsystem.

## Target Design

Introduce a canonical service-first subsystem:

`victor.agent.services.runtime_intelligence`

Primary responsibilities:
- Provide a typed turn-analysis snapshot for runtime consumers.
- Centralize prompt-optimization bundle retrieval.
- Centralize decision-service budget reset/access.
- Keep heuristics, perception, and prompt-optimization policies behind one service boundary.

Primary types:
- `PromptOptimizationBundle`
- `RuntimeIntelligenceSnapshot`
- `RuntimeIntelligenceService`

Design rules:
- `RuntimeIntelligenceService` composes existing strong components instead of replacing them.
- `PerceptionIntegration`, `TaskAnalyzer`, `OptimizationInjector`, and the decision service remain reusable collaborators behind the service.
- Consumers depend on the service boundary, not the implementation details of each collaborator.
- Legacy constructor arguments remain temporarily supported for compatibility, but canonical paths should prefer the new service.

## Canonical Consumer Migrations

Phase 1:
- `UnifiedPromptPipeline`
  - Consume prompt-optimization bundles from `RuntimeIntelligenceService`
  - Use service-provided task classification helpers
- `StreamingChatPipeline`
  - Use service for decision-budget reset
  - Use service for pre-loop turn analysis / perception snapshot
- `AgenticLoop`
  - Use service for turn analysis instead of directly orchestrating perception

Phase 2:
- `ServiceStreamingRuntime`
  - Pass or create a shared runtime-intelligence service for the live service-owned path
- `AgentOrchestrator`
  - Construct and reuse a shared `_runtime_intelligence` service

Phase 3:
- Decision-service lookup consolidation in secondary consumers:
  - `task_completion.py`
  - `thinking_detector.py`
  - `continuation_strategy.py`
  - `streaming/intent_classification.py`
  - `agent/factory/coordination_builders.py`
- Remaining direct decision consumers after this slice:
  - `error_classifier.py`
  - `unified_classifier.py`
  - `context_compactor.py`
  - `tool_selection.py`
  - `conversation/state_machine.py`

Phase 4:
- Collapse remaining duplicated low-confidence / retry / clarification policies into typed runtime policies behind the same subsystem.
  - 4.1 Clarification policy normalization
  - 4.2 Low-confidence retry-budget policy normalization
  - 4.3 Enhanced-evaluator policy emission normalization

## First Implementation Slice

Scope:
- Add `RuntimeIntelligenceService` with typed snapshot and prompt-optimization bundle.
- Migrate `UnifiedPromptPipeline`, `StreamingChatPipeline`, and `AgenticLoop`.
- Keep old constructor arguments working as compatibility inputs.

TDD checkpoints:
1. Add unit tests for the new service.
2. Add migration tests proving prompt pipeline, streaming pipeline, and agentic loop can consume the new service.
3. Implement the service.
4. Migrate the active calling sites.
5. Run focused and broader regression suites.

## Expected Outcome

After Phase 1:
- Prompt optimization, perception, and decision-budget handling will have one canonical service seam.
- The three highest-value runtime consumers will depend on a shared abstraction instead of stitching low-level collaborators together independently.
- Future work on GEPA/MiPRO/CoT/credit-driven policies can attach to the runtime-intelligence service without growing more cross-cutting constructor clutter.

## Progress Checkpoint

Completed on 2026-04-25:
- Phase 1 and 2 migrations:
  - `UnifiedPromptPipeline`
  - `StreamingChatPipeline`
  - `ServiceStreamingRuntime`
  - `AgenticLoop`
  - orchestrator/factory runtime-intelligence wiring
- Phase 3.1 secondary decision-consumer migration:
  - `task_completion.py`
  - `thinking_detector.py`
  - `continuation_strategy.py`
  - `streaming/intent_classification.py`
  - `agent/factory/coordination_builders.py`
  - direct orchestrator task-completion initialization
- Phase 3.2 classifier/runtime consolidation:
  - `error_classifier.py`
  - `unified_classifier.py`
  - `task_analyzer.py`
  - runtime-service binding of `TaskAnalyzer` to the canonical runtime boundary
- Phase 3.3 compaction/runtime consolidation:
  - `context_compactor.py`
  - `agent/factory/coordination_builders.py` compactor construction path
- Phase 3.4 state-machine/runtime consolidation:
  - `conversation/state_machine.py`
  - `service_provider.py` scoped state-machine construction path
- Phase 3.5 tool-selection/runtime consolidation:
  - `tool_selection.py`
  - `agent/factory/tool_builders.py`
  - `service_provider.py` tool-selector construction path
- Phase 4.1 clarification-policy normalization:
  - `RuntimeIntelligenceService` now exposes a typed clarification decision
  - `StreamingChatPipeline` and `AgenticLoop` now consume the same canonical clarification prompt policy
- Phase 4.2 low-confidence retry-budget normalization:
  - `RuntimeIntelligenceService` now owns the canonical confidence-band and retry-budget policy
  - `AgenticLoop` now delegates both raw confidence fallback and enhanced low-confidence retry gating through the same runtime policy
- Phase 4.3 enhanced-evaluator policy emission normalization:
  - `RuntimeIntelligenceService` now exposes a budget-free confidence evaluation emitter
  - `EnhancedCompletionEvaluator` now emits the same canonical confidence vocabulary as the live loop without mutating retry state
- Phase 5.1 runtime evaluation-policy extraction:
  - Introduced `RuntimeEvaluationPolicy` as the shared threshold/wording object
  - `PerceptionIntegration`, `RuntimeIntelligenceService`, `StreamingChatPipeline`,
    `AgenticLoop`, and `EnhancedCompletionEvaluator` now share one policy model for
    clarification defaults and confidence-band decisions
- Phase 5.2 calibrated-completion policy consolidation:
  - `RuntimeEvaluationPolicy` now owns calibrated completion thresholds, penalties,
    and enhanced evaluation wording
  - `EnhancedCompletionEvaluator` now delegates calibrated completion math and
    enhanced COMPLETE/CONTINUE/RETRY result construction through that shared policy
  - `AgenticLoop` now relies on the shared policy for completion-threshold
    configuration instead of threading a separate evaluator-only threshold knob
- Phase 5.3 immutable policy overlays for compatibility helpers:
  - `RuntimeEvaluationPolicy` now supports immutable override overlays
  - `RuntimeIntelligenceService` static compatibility helpers now merge explicit
    threshold and prompt overrides into the shared policy instead of ignoring
    overrides whenever a policy instance is already present
- Phase 5.4 runtime calibration feedback integration:
  - `RuntimeEvaluationPolicy` now accepts calibrated runtime feedback overlays
  - `ConfidenceCalibrator`, `LLMDecisionService`, and `TieredDecisionService`
    can export runtime-evaluation feedback for task-completion thresholds
  - `RuntimeIntelligenceService.from_container(...)` now applies decision-service
    runtime feedback to the shared policy and synchronizes the perception path to
    that calibrated policy
- Phase 5.5 benchmark-truth feedback persistence:
  - `EvaluationHarness` now derives runtime-calibration feedback from persisted
    benchmark truth metrics and saves both per-run snapshots plus the canonical
    `runtime_evaluation_feedback.json` artifact under the evaluation results path
  - `RuntimeIntelligenceService` now loads persisted benchmark-truth feedback as
    a first-class calibration input before live decision-service feedback
  - Explicit live runtime config remains authoritative, so persisted or live
    calibration can tune future runs without silently overriding intentionally set
    thresholds on the active execution path
- Phase 5.6 validated evaluation-truth freshness and aggregation:
  - `EvaluationHarness` now persists normalized per-run validated-evaluation-truth
    payloads inside each evaluation result artifact
  - `runtime_feedback.py` now aggregates eligible evaluation-truth artifacts from
    the results directory using recency- and reliability-weighted scoring instead
    of trusting a single stale canonical file
  - Canonical `runtime_evaluation_feedback.json` is now refreshed from that
    aggregate view, and raw heuristic runtime sources remain excluded unless they
    are upgraded into explicit validated evaluation truth
- Phase 5.7 scoped relevance selection and explicit session-truth schema:
  - `runtime_feedback.py` now defines a canonical `RuntimeEvaluationFeedbackScope`
    schema and a typed builder for future validated session-truth emitters instead
    of relying on loose metadata conventions
  - Validated evaluation-truth aggregation now applies centralized source trust,
    freshness, reliability, and scope-adjacency weighting so project/model/task-
    adjacent evidence is preferred over unrelated artifacts
  - Active prompt/runtime call sites now pass available provider/model scope into
    `RuntimeIntelligenceService` so calibration can use scoped evidence whenever
    the caller already knows that context
- Phase 5.8 real post-hoc validator emission for coding workflows:
  - `runtime_feedback.py` now derives validated session-truth payloads directly
    from objective SWE-bench baseline validation plus correlated scoring, rather
    than requiring synthetic/manual artifact creation
  - `EvaluationOrchestrator` now persists those validated session-truth artifacts
    under its output `evaluations/` directory and refreshes the canonical runtime
    feedback aggregate from that real validator output
  - The strong-evidence gate for this slice is explicit: only valid baselines with
    actual post-change test evidence emit session truth; weak or unvalidated runs
    remain ineligible

Next recommended slice:
- Phase 5.9 completed: connect browser and research post-hoc validators to the
  same validated session-truth emitter path so non-coding workflows can
  contribute scoped calibration evidence under the same trust and gating rules
  - `runtime_feedback.py` now exposes canonical browser and deep-research
    validated session-truth builders on top of the shared scope schema and
    aggregation path
  - `EvaluationHarness` now persists per-task `eval_session_*` artifacts for
    browser-task and DR3-style runs, then refreshes the canonical aggregate from
    the same results directory without introducing a second persistence path
  - The non-coding admission contract remains explicit: only tasks with concrete
    post-hoc coverage evidence emit session truth; empty or heuristic-only runs
    still remain ineligible

Next recommended slice:
- Phase 5.10 completed: extract a registry-backed validated session-truth
  emitter strategy so future benchmark families can register their own evidence
  gates without growing benchmark-specific conditionals inside
  `EvaluationHarness`
  - `validated_session_truth_emitters.py` now defines the canonical emitter
    interface, artifact contract, default registry, and browser/research
    emitters while keeping trust, freshness, and scope weighting centralized in
    `runtime_feedback.py`
  - `EvaluationHarness` now resolves emitters through that registry instead of
    branching directly on benchmark families, so new evaluators can plug into
    the same persistence path without reopening harness orchestration

Next recommended slice:
- Phase 5.11 completed: migrate coding/SWE-bench validated session-truth
  emission onto the same emitter contract so `EvaluationOrchestrator` and
  `EvaluationHarness` share one extensibility model for all benchmark families
  - `validated_session_truth_emitters.py` now includes the canonical SWE-bench
    emitter and a shared emission-context contract used by browser, research,
    and coding workflows
  - `EvaluationOrchestrator` now resolves the SWE-bench emitter through the
    same registry contract instead of building coding session-truth artifacts
    through a direct ad hoc code path
  - The stronger coding evidence gate remains intact: valid baseline plus actual
    post-change test evidence is still the admission rule for coding feedback

Next recommended slice:
- Phase 5.12 completed: extract shared validated session-truth
  persistence/refresh orchestration so harnesses and orchestrators stop
  duplicating artifact write and aggregate refresh mechanics after emitter
  resolution
  - `validated_session_truth_persistence.py` now owns artifact write plus
    aggregate refresh, while emitters remain responsible only for evidence
    gating and artifact construction
  - `EvaluationHarness` and `EvaluationOrchestrator` now share that canonical
    persistence helper instead of hand-writing JSON and refresh logic in each
    runtime

Next recommended slice:
- Phase 5.13 completed: extract a small validated session-truth service that
  owns emitter resolution, context assembly, and persistence so runtimes only
  provide their native evidence objects and output directories
  - `validated_session_truth_service.py` now owns emitter resolution, context
    assembly, and persistence orchestration for evaluation results and
    validation outputs
  - `EvaluationHarness` and `EvaluationOrchestrator` now delegate to that
    service instead of building emission contexts themselves, while registry
    injection remains available only as a compatibility fallback

Next recommended slice:
- Phase 5.14 completed: centralize runtime error handling and directory
  preparation in the validated session-truth service so callers stop managing
  those edge cases directly
  - `ValidatedSessionTruthService` now owns directory preparation and safe
    degradation behavior for emitter and persistence failures
  - Parent evaluation runs now degrade by skipping session-truth capture rather
    than failing outright when emitter-specific capture goes wrong

Next recommended slice:
- Phase 5.15 completed: make artifact naming policy explicit inside the
  validated session-truth subsystem so future emitters cannot silently diverge
  in file layout conventions
  - `validated_session_truth_naming.py` now owns the canonical, backward-
    compatible artifact path policy
  - Browser, research, and coding emitters now route artifact path generation
    through that policy instead of open-coding filenames inside emitter logic

Next recommended slice:
- Phase 5.16 completed: make validated session-truth service construction
  canonical so runtime constructors no longer need to expose both service and
  registry wiring paths as permanent peers
  - `create_default_validated_session_truth_service(...)` now owns default
    service construction from an optional registry override
  - `EvaluationHarness` and `EvaluationOrchestrator` now center their public
    constructor wiring on the service while still accepting the legacy registry
    keyword as a compatibility shim

Next recommended slice:
- Phase 5.17 completed: promote the validated session-truth service through
  higher-level factories/exports so the broader evaluation stack depends on
  one canonical DI entrypoint instead of importing the concrete service
  opportunistically
  - `services.py` now exposes the canonical evaluation-level validated
    session-truth factory/export path
  - `EvaluationHarness` and `EvaluationOrchestrator` now resolve their default
    service through that higher-level entrypoint instead of importing the
    concrete service factory directly

Next recommended slice:
- Phase 5.18 completed: narrow validated session-truth constructor and factory
  typing to a protocol so evaluation runtimes depend on the service contract
  rather than the concrete implementation class
  - `services.py` now exposes `ValidatedSessionTruthServiceProtocol` alongside
    the canonical evaluation-level factory
  - `EvaluationHarness` and `EvaluationOrchestrator` now type their injected
    service dependency against the protocol while preserving existing runtime
    behavior

Next recommended slice:
- Phase 5.19: centralize validated session-truth service resolution so
  constructor compatibility shims and default factory fallback are owned by one
  evaluation-level helper instead of being open-coded in each runtime
  - Keep explicit service injection authoritative
  - Preserve the legacy registry keyword only through the shared resolution path
