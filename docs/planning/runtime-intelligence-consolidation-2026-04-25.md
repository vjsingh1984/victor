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

Next recommended slice:
- Phase 5.5: persist benchmark-truth feedback back into the runtime calibration
  path so offline evaluation outcomes can update future live policies without
  requiring ad hoc threshold wiring
