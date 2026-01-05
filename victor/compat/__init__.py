# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Backward compatibility aliases and shims.

This module centralizes all backward compatibility aliases to:
1. Make it clear what aliases exist
2. Avoid confusion about canonical sources
3. Enable future deprecation warnings
4. Document migration paths

CANONICAL SOURCES (import from here):
=====================================

CircuitBreaker Types:
    from victor.providers.circuit_breaker import (
        CircuitState,
        CircuitBreakerConfig,
        CircuitBreakerError,
        CircuitBreaker,
        CircuitBreakerRegistry,
    )

ToolCall Types:
    from victor.agent.tool_calling.base import ToolCall

LSP Types:
    from victor.protocols.lsp_types import (
        Position, Range, TextEdit, CompletionItemKind,
        DiagnosticSeverity, SymbolKind, ...
    )

Completion Protocol:
    from victor.completion.protocol import (
        CompletionItem, CompletionList, ...
    )


DEPRECATED ALIASES (will be removed in v1.0):
=============================================

These still work but should be migrated:

- victor.protocols.provider_adapter.ToolCall
  → victor.agent.tool_calling.base.ToolCall

- victor.integrations.protocols.provider_adapter.ToolCall
  → victor.agent.tool_calling.base.ToolCall

- victor.agent.resilience.CircuitState
  → victor.providers.circuit_breaker.CircuitState

- victor.observability.resilience.CircuitState
  → victor.providers.circuit_breaker.CircuitState


SEMANTIC RENAMES (different from aliases):
==========================================

These classes were renamed because they have distinct semantics:

ToolCall variants (all different semantics):
    - ToolCall: Basic request representation (canonical)
    - ToolInvocation: Request + response pair
    - ToolCallExecution: Full lifecycle with status/timing
    - EvalToolCall: Evaluation tracking
    - TrackedToolCall: Deduplication tracking

SearchResult variants (all different semantics):
    - DocumentSearchResult: RAG document chunks
    - EmbeddingSearchResult: Vector store results
    - UnifiedSearchResult: Multi-signal search (semantic+keyword+graph)

StreamChunk variants (all different semantics):
    - StreamChunk (victor.providers.base): Provider-level raw streaming (canonical)
    - OrchestratorStreamChunk: Orchestrator protocol with typed ChunkType enum
    - TypedStreamChunk: Safe typed accessor with nested StreamDelta
    - ClientStreamChunk: Protocol interface for clients (CLI/VS Code)

AgentMode variants (all different semantics):
    - AgentMode (victor.agent.mode_controller): Base 3 modes (BUILD, PLAN, EXPLORE) (canonical)
    - RLAgentMode: Extended 5 modes for RL state machine
    - AdaptiveAgentMode: Extended 5 modes for adaptive control

ModeConfig variants (all different semantics):
    - ModeConfig (victor.core.mode_config): Simple tool budget/iteration config (canonical)
    - OperationalModeConfig: Rich operational mode configuration with prompts and tool control

TaskType variants (all different semantics):
    - TaskType (victor.classification.pattern_registry): Canonical prompt classification (canonical)
    - TrackerTaskType: Progress tracking with milestones
    - LoopDetectorTaskType: Loop detection thresholds
    - ClassifierTaskType: Unified classification output
    - FrameworkTaskType: Framework-level task abstraction

Severity variants (all different semantics):
    - CVESeverity (victor.security.protocol): CVE/CVSS-based severity (NONE, LOW, MEDIUM, HIGH, CRITICAL)
    - AuditSeverity (victor.security.audit.protocol): Audit event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - IaCSeverity (victor.iac.protocol): IaC issue severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
    - ReviewSeverity (victor.coding.review.protocol): Code review severity (ERROR, WARNING, INFO, HINT)

RiskLevel variants (all different semantics):
    - OperationalRiskLevel (victor.agent.safety): Tool/command operational risk (5 values including SAFE)
    - PatternRiskLevel (victor.safety.code_patterns): Code pattern detection risk (4 uppercase values)

QualityDimension variants (all different semantics):
    - ResponseQualityDimension (victor.agent.response_quality): LLM response quality (7 dimensions incl. CODE_QUALITY)
    - ProtocolQualityDimension (victor.protocols.quality): Protocol-level quality (7 dimensions incl. GROUNDING, SAFETY)

Checkpoint variants (all different semantics):
    - GitCheckpoint (victor.agent.checkpoints): Git stash-based working tree snapshots
    - ExecutionCheckpoint (victor.agent.time_aware_executor): Progress tracking with TimePhase
    - WorkflowCheckpoint (victor.framework.graph): StateGraph DSL workflow state persistence
    - HITLCheckpoint (victor.framework.hitl): Human-in-the-loop pause/resume context

CircuitBreaker variants (all different semantics):
    - CircuitBreaker (victor.providers.circuit_breaker): Standalone with decorator/context manager (canonical)
    - MultiCircuitBreaker (victor.agent.resilience): Manages multiple named circuits (Dict[str, CircuitStats])
    - ObservableCircuitBreaker (victor.observability.resilience): Metrics/callback focused with on_state_change
    - ProviderCircuitBreaker (victor.providers.resilience): ResilientProvider workflow with execute(), is_available

ValidationResult variants (all different semantics):
    - ToolValidationResult (victor.tools.base): Tool parameter validation with invalid_params
    - ConfigValidationResult (victor.core.validation): Configuration validation with ValidationIssue
    - ContentValidationResult (victor.framework.middleware): Content validation with auto-fix support
    - ParameterValidationResult (victor.agent.parameter_enforcer): Parameter enforcement with missing_required
    - CodeValidationResult (victor.evaluation.correction.types): Code validation with syntax/imports/language

Dependency variants (all different semantics):
    - SecurityDependency (victor.security.protocol): Security scanning with ecosystem, license, package_url
    - PackageDependency (victor.deps.protocol): Package management with version tracking (installed/latest)

BaseDependencyParser variants (all different semantics):
    - BaseSecurityDependencyParser (victor.security.scanner): Security scanning with ecosystem property
    - BasePackageDependencyParser (victor.deps.parsers): Package management with PackageManager enum

RetryConfig variants (all different semantics):
    - ProviderRetryConfig (victor.providers.resilience): Provider-specific with retryable_patterns
    - AgentRetryConfig (victor.agent.resilience): Agent-specific with jitter flag
    - ObservabilityRetryConfig (victor.observability.resilience): With BackoffStrategy and on_retry callback

RetryStrategy variants (all different semantics):
    - BaseRetryStrategy (victor.core.retry): Abstract base with should_retry(), get_delay() methods
    - ProviderRetryStrategy (victor.providers.resilience): Concrete provider retry with execute()
    - BatchRetryStrategy (victor.workflows.batch_executor): Enum for batch retry modes (NONE, IMMEDIATE, etc.)

NodeType variants (all different semantics):
    - WorkflowNodeType (victor.workflows.definition): Workflow definition nodes (COMPUTE, HITL, START, END)
    - GraphNodeType (victor.workflows.graph_dsl): Graph DSL nodes (FUNCTION, CONDITIONAL, SUBGRAPH)
    - YAMLNodeType (victor.workflows.yaml_loader): YAML loader validation nodes

NodeStatus variants (all different semantics):
    - ProtocolNodeStatus (victor.workflows.protocols): Workflow protocol node status
    - ExecutorNodeStatus (victor.workflows.executor): Executor node status
    - FrameworkNodeStatus (victor.framework.graph): Framework graph node status

StopReason variants (all different semantics):
    - TrackerStopReason (victor.agent.unified_task_tracker): Task tracker stop reasons (budget, loop, iterations)
    - LoopStopRecommendation (victor.agent.loop_detector): Loop detection recommendation dataclass
    - DebugStopReason (victor.observability.debug.protocol): Debugger stop reasons

VerificationResult variants (all different semantics):
    - GroundingVerificationResult (victor.agent.grounding_verifier): Agent grounding with issues list
    - ClaimVerificationResult (victor.protocols.grounding): Protocol claim verification

ToolSelectionContext variants (all different semantics):
    - AgentToolSelectionContext (victor.agent.protocols): Basic agent-level context
    - VerticalToolSelectionContext (victor.core.verticals.protocols.tool_provider): Vertical-specific
    - CrossVerticalToolSelectionContext (victor.tools.selection.protocol): Extended cross-vertical

Symbol variants (all different semantics):
    - NativeSymbol (victor.native.protocols): Rust-extracted symbols (frozen, hashable)
    - IndexedSymbol (victor.coding.codebase.indexer): Pydantic model for index storage
    - RefactorSymbol (victor.coding.refactor.protocol): Refactoring symbol with SourceLocation

ScanResult variants (all different semantics):
    - SafetyScanResult (victor.safety.code_patterns): Safety pattern matching results
    - IaCScanResult (victor.iac.protocol): Infrastructure-as-Code scan results

RecoveryAction variants (all different semantics):
    - ErrorRecoveryAction (victor.agent.error_recovery): Tool error recovery actions (string values)
    - StrategyRecoveryAction (victor.agent.recovery.protocols): Recovery strategy actions (auto enum)
    - OrchestratorRecoveryAction (victor.agent.orchestrator_recovery): Orchestrator recovery dataclass
"""

from typing import Any, Dict

# Version for tracking compat changes
COMPAT_VERSION = "0.4.1"

# Registry of all deprecated aliases for runtime warnings
DEPRECATED_ALIASES: Dict[str, Dict[str, Any]] = {
    # Format: "old.import.path": {"canonical": "new.import.path", "deprecated_in": "version"}
    "victor.protocols.provider_adapter.ToolCall": {
        "canonical": "victor.agent.tool_calling.base.ToolCall",
        "deprecated_in": "0.4.1",
    },
    "victor.integrations.protocols.provider_adapter.ToolCall": {
        "canonical": "victor.agent.tool_calling.base.ToolCall",
        "deprecated_in": "0.4.1",
    },
    "victor.agent.resilience.CircuitState": {
        "canonical": "victor.providers.circuit_breaker.CircuitState",
        "deprecated_in": "0.4.1",
    },
    "victor.observability.resilience.CircuitState": {
        "canonical": "victor.providers.circuit_breaker.CircuitState",
        "deprecated_in": "0.4.1",
    },
    "victor.integrations.protocols.lsp_types": {
        "canonical": "victor.protocols.lsp_types",
        "deprecated_in": "0.4.1",
        "note": "Deleted - was exact duplicate",
    },
}

# Registry of semantic renames (not aliases - different behavior)
SEMANTIC_RENAMES: Dict[str, Dict[str, str]] = {
    "ToolCall": {
        "victor.integrations.protocol.interface": "ToolInvocation",
        "victor.integrations.protocol.messages": "ToolCallExecution",
        "victor.evaluation.agentic_harness": "EvalToolCall",
        "victor.agent.tool_deduplication": "TrackedToolCall",
    },
    "SearchResult": {
        "victor.rag.document_store": "DocumentSearchResult",
        "victor.storage.vector_stores.base": "EmbeddingSearchResult",
        "victor.storage.unified.protocol": "UnifiedSearchResult",
        "victor.integrations.protocol.interface": "CodeSearchResult",
        "victor.integrations.api.fastapi_server": "APISearchResult",
        "victor.coding.codebase.embeddings.base": "EmbeddingSearchResult (imported)",
    },
    "StreamChunk": {
        # Canonical: victor.providers.base.StreamChunk (Pydantic BaseModel)
        "victor.framework.protocols": "OrchestratorStreamChunk",
        "victor.core.typed_models": "TypedStreamChunk",
        "victor.integrations.protocol.interface": "ClientStreamChunk",
    },
    "AgentMode": {
        # Canonical: victor.agent.mode_controller.AgentMode (3 modes: BUILD, PLAN, EXPLORE)
        "victor.agent.rl.learners.mode_transition": "RLAgentMode",
        "victor.agent.adaptive_mode_controller": "AdaptiveAgentMode",
        "victor.integrations.protocol.interface": "(imported from canonical)",
    },
    "ModeConfig": {
        # Canonical: victor.core.mode_config.ModeConfig (simple tool budget/iteration config)
        "victor.agent.mode_controller": "OperationalModeConfig",
    },
    "TaskType": {
        # Canonical: victor.classification.pattern_registry.TaskType (prompt classification)
        "victor.agent.unified_task_tracker": "TrackerTaskType",
        "victor.agent.loop_detector": "LoopDetectorTaskType",
        "victor.agent.unified_classifier": "ClassifierTaskType",
        "victor.framework.task.core": "FrameworkTaskType",
    },
    "Severity": {
        # Canonical: victor.security.protocol.CVESeverity (CVE/CVSS-based)
        # Each variant has different values for different domains
        "victor.security.protocol": "CVESeverity",
        "victor.security.audit.protocol": "AuditSeverity",
        "victor.iac.protocol": "IaCSeverity",
        "victor.coding.review.protocol": "ReviewSeverity",
    },
    "RiskLevel": {
        # Each variant has different values for different domains
        "victor.agent.safety": "OperationalRiskLevel (5 values incl. SAFE)",
        "victor.safety.code_patterns": "PatternRiskLevel (4 uppercase values)",
        "victor.safety.infrastructure": "(imports from code_patterns)",
    },
    "QualityDimension": {
        # Each variant has different dimensions for different purposes
        "victor.agent.response_quality": "ResponseQualityDimension (7 dims incl. CODE_QUALITY)",
        "victor.protocols.quality": "ProtocolQualityDimension (7 dims incl. GROUNDING, SAFETY)",
    },
    "Checkpoint": {
        # Each variant has different semantics for different checkpoint purposes
        "victor.agent.checkpoints": "GitCheckpoint (git stash-based)",
        "victor.agent.time_aware_executor": "ExecutionCheckpoint (progress tracking)",
        "victor.framework.graph": "WorkflowCheckpoint (StateGraph state)",
        "victor.framework.hitl": "HITLCheckpoint (human-in-the-loop)",
    },
    "CircuitBreaker": {
        # Canonical: victor.providers.circuit_breaker.CircuitBreaker (standalone)
        "victor.agent.resilience": "MultiCircuitBreaker (multiple named circuits)",
        "victor.observability.resilience": "ObservableCircuitBreaker (metrics/callbacks)",
        "victor.providers.resilience": "ProviderCircuitBreaker (ResilientProvider)",
    },
    "ValidationResult": {
        # All variants have different attributes for different validation purposes
        "victor.tools.base": "ToolValidationResult (invalid_params Dict)",
        "victor.core.validation": "ConfigValidationResult (ValidationIssue list)",
        "victor.framework.middleware": "ContentValidationResult (auto-fix support)",
        "victor.agent.parameter_enforcer": "ParameterValidationResult (missing_required)",
        "victor.evaluation.correction.types": "CodeValidationResult (syntax/imports)",
    },
    "Dependency": {
        # Different semantics for security vs package management
        "victor.security.protocol": "SecurityDependency (ecosystem, license, package_url)",
        "victor.deps.protocol": "PackageDependency (version tracking)",
    },
    "BaseDependencyParser": {
        # Different interface for security vs package management
        "victor.security.scanner": "BaseSecurityDependencyParser (ecosystem property)",
        "victor.deps.parsers": "BasePackageDependencyParser (PackageManager enum)",
    },
    "RetryConfig": {
        # Different attributes for different retry domains
        "victor.providers.resilience": "ProviderRetryConfig (retryable_patterns)",
        "victor.agent.resilience": "AgentRetryConfig (jitter flag)",
        "victor.observability.resilience": "ObservabilityRetryConfig (BackoffStrategy)",
    },
    "RetryStrategy": {
        # Canonical: victor.core.retry.BaseRetryStrategy (abstract base)
        "victor.core.retry": "BaseRetryStrategy (abstract base)",
        "victor.providers.resilience": "ProviderRetryStrategy (concrete with execute())",
        "victor.workflows.batch_executor": "BatchRetryStrategy (enum: NONE, IMMEDIATE, etc.)",
    },
    "NodeType": {
        # Different node types for different workflow systems
        "victor.workflows.definition": "WorkflowNodeType (COMPUTE, HITL, START, END)",
        "victor.workflows.graph_dsl": "GraphNodeType (FUNCTION, CONDITIONAL, SUBGRAPH)",
        "victor.workflows.yaml_loader": "YAMLNodeType (YAML validation)",
    },
    "NodeStatus": {
        # Same values, different contexts
        "victor.workflows.protocols": "ProtocolNodeStatus",
        "victor.workflows.executor": "ExecutorNodeStatus",
        "victor.framework.graph": "FrameworkNodeStatus",
    },
    "StopReason": {
        # Different semantics for different stopping contexts
        "victor.agent.unified_task_tracker": "TrackerStopReason (budget, loop, iterations)",
        "victor.agent.loop_detector": "LoopStopRecommendation (dataclass recommendation)",
        "victor.observability.debug.protocol": "DebugStopReason (debugger reasons)",
    },
    "VerificationResult": {
        # Different verification contexts
        "victor.agent.grounding_verifier": "GroundingVerificationResult (issues list)",
        "victor.protocols.grounding": "ClaimVerificationResult (claim verification)",
        "victor.integrations.protocols.grounding": "ClaimVerificationResult (duplicate)",
    },
    "ToolSelectionContext": {
        # Different contexts for tool selection
        "victor.agent.protocols": "AgentToolSelectionContext (basic agent-level)",
        "victor.core.verticals.protocols.tool_provider": "VerticalToolSelectionContext (vertical-specific)",
        "victor.tools.selection.protocol": "CrossVerticalToolSelectionContext (cross-vertical)",
    },
    "Symbol": {
        # Different symbol representations for different purposes
        "victor.native.protocols": "NativeSymbol (Rust-extracted, frozen)",
        "victor.coding.codebase.indexer": "IndexedSymbol (Pydantic for index storage)",
        "victor.coding.refactor.protocol": "RefactorSymbol (SourceLocation + references)",
    },
    "ScanResult": {
        # Different scan result types for different domains
        "victor.safety.code_patterns": "SafetyScanResult (safety pattern matching)",
        "victor.security.safety.code_patterns": "SafetyScanResult (duplicate)",
        "victor.iac.protocol": "IaCScanResult (IaC scan results)",
    },
    "RecoveryAction": {
        # Different recovery action types for different contexts
        "victor.agent.error_recovery": "ErrorRecoveryAction (tool error recovery, string enum)",
        "victor.agent.recovery.protocols": "StrategyRecoveryAction (strategy actions, auto enum)",
        "victor.agent.orchestrator_recovery": "OrchestratorRecoveryAction (dataclass)",
    },
}


# =============================================================================
# MODULE-LEVEL DUPLICATIONS (for future cleanup)
# =============================================================================
# These are full module duplications that should be consolidated in a future sprint.
# One module should be canonical, others should import from it.

MODULE_DUPLICATIONS = {
    "safety_patterns": {
        "canonical": "victor.safety",
        "duplicates": ["victor.security.safety"],
        "note": "Near-identical modules with different import paths",
    },
}


def get_canonical_source(alias_path: str) -> str:
    """Get the canonical source for a deprecated alias.

    Args:
        alias_path: Full import path of the alias

    Returns:
        Canonical import path, or original if not deprecated
    """
    info = DEPRECATED_ALIASES.get(alias_path)
    if info:
        return info["canonical"]
    return alias_path


def list_all_aliases() -> Dict[str, str]:
    """List all deprecated aliases and their canonical sources."""
    return {path: info["canonical"] for path, info in DEPRECATED_ALIASES.items()}
