# Tiered Confidence-Based Classification System

## Executive Summary

Implement a **confidence-based triage system** for fuzzy matching + weighted similarity classification that optimizes for speed, accuracy, and efficiency through intelligent confidence banding.

## Problem Statement

Current fuzzy matching + weighted similarity provides good results but lacks **confidence-aware decision making**:
- **High-confidence matches** (>0.8) should be accepted immediately (fast path)
- **Grey area matches** (0.6-0.8) need additional verification (edge LLM)
- **Low-confidence matches** (<0.6) should be rejected early (avoid waste)

## Solution: Tiered Classification Triage

### Confidence Bands

| Confidence Range | Action | Rationale | Performance |
|-----------------|--------|-----------|-------------|
| **> 0.8** | **ACCEPT** (fast path) | High confidence, no verification needed | ~4-5μs |
| **0.6 - 0.8** | **VERIFY** (edge LLM if available) | Grey area, needs LLM confirmation | ~50-200ms (edge LLM) |
| **< 0.6** | **REJECT** (early rejection) | Too uncertain, waste of resources | ~1μs |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Classification Request                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Fuzzy + Weighted Similarity Check                  │
│         (Adaptive thresholds + 75% min similarity)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                    ┌─────────┐
                    │ Score   │
                    └────┬────┘
                         │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
        Score > 0.8   0.6 ≤ Score ≤ 0.8  Score < 0.6
            │              │              │
            ▼              ▼              ▼
      ┌─────────┐    ┌──────────┐   ┌─────────┐
      │ ACCEPT  │    │ VERIFY   │   │ REJECT  │
      │ (fast)  │    │ (edge    │   │ (early) │
      └────┬────┘    │ LLM?)    │   └────┬────┘
           │         └────┬─────┘        │
           │              │              │
           │         ┌────┴─────┐        │
           │         │ Edge LLM │        │
           │         │Available? │        │
           │         └────┬─────┘        │
           │              │              │
           │         ┌────┴─────┐        │
           │         │   YES    │   NO   │
           │         └────┬─────┘        │
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Return   │   │ Edge LLM │   │ Return   │
    │ Accepted │   │ Decision │   │ Rejected │
    │ Result   │   │          │   │ Result   │
    └──────────┘   └──────────┘   └──────────┘
```

## Implementation

### 1. Configuration

**File**: `victor/config/groups/fuzzy_matching_config.py`

```python
@dataclass
class TieredClassificationConfig(BaseModel):
    """Configuration for tiered confidence-based classification."""

    # Master switch for tiered classification
    enabled: bool = True

    # Confidence thresholds
    high_confidence_threshold: float = 0.8  # Fast path accept
    low_confidence_threshold: float = 0.6   # Early rejection
    verification_min_threshold: float = 0.6  # Minimum for verification

    # Edge LLM configuration
    use_edge_llm_for_verification: bool = True  # Use edge LLM in grey area
    edge_llm_timeout_ms: int = 5000  # Max wait time for edge LLM
    edge_llm_fallback_to_reject: bool = True  # If edge LLM unavailable, reject?

    # Statistics
    log_tiered_decisions: bool = False  # Log triage decisions for debugging
    track_band_distribution: bool = True  # Track confidence band stats
```

### 2. Tiered Classification Service

**File**: `victor/storage/embeddings/tiered_classifier.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any
from victor.config.groups.fuzzy_matching_config import FuzzyMatchingSettings

class ConfidenceBand(Enum):
    """Confidence bands for triage."""
    HIGH = "high"       # > 0.8: Accept immediately
    GREY = "grey"       # 0.6-0.8: Verify with edge LLM
    LOW = "low"         # < 0.6: Reject immediately

@dataclass
class TieredClassificationResult:
    """Result from tiered classification."""
    accepted: bool
    confidence: float
    band: ConfidenceBand
    method: str  # "fuzzy_similarity", "edge_llm", "rejected"
    edge_llm_used: bool = False
    edge_llm_confidence: Optional[float] = None
    rejection_reason: Optional[str] = None

class TieredClassificationService:
    """Tiered classification with confidence-based triage."""

    def __init__(
        self,
        settings: FuzzyMatchingSettings,
        edge_llm_service: Optional[Any] = None,
    ):
        self.settings = settings
        self.edge_llm_service = edge_llm_service
        self.stats = {
            "high_band": 0,
            "grey_band": 0,
            "low_band": 0,
            "edge_llm_calls": 0,
            "edge_llm_accepts": 0,
            "edge_llm_rejects": 0,
        }

    def classify_with_triage(
        self,
        query: str,
        fuzzy_similarity_score: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> TieredClassificationResult:
        """Classify with confidence-based triage.

        Args:
            query: The query text
            fuzzy_similarity_score: Score from fuzzy + weighted similarity
            context: Optional context for edge LLM

        Returns:
            TieredClassificationResult with triage decision
        """
        confidence = fuzzy_similarity_score

        # Band 1: High confidence (> 0.8) - Accept immediately
        if confidence > self.settings.tiered_high_confidence_threshold:
            self.stats["high_band"] += 1
            return TieredClassificationResult(
                accepted=True,
                confidence=confidence,
                band=ConfidenceBand.HIGH,
                method="fuzzy_similarity",
            )

        # Band 3: Low confidence (< 0.6) - Reject immediately
        elif confidence < self.settings.tiered_low_confidence_threshold:
            self.stats["low_band"] += 1
            return TieredClassificationResult(
                accepted=False,
                confidence=confidence,
                band=ConfidenceBand.LOW,
                method="rejected",
                rejection_reason="Confidence below threshold",
            )

        # Band 2: Grey area (0.6-0.8) - Verify with edge LLM
        else:
            self.stats["grey_band"] += 1
            return self._verify_with_edge_llm(query, confidence, context)

    def _verify_with_edge_llm(
        self,
        query: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> TieredClassificationResult:
        """Verify grey area classification with edge LLM.

        Args:
            query: The query text
            confidence: Current confidence score
            context: Optional context for edge LLM

        Returns:
            TieredClassificationResult based on edge LLM decision
        """
        # Check if edge LLM is available
        if not self.settings.tiered_use_edge_llm_for_verification:
            # Edge LLM disabled, reject conservatively
            return TieredClassificationResult(
                accepted=False,
                confidence=confidence,
                band=ConfidenceBand.GREY,
                method="rejected",
                rejection_reason="Edge LLM verification disabled",
            )

        if self.edge_llm_service is None:
            # Edge LLM not available
            if self.settings.tiered_edge_llm_fallback_to_reject:
                return TieredClassificationResult(
                    accepted=False,
                    confidence=confidence,
                    band=ConfidenceBand.GREY,
                    method="rejected",
                    rejection_reason="Edge LLM unavailable",
                )
            else:
                # Accept anyway (risky)
                return TieredClassificationResult(
                    accepted=True,
                    confidence=confidence,
                    band=ConfidenceBand.GREY,
                    method="fuzzy_similarity",
                )

        # Use edge LLM for verification
        self.stats["edge_llm_calls"] += 1

        try:
            edge_decision = self.edge_llm_service.verify_classification(
                query=query,
                initial_confidence=confidence,
                context=context,
                timeout_ms=self.settings.tiered_edge_llm_timeout_ms,
            )

            self.stats["edge_llm_accepts" if edge_decision.accepted else "edge_llm_rejects"] += 1

            return TieredClassificationResult(
                accepted=edge_decision.accepted,
                confidence=edge_decision.confidence,
                band=ConfidenceBand.GREY,
                method="edge_llm",
                edge_llm_used=True,
                edge_llm_confidence=edge_decision.confidence,
            )

        except Exception as e:
            # Edge LLM failed
            if self.settings.tiered_edge_llm_fallback_to_reject:
                return TieredClassificationResult(
                    accepted=False,
                    confidence=confidence,
                    band=ConfidenceBand.GREY,
                    method="rejected",
                    rejection_reason=f"Edge LLM error: {str(e)}",
                )
            else:
                # Accept anyway (fallback)
                return TieredClassificationResult(
                    accepted=True,
                    confidence=confidence,
                    band=ConfidenceBand.GREY,
                    method="fuzzy_similarity",
                )

    def get_stats(self) -> Dict[str, int]:
        """Get triage statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset triage statistics."""
        self.stats = {
            "high_band": 0,
            "grey_band": 0,
            "low_band": 0,
            "edge_llm_calls": 0,
            "edge_llm_accepts": 0,
            "edge_llm_rejects": 0,
        }
```

### 3. Edge LLM Verification Protocol

**File**: `victor/agent/services/protocols/edge_llm_verifier.py`

```python
from typing import Protocol, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EdgeLLMVerificationResult:
    """Result from edge LLM verification."""
    accepted: bool
    confidence: float
    reasoning: Optional[str] = None
    suggested_task_type: Optional[str] = None

class EdgeLLMVerifierProtocol(Protocol):
    """Protocol for edge LLM verification service."""

    def verify_classification(
        self,
        query: str,
        initial_confidence: float,
        context: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 5000,
    ) -> EdgeLLMVerificationResult:
        """Verify classification with edge LLM.

        Args:
            query: The query text to verify
            initial_confidence: Initial confidence from fuzzy similarity
            context: Optional context (conversation history, etc.)
            timeout_ms: Maximum time to wait for edge LLM

        Returns:
            EdgeLLMVerificationResult with edge LLM decision
        """
        ...
```

### 4. Integration Points

#### 4.1 TaskTypeClassifier Integration

**File**: `victor/storage/embeddings/task_classifier.py`

```python
from victor.storage.embeddings.tiered_classifier import TieredClassificationService

class TaskTypeClassifier:
    def __init__(...):
        # ... existing code ...
        self._tiered_service: Optional[TieredClassificationService] = None

    def initialize_sync(self):
        # ... existing code ...
        # Initialize tiered service if enabled
        if self.settings.tiered_enabled:
            edge_llm = get_container().get(EdgeLLMVerifierProtocol, None)
            self._tiered_service = TieredClassificationService(
                settings=self.settings,
                edge_llm_service=edge_llm,
            )

    def classify_sync(self, prompt: str) -> TaskTypeResult:
        """Classify with tiered confidence-based triage."""
        # ... existing fuzzy + weighted similarity logic ...
        base_score = weighted_similarity_score

        # Apply tiered classification if enabled
        if self._tiered_service:
            triage_result = self._tiered_service.classify_with_triage(
                query=prompt,
                fuzzy_similarity_score=base_score,
                context={"prompt": prompt},
            )

            if not triage_result.accepted:
                # Return low-confidence result
                return TaskTypeResult(
                    task_type=TaskType.GENERAL,
                    confidence=triage_result.confidence,
                    method="tiered_rejected",
                    rejection_reason=triage_result.rejection_reason,
                )

            # Update confidence if edge LLM was used
            if triage_result.edge_llm_used and triage_result.edge_llm_confidence:
                final_confidence = triage_result.edge_llm_confidence
            else:
                final_confidence = base_score
        else:
            # No tiered classification, use original logic
            final_confidence = base_score

        # ... rest of classification logic ...
```

#### 4.2 IntentClassifier Integration

**File**: `victor/storage/embeddings/intent_classifier.py`

```python
class IntentClassifier:
    def __init__(...):
        # ... existing code ...
        self._tiered_service: Optional[TieredClassificationService] = None

    def initialize_sync(self):
        # ... existing code ...
        if self.settings.tiered_enabled:
            edge_llm = get_container().get(EdgeLLMVerifierProtocol, None)
            self._tiered_service = TieredClassificationService(
                settings=self.settings,
                edge_llm_service=edge_llm,
            )

    def classify_intent_sync(self, text: str) -> IntentResult:
        """Classify intent with tiered confidence-based triage."""
        # ... existing similarity logic ...
        base_score = best_continuation  # or best_completion, etc.

        # Apply tiered classification if enabled
        if self._tiered_service:
            triage_result = self._tiered_service.classify_with_triage(
                query=text,
                fuzzy_similarity_score=base_score,
                context={"text": text},
            )

            if not triage_result.accepted:
                # Return NEUTRAL intent
                return IntentResult(
                    intent=IntentType.NEUTRAL,
                    confidence=triage_result.confidence,
                    method="tiered_rejected",
                )

            # Update confidence if edge LLM was used
            if triage_result.edge_llm_used:
                final_confidence = triage_result.edge_llm_confidence
            else:
                final_confidence = base_score
        else:
            final_confidence = base_score

        # ... rest of intent logic ...
```

### 5. Settings Configuration

**File**: `victor/config/groups/fuzzy_matching_config.py`

Add to `FuzzyMatchingSettings`:

```python
# Tiered classification settings
tiered_enabled: bool = True
tiered_high_confidence_threshold: float = 0.8
tiered_low_confidence_threshold: float = 0.6
tiered_use_edge_llm_for_verification: bool = True
tiered_edge_llm_timeout_ms: int = 5000
tiered_edge_llm_fallback_to_reject: bool = True
tiered_log_tiered_decisions: bool = False
tiered_track_band_distribution: bool = True
```

### 6. Feature Flag

**File**: `victor/core/feature_flags.py`

```python
class FeatureFlag(Enum):
    # ... existing flags ...
    USE_TIERED_CLASSIFICATION = "use_tiered_classification"  # Phase: Tiered Classification
```

## Performance Characteristics

### Expected Distribution

Based on typical classification distributions:

| Band | Expected % | Action | Avg Latency |
|------|-----------|--------|-------------|
| **High (>0.8)** | 60-70% | Accept (fast) | ~4-5μs |
| **Grey (0.6-0.8)** | 20-30% | Verify (edge LLM) | ~50-200ms |
| **Low (<0.6)** | 10-20% | Reject (early) | ~1μs |

**Overall average latency**: ~15-60μs (dominated by fuzzy matching)

### Without Edge LLM

If edge LLM is not available or disabled:
- **Grey area**: Reject conservatively (or accept if `fallback_to_reject=False`)
- **Overall accuracy**: Slightly lower but much faster
- **Latency**: ~4-5μs (all fast path)

## Benefits

1. **Speed**: High-confidence cases (~60-70%) bypass all verification
2. **Accuracy**: Grey area (~20-30%) gets LLM verification for certainty
3. **Efficiency**: Low-confidence cases (~10-20%) rejected early
4. **Flexibility**: Edge LLM is **optional** - works without it
5. **Observability**: Track band distribution and edge LLM effectiveness
6. **Robustness**: Graceful degradation when edge LLM unavailable

## Testing Strategy

### Unit Tests

```python
def test_high_confidence_fast_path():
    """Test high confidence (>0.8) is accepted immediately."""
    service = TieredClassificationService(settings=mock_settings, edge_llm=None)
    result = service.classify_with_triage("analyze code", fuzzy_similarity_score=0.85)
    assert result.accepted is True
    assert result.band == ConfidenceBand.HIGH
    assert result.method == "fuzzy_similarity"

def test_low_confidence_early_rejection():
    """Test low confidence (<0.6) is rejected immediately."""
    service = TieredClassificationService(settings=mock_settings, edge_llm=None)
    result = service.classify_with_triage("xyz abc", fuzzy_similarity_score=0.4)
    assert result.accepted is False
    assert result.band == ConfidenceBand.LOW
    assert result.method == "rejected"

def test_grey_area_with_edge_llm():
    """Test grey area (0.6-0.8) uses edge LLM verification."""
    service = TieredClassificationService(
        settings=mock_settings,
        edge_llm=mock_edge_llm_accepting,
    )
    result = service.classify_with_triage("analize code", fuzzy_similarity_score=0.7)
    assert result.band == ConfidenceBand.GREY
    assert result.edge_llm_used is True

def test_grey_area_edge_llm_unavailable():
    """Test grey area rejects when edge LLM unavailable."""
    service = TieredClassificationService(
        settings=mock_settings_with_reject_fallback,
        edge_llm=None,
    )
    result = service.classify_with_triage("analize code", fuzzy_similarity_score=0.7)
    assert result.accepted is False
    assert result.band == ConfidenceBand.GREY
    assert "unavailable" in result.rejection_reason
```

### Integration Tests

```python
@pytest.mark.integration
def test_end_to_end_tiered_classification():
    """Test full tiered classification with real classifiers."""
    classifier = TaskTypeClassifier.get_instance()
    classifier.initialize_sync()

    # High confidence case
    result1 = classifier.classify_sync("analyze the code structure")
    assert result1.confidence > 0.8
    assert result1.task_type == TaskType.ANALYZE

    # Grey area case (with typo)
    result2 = classifier.classify_sync("analize the structre")
    # Should use edge LLM or reject depending on config
    assert 0.6 <= result2.confidence <= 0.8

    # Low confidence case
    result3 = classifier.classify_sync("xyz random words")
    assert result3.confidence < 0.6
```

## Migration Path

### Phase 1: Implementation (Current PR)
- Implement `TieredClassificationService`
- Add configuration and feature flag
- Write unit and integration tests
- **Feature flag OFF by default**

### Phase 2: Testing & Validation
- Enable in staging with feature flag
- Collect metrics on band distribution
- Validate edge LLM effectiveness
- Tune thresholds based on real data

### Phase 3: Gradual Rollout
- Enable for 10% of traffic
- Monitor accuracy and latency
- Increase to 50%, then 100%
- Remove feature flag once stable

## Monitoring & Observability

### Metrics to Track

```python
# Band distribution
tiered_high_band_count = Counter("tiered_classification.high_band")
tiered_grey_band_count = Counter("tiered_classification.grey_band")
tiered_low_band_count = Counter("tiered_classification.low_band")

# Edge LLM effectiveness
tiered_edge_llm_calls = Counter("tiered_classification.edge_llm_calls")
tiered_edge_llm_accepts = Counter("tiered_classification.edge_llm_accepts")
tiered_edge_llm_rejects = Counter("tiered_classification.edge_llm_rejects")

# Overall accuracy
tiered_classification_accuracy = Histogram("tiered_classification.accuracy")
```

### Logging

```python
if settings.tiered_log_tiered_decisions:
    logger.debug(
        f"Tiered classification: query='{query[:50]}...', "
        f"confidence={confidence:.3f}, band={band.value}, "
        f"method={method}, accepted={accepted}"
    )
```

## Conclusion

This tiered confidence-based classification system provides:
- ✅ **Speed**: Fast path for high-confidence cases
- ✅ **Accuracy**: Edge LLM verification for grey area
- ✅ **Efficiency**: Early rejection for low-confidence cases
- ✅ **Flexibility**: Edge LLM is optional
- ✅ **Robustness**: Graceful degradation

The system balances **accuracy with efficiency** by only using the expensive edge LLM when it matters most (grey area cases), while fast-tracking obvious acceptances and rejections.
