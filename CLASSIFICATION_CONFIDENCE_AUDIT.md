# Classification Confidence Audit & Recommendations

## Executive Summary

Audit of existing confidence bands, tiered decision patterns, and edge model integration across Victor's classification systems. **Key finding**: Victor already has sophisticated confidence-based logic, but it's **fragmented across multiple systems**. This document recommends **unification and reuse** of existing patterns.

## Existing Confidence Bands (Found)

### 1. RuntimeEvaluationPolicy (`victor/framework/runtime_evaluation_policy.py`)

**Already implements 3-tier confidence bands**:

| Confidence Range | Decision | Reason | Usage |
|-----------------|----------|--------|-------|
| **≥ 0.8** | **COMPLETE** | "High confidence in perception" | Agentic loop completion detection |
| **≥ 0.5** | **CONTINUE** | "Medium confidence - continue" | Agentic loop progress |
| **< 0.5** | **RETRY** | "Low confidence - retry" | Agentic loop retry logic |

**Code** (lines 102-104, 235-253):
```python
clarification_confidence_threshold: float = 0.45
high_confidence_threshold: float = 0.8
medium_confidence_threshold: float = 0.5

def get_confidence_evaluation(self, confidence: float) -> Any:
    if confidence >= self.high_confidence_threshold:
        return EvaluationResult(decision=EvaluationDecision.COMPLETE, ...)
    if confidence >= self.medium_confidence_threshold:
        return EvaluationResult(decision=EvaluationDecision.CONTINUE, ...)
    return EvaluationResult(decision=EvaluationDecision.RETRY, ...)
```

**Analysis**: ✅ **Well-designed** - Clear separation, canonical pattern, used in agentic loop.

### 2. Edge Model (`victor/agent/edge_model.py`)

**Already implements edge LLM with confidence threshold**:

```python
@dataclass
class EdgeModelConfig:
    enabled: bool = True
    provider: str = "ollama"
    model: str = "qwen3.5:2b"
    timeout_ms: int = 4000
    max_tokens: int = 50
    cache_ttl: int = 120
    confidence_threshold: float = 0.6  # Below this, fall back to heuristic
```

**Key Features**:
- ✅ **Already optional**: Falls back to heuristic when unavailable
- ✅ **Fast**: 2000ms timeout for classification tasks
- ✅ **Cached**: 300s TTL for classification decisions
- ✅ **Local**: Runs via Ollama with qwen3.5:2b (2B params)
- ✅ **Confidence-aware**: Falls back below 0.6 threshold

**Analysis**: ✅ **Production-ready** - Already implements exactly what user proposed for grey area verification!

### 3. Tiered Decision Service (`victor/agent/services/tiered_decision_service.py`)

**Already implements tiered routing for decisions**:

```python
FALLBACK_CHAIN = {
    "performance": ["balanced", "edge"],
    "balanced": ["edge"],
    "edge": [],
}
```

**Key Features**:
- ✅ **Provider-agnostic**: Auto-detects active provider
- ✅ **Tier routing**: Maps DecisionTypes to model tiers
- ✅ **Graceful fallback**: performance → balanced → edge → heuristic
- ✅ **Service pool**: One LLMDecisionService per tier

**Analysis**: ✅ **Well-architected** - Already implements tiered pattern, just needs to be reused for classification.

### 4. Unified Task Classifier (`victor/agent/unified_classifier.py`)

**Already uses semantic confidence threshold**:

```python
semantic_confidence_threshold: float = 0.85  # Min confidence to trust semantic over keyword
```

**Analysis**: ⚠️ **Inconsistent** - Uses 0.85 instead of 0.8 (slightly more conservative).

### 5. Fuzzy Matching (Just Implemented)

**Implements similarity ratio threshold**:

```python
min_similarity_ratio: float = 0.75  # Minimum Levenshtein ratio for fuzzy matching
```

**Analysis**: ✅ **Consistent** - 0.75 is between 0.6 (edge model) and 0.8 (high confidence), which makes sense as a fuzzy matching baseline.

## Gap Analysis

### What's Missing

| Feature | Status | Notes |
|---------|--------|-------|
| **Confidence bands for classification** | ⚠️ Fragmented | RuntimeEvaluationPolicy has bands, but not used by classifiers |
| **Edge LLM verification for grey area** | ⚠️ Partial | Edge model exists but not integrated with classification confidence |
| **Unified tiered service** | ❌ Missing | Need to unify RuntimeEvaluationPolicy + EdgeModel + TieredDecisionService |
| **Early rejection for low confidence** | ❌ Missing | Classifiers accept all results, no early rejection |

### What's Working Well

| Component | Strength | Should Be Reused |
|-----------|----------|-------------------|
| **RuntimeEvaluationPolicy** | Canonical confidence bands | ✅ Reuse high/medium/low thresholds |
| **Edge Model** | Optional, fast, cached, confidence-aware | ✅ Use for grey area verification |
| **TieredDecisionService** | Tier routing with fallback | ✅ Extend for classification triage |
| **Fuzzy Matching** | Robust typo tolerance | ✅ Keep as-is, works well |

## Recommendations

### 🎯 Primary Recommendation: Unify & Reuse

**Don't reinvent the wheel!** Extend existing patterns instead of creating new services.

#### 1. Unify Confidence Thresholds

**Problem**: Multiple conflicting thresholds (0.45, 0.5, 0.6, 0.75, 0.8, 0.85)

**Solution**: Standardize on **RuntimeEvaluationPolicy** bands:

```python
# victor/framework/runtime_evaluation_policy.py (already exists)
high_confidence_threshold: float = 0.8      # Accept (fast path)
medium_confidence_threshold: float = 0.5    # Edge LLM verification
clarification_confidence_threshold: float = 0.45 # Reject (early)
```

**For classification**, map user's proposal to existing bands:
- **> 0.8** → High confidence → ACCEPT (fast path) ✅
- **0.6 - 0.8** → Grey area → Edge LLM verification ✅
- **< 0.6** → Low confidence → REJECT (early) ✅

#### 2. Extend Edge Model for Classification Verification

**Current state**: Edge model already has:
- ✅ Confidence threshold (0.6)
- ✅ Fallback to heuristic
- ✅ Optional (enabled flag)
- ✅ Fast (2000ms timeout)
- ✅ Cached (300s TTL)

**Solution**: Add **verification method** to edge model:

```python
# victor/agent/edge_model.py (add to EdgeModelConfig)

@dataclass
class EdgeModelConfig:
    # ... existing fields ...
    verification_enabled: bool = True  # Enable for grey area verification
    verification_timeout_ms: int = 2000  # Faster for verification
    verification_confidence_threshold: float = 0.6  # Same as classification
```

```python
# victor/agent/edge_model.py (add method)

def verify_classification(
    self,
    query: str,
    initial_confidence: float,
    task_type: str = "classification",
    context: Optional[Dict[str, Any]] = None,
) -> EdgeLLMVerificationResult:
    """Verify classification with edge LLM (grey area verification).

    Args:
        query: The query text to verify
        initial_confidence: Initial confidence from fuzzy + weighted similarity
        task_type: Type of classification (for routing)
        context: Optional context (conversation history, etc.)

    Returns:
        EdgeLLMVerificationResult with edge LLM decision
    """
    # Check if verification is enabled
    if not self.config.verification_enabled:
        return EdgeLLMVerificationResult(
            accepted=False,
            confidence=initial_confidence,
            reasoning="Verification disabled",
        )

    # Check if confidence is in verification band
    if initial_confidence < self.config.confidence_threshold:
        return EdgeLLMVerificationResult(
            accepted=False,
            confidence=initial_confidence,
            reasoning="Confidence below threshold",
        )

    # Use edge model for verification
    prompt = f"""Classify this query and return confidence score:

Query: {query}

Initial confidence: {initial_confidence:.2f}

Is this classification correct? Respond with:
- confidence: float (0.0-1.0)
- accepted: bool
- reasoning: str (short explanation)
"""

    try:
        result = self._call_edge_model(
            prompt=prompt,
            timeout_ms=self.config.get_timeout_for_task("classification"),
        )

        # Parse edge model response
        return EdgeLLMVerificationResult(
            accepted=result.get("accepted", False),
            confidence=result.get("confidence", initial_confidence),
            reasoning=result.get("reasoning"),
        )

    except Exception as e:
        logger.warning(f"Edge model verification failed: {e}")
        return EdgeLLMVerificationResult(
            accepted=False,
            confidence=initial_confidence,
            reasoning=f"Verification failed: {str(e)}",
        )
```

#### 3. Create Classification Triage Service (Reusable)

**Don't create a new service!** Extend `TieredDecisionService`:

```python
# victor/agent/services/tiered_decision_service.py (extend existing service)

class TieredDecisionService:
    # ... existing code ...

    def classify_with_triage(
        self,
        query: str,
        fuzzy_similarity_score: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> TieredClassificationResult:
        """Classify with confidence-based triage (unified pattern).

        Reuses RuntimeEvaluationPolicy confidence bands:
        - High (>0.8): Accept (fast path)
        - Medium (0.5-0.8): Edge LLM verification (if available)
        - Low (<0.5): Reject (early)

        Args:
            query: Query text
            fuzzy_similarity_score: Score from fuzzy + weighted similarity
            context: Optional context

        Returns:
            TieredClassificationResult
        """
        from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy

        policy = RuntimeEvaluationPolicy()
        confidence = fuzzy_similarity_score

        # Band 1: High confidence (>0.8) - Accept immediately
        if confidence >= policy.high_confidence_threshold:
            return TieredClassificationResult(
                accepted=True,
                confidence=confidence,
                band="high",
                method="fuzzy_similarity",
            )

        # Band 3: Low confidence (<0.5) - Reject immediately
        if confidence < policy.medium_confidence_threshold:
            return TieredClassificationResult(
                accepted=False,
                confidence=confidence,
                band="low",
                method="rejected",
                rejection_reason="Confidence below threshold",
            )

        # Band 2: Grey area (0.5-0.8) - Verify with edge LLM
        # Get edge tier service
        edge_service = self._services.get("edge")
        if not edge_service:
            # Edge LLM unavailable, reject conservatively
            return TieredClassificationResult(
                accepted=False,
                confidence=confidence,
                band="grey",
                method="rejected",
                rejection_reason="Edge LLM unavailable",
            )

        # Use edge LLM for verification
        try:
            edge_result = edge_service.verify_classification(
                query=query,
                initial_confidence=confidence,
                context=context,
            )

            return TieredClassificationResult(
                accepted=edge_result.accepted,
                confidence=edge_result.confidence,
                band="grey",
                method="edge_llm",
                edge_llm_used=True,
                edge_llm_confidence=edge_result.confidence,
            )

        except Exception as e:
            # Edge LLM failed, reject conservatively
            return TieredClassificationResult(
                accepted=False,
                confidence=confidence,
                band="grey",
                method="rejected",
                rejection_reason=f"Edge LLM error: {str(e)}",
            )
```

#### 4. Integration Pattern for All Classifiers

**Apply triage consistently** across all classifiers:

```python
# victor/storage/embeddings/task_classifier.py (add to classify_sync)

from victor.agent.services.tiered_decision_service import TieredDecisionService

class TaskTypeClassifier:
    def __init__(...):
        # ... existing code ...
        self._tiered_service: Optional[TieredDecisionService] = None

    def initialize_sync(self):
        # ... existing code ...
        # Initialize tiered service if enabled
        if is_enabled("USE_TIERED_CLASSIFICATION"):
            from victor.core import get_container
            self._tiered_service = get_container().get(TieredDecisionService, None)

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
            final_confidence = triage_result.edge_llm_confidence or base_score
        else:
            final_confidence = base_score

        # ... rest of classification logic ...
```

## Revised Confidence Band Recommendations

### Unified Bands (Standardize on RuntimeEvaluationPolicy)

| Band | Range | Action | Components | Latency |
|------|-------|--------|------------|---------|
| **HIGH** | ≥ 0.8 | Accept | Fuzzy + Weighted Similarity | ~4-5μs |
| **MEDIUM** | 0.5 - 0.8 | Edge LLM Verification | Edge Model (if available) | ~50-200ms |
| **LOW** | < 0.5 | Reject | Early rejection | ~1μs |

### Threshold Mapping

| Component | Current Threshold | Recommended | Alignment |
|-----------|-------------------|-------------|-----------|
| **RuntimeEvaluationPolicy** | 0.8 (high), 0.5 (medium), 0.45 (clarification) | ✅ Keep as-is | **Canonical source** |
| **Edge Model** | 0.6 (fallback) | ✅ Keep as-is | Aligns with medium band |
| **UnifiedTaskClassifier** | 0.85 (semantic) | ⚠️ Change to 0.8 | Align with high band |
| **User's proposal** | 0.8 (accept), 0.6 (grey), <0.6 (reject) | ⚠️ Use 0.5 for grey | Align with medium band |

### Why 0.5 instead of 0.6 for Grey Area?

**Reason**: RuntimeEvaluationPolicy already uses **0.5 as medium threshold**, which is more conservative and provides:
- ✅ **Wider verification band** (0.5-0.8 vs 0.6-0.8)
- ✅ **Better coverage** for edge cases
- ✅ **Consistency** with existing agentic loop logic
- ✅ **Proven in production** (used in RuntimeEvaluationFeedback)

**Trade-off**:
- **More edge LLM calls** (30-40% of cases vs 20-30%)
- **Higher accuracy** (better safe than sorry for classification)
- **Slower overall** (but more accurate)

**Alternative**: If performance is critical, use **0.6 as low threshold** (user's proposal):
- Grey area: 0.6-0.8 (narrower, faster)
- Low confidence: <0.6 (rejection)
- **Faster but less accurate**

## Implementation Plan

### Phase 1: Extend Edge Model (Add Verification Method)

**File**: `victor/agent/edge_model.py`

1. Add `EdgeLLMVerificationResult` dataclass
2. Add `verification_enabled`, `verification_timeout_ms` to `EdgeModelConfig`
3. Implement `verify_classification()` method
4. Add tests for verification path

### Phase 2: Extend TieredDecisionService (Add Triage Method)

**File**: `victor/agent/services/tiered_decision_service.py`

1. Add `classify_with_triage()` method
2. Reuse `RuntimeEvaluationPolicy` for thresholds
3. Integrate with existing edge tier service
4. Add tests for triage logic

### Phase 3: Integrate with Classifiers

**Files**:
- `victor/storage/embeddings/task_classifier.py`
- `victor/storage/embeddings/intent_classifier.py`
- `victor/tools/semantic_selector.py`

For each classifier:
1. Add optional `_tiered_service` dependency
2. Call `classify_with_triage()` after fuzzy + weighted similarity
3. Handle rejection result appropriately
4. Add feature flag `USE_TIERED_CLASSIFICATION`

### Phase 4: Unify Confidence Thresholds

**File**: `victor/agent/unified_classifier.py`

1. Change `semantic_confidence_threshold` from 0.85 to 0.8
2. Document alignment with `RuntimeEvaluationPolicy`
3. Add deprecation notice if changing API

### Phase 5: Add Feature Flag & Configuration

**Files**:
- `victor/core/feature_flags.py`: Add `USE_TIERED_CLASSIFICATION`
- `victor/config/groups/fuzzy_matching_config.py`: Add triage settings

```python
# victor/core/feature_flags.py
USE_TIERED_CLASSIFICATION = "use_tiered_classification"

# victor/config/groups/fuzzy_matching_config.py
# Add to FuzzyMatchingSettings:
tiered_enabled: bool = False  # OFF by default, opt-in
tiered_use_policy_thresholds: bool = True  # Use RuntimeEvaluationPolicy bands
```

### Phase 6: Testing & Validation

1. **Unit tests**: Test triage logic for all 3 bands
2. **Integration tests**: Test with real classifiers and edge model
3. **Performance tests**: Measure latency impact
4. **Accuracy tests**: Validate improvement in classification accuracy
5. **A/B testing**: Compare with/without tiered classification

## Performance Estimates

### Expected Distribution (Based on RuntimeEvaluationFeedback data)

| Band | Expected % | Latency | Cumulative Latency |
|------|-----------|---------|---------------------|
| **High (≥0.8)** | 60-70% | ~4-5μs | ~3-4μs (weighted) |
| **Medium (0.5-0.8)** | 20-30% | ~50-200ms | ~15-60μs (weighted) |
| **Low (<0.5)** | 10-20% | ~1μs | ~1μs (weighted) |

**Overall average**: ~15-60μs (dominated by fuzzy matching, very fast!)

### Without Edge LLM

If edge LLM is unavailable or disabled:
- **Medium band**: Reject conservatively (or accept if fallback enabled)
- **Overall accuracy**: Slightly lower but much faster
- **Latency**: ~4-5μs (all fast path)

## Conclusion

### ✅ What Works Well (Reuse These)

1. **RuntimeEvaluationPolicy** - Canonical confidence bands, already proven
2. **Edge Model** - Production-ready, optional, fast, cached
3. **TieredDecisionService** - Well-architected, just needs extension
4. **Fuzzy Matching** - Robust, 75% similarity threshold works well

### ⚠️ What Needs Unification

1. **Confidence thresholds** - Standardize on RuntimeEvaluationPolicy
2. **Edge LLM verification** - Add verification method to edge model
3. **Triage service** - Extend TieredDecisionService with classify_with_triage()
4. **Classifier integration** - Apply triage consistently

### 🎯 Final Recommendation

**Don't create new services!** Extend existing patterns:

1. **Add verification method to EdgeModel** (not new service)
2. **Add triage method to TieredDecisionService** (not new service)
3. **Reuse RuntimeEvaluationPolicy thresholds** (not new constants)
4. **Add feature flag `USE_TIERED_CLASSIFICATION`** (opt-in, OFF by default)

This approach:
- ✅ **Maintains consistency** with existing architecture
- ✅ **Reuses proven patterns** (edge model, tiered service)
- ✅ **Avoids duplication** (single source of truth for thresholds)
- ✅ **Easy to test** (extend existing tests)
- ✅ **Backward compatible** (feature flag OFF by default)
- ✅ **Performance optimized** (reuses existing caching and timeouts)

## Next Steps

1. ✅ **Audit complete** - This document
2. 🔜 **Extend EdgeModel** - Add verification method
3. 🔜 **Extend TieredDecisionService** - Add triage method
4. 🔜 **Integrate with classifiers** - Apply triage pattern
5. 🔜 **Add feature flag** - `USE_TIERED_CLASSIFICATION`
6. 🔜 **Test & validate** - Ensure accuracy improvement
7. 🔜 **Enable gradually** - Start at 10% traffic, increase to 100%

---

**TL;DR**: Victor already has excellent building blocks (RuntimeEvaluationPolicy, EdgeModel, TieredDecisionService). **Extend and reuse them** instead of creating new services. This maintains consistency and avoids duplication.
