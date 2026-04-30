# Task Classification Improvements - Summary

## Problem Statement

The agent was unable to complete a comprehensive framework structural analysis task because it was misclassified as a general task instead of an analysis task, resulting in severe tool budget restrictions.

## Root Cause Analysis

### The Failure Chain

1. **User Message**: "framework structural analysis"
2. **TaskTypeClassifier** (embedding-based) → Classified as `GENERAL` instead of `ANALYZE`
3. **unified_tracker.detect_task_type()** → Returned `TrackerTaskType.GENERAL`
4. **chat_stream_helpers** → Set `is_analysis_task=False`
5. **Pipeline** → Applied `tool_budget=10` (vs 200 for analysis tasks)
6. **Recovery Service** → Truncated tool calls: "Truncating 4 tool calls to budget limit of 2"

### Log Evidence

```
2026-04-30 11:30:51,553 - Task type classification: coarse=default, unified=general, is_analysis=False, is_action=False
2026-04-30 11:30:51,553 - Stream chat limits: tool_budget=10, max_total_iterations=50, max_exploration_iterations=50
2026-04-30 11:31:08,931 - Truncating 3 tool calls to budget limit of 2
2026-04-30 11:31:18,431 - Truncating 4 tool calls to budget limit of 2
```

### Why Classification Failed

The `TaskTypeClassifier` in `victor/storage/embeddings/task_classifier.py` uses embedding similarity to match user messages to predefined phrase patterns. While it had `"analyze the structure"` in ANALYZE_PHRASES, the phrase `"framework structural analysis"` didn't match well enough due to:
- Different word order ("structural analysis" vs "analyze the structure")
- More abstract phrasing than concrete patterns
- Possible competition with DESIGN_PHRASES containing architecture terms

## Implemented Fixes

### 1. Enhanced ANALYZE_PHRASES (task_classifier.py:374-385)

Added structural/architecture analysis patterns:
```python
# Structural/architecture analysis (abstract patterns)
"structural analysis",
"framework analysis",
"architecture analysis",
"code structure review",
"system architecture review",
"analyze the framework",
"review the framework",
"examine the architecture",
"assess the architecture",
"analyze system design",
"review system structure",
```

### 2. Improved Fallback Logic (chat_stream_helpers.py:266-277)

Added keyword-based fallback when embedding classifier returns GENERAL but keywords suggest analysis:
```python
# Fallback: if unified says GENERAL but keywords say analysis, trust keywords
ctx.is_analysis_task = keyword_is_analysis or unified_is_analysis or (
    unified_task_type.value == "general" and keyword_is_analysis
)
```

This ensures that even if the embedding classifier fails, the keyword-based classifier (which already has "analysis", "architecture", "structure" keywords) can still correctly identify analysis tasks.

## Additional Recommendations

### 1. Add Classification Confidence Logging

Add logging when classification confidence is low to catch future issues:
```python
if result.confidence < 0.7:
    logger.warning(
        f"Low classification confidence: {result.confidence:.2f}, "
        f"type={result.task_type}, message={user_message[:50]}"
    )
```

### 2. Consider Hybrid Classification Approach

For critical tasks (tool budget allocation), use ensemble voting:
- Embedding classifier (semantic similarity)
- Keyword classifier (pattern matching)
- Edge model (LLM-based classification via FEP-0001)

Only override to GENERAL if **all three** agree, otherwise use the most permissive classification.

### 3. Add Tool Budget Adjustment Based on Progress

For long-running analysis tasks, dynamically adjust budget based on:
- Number of unique files read
- Progress indicators (graphs, searches)
- User "continue" signals

If the agent is actively making progress (reading new files, exploring new areas), increase budget instead of truncating.

### 4. Improve AgenticLoop Integration

The streaming path has its own iteration loop and is not yet integrated with AgenticLoop. Consider:
- Merging streaming and batch paths for consistent behavior
- Using AgenticLoop's PERCEIVE → PLAN → ACT → EVALUATE → DECIDE phases
- Sharing progress tracking and budget management logic

### 5. Add Classification Unit Tests

Add tests for edge cases:
```python
def test_structural_analysis_classification():
    """Verify structural analysis tasks are classified correctly."""
    classifier = TaskTypeClassifier.get_instance()
    result = classifier.classify_sync("framework structural analysis")
    assert result.task_type == TaskType.ANALYZE

def test_framework_analysis_classification():
    """Verify framework analysis tasks are classified correctly."""
    classifier = TaskTypeClassifier.get_instance()
    result = classifier.classify_sync("analyze the framework architecture")
    assert result.task_type == TaskType.ANALYZE
```

## Testing the Fixes

To verify the fixes work:

1. Restart the agent to pick up the new phrase patterns
2. Test with: "framework structural analysis"
3. Verify logs show:
   ```
   Task type classification: ..., is_analysis=True, ...
   Stream chat limits: tool_budget=200, ...
   ```

## Related Files Modified

- `victor/storage/embeddings/task_classifier.py` - Added structural analysis phrases
- `victor/agent/services/chat_stream_helpers.py` - Added fallback logic

## Impact

These changes should:
1. ✅ Correctly classify "framework structural analysis" as an analysis task
2. ✅ Allocate appropriate tool budget (200 instead of 10)
3. ✅ Prevent premature tool call truncation
4. ✅ Allow comprehensive codebase analysis tasks to complete successfully

---

Generated: 2025-04-30
Session: codingagent-ab9a4cd7
