# Victor Coding Agent - DEMERITS Improvement Plan

## Executive Summary

This document outlines a comprehensive improvement plan to address the design demerits identified in Victor's self-analysis. The plan prioritizes fixes by impact, feasibility, and risk.

## Identified DEMERITS (from Self-Analysis)

### 1. Tool Broadcasting Fallback Issue
**Problem**: When semantic selection returns 0 tools, Victor broadcasts all 25+ tools to the LLM, defeating the purpose of intelligent selection and overwhelming small models.

**Impact**: HIGH - Wastes tokens, increases latency, confuses smaller models
**Feasibility**: HIGH - Simple fallback logic change
**Risk**: LOW - Easy to test and verify

**Current Code**: `victor/tools/semantic_selector.py`

### 2. Limited Tool Result Caching
**Problem**: No caching system for tool results, leading to redundant execution of similar operations (e.g., reading the same file multiple times).

**Impact**: HIGH - Reduces API calls, improves performance
**Feasibility**: MEDIUM - Requires cache key design and invalidation strategy
**Risk**: LOW - Can be disabled if issues arise

### 3. Cost-Aware Tool Selection Missing
**Problem**: No consideration of API costs when selecting tools, potentially leading to expensive operations without user awareness.

**Impact**: MEDIUM - Prevents unexpected costs, better UX
**Feasibility**: MEDIUM - Requires cost modeling per tool
**Risk**: LOW - Additive feature, won't break existing functionality

### 4. Stage-Based Pruning Fragility
**Problem**: Conversation stage detection is fragile and requires manual management rather than automatic detection.

**Impact**: MEDIUM - Better tool selection accuracy
**Feasibility**: MEDIUM - Requires better heuristics
**Risk**: MEDIUM - Changes tool selection behavior

**Current Code**: `victor/agent/orchestrator.py:953`

### 5. No Dynamic Tool Loading
**Problem**: All 25+ tools loaded at startup with no lazy loading or plugin system.

**Impact**: LOW - Marginal performance improvement
**Feasibility**: LOW - Requires significant refactoring
**Risk**: HIGH - Major architectural change

### 6. Incomplete MCP Integration
**Problem**: MCP server implemented but not automatically exposed to LLMs like Ollama, limiting usefulness.

**Impact**: LOW - Niche use case
**Feasibility**: LOW - Complex integration
**Risk**: MEDIUM - Requires provider changes

### 7. Manual MCP Configuration
**Problem**: MCP client requires manual configuration in settings, with no auto-discovery capabilities.

**Impact**: LOW - UX improvement for MCP users
**Feasibility**: MEDIUM - Requires discovery protocol
**Risk**: MEDIUM - Network operations, security concerns

---

## Prioritized Implementation Plan

### Phase 1: High-Impact, Low-Risk Fixes (Implement Now)

#### Fix 1.1: Tool Broadcasting Fallback Improvement
**Priority**: P0
**Estimated Effort**: 1-2 hours
**Files**: `victor/tools/semantic_selector.py`

**Current Behavior**:
```python
if not relevant_tools:
    # Fall back to all tools
    return list(self.tool_registry.tools.keys())
```

**Proposed Solution**:
```python
if not relevant_tools:
    # Option A: Use keyword-based fallback
    relevant_tools = self._fallback_to_keywords(query, threshold, max_tools)

    # Option B: Return most commonly used tools
    if not relevant_tools:
        return self._get_common_tools(max_tools=5)
```

**Benefits**:
- Reduces token waste from broadcasting all tools
- Improves performance for small models
- Maintains fallback safety net

#### Fix 1.2: Tool Result Caching System
**Priority**: P0
**Estimated Effort**: 3-4 hours
**Files**: `victor/agent/orchestrator.py`, new `victor/cache/tool_cache.py`

**Implementation**:
1. Create simple in-memory LRU cache for tool results
2. Cache key: hash(tool_name + sorted_args_json)
3. TTL: 60 seconds (configurable)
4. Max size: 100 entries
5. Invalidation: automatic (LRU + TTL)

**Benefits**:
- Eliminates redundant file reads
- Reduces API calls for web tools
- Improves response time

### Phase 2: Medium-Impact Enhancements (Next Iteration)

#### Fix 2.1: Cost-Aware Tool Selection
**Priority**: P1
**Estimated Effort**: 4-6 hours
**Files**: `victor/tools/base.py`, `victor/tools/semantic_selector.py`

**Implementation**:
1. Add `cost_estimate` metadata to tool definitions
2. Track cumulative costs per session
3. Warn when approaching cost thresholds
4. Deprioritize expensive tools when cheaper alternatives exist

**Cost Tiers**:
- **Free**: filesystem, bash, git (local operations)
- **Low**: code_review, refactor (compute only)
- **Medium**: web_search, web_fetch (API calls)
- **High**: batch_processor with 100+ files

#### Fix 2.2: Improved Stage Detection
**Priority**: P1
**Estimated Effort**: 3-4 hours
**Files**: `victor/agent/orchestrator.py`

**Current Logic**:
```python
stage = "initial" if first_pass else ("post_read" if self.observed_files else "post_plan")
```

**Proposed Improvements**:
- Detect planning stage from tool_calls (plan_files, code_search)
- Detect reading stage from observed_files count
- Detect execution stage from executed_tools (execute_bash, file_editor)
- Use conversation depth and message history

### Phase 3: Architectural Improvements (Future Work)

#### Fix 3.1: Dynamic Tool Loading (Plugin System)
**Priority**: P2
**Estimated Effort**: 2-3 days
**Risk**: HIGH

**Approach**:
1. Lazy load tool modules on first use
2. Plugin discovery from `~/.victor/plugins/`
3. Tool dependency resolution
4. Hot reload capability

#### Fix 3.2: MCP Auto-Exposure
**Priority**: P3
**Estimated Effort**: 1-2 days

**Implementation**:
- Auto-start MCP server when Ollama provider detected
- Expose tools via MCP protocol
- Auto-configure client discovery

---

## Implementation Strategy

### Test-Driven Development
All fixes will follow TDD approach:
1. Write failing tests first
2. Implement minimal fix
3. Verify tests pass
4. Refactor if needed
5. Update documentation

### Metrics for Success

**Fix 1.1 (Tool Broadcasting)**:
- ✅ Zero fallbacks to "all tools" in test suite
- ✅ Average tools selected ≤ 7 per query
- ✅ 90%+ queries use semantic or keyword selection

**Fix 1.2 (Tool Caching)**:
- ✅ Cache hit rate ≥ 30% in typical sessions
- ✅ 50%+ reduction in redundant file reads
- ✅ No cache-related bugs in test suite

**Fix 2.1 (Cost Awareness)**:
- ✅ Cost estimates displayed for expensive tools
- ✅ Warning shown when cost threshold exceeded
- ✅ User opt-in for high-cost operations

**Fix 2.2 (Stage Detection)**:
- ✅ Correct stage detection ≥ 80% of time
- ✅ Improved tool selection relevance
- ✅ Fewer unnecessary tools loaded

### Rollout Plan

1. **Week 1**: Implement and test Phase 1 fixes (1.1, 1.2)
2. **Week 2**: Deploy Phase 1, monitor metrics
3. **Week 3**: Implement Phase 2 fixes (2.1, 2.2)
4. **Week 4**: Deploy Phase 2, gather feedback
5. **Month 2+**: Evaluate Phase 3 based on user feedback

---

## Risk Mitigation

### Backwards Compatibility
- All fixes are additive or use feature flags
- Existing behavior preserved with `use_legacy_fallback=True` setting
- Gradual rollout with monitoring

### Testing Strategy
- Unit tests for each component
- Integration tests for end-to-end flows
- Performance benchmarks before/after
- A/B testing for tool selection changes

### Rollback Plan
- Feature flags for each fix
- Ability to disable cache if issues arise
- Logging for debugging tool selection
- Metrics dashboard for monitoring

---

## Next Steps

1. ✅ Review and approve improvement plan
2. ⏳ Implement Fix 1.1 (Tool Broadcasting Fallback)
3. ⏳ Implement Fix 1.2 (Tool Result Caching)
4. ⏳ Create comprehensive test suite
5. ⏳ Update documentation
6. ⏳ Deploy and monitor

---

## Conclusion

This improvement plan addresses the most impactful design demerits while minimizing risk and maintaining backwards compatibility. Phase 1 fixes provide immediate value with minimal effort, while Phase 2 and 3 build on that foundation for long-term architectural improvements.

**Estimated Total Effort (Phase 1 + 2)**: 15-20 hours
**Expected Impact**: 40-50% improvement in tool selection efficiency, 30%+ cache hit rate, better cost awareness
