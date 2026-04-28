# Context Compaction Strategies Analysis & Recommendations

**Date**: 2026-04-28
**Purpose**: Analyze compaction-related abrupt stops in Victor and recommend solutions

---

## Executive Summary

After analyzing research papers, Claude Code's implementation, and Victor's current system, the root cause of the "agent stops after compaction" issue appears to be **insufficient continuation prompting after aggressive compaction** combined with **lack of compaction context injection**. DeepSeek Chat may be more sensitive to context loss than other models.

**Key Finding**: Victor has sophisticated compaction logic but fails to provide adequate continuity signals after compaction, causing the model to lose track of the ongoing task.

---

## 1. Research Paper Findings

### 1.1 ContextBudget (Wu et al., 2604.01664)

**Core Contribution**: Budget-Aware Context Management (BACM)

| Aspect | Finding |
|--------|---------|
| **Problem** | Budget-free compaction causes over-compression (information loss) or under-compression (overflow) |
| **Solution** | Budget-conditioned state: `b_t = (s_t, r_t, \|o_t\|)` where `r_t` is remaining budget |
| **Key Mechanism** | Deferred observation loading - compress BEFORE loading new observation |
| **Actions** | NULL (skip), PARTIAL (selective aggregation), FULL (complete aggregation) |
| **Training** | GRPO with progressively tightened budget curriculum (8k → 4k tokens) |
| **Results** | 1.6× gains over baselines in high-complexity settings |

**Actionable Insight for Victor**:
- Add budget-aware signal to compaction decision (`remaining_tokens` parameter)
- Consider compression before new tool results, not after
- Progressive tightening: start conservative, get more aggressive

### 1.2 AgentSwing (Feng et al., 2603.27490)

**Core Contribution**: Adaptive Parallel Context Management Routing

| Aspect | Finding |
|--------|---------|
| **Problem** | Static context management strategies can't adapt to trajectory quality changes |
| **Framework** | Success = η (search efficiency) × ρ (terminal precision) |
| **Key Mechanism** | Parallel execution of multiple strategies, lookahead routing |
| **Strategies Compared** | No CM, Summary, Keep-Last-N, Discard-All, AgentSwing |
| **Results** | Up to 3× fewer turns while matching or exceeding static strategies |

**Actionable Insight for Victor**:
- Consider multiple compaction strategies and route between them
- Track "context rot" - larger contexts reduce terminal precision
- Discard-All (reset) can outperform gradual compaction in some cases

### 1.3 Building Effective AI Coding Agents (Bui, 2603.05344) - OpenDev

**Core Contribution**: Terminal-native agent architecture

| Aspect | Finding |
|--------|---------|
| **Context Strategy** | Adaptive context compaction - progressively reduces older observations |
| **Mechanism** | Compound AI system with workload-specialized model routing |
| **Key Feature** | Dual-agent architecture (planning vs execution separation) |
| **Memory System** | Automated memory accumulation across sessions |
| **Countermeasures** | Event-driven system reminders to counter instruction fade-out |

**Actionable Insight for Victor**:
- System reminders after compaction to maintain task continuity
- Consider dual-agent pattern for planning vs execution
- Cross-session memory for recurring patterns

### 1.4 Inside the Scaffold (Rombaut, 2604.03515)

**Core Contribution**: Source-code taxonomy of coding agent architectures

| Aspect | Finding |
|--------|---------|
| **Analysis** | 13 open-source coding agents analyzed at source level |
| **Compaction Spectrum** | Seven distinct compaction strategies identified |
| **Finding** | Context compaction is where designs diverge most |
| **Loop Primitives** | ReAct, generate-test-repair, plan-execute, multi-attempt retry, tree search |
| **Composability** | 11/13 agents compose multiple primitives |

**Compaction Strategies Identified**:
1. None (no compaction)
2. Truncation (keep last N)
3. Summarization (LLM-based)
4. Sliding window
5. Hierarchical
6. Selective pruning
7. Reset-based (Discard-All)

### 1.5 Codified Context (Vasilopoulos, 2602.20478)

**Core Contribution**: Three-tier codified context infrastructure

| Tier | Purpose | Size | Load Strategy |
|------|---------|------|---------------|
| **Tier 1: Constitution** | Hot memory - always loaded | ~660 lines | Every session |
| **Tier 2: Specialist Agents** | Domain experts | ~9,300 lines | Per task |
| **Tier 3: Knowledge Base** | Cold memory | ~16,250 lines | On demand |

**Key Metrics**:
- 283 development sessions
- 2,801 human prompts, 1,197 agent invocations
- 16,522 agent turns
- AGENTS.md presence = 29% reduction in runtime, 17% reduction in tokens

---

## 2. Claude Code's Compaction Approach

### 2.1 Implementation (src/query_engine.py)

```python
class QueryEngineConfig:
    max_turns: int = 8           # Hard limit on conversation length
    max_budget_tokens: int = 2000  # Total token budget
    compact_after_turns: int = 12  # When to trigger compaction
```

### 2.2 Compaction Logic

```python
def compact_messages_if_needed(self) -> None:
    if len(self.mutable_messages) > self.config.compact_after_turns:
        self.mutable_messages[:] = self.mutable_messages[-self.config.compact_after_turns:]
    self.transcript_store.compact(self.config.compact_after_turns)
```

**Key Characteristics**:
| Aspect | Claude Code | Victor |
|--------|-------------|--------|
| **Strategy** | Simple sliding window | Multi-strategy (tiered, semantic, hybrid) |
| **Trigger** | Fixed turn count (12) | Token-based (configurable threshold) |
| **Retention** | Keep last N messages | Scoring-based (importance, recency, semantics) |
| **Continuation** | Implied (last messages preserved) | Explicit strategy required |
| **Complexity** | O(1) | O(n log n) for sorting |

**Why Claude Code Works**:
- Simple, predictable behavior
- Last N messages always contain immediate context
- No complex state to manage after compaction
- Budget checked BEFORE each turn

---

## 3. Victor's Current Compaction Implementation

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ConversationController                    │
├─────────────────────────────────────────────────────────────┤
│  Compaction Strategies:                                      │
│  - SIMPLE:     Keep last N (like Claude Code)                │
│  - TIERED:     Score-based with tiered retention             │
│  - SEMANTIC:   Embedding-based relevance scoring             │
│  - HYBRID:     Combination of tiered + semantic              │
├─────────────────────────────────────────────────────────────┤
│  Supporting Components:                                      │
│  - CompactionRouter:     Strategy selection                  │
│  - HierarchicalManager:  Multi-level summaries              │
│  - ContextReminderManager: Post-compaction context injection │
│  - ContinuationStrategy: Post-compaction prompting          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Configuration (victor/config/compaction_settings.py)

```python
class CompactionSettings(BaseModel):
    compaction_enabled: bool = True
    compaction_preserve_recent: int = 4
    compaction_max_estimated_tokens: int = 10000
    compaction_auto_compact: bool = False

    # New hybrid settings
    strategy: Optional[CompactionStrategySettings] = None
    compaction_feature_flags: Optional[CompactionFeatureFlags] = None
    adaptive_threshold: Optional[AdaptiveCompactionSettings] = None
```

### 3.3 Scoring System (victor/agent/conversation/controller.py)

```python
def _score_messages(self, messages, current_query=None) -> List[MessageImportance]:
    # Scores based on:
    # 1. Role (system = 1000.0, pinned = 999.0)
    # 2. Recency (recent_message_weight)
    # 3. Tool results (tool_result_retention_weight)
    # 4. Semantic relevance (if embeddings available)
```

### 3.4 Identified Issues

| Issue | Location | Impact |
|-------|----------|--------|
| **Compaction context not injected** | `inject_compaction_context()` may not be called | Model loses track of task |
| **Continuation not triggered after compaction** | Streaming pipeline doesn't detect compaction | Agent stops |
| **Tool budget reduced** | 60 → 50 after compaction | May stop too early |
| **DeepSeek sensitivity** | Model-specific behavior unknown | May need special handling |
| **No explicit "compaction happened" signal** | Pipeline doesn't know | Can't prompt continuation |

---

## 4. Root Cause Analysis

### 4.1 The Stop Sequence (from console transcript)

```
ITER 13:
  - 4 tool calls executed
  - Compaction triggered: 64 messages removed (~52852 tokens freed)
  - Context reduced significantly

ITER 14:
  - 2 tool calls (graph_analytics, graph_query)
  - Tool budget updated: 60 → 50
  - Agent response: Short/no tool calls
  - Agent stops (no continuation)
```

### 4.2 Why DeepSeek Chat Stops

1. **Context Loss**: DeepSeek may be more sensitive to context removal than Claude/GPT
2. **No Continuation Signal**: After compaction, no "continue what you were doing" message
3. **Task Drift**: Original user prompt may have been removed or de-prioritized
4. **Tool Budget Exhaustion**: 50 remaining tools may signal "near end" to the model
5. **Missing System Prompt**: System prompt may not include "you are in a multi-turn task"

### 4.3 Comparison with Successful Models

| Model | Context Behavior | Compaction Response |
|-------|------------------|---------------------|
| **Claude 3.5** | Robust to context loss | Continues with partial context |
| **GPT-4** | Token-efficient | Resumes after summary |
| **DeepSeek Chat** | May need more context | Stops without explicit continuation |

---

## 5. Recommended Solutions

### 5.1 Immediate Fixes (High Priority)

#### Fix 1: Inject Compaction Summary After Compaction

**File**: `victor/agent/streaming/pipeline.py`

```python
# After compaction, inject a system message
if compaction_occurred:
    summary = self._conversation_controller.get_compaction_summaries()[-1]
    self._conversation_controller.add_assistant_message(
        content=f"[Context compacted. Previous work summary: {summary}]"
    )
```

#### Fix 2: Force Continuation Prompt After Compaction

**File**: `victor/agent/continuation_strategy.py`

```python
# Detect compaction and force continuation
if stream_ctx.compaction_occurred and not is_completion:
    return {
        "action": "prompt_tool_call",
        "message": (
            "Context was compacted to free space. "
            "You were in the middle of a task. "
            "Please continue with what you were doing - "
            "use tools to gather more information or complete your analysis."
        ),
        "reason": "Post-compaction continuation",
    }
```

#### Fix 3: Preserve User Intent Across Compaction

**File**: `victor/agent/conversation/controller.py`

```python
# Always keep the original user prompt
def smart_compact_history(self, ...):
    # Pin original user message with highest priority
    for i, msg in enumerate(self.messages):
        if msg.role == "user" and i == first_user_index:
            scored_messages.append(MessageImportance(
                msg, i, 2000.0, "original_user_intent"
            ))
```

#### Fix 4: Model-Specific Continuation Budgets

**File**: `victor/config/compaction_settings.py`

```python
# DeepSeek may need higher continuation budget after compaction
DEEPSEEK_CONTINUATION_BONUS = 3  # Add 3 extra continuation prompts

def get_continuation_budget(provider, model, base_budget):
    if "deepseek" in provider.lower() or "deepseek" in model.lower():
        return base_budget + DEEPSEEK_CONTINUATION_BONUS
    return base_budget
```

### 5.2 Medium-Term Improvements

#### Improvement 1: Budget-Aware Compaction (from ContextBudget paper)

```python
class BudgetAwareCompaction:
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.warning_threshold = total_budget * 0.7
        self.critical_threshold = total_budget * 0.9

    def should_compact(self, current_tokens: int, pending_tokens: int) -> bool:
        """Compact BEFORE adding pending observation."""
        projected = current_tokens + pending_tokens
        return projected > self.warning_threshold

    def get_compaction_level(self, remaining_tokens: int) -> str:
        """Return NULL, PARTIAL, or FULL based on remaining budget."""
        if remaining_tokens > self.total_budget * 0.5:
            return "NULL"  # Skip compaction
        elif remaining_tokens > self.total_budget * 0.2:
            return "PARTIAL"  # Selective aggregation
        else:
            return "FULL"  # Full aggregation
```

#### Improvement 2: Context Reminder System (from Codified Context paper)

```python
class ContextReminderManager:
    """Inject reminders after compaction to maintain task continuity."""

    REMINDER_TEMPLATES = {
        "analysis": "You were analyzing {files}. Continue your investigation.",
        "implementation": "You were implementing {feature}. Continue coding.",
        "debugging": "You were debugging {issue}. Continue troubleshooting.",
    }

    def get_reminder(self, task_type: str, context: dict) -> str:
        """Generate a context-aware reminder after compaction."""
        template = self.REMINDER_TEMPLATES.get(task_type, "Continue your task.")
        return template.format(**context)
```

#### Improvement 3: Adaptive Strategy Selection (from AgentSwing paper)

```python
class AdaptiveCompactionRouter:
    """Select compaction strategy based on trajectory quality."""

    def __init__(self):
        self.strategies = {
            "keep_last_n": KeepLastNStrategy(),
            "summarize": SummarizeStrategy(),
            "discard_all": DiscardAllStrategy(),
        }

    def select_strategy(self, trajectory_quality: float) -> str:
        """
        High quality → keep_last_n (preserve context)
        Medium quality → summarize (balance)
        Low quality → discard_all (reset and retry)
        """
        if trajectory_quality > 0.7:
            return "keep_last_n"
        elif trajectory_quality > 0.4:
            return "summarize"
        else:
            return "discard_all"
```

### 5.3 Long-Term Research Directions

1. **Learned Compaction Policies**: Use RL to learn when and how to compact (ContextBudget approach)
2. **Parallel Branching**: Try multiple compaction strategies in parallel, route to best (AgentSwing)
3. **Hierarchical Memory**: Multi-level summaries with tiered retrieval (Codified Context)
4. **Model-Specific Tuning**: Learn optimal compaction parameters per model

---

## 6. Implementation Priority Matrix

| Priority | Fix | Impact | Effort | Risk |
|----------|-----|--------|--------|------|
| **P0** | Inject compaction summary | High | Low | Low |
| **P0** | Force continuation after compaction | High | Low | Low |
| **P1** | Preserve original user intent | High | Medium | Low |
| **P1** | Model-specific continuation budgets | Medium | Low | Low |
| **P2** | Budget-aware compaction | High | High | Medium |
| **P2** | Context reminder system | Medium | Medium | Low |
| **P3** | Adaptive strategy selection | Medium | High | High |
| **P3** | Learned compaction policies | High | Very High | High |

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# Test compaction continuation
def test_compaction_triggers_continuation():
    controller = ConversationController()
    # Add messages until compaction
    for _ in range(100):
        controller.add_assistant_message(f"Message {_}")
    # Compact
    removed = controller.smart_compact_history(target_messages=10)
    assert removed > 0
    # Check continuation signal is set
    assert controller.compaction_occurred == True

# Test user intent preservation
def test_original_user_message_preserved():
    controller = ConversationController()
    user_msg = controller.add_user_message("Analyze this file")
    # Add many messages and compact
    for _ in range(100):
        controller.add_assistant_message(f"Message {_}")
    controller.smart_compact_history(target_messages=10)
    # Original user message should still be present
    messages = controller.messages
    assert any(m.content == "Analyze this file" for m in messages)
```

### 7.2 Integration Tests

```python
# Test DeepSeek Chat with compaction
async def test_deepseek_compaction_continuation():
    orchestrator = create_orchestrator(provider="deepseek", model="chat")
    # Create long conversation that triggers compaction
    result = await orchestrator.chat(
        "Analyze the codebase and create a findings table",
        max_turns=50
    )
    # Should complete without stopping early
    assert "findings table" in result.lower() or "analysis" in result.lower()
```

---

## 8. Configuration Recommendations

### 8.1 DeepSeek-Specific Settings

```yaml
# profiles/deepseek_compact.yaml
compaction:
  enabled: true
  strategy: "tiered"
  preserve_recent: 6  # Higher for DeepSeek (default 4)
  max_estimated_tokens: 12000  # Allow more before compaction

continuation:
  max_prompts_analysis: 9  # Higher default (6)
  max_prompts_action: 8  # Higher default (5)
  post_compaction_bonus: 3  # Extra prompts after compaction

context_reminder:
  enabled: true
  inject_after_compaction: true
  template: "Context was compacted. You were working on: {task_summary}. Please continue."
```

### 8.2 Conservative vs Aggressive Profiles

```yaml
# profiles/compaction_conservative.yaml
compaction:
  trigger_threshold: 0.80  # Compact at 80% of max
  preserve_recent: 10
  strategy: "keep_last_n"

# profiles/compaction_aggressive.yaml
compaction:
  trigger_threshold: 0.60  # Compact at 60% of max
  preserve_recent: 4
  strategy: "semantic"
```

---

## 9. Monitoring and Observability

### 9.1 Metrics to Track

```python
@dataclass
class CompactionMetrics:
    compaction_count: int
    total_tokens_freed: int
    messages_removed_per_compaction: List[int]
    post_compaction_continuation_success: float  # % of times agent continued
    model_specific_success_rates: Dict[str, float]
```

### 9.2 Logging

```python
logger.info(
    "Compaction occurred",
    extra={
        "iteration": iteration,
        "messages_before": before_count,
        "messages_after": after_count,
        "tokens_freed": tokens_freed,
        "strategy_used": strategy,
        "model": provider_model,
        "continuation_prompted": continuation_triggered,
    }
)
```

---

## 10. Conclusion

The root cause of DeepSeek Chat stopping after compaction is **insufficient continuity signaling** after aggressive context reduction. Victor has sophisticated compaction logic but lacks the post-compaction continuation mechanism that Claude Code's simpler approach implicitly provides.

**Recommended Action Plan**:
1. **Immediate**: Implement P0 fixes (summary injection, forced continuation)
2. **Short-term**: Add P1 fixes (user intent preservation, model-specific budgets)
3. **Medium-term**: Implement P2 improvements (budget-aware compaction, reminders)
4. **Long-term**: Research P3 features (learned policies, adaptive routing)

The research shows that **simple, predictable compaction with explicit continuation signals** outperforms complex strategies without proper continuity handling. Victor should combine its sophisticated scoring with Claude Code's simplicity: compact aggressively but always tell the model what it was doing.

---

## References

1. Wu et al. (2026). "ContextBudget: Budget-Aware Context Management for Long-Horizon Search Agents." arXiv:2604.01664
2. Feng et al. (2026). "AgentSwing: Adaptive Parallel Context Management Routing for Long-Horizon Web Agents." arXiv:2603.27490
3. Bui (2026). "Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned." arXiv:2603.05344
4. Rombaut (2026). "Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures." arXiv:2604.03515
5. Vasilopoulos (2026). "Codified Context: Infrastructure for AI Agents in a Complex Codebase." arXiv:2602.20478
