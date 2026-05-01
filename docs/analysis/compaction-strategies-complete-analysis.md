# Context Compaction Strategies: Complete Analysis & Recommendations for Victor

**Date**: 2026-04-28
**Purpose**: Analyze compaction-related abrupt stops in Victor and recommend solutions based on research papers and production systems

---

## Executive Summary

After analyzing 10 research papers (1809-4407 lines of full text), Claude Code's implementation, and Victor's current system, the root cause of DeepSeek Chat stopping after compaction is **insufficient continuation signaling after aggressive context reduction**.

**Key Finding**: Victor has sophisticated compaction logic but lacks the post-compaction continuation mechanism that simpler systems like Claude Code implicitly provide through their "last N messages" approach.

**Recommended Solution**: Implement explicit continuation prompts after compaction, preserve user intent, and use model-specific budgets.

---

## Part 1: Research Paper Findings (Complete Analysis)

### 1.1 ContextBudget (Wu et al., 2604.01664) - 1809 lines

**Core Contribution**: Budget-Aware Context Management (BACM)

| Aspect | Finding | Relevance to Victor |
|--------|---------|---------------------|
| **Problem Statement** | Budget-free compaction causes over-compression (information loss) or under-compression (overflow) | Victor's tiered compaction may over-compress |
| **Key Innovation** | **Deferred observation loading**: compress BEFORE loading new observation | Victor compacts AFTER - consider reversing |
| **Budget State** | `b_t = (s_t, r_t, \|o_t\|)` where `r_t = B - \|C_t\|` is remaining budget | Add explicit remaining_tokens signal |
| **Actions** | NULL (skip), PARTIAL (selective), FULL (complete aggregation) | Victor has similar but no budget conditioning |
| **Training** | GRPO with progressively tightened curriculum (8k → 4k) | Consider for learned compaction policies |
| **Results** | 1.6× gains in high-complexity, maintains advantage as budgets shrink | Strong validation for budget-aware approach |

**Key Quote**: "Budget-aware context management provides a promising direction for enabling robust long-horizon reasoning under strict resource constraints."

**Actionable Insights**:
1. Add `remaining_budget` to compaction decision state
2. Compact BEFORE new tool results, not after
3. Progressive tightening: start conservative, get more aggressive
4. Use budget to select NULL/PARTIAL/FULL compression level

---

### 1.2 AgentSwing (Feng et al., 2603.27490) - 2322 lines

**Core Contribution**: Adaptive Parallel Context Management Routing

| Aspect | Finding | Relevance to Victor |
|--------|---------|---------------------|
| **Problem** | Static context management can't adapt to trajectory quality | Victor uses single strategy - consider routing |
| **Framework** | Success = η (search efficiency) × ρ (terminal precision) | New way to measure compaction success |
| **Key Innovation** | Parallel execution of multiple strategies + lookahead routing | Could try multiple strategies in parallel |
| **Strategies Compared** | No CM, Summary, Keep-Last-N, Discard-All, AgentSwing | Victor has all except Discard-All |
| **Results** | Up to 3× fewer turns, matches/exceeds static with less budget | Adaptive routing beats static strategies |
| **DeepSeek-v3.2 Results** | 71.3 on BrowseComp-ZH with AgentSwing (vs 61.6 baseline) | DeepSeek benefits significantly from good CM |

**Context Rot Phenomenon** (Figure 2):
```
Aligned Terminal Precision decreases as Context Budget increases:
- 25.6k budget: ~90% precision
- 51.2k budget: ~80% precision
- 102.4k budget: ~70% precision
```

**Key Quote**: "Larger working contexts lead to more severe context rot at termination."

**Strategy Transition Probabilities** (Figure 9):
- GPT-OSS-120B favors: Discard-All (0.55) > Keep-Last-N (0.25) > Summary (0.20)
- DeepSeek-v3.2 favors: Summary (0.49) > Keep-Last-N (0.27) > Discard-All (0.24)
- Tongyi-DR favors: Keep-Last-N (0.58) > Summary (0.28) > Discard-All (0.14)

**Actionable Insights**:
1. **Model-specific strategy preferences**: DeepSeek prefers Summary over Discard-All
2. Adaptive routing based on trajectory quality outperforms any single strategy
3. Consider implementing Discard-All as an option for "too much noise" scenarios
4. Track both efficiency (can I continue?) and precision (am I correct?)

---

### 1.3 OpenDev (Bui, 2603.05344) - 4407 lines

**Core Contribution**: Terminal-native agent with adaptive context compaction

| Aspect | Finding | Relevance to Victor |
|--------|---------|---------------------|
| **Context Strategy** | Adaptive context compaction - progressively reduces older observations | Similar to Victor's tiered approach |
| **Mechanism** | Compound AI system with workload-specialized model routing | Victor has similar multi-model routing |
| **Key Feature** | Dual-agent architecture (planning vs execution separation) | Consider for Victor's architecture |
| **Memory System** | Automated memory accumulation across sessions | Counteracts instruction fade-out |
| **Countermeasures** | Event-driven system reminders to counter instruction fade-out | **Critical for post-compaction continuity** |

**Key Design Tension** (Section 3.1): "Context Pressure as the Central Design Constraint"
- Tool outputs consume 70-80% of context in typical sessions
- Context utilization is the single most important metric for agent longevity
- "Treat context as a budget, not a buffer"

**Critical Lesson** (Section 3.2): "Inject reminders at the point of decision, not upfront"
- Short, targeted reminders at maximum recency are more effective than long system prompt sections
- Use `role: user` rather than `role: system` for reminders (higher salience)
- Reminder frequency must be capped per type to avoid becoming background noise

**Fast Pruning** (Section 2.3.6):
```python
# Walk backwards through tool results, replace with [pruned] marker
for result in reversed(tool_results):
    if beyond_working_horizon(result):
        result.content = "[pruned]  # Original was {len(result)} chars"
```

**Actionable Insights**:
1. **User-role reminders** after compaction (not system role)
2. Fast pruning before expensive LLM compaction
3. Separate planning from execution (dual-agent pattern)
4. Calibrate from API-reported token counts, not local estimates

---

### 1.4 Inside the Scaffold (Rombaut, 2604.03515) - 2368 lines

**Core Contribution**: Source-code taxonomy of 13 coding agent architectures

| Aspect | Finding | Relevance to Victor |
|--------|---------|---------------------|
| **Compaction Spectrum** | Seven distinct strategies identified | Victor implements most |
| **Finding** | Context compaction is where designs diverge most | Confirms importance of this area |
| **Loop Primitives** | ReAct, generate-test-repair, plan-execute, multi-attempt retry, tree search | Victor uses ReAct + planning |
| **Composability** | 11/13 agents compose multiple primitives | Victor's multi-strategy approach is validated |

**Seven Compaction Strategies Identified**:
1. None (no compaction)
2. Truncation (keep last N) - Claude Code uses this
3. Summarization (LLM-based)
4. Sliding window
5. Hierarchical - Victor has this
6. Selective pruning - Victor has scoring-based
7. Reset-based (Discard-All)

**Key Finding**: "Context compaction spans seven distinct strategies" and "designs diverge most" in this area.

**Actionable Insights**:
1. Victor's multi-strategy approach is research-aligned
2. Consider adding Discard-All as emergency option
3. Composition of primitives is the norm, not exception

---

### 1.5 Codified Context (Vasilopoulos, 2602.20478) - 1050 lines

**Core Contribution**: Three-tier codified context infrastructure

| Tier | Purpose | Size | Load Strategy | Victor Equivalent |
|------|---------|------|---------------|-------------------|
| **Tier 1: Constitution** | Hot memory - always loaded | ~660 lines | Every session | System prompt |
| **Tier 2: Specialist Agents** | Domain experts | ~9,300 lines | Per task | Victor's subagents? |
| **Tier 3: Knowledge Base** | Cold memory | ~16,250 lines | On demand | Victor's retrieval? |

**Key Metrics**:
- 283 development sessions
- AGENTS.md presence = 29% reduction in runtime, 17% reduction in tokens
- "Brevity bias" - iterative optimization collapses toward short, generic prompts

**Actionable Insights**:
1. Three-tier memory: hot (always), warm (per-task), cold (on-demand)
2. Specialist agents require substantial embedded domain knowledge (>50%)
3. AGENTS.md files provide measurable token savings

---

### 1.6 The Pensieve Paradigm (Liu et al., 2602.12108) - 1852 lines

**Core Contribution**: StateLM - models with internal reasoning loop to manage their own state

| Aspect | Finding | Relevance to Victor |
|--------|---------|---------------------|
| **Memory Tools** | Context pruning, document indexing, note-taking | Victor has similar tools |
| **Training** | Model learns to actively manage memory tools | Future direction for Victor |
| **Results** | 10-20% absolute accuracy improvement on chat memory | Up to 52% vs 5% on BrowseComp-Plus |

**Key Innovation**: Models actively manage their own context through tools rather than passively receiving engineered context.

**Actionable Insights**:
1. Consider training models to manage their own compaction
2. Tool-based context management is promising direction

---

### 1.7 Additional Papers (Summarized)

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| ACE-Bench | Agent Configurable Evaluation with scalable horizons | Test compaction under various horizons |
| Natural-Language Agent Harnesses | NLAH - harnesses for long-horizon agents | Consider for testing |
| Dynamic Attentional Context Scoping | Agent-triggered focus sessions for multi-agent | Relevant for Victor's teams |
| From Static Templates to Dynamic Runtime Graphs | Workflow optimization survey | Victor's workflows could benefit |

---

## Part 2: Production System Analysis

### 2.1 Claude Code (Simple & Effective)

```python
class QueryEngineConfig:
    max_turns: int = 8
    max_budget_tokens: int = 2000
    compact_after_turns: int = 12

def compact_messages_if_needed(self):
    if len(self.messages) > self.compact_after_turns:
        self.messages[:] = self.messages[-self.compact_after_turns:]
```

**Why It Works**:
- Last N messages always contain immediate context
- No complex state to manage after compaction
- Budget checked BEFORE each turn
- Implicit continuation (model sees recent conversation)

### 2.2 Victor (Sophisticated but Complex)

```python
class ConversationController:
    def smart_compact_history(self, target_messages, current_query, task_type):
        # Score messages by:
        # - Role (system=1000.0, pinned=999.0)
        # - Recency (recent_message_weight)
        # - Tool results (tool_result_retention_weight)
        # - Semantic relevance (if embeddings available)
        scored = self._score_messages(messages, current_query)
        # Keep top N, preserve tool-call pairs, ensure USER/ASSISTANT present
```

**Strengths**:
- Multi-strategy (SIMPLE, TIERED, SEMANTIC, HYBRID)
- Scoring-based importance
- Tool-pair preservation
- Hierarchical summaries

**Weaknesses**:
- No explicit continuation after compaction
- Compaction context may not be injected
- User intent may be lost
- No model-specific handling

---

## Part 3: Root Cause Analysis - Why DeepSeek Chat Stops

### 3.1 The Stop Sequence (from console transcript)

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

### 3.2 Root Causes

| Cause | Evidence | Fix Priority |
|-------|----------|--------------|
| **No continuation signal** | Agent doesn't know to continue after compaction | P0 |
| **User intent lost** | Original prompt may have been removed/de-prioritized | P0 |
| **No compaction summary injected** | Model doesn't know what was removed | P0 |
| **Tool budget reduced** | 60 → 50 signals "near end" | P1 |
| **DeepSeek sensitivity** | Research shows DeepSeek prefers Summary, may need more context | P1 |
| **System prompt decay** | OpenDev shows instructions decay after ~30 turns | P1 |

### 3.3 Why Claude Code Works But Victor Doesn't

| Aspect | Claude Code | Victor | Issue |
|--------|-------------|-------|-------|
| Continuation | Implicit (last messages preserve context) | None required | Model stops |
| User Intent | Always in last N messages | May be removed | Model forgets task |
| Complexity | O(1) simple slice | O(n log n) scoring | More failure modes |
| Model Awareness | None (works for all models) | Some (incomplete) | DeepSeek not handled |

---

## Part 4: Comprehensive Recommendations

### 4.1 Immediate Fixes (P0 - Implement Now)

#### Fix 1: Inject Compaction Summary After Compaction

**File**: `victor/agent/conversation/controller.py`

```python
def smart_compact_history(self, ...):
    # ... existing compaction logic ...

    # NEW: Inject compaction summary immediately
    if removed_count > 0:
        summary = self._get_last_compaction_summary()
        reminder_msg = Message(
            role="user",  # User role for higher salience (OpenDev finding)
            content=f"[Context was compacted to free space. Previous work: {summary}]"
        )
        self._history._messages.insert(1, reminder_msg)  # After system prompt
```

#### Fix 2: Force Continuation Prompt After Compaction

**File**: `victor/agent/streaming/pipeline.py` or `victor/agent/continuation_strategy.py`

```python
# Add compaction detection
if hasattr(stream_ctx, 'compaction_occurred') and stream_ctx.compaction_occurred:
    if not is_completion and not is_asking_input:
        return {
            "action": "prompt_tool_call",
            "message": (
                "Context was compacted to continue the session. "
                "You were in the middle of a task. "
                "Please continue with what you were doing - "
                "use tools to gather more information or complete your analysis."
            ),
            "reason": "Post-compaction continuation",
        }
```

#### Fix 3: Always Preserve Original User Intent

**File**: `victor/agent/conversation/controller.py`

```python
def _score_messages(self, messages, current_query=None):
    scored = []
    original_user_msg = None

    for i, msg in enumerate(messages):
        # Find and pin the original user message
        if msg.role == "user" and original_user_msg is None:
            original_user_msg = msg
            scored.append(MessageImportance(
                msg, i, 2000.0, "original_user_intent"  # Highest priority
            ))
        elif msg == original_user_msg:
            scored.append(MessageImportance(msg, i, 2000.0, "original_user_intent"))
        # ... rest of scoring ...
```

#### Fix 4: Track Compaction State in Stream Context

**File**: `victor/agent/streaming/context.py`

```python
@dataclass
class StreamingContext:
    # ... existing fields ...
    compaction_occurred: bool = False
    compaction_summary: str = ""
    last_compaction_turn: int = -1
```

### 4.2 Short-Term Improvements (P1)

#### Improvement 1: Model-Specific Continuation Budgets

**File**: `victor/config/compaction_settings.py`

```python
# DeepSeek may need higher continuation budget after compaction
MODEL_CONTINUATION_BONUS = {
    "deepseek": 3,      # Add 3 extra continuation prompts
    "deepseek-chat": 3,
    "deepseek-coder": 2,
}

def get_continuation_budget(provider, model, base_budget, compaction_occurred):
    bonus = 0
    if compaction_occurred:
        for model_key, bonus_val in MODEL_CONTINUATION_BONUS.items():
            if model_key in provider.lower() or model_key in model.lower():
                bonus = bonus_val
                break
    return base_budget + bonus
```

#### Improvement 2: Budget-Aware Compaction (from ContextBudget)

**File**: `victor/agent/conversation/controller.py`

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
        """NULL, PARTIAL, or FULL based on remaining budget."""
        if remaining_tokens > self.total_budget * 0.5:
            return "NULL"  # Skip
        elif remaining_tokens > self.total_budget * 0.2:
            return "PARTIAL"  # Selective
        else:
            return "FULL"  # Aggressive
```

#### Improvement 3: Context Reminder System (from OpenDev)

**File**: `victor/agent/context_reminder.py` (new)

```python
class ContextReminderManager:
    """Inject reminders after compaction to maintain task continuity."""

    REMINDER_TEMPLATES = {
        "analysis": "You were analyzing {files}. Continue your investigation.",
        "implementation": "You were implementing {feature}. Continue coding.",
        "debugging": "You were debugging {issue}. Continue troubleshooting.",
    }

    def get_post_compaction_reminder(
        self,
        task_type: str,
        context: dict
    ) -> str:
        """Generate a context-aware reminder after compaction."""
        template = self.REMINDER_TEMPLATES.get(task_type, "Continue your task.")
        return template.format(**context)
```

### 4.3 Medium-Term Improvements (P2)

#### Improvement 1: Adaptive Strategy Selection (from AgentSwing)

```python
class AdaptiveCompactionRouter:
    """Select compaction strategy based on trajectory quality."""

    def __init__(self):
        self.strategies = {
            "keep_last_n": KeepLastNStrategy(),
            "summarize": SummarizeStrategy(),
            "discard_all": DiscardAllStrategy(),
        }

    def select_strategy(
        self,
        trajectory_quality: float,
        model: str
    ) -> str:
        """
        High quality → keep_last_n (preserve context)
        Medium quality → summarize (balance)
        Low quality → discard_all (reset and retry)

        Model-specific preferences (from AgentSwing):
        - DeepSeek: prefers Summary
        - GPT: prefers Discard-All
        - Others: prefer Keep-Last-N
        """
        if trajectory_quality > 0.7:
            return "keep_last_n"
        elif trajectory_quality > 0.4:
            # Model-specific routing
            if "deepseek" in model.lower():
                return "summarize"
            return "keep_last_n"
        else:
            return "discard_all"
```

#### Improvement 2: Fast Pruning Before LLM Compaction (from OpenDev)

```python
def fast_prune_tool_results(controller, max_age: int = 5):
    """Prune old tool results before expensive LLM compaction."""
    pruned_count = 0
    for msg in reversed(controller._history._messages):
        if msg.role == "tool" and msg.tool_call_id:
            # Find how many turns ago this was
            age = controller.message_count - msg.index
            if age > max_age:
                msg.content = f"[pruned]  # Original was {len(msg.content)} chars"
                pruned_count += 1
    return pruned_count
```

### 4.4 Long-Term Research Directions (P3)

1. **Learned Compaction Policies**: Use RL to learn when and how to compact (ContextBudget approach)
2. **Parallel Branching**: Try multiple compaction strategies, route to best (AgentSwing)
3. **Hierarchical Memory**: Multi-level summaries with tiered retrieval (Codified Context)
4. **Model-Specific Tuning**: Learn optimal compaction parameters per model
5. **StateLM Integration**: Models that actively manage their own context (Pensieve)

---

## Part 5: Configuration Recommendations

### 5.1 DeepSeek-Specific Profile

```yaml
# profiles/deepseek_compact.yaml
compaction:
  enabled: true
  strategy: "summary"  # DeepSeek prefers summary per AgentSwing
  preserve_recent: 8  # Higher for DeepSeek (default 4)
  max_estimated_tokens: 12000  # Allow more before compaction
  trigger_threshold: 0.70  # Compact at 70% (more conservative)

continuation:
  max_prompts_analysis: 9  # Higher default (6)
  max_prompts_action: 8  # Higher default (5)
  post_compaction_bonus: 3  # Extra prompts after compaction
  force_after_compaction: true

context_reminder:
  enabled: true
  inject_after_compaction: true
  role: "user"  # User role for higher salience
  template: "Context was compacted. You were working on: {task_summary}. Please continue."
```

### 5.2 Conservative vs Aggressive Profiles

```yaml
# profiles/compaction_conservative.yaml
compaction:
  trigger_threshold: 0.80  # Compact at 80%
  preserve_recent: 12  # Keep more messages
  strategy: "keep_last_n"  # Simple, predictable

# profiles/compaction_aggressive.yaml
compaction:
  trigger_threshold: 0.60  # Compact at 60%
  preserve_recent: 4
  strategy: "semantic"  # Smartest but most complex
```

---

## Part 6: Implementation Priority Matrix

| Priority | Fix | Impact | Effort | Risk | Dependencies |
|----------|-----|--------|--------|------|---------------|
| **P0** | Inject compaction summary after compaction | High | Low | Low | None |
| **P0** | Force continuation prompt after compaction | High | Low | Low | None |
| **P0** | Always preserve original user message | High | Medium | Low | None |
| **P0** | Track compaction state in stream context | High | Low | Low | None |
| **P1** | Model-specific continuation budgets | Medium | Low | Low | None |
| **P1** | Budget-aware compaction (deferred loading) | High | High | Medium | Settings update |
| **P1** | Context reminder system | Medium | Medium | Low | None |
| **P1** | Fast pruning before LLM compaction | Medium | Low | Low | None |
| **P2** | Adaptive strategy selection | Medium | High | High | New router |
| **P2** | Discard-All emergency option | Low | Medium | Low | None |
| **P3** | Learned compaction policies | High | Very High | High | RL infrastructure |
| **P3** | Parallel branching (AgentSwing) | Medium | Very High | High | Major refactoring |

---

## Part 7: Testing Strategy

### 7.1 Unit Tests

```python
# Test compaction continuation
def test_compaction_triggers_continuation():
    controller = ConversationController()
    for _ in range(100):
        controller.add_assistant_message(f"Message {_}")
    removed = controller.smart_compact_history(target_messages=10)
    assert removed > 0
    assert controller.compaction_occurred == True
    assert controller.get_compaction_summaries()  # Summary available

# Test user intent preservation
def test_original_user_message_preserved():
    controller = ConversationController()
    user_msg = controller.add_user_message("Analyze this file")
    for _ in range(100):
        controller.add_assistant_message(f"Message {_}")
    controller.smart_compact_history(target_messages=10)
    messages = controller.messages
    assert any(m.content == "Analyze this file" for m in messages)

# Test DeepSeek-specific continuation budget
def test_deepseek_extra_continuation():
    budget = get_continuation_budget(
        provider="deepseek",
        model="chat",
        base_budget=6,
        compaction_occurred=True
    )
    assert budget == 9  # 6 + 3 bonus
```

### 7.2 Integration Tests

```python
async def test_deepseek_compaction_continuation():
    orchestrator = create_orchestrator(provider="deepseek", model="chat")
    result = await orchestrator.chat(
        "Analyze the codebase and create a findings table",
        max_turns=50
    )
    # Should complete without stopping early
    assert "findings table" in result.lower() or "analysis" in result.lower()
```

---

## Part 8: Key Insights Summary

### 8.1 From Research Papers

1. **Budget-awareness matters** (ContextBudget): Conditioning on remaining budget outperforms static strategies
2. **Model preferences differ** (AgentSwing): DeepSeek prefers Summary, GPT prefers Discard-All
3. **Context rot is real** (AgentSwing): Larger contexts reduce terminal precision
4. **Reminders at decision point** (OpenDev): User-role reminders are more effective than system-role
5. **Progressive compression** (ContextBudget): Start conservative, get more aggressive

### 8.2 From Production Systems

1. **Claude Code**: Simplicity wins - last N messages always contains context
2. **OpenDev**: Fast pruning before expensive LLM compaction
3. **AgentSwing**: Adaptive routing beats any single static strategy

### 8.3 For Victor

1. **Current strength**: Sophisticated multi-strategy compaction is research-aligned
2. **Current weakness**: No explicit continuation after compaction
3. **Quick wins**: Add continuation prompts and preserve user intent
4. **Long-term**: Consider adaptive routing and model-specific handling

---

## Part 9: Recommended Action Plan

### Phase 1: Immediate (This Week)
1. Implement compaction summary injection
2. Add forced continuation after compaction
3. Always preserve original user message
4. Track compaction state in stream context

### Phase 2: Short-term (This Month)
1. Add model-specific continuation budgets
2. Implement fast pruning before LLM compaction
3. Add context reminder system
4. Create DeepSeek-specific profile

### Phase 3: Medium-term (This Quarter)
1. Implement budget-aware compaction with deferred loading
2. Add adaptive strategy selection
3. Implement Discard-All as emergency option
4. Add comprehensive testing

### Phase 4: Long-term (Future)
1. Research learned compaction policies
2. Explore parallel branching (AgentSwing)
3. Consider StateLM-style active context management

---

## Part 10: Conclusion

The root cause of DeepSeek Chat stopping after compaction is **insufficient continuity signaling** after aggressive context reduction. Victor has sophisticated compaction logic but lacks the post-compaction continuation mechanism that simpler systems provide implicitly.

**The key insight from all research**: Simple, predictable compaction with explicit continuation signals outperforms complex strategies without proper continuity handling.

Victor should combine its sophisticated scoring with Claude Code's simplicity: compact aggressively but **always tell the model what it was doing**.

**Recommended immediate action**: Implement the P0 fixes (summary injection, forced continuation, user intent preservation, compaction tracking) to resolve the DeepSeek stopping issue.

---

## References

1. Wu et al. (2026). "ContextBudget: Budget-Aware Context Management for Long-Horizon Search Agents." arXiv:2604.01664
2. Feng et al. (2026). "AgentSwing: Adaptive Parallel Context Management Routing for Long-Horizon Web Agents." arXiv:2603.27490
3. Bui (2026). "Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned." arXiv:2603.05344
4. Rombaut (2026). "Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures." arXiv:2604.03515
5. Vasilopoulos (2026). "Codified Context: Infrastructure for AI Agents in a Complex Codebase." arXiv:2602.20478
6. Liu et al. (2026). "The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context." arXiv:2602.12108
