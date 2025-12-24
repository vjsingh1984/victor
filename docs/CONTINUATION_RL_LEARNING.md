# Continuation Prompt RL Learning

Automated reinforcement learning system for optimizing continuation prompt limits per provider/model combination.

## Overview

Different LLM models have different continuation behaviors when performing multi-step tasks:
- **Some models** need many continuation prompts to complete thorough analysis
- **Other models** get stuck in narration loops and need fewer prompts with faster forcing
- **Quality vs Speed tradeoff** varies by model architecture and training

This RL system learns optimal continuation prompt limits automatically based on:
- Success rate vs continuation prompts used
- Quality score (from grounding verifier / user feedback)
- Stuck loop frequency
- Forced completion rate

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  AgentOrchestrator (During Session)                              │
├──────────────────────────────────────────────────────────────────┤
│  1. Check RL recommendations for provider:model:task_type        │
│  2. Apply learned limits (or use defaults)                       │
│  3. Track continuation prompts used                              │
│  4. Detect stuck loops, forced completions                       │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│  Session Completion                                              │
├──────────────────────────────────────────────────────────────────┤
│  _record_intelligent_outcome() → ContinuationPromptLearner       │
│  Records: prompts_used, quality, success, stuck_loop, forced     │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│  ContinuationPromptLearner                                       │
├──────────────────────────────────────────────────────────────────┤
│  1. Update statistics (weighted moving average)                  │
│  2. Recompute recommendation based on:                           │
│     - High stuck rate → Decrease prompts                         │
│     - Low quality + few prompts → Increase prompts               │
│     - High quality + many prompts → Decrease prompts             │
│  3. Persist to ~/.victor/rl_data/continuation_rl.json            │
└──────────────────────────────────────────────────────────────────┘
```

## Enabling RL Learning

### 1. Enable in Settings

```python
# In victor/config/settings.py (or override in profiles.yaml)
enable_continuation_rl_learning: bool = True
```

Or in `~/.victor/profiles.yaml`:
```yaml
enable_continuation_rl_learning: true
```

### 2. Run Sessions

Just use Victor normally! The system automatically:
- Records outcomes after each session
- Updates statistics and recommendations
- Persists learned data to disk

### 3. Monitor Learning

```bash
# Show current RL status and recommendations
python scripts/show_continuation_rl.py

# Filter by specific provider/model
python scripts/show_continuation_rl.py --provider ollama --model qwen3-coder-tools:30b-128k

# Export recommendations to YAML for manual application
python scripts/show_continuation_rl.py --export recommendations.yaml
```

## Learning Strategy

### Multi-Armed Bandit with Epsilon-Greedy Exploration

The system uses a **weighted moving average** approach:

```python
# Recent observations weighted more heavily (decay = 0.9)
avg_quality = (0.9 * old_avg + new_quality) / (1 + (sessions - 1) * 0.9)
avg_prompts = (0.9 * old_avg + new_prompts) / (1 + (sessions - 1) * 0.9)
```

### Adjustment Rules

After collecting ≥3 sessions, recommendations are made based on:

| Condition | Adjustment | Reason |
|-----------|------------|--------|
| Stuck rate > 50% | -2 prompts | Model gets stuck frequently, needs aggressive decrease |
| Stuck rate > 30% | -1 prompt | Model gets stuck often, decrease prompts |
| Quality < 0.4 AND avg_prompts < 70% of max | +2 prompts | Poor quality, model needs more encouragement |
| Quality < 0.6 AND avg_prompts < 70% of max | +1 prompt | Low quality, increase prompts |
| Quality > 0.8 AND avg_prompts > 80% of max | -1 prompt | High quality, wasting time with too many prompts |
| Success rate < 50% (≥5 sessions) | +1 prompt | Model struggling, give more prompts |

Bounds: Recommendations stay within **[2, 12]** range

## Example Learning Trajectory

### Initial State (No Data)
```
ollama:qwen3-coder-tools:30b-128k
  analysis: 6 (default)
  action: 5 (default)
  default: 3 (default)
```

### After 5 Sessions
```
📊 ollama:qwen3-coder-tools:30b-128k

  Task Type: ANALYSIS
    Sessions: 5
    Success Rate: 80.0%
    Avg Quality: 0.72
    Avg Prompts Used: 4.2
    Stuck Loop Rate: 40.0%  ⚠️ High!
    Current Max: 6
    ✨ Recommended: 4 (-2)   ← Decrease due to high stuck rate
```

### After 15 Sessions (Converged)
```
📊 ollama:qwen3-coder-tools:30b-128k

  Task Type: ANALYSIS
    Sessions: 15
    Success Rate: 93.3%
    Avg Quality: 0.81
    Avg Prompts Used: 3.1
    Stuck Loop Rate: 13.3%  ✓ Low
    Current Max: 4
    ✨ Recommended: 4 (0)    ← Stable, optimal found
```

## Manual Override vs RL

**Priority order** (highest to lowest):
1. **Manual overrides** in `profiles.yaml` (always take precedence)
2. **RL-learned recommendations** (if learning enabled and enough data)
3. **Global defaults** from settings (fallback)

This allows you to:
- Let RL learn for most models
- Override specific models manually if needed
- Start with safe defaults

## Monitoring & Diagnostics

### View Learning Report

```bash
$ python scripts/show_continuation_rl.py

================================================================================
Continuation Prompt RL Learning Report
================================================================================

📊 ollama:qwen3-coder-tools:30b-128k
--------------------------------------------------------------------------------

  Task Type: analysis
    Sessions: 12
    Success Rate: 91.7%
    Avg Quality: 0.79
    Avg Prompts Used: 3.4
    Stuck Loop Rate: 16.7%
    Current Max: 4
    ✨ Recommended: 4 (0)

  Task Type: action
    Sessions: 8
    Success Rate: 87.5%
    Avg Quality: 0.76
    Avg Prompts Used: 2.9
    Stuck Loop Rate: 12.5%
    Current Max: 5
    ✨ Recommended: 4 (-1)

================================================================================
Total outcomes recorded: 20
Last updated: 2025-12-18T02:30:45.123456
================================================================================

RECENT OUTCOMES (last 10)
================================================================================

 1. ✓ ollama:qwen3-coder-tools:30b-128k (analysis) - prompts=2/4, quality=0.82, tools=15
 2. ✓ ollama:qwen3-coder-tools:30b-128k (analysis) - prompts=3/4, quality=0.79, tools=18
 3. ✗ ollama:qwen3-coder-tools:30b-128k (analysis) - prompts=2/4, quality=0.45, tools=8 [STUCK] [FORCED]
 4. ✓ ollama:qwen3-coder-tools:30b-128k (action) - prompts=2/5, quality=0.81, tools=12
...
```

### Export Recommendations

```bash
$ python scripts/show_continuation_rl.py --export learned_overrides.yaml

✅ Exported recommendations to: learned_overrides.yaml
   Provider:Model combinations: 3
   Based on 42 outcomes

To use these recommendations:
1. Merge learned_overrides.yaml into your ~/.victor/profiles.yaml
2. Or copy the continuation_prompt_overrides section manually
```

Generated `learned_overrides.yaml`:
```yaml
# Auto-generated continuation prompt recommendations
# Generated by: python scripts/show_continuation_rl.py --export
# Based on 42 recorded outcomes

continuation_prompt_overrides:
  ollama:qwen3-coder-tools:30b-128k:
    analysis: 4
    action: 4
    default: 3
  anthropic:claude-sonnet-4-20250514:
    analysis: 3
    action: 3
    default: 2
enable_continuation_rl_learning: true
```

## Testing the System

### Test Case: High Stuck Rate Model

```bash
# 1. Enable RL learning
echo "enable_continuation_rl_learning: true" >> ~/.victor/profiles.yaml

# 2. Run analysis task that triggers stuck loop
victor chat --provider ollama --model qwen3-coder-tools:30b-128k \
  "Analyze the codebase comprehensively and suggest all improvements."

# 3. Check learning status
python scripts/show_continuation_rl.py --provider ollama

# 4. Run more sessions to see learning adapt
for i in {1..5}; do
  victor chat --provider ollama --model qwen3-coder-tools:30b-128k \
    "Analyze module X and suggest improvements."
done

# 5. Check final recommendations
python scripts/show_continuation_rl.py --provider ollama
```

### Expected Behavior

**Session 1-3:** System uses default (6 prompts for analysis)
- Stuck loops detected frequently
- System starts lowering recommendations

**Session 4-8:** System applies learned recommendations (4-5 prompts)
- Fewer stuck loops
- Better completion rates
- Quality scores improve

**Session 9+:** System converges to optimal value
- Recommendations stabilize
- Success rate >90%
- Stuck rate <15%

## Data Persistence

**Location:** `~/.victor/rl_data/continuation_rl.json`

**Format:**
```json
{
  "stats": {
    "ollama:qwen3-coder-tools:30b-128k:analysis": {
      "total_sessions": 12,
      "successful_sessions": 11,
      "stuck_loop_count": 2,
      "forced_completion_count": 1,
      "avg_quality_score": 0.79,
      "avg_prompts_used": 3.4,
      "current_max_prompts": 4,
      "recommended_max_prompts": 4,
      "last_updated": "2025-12-18T02:30:45.123456",
      "_quality_sum": 8.48,
      "_prompts_sum": 36.5
    }
  },
  "outcomes": [
    {
      "provider": "ollama",
      "model": "qwen3-coder-tools:30b-128k",
      "task_type": "analysis",
      "continuation_prompts_used": 2,
      "max_prompts_configured": 4,
      "success": true,
      "quality_score": 0.82,
      "stuck_loop_detected": false,
      "forced_completion": false,
      "tool_calls_total": 15,
      "timestamp": "2025-12-18T02:25:30.123456"
    }
  ]
}
```

## Integration Points

### 1. Orchestrator Initialization

```python
# In AgentOrchestrator.__init__()
if getattr(settings, "enable_continuation_rl_learning", False):
    self._continuation_learner = ContinuationPromptLearner()
```

### 2. Applying Recommendations

```python
# In _determine_continuation_action()
if self._continuation_learner:
    learned_val = self._continuation_learner.get_recommendation(
        self.provider.name, self.model, task_type
    )
    if learned_val is not None:
        max_continuation_prompts = learned_val
```

### 3. Recording Outcomes

```python
# In _record_intelligent_outcome()
if self._continuation_learner:
    self._continuation_learner.record_outcome(
        provider=self.provider.name,
        model=self.model,
        task_type=task_type,
        continuation_prompts_used=continuation_prompts_used,
        max_prompts_configured=max_prompts_configured,
        success=success and completed,
        quality_score=quality_score,
        stuck_loop_detected=self._stuck_loop_detected,
        forced_completion=ctx.force_completion,
        tool_calls_total=self.tool_calls_used,
    )
```

## Future Enhancements

1. **Provider-Level Defaults**: Learn defaults per provider (not just per model)
2. **Context-Aware Tuning**: Adjust based on task complexity, context size
3. **User Feedback**: Incorporate explicit user satisfaction ratings
4. **A/B Testing**: Compare RL-tuned vs default configurations
5. **Multi-Objective Optimization**: Balance quality, speed, cost
6. **Transfer Learning**: Apply learnings from similar models

## Troubleshooting

### RL not learning
```bash
# Check if RL is enabled
grep enable_continuation_rl_learning ~/.victor/profiles.yaml

# Check if data is being recorded
ls -lh ~/.victor/rl_data/continuation_rl.json
cat ~/.victor/rl_data/continuation_rl.json | jq '.outcomes | length'
```

### Recommendations not applying
```bash
# Manual overrides take precedence - check for conflicts
grep continuation_prompt_overrides ~/.victor/profiles.yaml

# Check debug logs
victor chat --log-level DEBUG ... 2>&1 | grep "RL:"
```

### Reset learning
```bash
# Clear learned data (start fresh)
rm ~/.victor/rl_data/continuation_rl.json

# Learning will restart from defaults
```

## References

- **Reinforcement Learning**: [Sutton & Barto, "Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/the-book.html)
- **Multi-Armed Bandits**: [Epsilon-Greedy Algorithm](https://en.wikipedia.org/wiki/Multi-armed_bandit#Epsilon-greedy_strategy)
- **Weighted Moving Average**: [Exponential Moving Average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average)
