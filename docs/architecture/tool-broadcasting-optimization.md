# Tool Broadcasting Optimization - Context-Aware, Economy-First Strategy

## Overview

This document describes Victor's context-window-aware, economy-first tool broadcasting strategy (v2), which optimizes tool selection for both cloud and local providers.

## Key Principles

### 1. Context Window Awareness
- **HARD CONSTRAINT**: Tools must not exceed 25% of available context window
- Respects provider-specific context limits (e.g., 32768 for Ollama, 200000 for Anthropic)
- Prevents context overflow errors with small local models

### 2. Economy-First Strategy
- **Session-locking**: Freeze tools in system prompt when beneficial
- Minimizes cache invalidations for caching providers (90% API discount)
- Reduces per-turn token costs for all providers

### 3. Data-Driven Tier Assignments
- **FULL** schema (~125 tokens): Core tools used in >85% of sessions
- **COMPACT** schema (~70 tokens): Secondary tools used in 50-85% of sessions
- **STUB** schema (~32 tokens): Specialty tools used in <50% of sessions
- Based on actual usage data from ConversationStore and JSONL logs

## Provider Categories

### Cloud Providers (Caching)
- **Providers**: Anthropic, OpenAI, DeepSeek, Google, Azure OpenAI
- **Characteristics**:
  - 90% API-level cache discount on cached tokens
  - Support KV prefix caching for latency optimization
  - Large context windows (128K-200K tokens)

- **Strategy**: Session-lock all tools in system prompt
  - Maximize cache hit rate
  - Minimize cache invalidations
  - Accept larger system prompt for long-term savings

### Local Providers (Non-Caching)
- **Providers**: Ollama, LMStudio, llama.cpp, MLX, vLLM
- **Characteristics**:
  - No API-level cache discount (cost is $0)
  - Support KV prefix caching for latency reduction
  - Smaller context windows (32K-128K tokens)

- **Strategy**: Context-budgeted selection
  - Respect 25% context window constraint
  - Use semantic selection for relevance
  - Optimize for prefill latency reduction

## Implementation

### Core Components

#### 1. Context Window Detection (`BaseProvider.context_window()`)
```python
def context_window(self, model: str) -> int:
    """Get context window size for a given model."""
    # Returns model-specific context window size
    # Safe default: 8192 for unknown models
```

#### 2. Context-Aware Strategy (`Orchestrator._apply_context_aware_strategy()`)
```python
def _apply_context_aware_strategy(self, tools):
    """Apply context-window-aware, economy-first tool selection."""
    # HARD CONSTRAINT: Tools ≤ 25% of context window
    max_tool_tokens = int(context_window * 0.25)

    # ECONOMY STRATEGY: Session-lock when beneficial
    if self._should_session_lock_all_tools(provider, context_window, tool_tokens):
        return tools  # Cache providers or large local models

    # SMALL LOCAL MODELS: Context-budgeted semantic selection
    return self._semantic_select_tools(tools, max_tool_tokens)
```

#### 3. Tool Tier Management (`tool_tiers.py`)
```python
def get_tool_tier(tool_name: str) -> str:
    """Get schema tier for a specific tool."""
    # Returns "FULL", "COMPACT", or "STUB"
    # Defaults to STUB for unknown tools
```

#### 4. Data-Driven Tier Generation (`analyze_tool_usage.py`)
```bash
python -m victor.scripts.analyze_tool_usage --days 30
```
- Pulls actual tool call frequency from ConversationStore
- Analyzes JSONL logs for usage patterns
- Merges data from multiple sources
- Assigns tiers based on usage frequency (decile/quartile)
- Generates `tool_tiers.yaml` with provenance metadata

#### 5. Validation (`validate_tool_strategy.py`)
```bash
python -m victor.scripts.validate_tool_strategy --provider anthropic --model claude-sonnet-4-20250514
```
- Validates context_window() method for providers
- Validates tier assignments exist and are reasonable
- Validates schema token costs
- Tests context constraints for typical tool sets
- Provides comprehensive validation report

## Configuration

### Feature Flag
```bash
# Enable tool strategy v2
export VICTOR_TOOL_STRATEGY_V2=true

# Or in settings.yaml
tool_strategy_v2_enabled: true
```

### Tier Configuration (`tool_tiers.yaml`)
```yaml
provenance:
  generated_at: "2026-04-24T16:00:00Z"
  data_source: "default configuration"
  sample_size_days: 0
  total_sessions: 0
  total_tool_calls: 0
  tools_analyzed: 0

tool_tiers:
  FULL:
    - read
    - write
    - edit
    - code_search
    - shell

  COMPACT:
    - git_status
    - git_diff
    - test
    - ls
    - find
    - web_search

  STUB: "*"
```

## Validation Results

### Cloud Provider (Anthropic)
```
✅ context_window_detection: 200000 tokens
✅ context_window_large: Large context window - good for tool diversity
✅ tier_assignments_exist: 5 FULL, 6 COMPACT tools
✅ context_budget_calculation: 50000 max tool tokens (25% of context)
✅ typical_tool_set_fits: 2090 tokens fits within budget (50000)
```

### Local Provider (Ollama)
```
✅ context_window_detection: 32768 tokens
✅ tier_assignments_exist: 5 FULL, 6 COMPACT tools
✅ context_budget_calculation: 8192 max tool tokens (25% of context)
✅ typical_tool_set_fits: 2090 tokens fits within budget (8192)
```

## Usage Examples

### Generate Data-Driven Tiers
```bash
# Analyze last 30 days of usage
python -m victor.scripts.analyze_tool_usage --days 30

# Output: victor/config/tool_tiers.yaml with provenance
```

### Validate Implementation
```bash
# Validate for cloud provider
python -m victor.scripts.validate_tool_strategy --provider anthropic --model claude-sonnet-4-20250514

# Validate for local provider
python -m victor.scripts.validate_tool_strategy --provider ollama --model qwen2.5-coder:7b
```

### Enable in Production
```bash
# Enable feature flag
export VICTOR_TOOL_STRATEGY_V2=true

# Run Victor with new strategy
victor chat --provider anthropic --model claude-sonnet-4-20250514
```

## Monitoring and Metrics

### Key Metrics
- Context window utilization (%)
- Tool token count (actual vs budget)
- Schema level distribution (FULL/COMPACT/STUB)
- Session-lock hit rate
- Cache hit rate (for caching providers)

### Observability
The implementation includes structured logging for:
- Context window detection
- Tool tier assignments
- Schema selection decisions
- Budget enforcement actions

## Migration from v1

### Key Differences
1. **Context Awareness**: v2 respects provider context windows; v1 did not
2. **Economy-First**: v2 session-locks when beneficial; v1 used dynamic injection
3. **Data-Driven**: v2 uses actual usage data; v1 used hardcoded assignments
4. **Provider-Specific**: v2 optimizes per provider type; v1 was one-size-fits-all

### Migration Steps
1. Generate data-driven tiers: `python -m victor.scripts.analyze_tool_usage --days 30`
2. Validate implementation: `python -m victor.scripts.validate_tool_strategy`
3. Enable feature flag: `export VICTOR_TOOL_STRATEGY_V2=true`
4. Monitor metrics and adjust tier assignments as needed

## Troubleshooting

### Issue: "Context window detection failed"
**Solution**: Ensure provider implements `context_window()` method
```python
class MyProvider(BaseProvider):
    def context_window(self, model: str) -> int:
        # Return model-specific context window
        return 128000
```

### Issue: "Tool tier assignments missing"
**Solution**: Generate tier configuration from usage data
```bash
python -m victor.scripts.analyze_tool_usage --days 30
```

### Issue: "Tool budget exceeded"
**Solution**: Review tier assignments - demote tools to COMPACT or STUB
```yaml
tool_tiers:
  FULL:
    - only_essential_tools
  COMPACT:
    - secondary_tools
  STUB: "*"  # All others
```

## References

- Implementation: `victor/agent/orchestrator.py` - `_apply_context_aware_strategy()`
- Provider base: `victor/providers/base.py` - `context_window()` method
- Tier management: `victor/config/tool_tiers.py`
- Analysis script: `victor/scripts/analyze_tool_usage.py`
- Validation script: `victor/scripts/validate_tool_strategy.py`
- Configuration: `victor/config/tool_tiers.yaml`
