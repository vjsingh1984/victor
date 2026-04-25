# Provider-Specific Tier Integration Testing Guide

This guide provides step-by-step instructions for testing provider-specific tool tier optimization with real providers.

## Prerequisites

1. **Feature Flag Enabled**:
   ```bash
   export VICTOR_TOOL_STRATEGY_V2=true
   ```

2. **Provider Setup**:
   - Edge models: Ollama running with qwen3.5:2b or similar
   - Standard models: Ollama running with qwen2.5:7b or similar
   - Large models: API keys for Anthropic/OpenAI/Google

3. **Monitoring Enabled**:
   ```bash
   export VICTOR_LOG_LEVEL=DEBUG
   export VICTOR_LOG_FILE=~/.victor/logs/provider-tier-test.log
   ```

## Test 1: Edge Model (qwen3.5:2b)

### Objective
Verify edge models use only 2 FULL tools (read, shell) and achieve 80% token reduction.

### Steps

1. **Start Ollama with edge model**:
   ```bash
   ollama serve
   ollama pull qwen3.5:2b
   ```

2. **Run Victor with edge model**:
   ```bash
   victor chat --provider ollama --model qwen3.5:2b --enable-tool-strategy-v2
   ```

3. **Test prompts**:
   ```
   # Should trigger read tool (FULL)
   "Read the file README.md and tell me what it's about"

   # Should trigger shell tool (FULL)
   "List all Python files in the current directory"

   # Should NOT trigger edit tool (STUB for edge)
   "Add a comment to the top of main.py"
   ```

4. **Verify metrics**:
   ```bash
   grep "TOOL_STRATEGY" ~/.victor/logs/provider-tier-test.log | tail -5
   ```

   **Expected output**:
   ```
   TOOL_STRATEGY: provider_category=edge, tool_count=2, tool_tokens=250
   TOOL_STRATEGY: tier_distribution={"FULL": 2, "STUB": N}
   ```

5. **Check budget compliance**:
   - Context window: 8,192 tokens
   - Max tool tokens: 2,048 (25%)
   - Actual tool tokens: 250
   - Utilization: 12.2% ✅

### Success Criteria
- ✅ Only 2 tools in FULL tier (read, shell)
- ✅ Tool tokens ≤ 2,048
- ✅ No budget warnings
- ✅ 80% token reduction vs global tiers

## Test 2: Standard Model (qwen2.5:7b)

### Objective
Verify standard models use 7 tools (5 FULL + 2 COMPACT) and achieve 40% token reduction.

### Steps

1. **Start Ollama with standard model**:
   ```bash
   ollama pull qwen2.5:7b
   ```

2. **Run Victor with standard model**:
   ```bash
   victor chat --provider ollama --model qwen2.5:7b --enable-tool-strategy-v2
   ```

3. **Test prompts**:
   ```
   # Should trigger read tool (FULL)
   "Read the file src/main.py"

   # Should trigger code_search tool (FULL)
   "Find all functions that call database.connect()"

   # Should trigger edit tool (FULL)
   "Change the function name from process() to handle()"

   # Should trigger write tool (COMPACT)
   "Create a new file utils.py with helper functions"

   # Should trigger test tool (COMPACT)
   "Run the unit tests for this module"
   ```

4. **Verify metrics**:
   ```bash
   grep "TOOL_STRATEGY" ~/.victor/logs/provider-tier-test.log | tail -5
   ```

   **Expected output**:
   ```
   TOOL_STRATEGY: provider_category=standard, tool_count=7, tool_tokens=765
   TOOL_STRATEGY: tier_distribution={"FULL": 5, "COMPACT": 2}
   ```

5. **Check budget compliance**:
   - Context window: 32,768 tokens
   - Max tool tokens: 8,192 (25%)
   - Actual tool tokens: 765
   - Utilization: 9.3% ✅

### Success Criteria
- ✅ 7 tools selected (5 FULL + 2 COMPACT)
- ✅ Tool tokens ≤ 8,192
- ✅ No budget warnings
- ✅ 38.8% token reduction vs global tiers

## Test 3: Large Model (claude-sonnet-4)

### Objective
Verify large models use all 10 FULL tools with no regression.

### Steps

1. **Set up Anthropic credentials**:
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. **Run Victor with large model**:
   ```bash
   victor chat --provider anthropic --model claude-sonnet-4-20250514 --enable-tool-strategy-v2
   ```

3. **Test prompts**:
   ```
   # Should trigger multiple FULL tools
   "Analyze the codebase, find all test files, and run the tests"

   # Should trigger symbol/find tools (FULL)
   "Find the definition of the Database class and all its usages"

   # Should trigger all tools as needed
   "Create a comprehensive test suite for the authentication module"
   ```

4. **Verify metrics**:
   ```bash
   grep "TOOL_STRATEGY" ~/.victor/logs/provider-tier-test.log | tail -5
   ```

   **Expected output**:
   ```
   TOOL_STRATEGY: provider_category=large, tool_count=10, tool_tokens=1250
   TOOL_STRATEGY: tier_distribution={"FULL": 10}
   ```

5. **Check budget compliance**:
   - Context window: 200,000 tokens
   - Max tool tokens: 50,000 (25%)
   - Actual tool tokens: 1,250
   - Utilization: 2.5% ✅

### Success Criteria
- ✅ All 10 tools in FULL tier
- ✅ Tool tokens ≤ 50,000
- ✅ No budget warnings
- ✅ No regression (same as global tiers)

## Metrics Verification

### Extract Tier Distribution

```bash
# Extract tier distribution from logs
grep "tier_distribution" ~/.victor/logs/provider-tier-test.log | jq -r '.tier_distribution'
```

### Monitor Token Usage

```bash
# Calculate actual token savings
python3 << 'EOF'
import re

log_file = "~/.victor/logs/provider-tier-test.log"
with open(log_file, "r") as f:
    for line in f:
        if "TOOL_STRATEGY" in line and "tool_tokens" in line:
            print(line)

# Compare global vs provider-specific tokens
global_tokens = 1250
edge_tokens = 250
standard_tokens = 765
large_tokens = 1250

edge_savings = ((global_tokens - edge_tokens) / global_tokens) * 100
standard_savings = ((global_tokens - standard_tokens) / global_tokens) * 100

print(f"Edge savings: {edge_savings:.1f}%")
print(f"Standard savings: {standard_savings:.1f}%")
EOF
```

### Check Provider Category Detection

```bash
# Verify provider category detection
grep "provider_category" ~/.victor/logs/provider-tier-test.log | sort | uniq -c
```

**Expected output**:
```
  5 TOOL_STRATEGY: provider_category=edge
  5 TOOL_STRATEGY: provider_category=standard
  5 TOOL_STRATEGY: provider_category=large
```

## Rollback Testing

### Test Feature Flag Rollback

1. **Disable feature flag**:
   ```bash
   export VICTOR_TOOL_STRATEGY_V2=false
   ```

2. **Run Victor**:
   ```bash
   victor chat --provider ollama --model qwen3.5:2b
   ```

3. **Verify global tiers used**:
   ```bash
   grep "tool_tokens" ~/.victor/logs/provider-tier-test.log | tail -1
   ```

   **Expected output**:
   ```
   # Should show 1,250 tokens (global tiers) instead of 250 (provider-specific)
   ```

### Success Criteria
- ✅ System reverts to global tiers when flag disabled
- ✅ No errors or crashes
- ✅ Backward compatibility maintained

## Performance Monitoring

### Latency Comparison

```bash
# Measure time to first response with provider-specific tiers
time victor chat --provider ollama --model qwen3.5:2b --enable-tool-strategy-v2 << EOF
What files are in the current directory?
EOF

# Compare with global tiers
time victor chat --provider ollama --model qwen3.5:2b << EOF
What files are in the current directory?
EOF
```

**Expected**: Provider-specific tiers should be faster (less tool tokens = faster prefill)

### Token Cost Comparison

```bash
# Calculate cost per request
# Edge model: 250 tool tokens vs 1,250 global
# Savings: 1,000 tokens per request

# For 1,000 requests:
# Global tiers: 1,250,000 tokens
# Provider tiers: 250,000 tokens
# Savings: 1,000,000 tokens (80% reduction)
```

## Common Issues

### Issue 1: Provider Category Not Detected

**Symptom**: `provider_category=unknown` in logs

**Solution**:
```bash
# Check if model is in CONTEXT_WINDOWS mapping
grep -r "qwen3.5:2b" victor/providers/base.py

# If not found, add it:
# CONTEXT_WINDOWS = {
#     "qwen3.5:2b": 8192,
# }
```

### Issue 2: Token Budget Exceeded

**Symptom**: Budget warning in logs

**Solution**:
```bash
# Verify tool tier assignments
python3 << 'EOF'
from victor.config.tool_tiers import get_provider_tool_tier

# Check edge tiers
print("read:", get_provider_tool_tier("read", "edge"))
print("shell:", get_provider_tool_tier("shell", "edge"))
print("edit:", get_provider_tool_tier("edit", "edge"))
EOF

# Expected: read=FULL, shell=FULL, edit=STUB
```

### Issue 3: Wrong Tier Assignment

**Symptom**: Tool in wrong tier (e.g., edit is FULL for edge)

**Solution**:
```bash
# Reload tier configuration
python3 << 'EOF'
from victor.config.tool_tiers import reload_provider_tiers
reload_provider_tiers()
print("Tiers reloaded")
EOF
```

## Next Steps

After successful integration testing:

1. **Enable for beta users** (10% rollout):
   ```bash
   # Update deployment config
   VICTOR_TOOL_STRATEGY_V2=true  # for 10% of users
   ```

2. **Monitor production metrics**:
   - Token usage per provider category
   - Error rates
   - User feedback

3. **Gradual rollout**:
   - Week 1: 10%
   - Week 2: 50%
   - Week 3: 100%

4. **Document findings**:
   - Actual token savings
   - Performance improvements
   - User feedback summary

## References

- **Configuration**: `victor/config/tool_tiers.yaml`
- **Implementation**: `victor/config/tool_tiers.py`, `victor/agent/orchestrator.py`
- **Validation**: `python -m victor.scripts.validate_provider_tiers`
- **Documentation**: `docs/architecture/provider-specific-tier-optimization-complete.md`
