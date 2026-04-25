# Provider-Specific Tier Metrics Verification Guide

This guide provides instructions for monitoring and verifying provider-specific tool tier optimization in production.

## Metrics to Track

### 1. Tier Distribution Metrics

**What**: Distribution of tools across FULL/COMPACT/STUB tiers per provider category

**Why**: Verify correct tier assignments and token savings

**How to extract**:
```bash
# From logs
grep "tier_distribution" ~/.victor/logs/victor.log | jq -r '.tier_distribution' | sort | uniq -c

# Expected output:
#   50 {"FULL": 2, "STUB": 8}        # Edge category
#  100 {"FULL": 5, "COMPACT": 2}      # Standard category
#  200 {"FULL": 10}                   # Large category
```

### 2. Provider Category Distribution

**What**: Distribution of requests across edge/standard/large categories

**Why**: Understand usage patterns and token savings impact

**How to extract**:
```bash
# From logs
grep "provider_category" ~/.victor/logs/victor.log | \
  jq -r '.provider_category' | \
  sort | uniq -c | \
  sort -rn

# Example output:
#   500 edge
#  1200 standard
#   300 large
```

### 3. Token Usage Per Category

**What**: Actual token consumption per provider category

**Why**: Verify token savings targets are met

**How to extract**:
```bash
# From ConversationStore
sqlite3 ~/.victor/victor.db << 'EOF'
SELECT
    provider_model,
    COUNT(*) as requests,
    SUM(input_tokens) as total_input,
    SUM(output_tokens) as total_output,
    SUM(input_tokens + output_tokens) as total_tokens,
    AVG(input_tokens + output_tokens) as avg_tokens_per_request
FROM messages
WHERE timestamp > datetime('now', '-7 days')
  AND role = 'assistant'
GROUP BY provider_model
ORDER BY total_tokens DESC;
EOF
```

### 4. Budget Utilization

**What**: Percentage of context budget used for tools

**Why**: Ensure tools fit within 25% budget constraint

**How to extract**:
```bash
# From metrics events
grep "TOOL_STRATEGY" ~/.victor/logs/victor.log | \
  jq -r 'select(.tool_tokens) | \
    "\(.provider_category) \(.tool_tokens) \(.max_tool_tokens) \(.tool_tokens / .max_tool_tokens * 100)"' | \
  awk '{sum[$1] += $2; max[$1] = $3; count[$1]++} END \
    {for (cat in sum) printf "%s: %.1f%% avg utilization\n", cat, (sum[cat]/max[cat]/count[cat])*100}'

# Expected output:
# edge: 12.2% avg utilization
# standard: 9.3% avg utilization
# large: 2.5% avg utilization
```

### 5. Token Savings Achieved

**What**: Percentage reduction in tool tokens vs global tiers

**Why**: Validate optimization effectiveness

**How to calculate**:
```python
# calculate_savings.py
import sqlite3
from collections import defaultdict

def calculate_token_savings(days=7):
    conn = sqlite3.connect('~/.victor/victor.db')
    cursor = conn.cursor()

    # Get tool tokens per provider category
    cursor.execute("""
        SELECT
            m.provider_model,
            m.tool_tokens,
            p.context_window
        FROM messages m
        JOIN providers p ON m.provider_model = p.model
        WHERE m.timestamp > datetime('now', '-' || ? || ' days')
          AND m.tool_tokens IS NOT NULL
    """, (days,))

    category_tokens = defaultdict(int)
    category_requests = defaultdict(int)

    for model, tool_tokens, context_window in cursor.fetchall():
        # Determine category
        if context_window < 16384:
            category = 'edge'
        elif context_window < 131072:
            category = 'standard'
        else:
            category = 'large'

        category_tokens[category] += tool_tokens
        category_requests[category] += 1

    # Calculate savings
    global_tokens_per_request = 1250  # Global tier cost

    print(f"Token Savings (Last {days} days)")
    print("=" * 60)

    for category in ['edge', 'standard', 'large']:
        if category not in category_tokens:
            continue

        total_tokens = category_tokens[category]
        requests = category_requests[category]
        avg_tokens = total_tokens / requests

        global_total = global_tokens_per_request * requests
        savings = global_total - total_tokens
        savings_pct = (savings / global_total) * 100

        print(f"{category.capitalize():10} {avg_tokens:6.0f} avg tokens/request")
        print(f"{'':10} {savings_pct:5.1f}% reduction vs global tiers")
        print(f"{'':10} {savings:8.0f} tokens saved ({requests} requests)")
        print()

if __name__ == '__main__':
    calculate_token_savings()
```

**Run**:
```bash
python3 calculate_savings.py
```

**Expected output**:
```
Token Savings (Last 7 days)
============================================================
Edge        250.0 avg tokens/request
             80.0% reduction vs global tiers
          100000 tokens saved (500 requests)

Standard    765.0 avg tokens/request
             38.8% reduction vs global tiers
          232000 tokens saved (1200 requests)

Large      1250.0 avg tokens/request
              0.0% reduction vs global tiers
               0 tokens saved (300 requests)
```

## Real-Time Monitoring

### Dashboard Metrics

Create a monitoring dashboard to track:

1. **Request Rate by Category**:
   ```bash
   # Requests per minute per provider category
   grep "provider_category" ~/.victor/logs/victor.log | \
     jq -r '.timestamp + " " + .provider_category' | \
     awk '{print $1" "$2}' | \
     sort | uniq -c | \
     awk '{print $2" "$1}' > /tmp/category_rate.txt
   ```

2. **Token Usage Rate**:
   ```bash
   # Tokens per minute
   grep "tool_tokens" ~/.victor/logs/victor.log | \
     jq -r '.timestamp + " " + (.tool_tokens | tostring)' | \
     awk '{tokens[$1] += $2} END {for (t in tokens) print t, tokens[t]}' | \
     sort -rn | head -10
   ```

3. **Error Rate by Category**:
   ```bash
   # Errors per provider category
   grep "ERROR" ~/.victor/logs/victor.log | \
     grep "provider_category" | \
     jq -r '.provider_category' | \
     sort | uniq -c | \
     sort -rn
   ```

### Alerting Thresholds

Set up alerts for:

1. **Budget Exceeded**:
   ```bash
   # Alert if tool tokens > 25% of context window
   grep "TOOL_STRATEGY" ~/.victor/logs/victor.log | \
     jq 'select(.tool_tokens > .max_tool_tokens) | \
       "ALERT: Budget exceeded for " + .provider_category + \
       " (" + (.tool_tokens | tostring) + " > " + (.max_tool_tokens | tostring) + ")"'
   ```

2. **Wrong Tier Assignment**:
   ```bash
   # Alert if edge category has >2 FULL tools
   grep "tier_distribution" ~/.victor/logs/victor.log | \
     jq 'select(.provider_category == "edge" and .tier_distribution.FULL > 2) | \
       "ALERT: Edge has " + (.tier_distribution.FULL | tostring) + " FULL tools"'
   ```

3. **High Error Rate**:
   ```bash
   # Alert if error rate > 5% for any category
   for category in edge standard large; do
     errors=$(grep "ERROR" ~/.victor/logs/victor.log | grep "$category" | wc -l)
     total=$(grep "$category" ~/.victor/logs/victor.log | wc -l)
     rate=$(echo "scale=2; $errors / $total * 100" | bc)
     if (( $(echo "$rate > 5" | bc -l) )); then
       echo "ALERT: $category error rate: $rate%"
     fi
   done
   ```

## Performance Metrics

### Latency Impact

**Measure**: Time to first response with provider-specific tiers

```bash
# Extract latency from logs
grep "latency_ms" ~/.victor/logs/victor.log | \
  jq -r 'select(.provider_category) | \
    .provider_category + " " + (.latency_ms | tostring)' | \
  awk '{latency[$1] += $2; count[$1]++} END \
    {for (cat in latency) printf "%s: %.0f ms avg latency\n", cat, latency[cat]/count[cat]}'
```

**Expected**: Edge models should show faster latency (less tool tokens = faster prefill)

### Cost Savings

**Calculate**: Actual cost reduction from token savings

```python
# calculate_cost_savings.py
import sqlite3

PRICING = {
    'edge': 0.0001,      # $0.10 per 1M tokens (free/local)
    'standard': 0.0001,  # $0.10 per 1M tokens (free/local)
    'large': 0.003,      # $3.00 per 1M tokens (claude-sonnet-4)
}

def calculate_cost_savings(days=30):
    conn = sqlite3.connect('~/.victor/victor.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            provider_model,
            SUM(input_tokens + output_tokens) as total_tokens
        FROM messages
        WHERE timestamp > datetime('now', '-' || ? || ' days')
          AND role = 'assistant'
        GROUP BY provider_model
    """)

    print(f"Cost Savings (Last {days} days)")
    print("=" * 60)

    for model, tokens in cursor.fetchall():
        # Determine pricing
        if any(x in model.lower() for x in ['claude', 'gpt', 'gemini']):
            price_per_m = PRICING['large']
        else:
            price_per_m = PRICING['standard']

        actual_cost = (tokens / 1_000_000) * price_per_m

        # Calculate cost with global tiers
        # Assume 20% of tokens are tool tokens
        tool_tokens = tokens * 0.20
        global_tool_tokens = tool_tokens * (1250 / 250)  # Edge: 5x increase
        global_total_tokens = tokens - tool_tokens + global_tool_tokens
        global_cost = (global_total_tokens / 1_000_000) * price_per_m

        savings = global_cost - actual_cost
        savings_pct = (savings / global_cost) * 100

        print(f"{model:30} ${actual_cost:8.4f}")
        print(f"{'':30} Would cost: ${global_cost:8.4f} (global tiers)")
        print(f"{'':30} Savings:    ${savings:8.4f} ({savings_pct:5.1f}%)")
        print()

if __name__ == '__main__':
    calculate_cost_savings()
```

## Regression Detection

### Compare Before/After

**Metric**: Token usage before vs after provider-specific tiers

```bash
# Before: Global tiers
sqlite3 ~/.victor/victor.db << 'EOF'
SELECT
    DATE(timestamp) as date,
    AVG(tool_tokens) as avg_tool_tokens
FROM messages
WHERE timestamp BETWEEN '2026-04-01' AND '2026-04-20'
  AND tool_tokens IS NOT NULL
GROUP BY date
ORDER BY date;
EOF

# After: Provider-specific tiers
sqlite3 ~/.victor/victor.db << 'EOF'
SELECT
    DATE(timestamp) as date,
    AVG(tool_tokens) as avg_tool_tokens
FROM messages
WHERE timestamp >= '2026-04-24'
  AND tool_tokens IS NOT NULL
GROUP BY date
ORDER BY date;
EOF
```

### A/B Testing

**Setup**: Run 50% of traffic with provider-specific tiers, 50% with global tiers

```python
# ab_test_config.py
import random

def should_use_provider_tiers(user_id):
    """50/50 A/B test split."""
    return hash(user_id) % 2 == 0

# In orchestrator
if should_use_provider_tiers(user_id):
    provider_category = get_provider_category(context_window)
else:
    provider_category = None  # Use global tiers
```

**Measure**: Compare metrics between groups

```sql
-- A/B test results
SELECT
    CASE
        WHEN tool_tokens < 500 THEN 'provider_tiers'
        ELSE 'global_tiers'
    END as group,
    COUNT(*) as requests,
    AVG(tool_tokens) as avg_tool_tokens,
    AVG(latency_ms) as avg_latency
FROM messages
WHERE timestamp >= '2026-04-24'
GROUP BY group;
```

## Health Checks

### Daily Health Report

```bash
#!/bin/bash
# daily_health_check.sh

echo "Provider-Specific Tier Health Report"
echo "===================================="
echo ""

# Check if feature flag is enabled
if [ "$VICTOR_TOOL_STRATEGY_V2" = "true" ]; then
  echo "✅ Feature flag enabled"
else
  echo "⚠️  Feature flag disabled"
fi

# Check tier assignments
echo ""
echo "Tier assignments (last 24h):"
grep "tier_distribution" ~/.victor/logs/victor.log | \
  jq -r '.tier_distribution' | \
  sort | uniq -c

# Check budget compliance
echo ""
echo "Budget compliance (last 24h):"
violations=$(grep "TOOL_STRATEGY" ~/.victor/logs/victor.log | \
  jq 'select(.tool_tokens > .max_tool_tokens)' | wc -l)

if [ $violations -eq 0 ]; then
  echo "✅ No budget violations"
else
  echo "❌ $violations budget violations detected"
fi

# Check error rates
echo ""
echo "Error rate by category (last 24h):"
for category in edge standard large; do
  errors=$(grep "ERROR" ~/.victor/logs/victor.log | grep "$category" | wc -l)
  total=$(grep "$category" ~/.victor/logs/victor.log | wc -l)
  if [ $total -gt 0 ]; then
    rate=$(echo "scale=2; $errors / $total * 100" | bc)
    echo "$category: $rate% ($errors errors, $total total)"
  fi
done

echo ""
echo "===================================="
echo "Report complete"
```

## References

- **Configuration**: `victor/config/tool_tiers.yaml`
- **Implementation**: `victor/config/tool_tiers.py`, `victor/agent/orchestrator.py`
- **Database**: `~/.victor/victor.db`, `~/.victor/project.db`
- **Logs**: `~/.victor/logs/victor.log`
- **Validation**: `python -m victor.scripts.validate_provider_tiers`
