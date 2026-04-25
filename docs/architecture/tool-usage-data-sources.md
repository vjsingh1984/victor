# Tool Usage Data Sources - Corrected Database Architecture

## Database Consolidation (Post-Refactoring)

Victor's databases were consolidated to eliminate redundancy. Tool usage data now lives in **two consolidated databases**, not the old `conversation.db`.

### Database Locations (Priority Order)

**1. Project-Specific Database** (Primary source)
```bash
<project>/.victor/project.db
```
- **Contains**: Project-specific tool usage, sessions, messages, graph data
- **Scope**: Single project/session
- **Priority**: Highest - most relevant data for tier generation
- **Example**: `/Users/vijaysingh/code/codingagent/.victor/project.db`

**2. Legacy Project Database** (Fallback)
```bash
<project>/.victor/conversation.db
```
- **Contains**: Old conversation format (pre-consolidation)
- **Scope**: Single project/session
- **Priority**: Medium - fallback for migrated projects
- **Status**: Deprecated - migrated to project.db

**3. Global Database** (Aggregate user data)
```bash
~/.victor/victor.db
```
- **Contains**: Global tool usage across projects, RL data, prompt optimization metrics, sessions
- **Scope**: All projects across all sessions
- **Priority**: Low - used when no project-specific data exists
- **Size**: ~3MB (includes RL cache, prompt optimization, etc.)

**4. Legacy Global Database** (Deprecated)
```bash
~/.victor/conversation.db
```
- **Contains**: Old global conversation format
- **Scope**: All sessions (pre-consolidation)
- **Priority**: Lowest - deprecated format
- **Status**: Deprecated - migrated to victor.db

## Data Schema

### Messages Table (Tool Usage Data)

```sql
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    token_count INTEGER,
    priority INTEGER,
    tool_name TEXT,                    -- ← Tool usage data
    tool_call_id TEXT,                 -- ← Tool call tracking
    metadata TEXT,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

### Tool Usage Query

```sql
SELECT
    tool_name,
    COUNT(*) as calls,
    COUNT(DISTINCT session_id) as sessions
FROM messages
WHERE tool_name IS NOT NULL
  AND timestamp > ?  -- Last N days
GROUP BY tool_name
ORDER BY calls DESC;
```

## Data Flow for Tool Tier Generation

### 1. Script Execution
```bash
python -m victor.scripts.analyze_tool_usage --days 30
```

### 2. Database Discovery (Priority Order)
```
1. Check <project>/.victor/project.db
2. Fallback to <project>/.victor/conversation.db
3. Fallback to ~/.victor/victor.db
4. Fallback to ~/.victor/conversation.db
```

### 3. Data Merging Strategy
- **Project-specific**: Use as-is (highest fidelity)
- **Multiple databases**: Aggregate by tool_name
  - Sum call counts
  - Max session counts (approximate)
- **JSONL logs**: Merge with database data (supplemental)

### 4. Tier Assignment
- **FULL**: Tools used in >50% of sessions (top decile)
- **COMPACT**: Tools used in >20% of sessions (top quartile)
- **STUB**: All other tools (remainder)

## Actual Data Example

### From Current Project (`/Users/vijaysingh/code/codingagent/.victor/project.db`)

```
Total tools analyzed: 13
Total tool calls: 1,237
Total sessions: 10
Time range: Last 30 days

Top Tools:
1. read        → 817 calls (100% sessions)
2. shell       → 121 calls (100% sessions)
3. ls          → 105 calls (100% sessions)
4. code_search →  86 calls (100% sessions)
5. edit        →  62 calls ( 90% sessions)
```

### Comparison: JSONL Logs vs Database

| Source      | Tool Calls | Sessions | Tools | Quality |
|-------------|------------|----------|-------|---------|
| JSONL logs  | 298        | 9        | 10    | Medium  |
| Project DB  | 939        | 10       | 12    | High    |
| **Merged**  | **1,237**  | **10**   | **13**| **Best** |

## Common Issues

### Issue 1: "No conversation databases found"
**Cause**: No databases in expected locations
**Solution**:
- Run Victor to generate tool usage data
- Check database locations with: `find ~/.victor -name "victor.db"`
- Check project database with: `ls -la .victor/project.db`

### Issue 2: "messages table not found"
**Cause**: Old database format or empty database
**Solution**:
- Use newer `project.db` instead of `conversation.db`
- Check tables with: `sqlite3 .victor/project.db ".tables"`

### Issue 3: Low tool count (< 10 tools)
**Cause**: Not enough usage data or too short time window
**Solution**:
- Increase `--days` parameter: `--days 90`
- Run more Victor sessions to accumulate data
- Check if tool tracking is enabled

## Best Practices

### 1. Use Project-Specific Data
- **Best**: Generate tiers from project-specific database
- **Reason**: Most relevant to actual usage patterns
- **Command**: Run from project directory

### 2. Include Sufficient Time Window
- **Minimum**: 30 days (default)
- **Better**: 60-90 days for stable patterns
- **Command**: `--days 60`

### 3. Validate Generated Tiers
- Review tier assignments manually
- Adjust based on domain knowledge
- Consider tool criticality, not just frequency

### 4. Update Regularly
- **Frequency**: Monthly or after major usage changes
- **Process**: Re-run script, review changes, commit YAML
- **Tracking**: Update provenance metadata

## Migration Notes

### Pre-Consolidation (Old)
```python
# Old database locations
~/.victor/conversation.db      # Global conversations
<project>/.victor/conversation.db  # Project conversations
```

### Post-Consolidation (New)
```python
# New database locations - TWO databases
~/.victor/victor.db            # Global user data (RL + prompt opt + tool usage + sessions)
<project>/.victor/project.db   # Project-specific data (conversations + graph + entities)
```

### Schema Changes
- **Added**: `tool_name` column to `messages` table
- **Added**: `tool_call_id` column for tracking
- **Removed**: Separate `tool_calls` table (integrated into messages)
- **Migrated**: All conversation data split between project.db (local) and victor.db (global)

## Monitoring Data Quality

### Check Database Size
```bash
# Project database (should be MBs, not KBs)
ls -lh .victor/project.db

# Global database (should be MBs)
ls -lh ~/.victor/victor.db
```

### Check Tool Usage Volume
```bash
# Count tool messages in project DB
sqlite3 .victor/project.db "SELECT COUNT(*) FROM messages WHERE tool_name IS NOT NULL;"

# Should return hundreds/thousands for active projects
```

### Check Session Count
```bash
# Count unique sessions
sqlite3 .victor/project.db "SELECT COUNT(DISTINCT session_id) FROM messages WHERE tool_name IS NOT NULL;"

# Should be >5 for meaningful tier assignments
```

## Provider-Specific Tier Optimization

### Overview

Victor now supports **provider-specific tool tier assignments** that optimize token usage based on model context window sizes. This replaces the single global tier assignment with category-specific optimizations.

### Provider Categories

**Edge** (<16K tokens):
- **Models**: qwen3.5:2b, gemma:2b, qwen2.5:0.5b, phi-2:2.7b, gemma3:4b, qwen2.5:3b, phi-3:4b, phi-3:3.8b, qwen2:1.5b, qwen2:1.5b-instruct, qwen2.5:1.5b, qwen2.5:1.5b-instruct
- **Tool Set**: 2 FULL tools (read, shell)
- **Token Cost**: 250 tokens (12.2% of 2K budget)
- **Savings**: 80% reduction vs global tiers

**Standard** (16K-128K tokens):
- **Models**: qwen2.5:7b, llama:7b, mistral:7b, and similar
- **Tool Set**: 5 FULL + 2 COMPACT tools (read, shell, ls, code_search, edit, write, test)
- **Token Cost**: 765 tokens (9.3% of 8K budget)
- **Savings**: 38.8% reduction vs global tiers

**Large** (>128K tokens):
- **Models**: claude-sonnet-4, gpt-4o, gemini-1.5, and similar
- **Tool Set**: 10 FULL tools (all core tools)
- **Token Cost**: 1,250 tokens (2.5% of 50K budget)
- **Savings**: No regression (full capability)

### Configuration

**File**: `victor/config/tool_tiers.yaml`

```yaml
# Provider-specific tiers
provider_tiers:
  edge:
    context_window_max: 16384
    FULL: [read, shell]
    COMPACT: []
    STUB: "*"

  standard:
    context_window_min: 16384
    context_window_max: 131072
    FULL: [read, shell, ls, code_search, edit]
    COMPACT: [write, test]
    STUB: "*"

  large:
    context_window_min: 131072
    FULL: [read, shell, ls, code_search, edit, write, _get_directory_summaries, symbol, find, test]
    COMPACT: []
    STUB: "*"
```

### Implementation

**Tier Loading** (`victor/config/tool_tiers.py`):
```python
def get_provider_category(context_window: int) -> str:
    """Get provider category based on context window size."""
    if context_window < 16384:
        return "edge"
    elif context_window < 131072:
        return "standard"
    else:
        return "large"

def get_provider_tool_tier(tool_name: str, provider_category: str) -> str:
    """Get tool tier for a specific provider category."""
    # Returns 'FULL', 'COMPACT', or 'STUB' based on category
```

**Orchestrator Integration** (`victor/agent/orchestrator.py`):
```python
def _apply_context_aware_strategy(self, tools):
    """Apply context-window-aware tool selection with provider-specific tiers."""
    # Get provider category from context window
    provider_category = get_provider_category(context_window)

    # Estimate tool tokens using provider-specific tiers
    tool_tokens = sum(self._estimate_tool_tokens(tool, provider_category) for tool in tools)

    # Select and optimize tool set based on budget
    ...
```

### Token Savings Comparison

| Category | Global Tiers | Provider Tiers | Savings | Budget Utilization |
|----------|-------------|----------------|---------|-------------------|
| Edge     | 1,250       | 250            | 80.0%   | 12.2%             |
| Standard | 1,250       | 765            | 38.8%   | 9.3%              |
| Large    | 1,250       | 1,250          | 0.0%    | 2.5%              |

### Feature Flag

**Control**: `VICTOR_TOOL_STRATEGY_V2=true` (default: false)

Enable provider-specific optimization:
```bash
export VICTOR_TOOL_STRATEGY_V2=true
victor chat --provider ollama --model qwen3.5:2b
```

### Validation

Run validation script to verify provider-specific tiers:
```bash
python -m victor.scripts.validate_provider_tiers
```

Expected output:
```
✅ Edge model: 250 tokens ≤ 2048 budget (12.2% utilization)
✅ Standard model: 765 tokens ≤ 8192 budget (9.3% utilization)
✅ Large model: 1250 tokens ≤ 50000 budget (2.5% utilization)
✅ Token Savings: 80% edge, 38.8% standard, 0% large
```

### Testing

**Unit Tests**:
```bash
# Run provider-specific tier tests
pytest tests/unit/config/test_provider_tool_tiers.py -v
pytest tests/unit/agent/test_provider_specific_tier_orchestrator.py -v
```

**Integration Tests** (with real providers):
```bash
# Edge model
victor chat --provider ollama --model qwen3.5:2b --enable-tool-strategy-v2
# Expected: 2 FULL tools, no budget warnings

# Standard model
victor chat --provider ollama --model qwen2.5:7b --enable-tool-strategy-v2
# Expected: 7 tools (5 FULL + 2 COMPACT), no budget warnings

# Large model
victor chat --provider anthropic --model claude-sonnet-4-20250514 --enable-tool-strategy-v2
# Expected: 10 FULL tools, no budget warnings
```

### Backward Compatibility

- ✅ `get_tool_tier(tool_name)` unchanged - existing code works
- ✅ `get_provider_tool_tier(tool_name, provider_category)` is additive
- ✅ Global tiers in YAML remain as fallback
- ✅ All orchestrator methods work with `provider_category=None`
- ✅ Feature flag enables instant rollback: `VICTOR_TOOL_STRATEGY_V2=false`

### Rollout Plan

**Week 1**: Internal testing with edge models
- Test with qwen3.5:2b, gemma:2b
- Verify token savings and budget compliance
- Monitor metrics

**Week 2**: Beta testing with standard models
- Test with qwen2.5:7b, llama:7b
- Verify balanced tool set performance
- Gather user feedback

**Week 3**: Gradual rollout (10% → 50% → 100%)
- Enable for 10% of users
- Monitor metrics and errors
- Increase to 50%, then 100%

**Week 4**: Production monitoring
- Track token savings across all categories
- Monitor latency improvements
- Adjust as needed

### References

- **Implementation**: `victor/scripts/analyze_tool_usage.py`
- **Configuration**: `victor/config/tool_tiers.yaml`
- **Validation**: `victor/scripts/validate_provider_tiers.py`
- **Database Schema**: `victor/agent/conversation/store.py`
- **Provider Tiers**: `victor/config/tool_tiers.py`
- **Orchestrator Integration**: `victor/agent/orchestrator.py`
