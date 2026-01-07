# Event Export Timing - Summary

## Flush Behavior

The JSONL exporter now uses **hybrid flushing**:

```python
buffer_size = 10 events
flush_interval = 60 seconds

# Flush happens when EITHER condition is met:
# 1. Buffer has 10 events
# 2. 60 seconds elapsed since last flush
```

## Timing Scenarios

### Low Activity (1-9 events/minute)
- **Flush trigger**: Time interval (60 seconds)
- **Max latency**: 60 seconds
- **Disk I/O**: Once per minute
- **Example**: 5 tool calls → flush after 60s

### High Activity (10+ events/minute)
- **Flush trigger**: Buffer full (10 events)
- **Max latency**: Depends on event rate
- **Disk I/O**: More frequent (efficient batching)
- **Example**: 20 tool calls → flush after 10 events (~30s)

### Bursts (many events quickly)
- **Flush trigger**: Buffer full
- **Latency**: Minimal (events batched)
- **Example**: 50 events in 10s → 5 flushes of 10 events each

## Dashboard File Watcher

```python
poll_interval = 100ms  # Checks file every 0.1 seconds
```

**Total latency**:
- Export: 0-60 seconds (flush behavior)
- File watcher: 0-100ms (polling)
- **Dashboard update**: 60.1 seconds max, usually much less

## Testing

### Quick Test (Low Activity)
```bash
# Run agent with few events
victor chat --log-events "List files in current directory"

# Events should appear within 60 seconds
# Dashboard monitors: ~/.victor/metrics/victor.jsonl
```

### High Activity Test
```bash
# Run agent with many events
victor chat --log-events "Analyze the entire codebase"

# Events should appear every 10 events (could be frequent)
```

### Manual Test Script
```bash
# Run the test script
python scripts/test_jsonl_logging.py

# This creates test events and verifies they're loaded
```

## Verification

### Check Events in File
```bash
# Watch events in real-time
tail -f ~/.victor/metrics/victor.jsonl | jq '{category, name}'

# Count events
wc -l ~/.victor/metrics/victor.jsonl

# View last 10 events
tail -n 10 ~/.victor/metrics/victor.jsonl | jq '.'
```

### Check Dashboard
```bash
# Start dashboard
victor dashboard

# Check Events tab - should show:
# - Historical events (loaded on startup)
# - Real-time events (polls every 100ms)
```

## Performance Impact

| Metric | Before (buffer=1) | After (buffer=10, 60s) |
|--------|-------------------|------------------------|
| Disk writes | Every event | Every 10 events or 60s |
| Disk I/O | High | Low (90% reduction) |
| Max latency | <1ms | 60 seconds |
| Typical latency | <1ms | 10-30 seconds (batched) |
| Memory usage | Minimal | +10 events in buffer |

## Tuning

If you need faster flushes, modify `victor/observability/bridge.py`:

```python
# Faster flushes (for debugging)
buffer_size=5,              # Flush every 5 events
flush_interval_seconds=10   # Flush every 10 seconds

# Slower flushes (for performance)
buffer_size=50,             # Flush every 50 events
flush_interval_seconds=300  # Flush every 5 minutes
```

## Files Modified

- `victor/observability/exporters.py` - Added time-based flushing
- `victor/observability/bridge.py` - Updated to use new settings
- `victor/observability/dashboard/file_watcher.py` - Fixed to always start watching

## Summary

✅ Events flush every **10 events OR 60 seconds** (whichever first)
✅ Dashboard polls every **100ms**
✅ Max latency to dashboard: **~60 seconds**
✅ Typical latency: **10-30 seconds** (for moderate activity)
✅ 90% reduction in disk I/O vs immediate flush
