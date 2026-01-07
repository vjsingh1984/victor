# Dashboard Debugging - Summary of Fixes

## Issues Found and Fixed

### 1. JSON Serialization Error ✅ FIXED

**Error**:
```
WARNING - Exporter error: Object of type set is not JSON serializable
```

**Root Cause**: Events with `set` fields in their data couldn't be serialized to JSON.

**Fix**: Added `_make_json_serializable()` method to convert non-JSON types:
- `set` → `list`
- Recursive handling for nested dicts and lists
- Fallback to `str()` for unknown types

**File**: `victor/observability/exporters.py`

### 2. File Watcher Not Starting ✅ FIXED

**Issue**: File watcher only started if file existed when dashboard mounted.

**Fix**: Now file watcher always starts, even if file doesn't exist yet:
```python
# Before: Only start if file exists
if self._file_path.exists():
    self._start_watching()

# After: Always start watching
if self._file_path.exists():
    self._load_existing_events()
self._start_watching()  # Always runs
```

**File**: `victor/observability/dashboard/file_watcher.py`

### 3. Added Debug Logging ✅ ADDED

**Added comprehensive logging** to diagnose issues:
- File watcher mount/startup
- Historical event loading
- Real-time event detection
- Event parsing errors

**Log location**: `~/.victor/logs/victor.log` (standard logger)

**File**: `victor/observability/dashboard/file_watcher.py`

## Testing the Fixes

### Step 1: Reinstall

```bash
pip install -e .
```

### Step 2: Run Agent with Logging

```bash
victor chat --no-tui --log-events "List all Python files"
```

### Step 3: Check Event File

```bash
# View events as they're written
tail -f ~/.victor/metrics/victor.jsonl | jq '{category, name}'

# Count events
wc -l ~/.victor/metrics/victor.jsonl
```

**Expected**: Events should appear without JSON serialization errors

### Step 4: Start Dashboard

```bash
# Normal mode (INFO-level logs)
victor dashboard

# Debug mode (see EventFileWatcher activity)
victor dashboard --log-level DEBUG
```

### Step 5: Check Logs

In another terminal, monitor the logs:

```bash
# Watch dashboard logs
tail -f ~/.victor/logs/victor.log | grep EventFileWatcher
```

**Expected output**:
```
INFO - [EventFileWatcher] Mounting with file_path: /Users/.../.victor/metrics/victor.jsonl
INFO - [EventFileWatcher] File exists, loading historical events
INFO - [EventFileWatcher] Loaded 20 historical events from /Users/.../.victor/metrics/victor.jsonl
INFO - [EventFileWatcher] Starting file watcher
INFO - [EventFileWatcher] File watcher started, monitoring: /Users/.../.victor/metrics/victor.jsonl
INFO - [EventFileWatcher] Loaded 10 new events from file
```

## What to Check

### 1. Event File Growth

```bash
# Watch file size grow
watch -n 5 'wc -l ~/.victor/metrics/victor.jsonl'
```

### 2. Dashboard Display

- **Events tab**: Should show events in chronological order
- **Table tab**: Should show events by category
- **Tools tab**: Should show tool execution stats
- **All 9 tabs**: Should be populated

### 3. Real-Time Updates

1. Start dashboard: `victor dashboard`
2. In another terminal: `victor chat --log-events "Test"`
3. **Within 10-60 seconds**: Events should appear in dashboard

## Flush Behavior

Events are flushed when either condition is met:
- **10 events buffered** → flush immediately
- **60 seconds elapsed** → flush automatically

This means:
- **Low activity** (< 10 events/minute): Flush every 60 seconds
- **High activity** (> 10 events/minute): Flush every 10 events

## Performance Metrics

| Metric | Value |
|--------|-------|
| File watcher poll interval | 100ms |
| Max flush latency | 60 seconds |
| Typical flush latency | 10-30 seconds |
| Dashboard update latency | +100ms |
| Total end-to-end latency | 10-60 seconds |

## Troubleshooting

### No Events in Dashboard

**Check 1**: Is event file being written?
```bash
tail -f ~/.victor/metrics/victor.jsonl
```

**Check 2**: Is file watcher running?
```bash
tail -f ~/.victor/logs/victor.log | grep EventFileWatcher
```

**Check 3**: Are there JSON serialization errors?
```bash
grep "Exporter error" ~/.victor/logs/victor.log
```

**Check 4**: Is EventBus receiving events?
```bash
# Run test script
python scripts/test_jsonl_logging.py
```

### Events Not Appearing Real-Time

**Expected behavior**:
- Events batched (10 events) → appear immediately
- Time-based flush (60 seconds) → appear after timeout

**If events take > 60 seconds**:
- Check if `flush()` is being called
- Check file watcher polling (should be every 100ms)

## Summary

✅ **Fixed JSON serialization** - Sets converted to lists
✅ **Fixed file watcher startup** - Always starts watching
✅ **Added debug logging** - Diagnose issues easily
✅ **Verified event file** - 20 events written successfully
✅ **Ready to test** - Reinstall and run dashboard

## Next Steps

1. **Reinstall**: `pip install -e .`
2. **Run agent**: `victor chat --log-events "Test"`
3. **Start dashboard**: `victor dashboard`
4. **Check logs**: `tail -f ~/.victor/logs/victor.log | grep EventFileWatcher`
5. **Verify events appear**: Should see events in all dashboard tabs

The dashboard should now work correctly! 🎯
