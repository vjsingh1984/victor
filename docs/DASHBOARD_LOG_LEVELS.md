# Dashboard Log Level Quick Reference

## Overview

The `victor dashboard` command now supports a `--log-level` flag to control logging verbosity.

## Usage

```bash
victor dashboard [--log-level LEVEL]
```

## Log Levels

| Level | When to Use | What You See |
|-------|-------------|--------------|
| **ERROR** | Production, only want to see failures | Only ERROR messages |
| **WARNING** | Normal operation, see potential issues | WARNING + ERROR |
| **INFO** | **Default for dashboard** | EventFileWatcher activity, event counts, startup info |
| **DEBUG** | Troubleshooting, deep diagnostics | All INFO + detailed polling, file position tracking |

## Examples

### Normal Operation
```bash
# See EventFileWatcher loading events, startup messages
victor dashboard --log-level INFO
# or just
victor dashboard  # INFO is default
```

**Expected logs**:
```
INFO - [EventFileWatcher] Mounting with file_path: /Users/.../.victor/metrics/victor.jsonl
INFO - [EventFileWatcher] File exists, loading historical events
INFO - [EventFileWatcher] Loaded 20 recent events (limit: 100) from ...
INFO - [EventFileWatcher] Starting file watcher
INFO - [EventFileWatcher] File watcher started, monitoring: ...
INFO - [EventFileWatcher] Loaded 5 new events from file
```

### Troubleshooting
```bash
# See detailed polling activity, file position changes
victor dashboard --log-level DEBUG
```

**Expected logs** (in addition to INFO):
```
DEBUG - [EventFileWatcher] File grew from 1024 to 2048 bytes
DEBUG - [EventFileWatcher] Read 3 new lines
DEBUG - [EventFileWatcher] Failed to parse line: {...}
```

### Quiet Mode
```bash
# Only show warnings and errors
victor dashboard --log-level WARNING
```

**Expected logs**: Only WARNING and ERROR messages (no EventFileWatcher activity logs)

## Monitoring Logs

### In Another Terminal
```bash
# Watch all dashboard logs
tail -f ~/.victor/logs/victor.log

# Watch only EventFileWatcher logs
tail -f ~/.victor/logs/victor.log | grep EventFileWatcher

# Watch only ERROR logs
tail -f ~/.victor/logs/victor.log | grep ERROR
```

## Troubleshooting Guide

### "Dashboard is empty, no events showing"

1. **Start dashboard with DEBUG logging**:
   ```bash
   victor dashboard --log-level DEBUG
   ```

2. **In another terminal, check logs**:
   ```bash
   tail -f ~/.victor/logs/victor.log | grep EventFileWatcher
   ```

3. **Look for these messages**:
   - ✅ "Mounting with file_path: ..." → File watcher initialized
   - ✅ "File exists, loading historical events" → Events being loaded
   - ✅ "Loaded X recent events" → Events loaded successfully
   - ❌ "File does not exist yet" → No event file, run agent first with `--log-events`

### "Events not appearing in real-time"

**Expected latency**: 0-60 seconds
- If agent generates 10+ events quickly → flush immediately
- If agent generates < 10 events → flush after 60 seconds

**Verify with DEBUG logging**:
```bash
victor dashboard --log-level DEBUG
# Should see:
# DEBUG - File grew from X to Y bytes
# DEBUG - Read N new lines
# INFO - Loaded N new events from file
```

### "Can't see any logs at all"

**Check log level**:
- Default for dashboard is `INFO`
- If you set global log level to `WARNING` or `ERROR`, you won't see INFO logs

**Solution**: Pass `--log-level INFO` or `--log-level DEBUG`

## Log File Location

All logs are written to:
```
~/.victor/logs/victor.log
```

## Related Documentation

- `DASHBOARD_FIXES_SUMMARY.md` - Complete list of fixes and testing
- `docs/DASHBOARD_DEBUGGING.md` - Detailed debugging guide
- `scripts/verify_dashboard_fixes.sh` - Automated verification script

## FAQ

**Q: What's the default log level?**
A: `INFO` for the dashboard command. This shows EventFileWatcher activity and startup messages.

**Q: Do I need to restart the dashboard to change log level?**
A: Yes, the `--log-level` flag is set at startup. Restart with a different level to change.

**Q: Can I change the log level without restarting?**
A: Not currently. You'll need to stop (Ctrl+C) and restart with `--log-level DEBUG`.

**Q: Why don't I see logs when I run `victor dashboard`?**
A: If you've configured Victor globally to use `WARNING` or `ERROR` log level, the default `INFO` might be suppressed. Use `--log-level DEBUG` to see all logs.

**Q: Where are the EventFileWatcher logs written?**
A: They're written to `~/.victor/logs/victor.log` via the standard Python logging system. Use `--log-level INFO` or higher to see them.
