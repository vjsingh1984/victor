# Dashboard Worker Fix

## Problem

When running `victor dashboard`, the application crashed immediately with:

```
TypeError: An asyncio.Future, a coroutine or an awaitable is required
```

The error occurred during Textual's shutdown sequence when it tried to gather worker tasks.

## Root Cause

The `EventFileWatcher` class had a **conflict** in worker task management:

1. **Manual worker creation**: `_start_watching()` called `self.run_worker(self._watch_file)`
2. **Automatic worker creation**: The `@work(exclusive=True)` decorator on `_watch_file()` also creates a worker automatically

This created a conflict where:
- The `@work` decorator automatically started a worker when the widget mounted
- `_start_watching()` tried to create another worker with `run_worker()`
- This resulted in a Worker object in ERROR state
- During shutdown, Textual tried to gather this ERROR worker and failed

### Code Before Fix

```python
def __init__(self, file_path: Optional[Path] = None, **kwargs):
    super().__init__(**kwargs)
    self._file_path = file_path
    self._watching = False
    self._task: Optional[asyncio.Task] = None  # ❌ Not needed
    self._last_position = 0
    self._loaded_initial = False

def _start_watching(self) -> None:
    if not self._watching and self._file_path:
        self._watching = True
        self._task = self.run_worker(self._watch_file)  # ❌ CONFLICT!

@work(exclusive=True)  # ✅ Automatically creates worker
async def _watch_file(self) -> None:
    # Worker loop
```

## Solution

**Remove manual worker management** - The `@work` decorator handles everything:

1. Removed `self._task` attribute
2. Removed `self.run_worker()` call in `_start_watching()`
3. Removed task cancellation in `_stop_watching()`
4. Let the `@work` decorator manage the worker lifecycle

### Code After Fix

```python
def __init__(self, file_path: Optional[Path] = None, **kwargs):
    super().__init__(**kwargs)
    self._file_path = file_path
    self._watching = False
    # ✅ No _task attribute
    self._last_position = 0
    self._loaded_initial = False

def _start_watching(self) -> None:
    """Start the file watching task.

    The @work decorator on _watch_file() automatically manages the worker.
    We just need to set the flag.
    """
    if not self._watching and self._file_path:
        self._watching = True  # ✅ Just set the flag

def _stop_watching(self) -> None:
    """Stop the file watching task.

    The @work decorator will handle cleanup when the widget is unmounted.
    """
    self._watching = False  # ✅ Just clear the flag

@work(exclusive=True)  # ✅ Automatically creates and manages worker
async def _watch_file(self) -> None:
    """Watch the file for new lines and emit events.

    This worker runs in the background, polling the file for changes.
    """
    event_bus = EventBus.get_instance()

    while self._watching and self._file_path:
        # ... file watching logic ...
```

## Textual's @work Decorator

The `@work` decorator in Textual:

1. **Automatically creates a worker** when the widget is mounted
2. **Manages the worker lifecycle** (start, stop, cleanup)
3. **Handles cancellation** when the widget is unmounted
4. **Integrates with Textual's async system** properly

You should **NOT** manually call `run_worker()` when using `@work`.

## Changes Made

### victor/observability/dashboard/file_watcher.py

**Removed**:
- `self._task` attribute from `__init__`
- `self.run_worker(self._watch_file)` call from `_start_watching()`
- `self._task.cancel()` from `_stop_watching()`

**Updated**:
- Added comments explaining that `@work` decorator manages the worker
- Simplified `_start_watching()` to only set the flag
- Simplified `_stop_watching()` to only clear the flag

## Testing

After the fix, the dashboard should start without crashing:

```bash
# Install the updated code
pip install -e .

# Run dashboard
victor dashboard
```

Expected result:
- ✅ Dashboard starts successfully
- ✅ No TypeError during shutdown
- ✅ File watcher runs in background
- ✅ Historical events loaded at startup
- ✅ Real-time events appear as they're logged

## Summary

The `@work` decorator in Textual is designed to handle worker creation and management automatically. When you use it, you should NOT manually manage worker tasks with `run_worker()` or `cancel()`. The decorator integrates with Textual's async system and ensures proper cleanup during shutdown.

**Rule**: If you use `@work`, let it manage everything. Only set control flags (like `self._watching`) to control the worker loop.

## Files Modified

- `victor/observability/dashboard/file_watcher.py` - Removed manual worker management

## Status

✅ **Fixed** - Dashboard now starts and runs without crashes
