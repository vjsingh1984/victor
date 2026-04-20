# Shell Output Limit Design Analysis

**Date**: 2026-04-20
**Question**: How should we handle shell command output limits?

---

## Current Implementation

**File**: `victor/tools/subprocess_executor.py` (lines 150-155, 314-317)

```python
def _truncate_output(text: str, max_bytes: int) -> Tuple[str, bool]:
    """Truncate output to *max_bytes* with a marker."""
    if max_bytes <= 0 or len(text.encode("utf-8")) <= max_bytes:
        return text, False
    truncated = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + "\n... [output truncated]", True

# Applied to both stdout and stderr
stdout_text, t1 = _truncate_output(stdout_text, max_output_bytes)
stderr_text, t2 = _truncate_output(stderr_text, max_output_bytes)
```

**Current Behavior**:
- Single limit applies to both stdout and stderr
- Byte-based truncation (not line-based)
- Same limit for all commands

---

## Design Considerations

### 1. Stdout vs Stderr - Should They Have Different Limits?

**YES - Here's why:**

**Stdout Characteristics**:
- Can be huge (e.g., `find /`, `git log`, `cat largefile.log`)
- Primary output (what user asked for)
- Often structured data, logs, or content
- **Priority**: Medium (need content, but can be truncated safely)

**Stderr Characteristics**:
- Usually smaller (errors, warnings, debug info)
- MORE critical for debugging (error messages, stack traces)
- Often unstructured but important
- **Priority**: HIGH (don't want to miss "Permission denied" or "Command failed")

**Real-World Examples**:

```bash
# Example 1: git log (huge stdout, small stderr)
git log --all  # stdout: 50,000 lines, stderr: 0 lines

# Example 2: Compilation (moderate stdout, critical stderr)
gcc code.c    # stdout: 50 lines, stderr: 3 lines (critical errors!)

# Example 3: Find command (huge stdout, small stderr)
find / -name "*.py"  # stdout: 100,000 lines, stderr: 5 lines

# Example 4: Failed command (small stdout, critical stderr)
gh run view 123  # stdout: 10 lines, stderr: 2 lines (ERROR!)
```

**Recommendation**: ✅ **Separate limits for stdout and stderr**

---

### 2. What Should the Defaults Be?

**Safety Considerations**:
- Need to prevent runaway commands from consuming gigabytes
- But must preserve critical error messages
- Research tasks need full output

**Proposed Defaults**:

```python
DEFAULT_STDOUT_LIMIT = 10000  # lines (~1MB at 100 chars/line)
DEFAULT_STDERR_LIMIT = 2000   # lines (~200KB at 100 chars/line)
```

**Rationale**:
- **Stdout (10K lines)**: Large enough for most commands, prevents runaway `find /`
- **Stderr (2K lines)**: Smaller but sufficient for error messages
- **Ratio**: 5:1 (stderr is 20% of stdout) - reflects relative importance

---

### 3. How to Specify Unlimited?

**Options**:
- `limit=None` - Pythonic, but requires special handling
- `limit=-1` - Common pattern for "unlimited" (e.g., in `tail -n -1`)
- `limit=0` - Natural "no limit" semantic
- `unlimited=True` - Explicit boolean flag

**Recommendation**: ✅ **Support multiple patterns**

```python
# All equivalent to unlimited:
limit=None           # Pythonic
limit=-1            # Unix convention
limit=0             # Natural semantic
unlimited=True      # Explicit flag

# Apply to both:
bash(cmd="git log", limit=None)

# Apply to specific stream:
bash(cmd="git log", stdout_limit=None, stderr_limit=2000)
```

---

### 4. Line-Based vs Byte-Based Truncation?

**Current**: Byte-based
**Proposed**: Line-based (more intuitive for text output)

**Comparison**:

| Approach | Pros | Cons |
|----------|------|------|
| **Byte-based** (current) | Precise memory control | Cuts messages mid-line (confusing) |
| **Line-based** (proposed) | Preserves line structure | Less precise memory control |

**Recommendation**: ✅ **Line-based with byte fallback**

```python
def _truncate_by_lines(text: str, max_lines: int) -> Tuple[str, bool]:
    """Truncate by lines, but also enforce byte limit."""
    if max_lines <= 0 or max_lines is None:
        return text, False
    
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text, False
    
    # Keep first max_lines lines
    truncated = ''.join(lines[:max_lines])
    
    # Also enforce byte limit (e.g., 1MB max)
    byte_limit = 1024 * 1024  # 1MB
    if len(truncated.encode("utf-8")) > byte_limit:
        truncated = truncated.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")
    
    return truncated + "\n... [truncated]", True
```

---

## Recommended Design

### Parameters

```python
@tool
def bash(
    cmd: str,
    limit: Optional[int] = None,
    stdout_limit: Optional[int] = None,
    stderr_limit: Optional[int] = None,
    unlimited: bool = False,
) -> Dict[str, Any]:
    """Execute shell command with output limits.
    
    Args:
        cmd: Command to execute
        limit: Limit for both stdout and stderr (lines). None=unlimited, -1=unlimited, 0=unlimited
        stdout_limit: Limit for stdout only (overrides `limit`)
        stderr_limit: Limit for stderr only (overrides `limit`)
        unlimited: If True, no limits apply (equivalent to limit=None)
    
    Returns:
        Command result with stdout and stderr
    """
```

### Priority Order

1. **Explicit parameters first**:
   - If `stdout_limit` provided, use it
   - If `stderr_limit` provided, use it
   - If `unlimited=True`, both unlimited

2. **Fallback to `limit`**:
   - If `limit` provided, use for both
   - If `limit=None/-1/0`, unlimited

3. **Task type defaults**:
   - Research tasks: unlimited (both)
   - Code generation: 1000 lines (stdout), 500 (stderr)
   - Debug: 500 lines (stdout), 2000 (stderr) - **more stderr!**
   - Default: 10000 lines (stdout), 2000 (stderr)

### Examples

```python
# Unlimited output (research)
bash(cmd="git log", unlimited=True)

# Different limits
bash(cmd="gcc code.c", stdout_limit=50, stderr_limit=200)

# Both unlimited
bash(cmd="cat file.log", stdout_limit=None, stderr_limit=None)

# Both limited to 100 lines
bash(cmd="find / -name '*.py'", limit=100)

# Legacy pattern (unlimited)
bash(cmd="make test", limit=-1)
```

---

## Implementation Strategy

### Phase 1: Update subprocess_executor.py (30 minutes)

1. Modify `_truncate_output()` to be line-based
2. Add `_truncate_output_separate()` with separate limits
3. Update `run_command()` to use new function

### Phase 2: Update bash tool (20 minutes)

1. Add `limit`, `stdout_limit`, `stderr_limit`, `unlimited` parameters
2. Pass limits through to executor
3. Handle task-type-specific defaults

### Phase 3: Update tool output pruner (10 minutes)

1. Respect explicit limits (don't re-truncate if limit specified)
2. Apply research task rule: unlimited by default

---

## Testing

```python
# Test 1: Separate limits
result = bash("gcc code.c", stdout_limit=10, stderr_limit=100)
assert len(result["stdout"].splitlines()) <= 10
assert len(result["stderr"].splitlines()) <= 100

# Test 2: Unlimited
result = bash("git log", unlimited=True)
assert "[truncated]" not in result["stdout"]
assert "[truncated]" not in result["stderr"]

# Test 3: Stderr priority (huge stdout, critical stderr)
result = bash("gcc huge.c", stdout_limit=50, stderr_limit=1000)
# Stdout truncated, stderr preserved even if huge
```

---

## Summary

**Recommendations**:
1. ✅ **Separate stdout/stderr limits** - Different characteristics and priorities
2. ✅ **Line-based truncation** - More intuitive, preserves structure
3. ✅ **Multiple unlimited patterns** - `limit=None/-1/0` or `unlimited=True`
4. ✅ **Task-type defaults** - Research=unlimited, Debug=more stderr
5. ✅ **Byte fallback** - Prevent runaway huge lines

**Proposed Defaults**:
- `stdout_limit`: 10000 lines (1MB)
- `stderr_limit`: 2000 lines (200KB)
- Research tasks: Unlimited
- Debug tasks: 500/2000 (more stderr priority)

**Estimated Time**: 1 hour (3 phases)
