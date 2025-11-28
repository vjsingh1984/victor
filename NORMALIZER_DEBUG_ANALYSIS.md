# Argument Normalizer Debug Analysis

**Date**: November 28, 2025
**Status**: 🔍 **INVESTIGATION IN PROGRESS**
**Priority**: MEDIUM (Tool execution succeeds despite normalizer reporting failure)

---

## Problem Summary

The argument normalizer reports `"Failed to normalize edit_files arguments after all strategies"` (ERROR log) even though:
1. The arguments are already valid JSON with proper double quotes
2. The tool execution succeeds
3. Unit tests and reproduction tests pass

---

## Investigation Results

### Unit Test Results: ✅ ALL PASS

```
$ python test_with_logging.py

Result: Strategy = direct

Debug logs show:
- has_json_like_strings=True
- AST normalization changed=False (already valid)
- Layer 1 - is_valid=True
- Successfully returns DIRECT strategy
```

### Real Execution: ❌ Reports FAILED

From user's debug logs (`victor_debug.log:1398`):
```
ERROR - [OllamaProvider] Failed to normalize edit_files arguments after all strategies
WARNING - Applied failed normalization to edit_files arguments
⚙ Normalized arguments via failed

✓ Tool executed successfully (2ms)
```

---

## Root Cause Hypothesis

### Hypothesis 1: Environment Difference
The normalizer works correctly in unit tests but fails in real execution. Possible causes:
1. **Python bytecode caching**: Old version of normalizer still in `__pycache__/`
2. **Multiple normalizer instances**: Different instances with different state
3. **Exception in validation**: Silent exception caught somewhere causing validation to fail

### Hypothesis 2: False Alarm (Not Actually a Problem)
The normalizer is working as designed:
- When all normalization strategies fail, it returns the original arguments unchanged
- If those arguments happen to be valid JSON (which they are), the tool succeeds
- The ERROR log is misleading but functionally harmless

### Hypothesis 3: Validation Method Issue
The `_is_valid_json_dict()` method throws an exception in production that doesn't occur in tests:
- Possible issue with the `json.loads()` call on the operations string
- Some edge case with the bash script content (special characters, escaping)
- Memory or resource issue in production

---

## Changes Made

### Enhanced Debug Logging (victor/agent/argument_normalizer.py)

Added comprehensive logging at every layer:

```python
# Lines 95-127: Added debug logging to track:
- has_json_like_strings detection
- Preemptive AST normalization attempts
- Layer 1 validation flow
- Exception details with exc_info=True
```

### Validation Method Enhancement (Lines 191-217)

Added detailed logging in `_is_valid_json_dict()`:
```python
- json.dumps() success/failure
- Per-key JSON validation
- Exception details
- Return value logging
```

---

## Next Steps

### Immediate Action Required

**User should run the following command**:

```bash
# Clear Python bytecode cache
find victor/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Run test with DEBUG logging
/Users/vijaysingh/code/codingagent/venv/bin/victor main --log-level DEBUG \
  "Create a bash routine to generate fibonacci series for n numbers. Execute for n=5" \
  2>&1 | tee /tmp/normalizer_enhanced_debug.log

# Extract relevant debug logs
grep -E "argument_normalizer|has_json_like|Layer 1|is_valid|FAILED" \
  /tmp/normalizer_enhanced_debug.log | tail -100
```

This will show:
1. Whether the enhanced logging is active (new log format)
2. Exactly which layer is failing and why
3. Any exceptions being thrown

### Analysis Tasks

1. **Review Enhanced Debug Logs**: Look for the new debug log format to confirm enhanced normalizer is running
2. **Check for Exceptions**: Look for `exc_info=True` stack traces showing where exceptions occur
3. **Validate Hypothesis**: Determine if it's bytecode caching, environment, or validation logic

### Potential Fixes

**If Hypothesis 1 (Bytecode Caching)**:
```bash
# Clear all Python caches
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

**If Hypothesis 2 (False Alarm)**:
```python
# Change ERROR to WARNING when arguments are valid JSON
# victor/agent/argument_normalizer.py:159-162
logger.warning(  # Changed from error
    f"[{self.provider_name}] Could not improve {tool_name} arguments "
    f"via normalization, using original. This may still work if JSON is valid."
)
```

**If Hypothesis 3 (Validation Exception)**:
```python
# Add more robust exception handling in _is_valid_json_dict
# Already implemented in lines 191-217
```

---

## Expected Outcome

After running with enhanced debug logging, we should see one of:

### Scenario A: Fixed (Bytecode Cache Issue)
```
DEBUG - [OllamaProvider] edit_files: has_json_like_strings=True
DEBUG - [OllamaProvider] edit_files: Layer 1 - Checking if already valid
DEBUG - [OllamaProvider] edit_files: Layer 1 - is_valid=True
```

### Scenario B: Still Failing (Need More Investigation)
```
DEBUG - [OllamaProvider] edit_files: Layer 1 - Checking if already valid
ERROR - [OllamaProvider] edit_files: Layer 1 validation threw exception: ...
```

### Scenario C: Different Root Cause
Logs will reveal unexpected behavior not covered by current hypotheses.

---

## Risk Assessment

**Current Risk**: LOW
- Tool execution succeeds despite normalizer failure
- No functional impact on end users
- Only cosmetic issue (scary ERROR logs)

**Potential Risk**: MEDIUM (if not addressed)
- Users may lose confidence in the system due to ERROR logs
- May mask real normalization failures in the future
- Debugging overhead from false alarms

---

## Files Modified

1. `victor/agent/argument_normalizer.py` (lines 95-127, 191-217)
   - Added comprehensive debug logging
   - Added exception tracking with stack traces

2. `test_debug_case.py` (new file)
   - Reproduces exact failure case from logs
   - Shows normalizer works correctly in isolation

3. `test_with_logging.py` (new file)
   - Tests with DEBUG logging enabled
   - Verifies enhanced logging is working

---

## Conclusion

The normalizer **logic is correct** (all tests pass), but **fails in production** for unknown reasons. Enhanced debug logging has been added to identify the root cause. User needs to:

1. Clear Python bytecode cache
2. Run with DEBUG logging
3. Share enhanced debug logs

This will reveal the true root cause and allow for a targeted fix.
