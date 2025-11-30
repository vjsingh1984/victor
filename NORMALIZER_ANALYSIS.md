# Argument Normalizer - Real-World Testing Analysis

**Date**: November 27, 2025
**Status**: 🔍 **INVESTIGATION COMPLETE** - Mixed Results

---

## Quick Test Results: ✅ 100% SUCCESS

```
Test 1: Valid JSON                  ✓ PASSED (strategy: direct)
Test 2: Python dict syntax          ✓ PASSED (strategy: python_ast)
Test 3: Nested structures           ✓ PASSED (strategy: python_ast)
Test 4: Real Qwen model output      ✓ PASSED (strategy: python_ast)

Statistics:
  Total calls: 4
  Normalizations: {'direct': 1, 'python_ast': 3, 'regex_quotes': 0, 'manual_repair': 0, 'failed': 0}
  Success rate: 100.0%
```

---

## Real-World Integration Test: ⚠️ PARTIAL FAILURE

### Test Command
```bash
victor main --log-level WARN "Create a bash routine to generate fibonacci series for n=27"
```

### Observed Behavior

**Timeline of Events**:

1. **Tool Call 1** (23:35:35): `edit_files` - ❌ **FAILED NORMALIZATION**
   ```
   ERROR - [OllamaProvider] Failed to normalize edit_files arguments after all strategies.
   Original: {'operations': '[{\'type\': \'modify\', \'path\': \'fibonacci.sh\', \'content\': \'#!/bin/bash\n\n...'}
   ```
   - Strategy used: `failed`
   - Arguments returned unchanged
   - Tool execution: ✓ Success (but arguments were malformed)

2. **Tool Call 2** (23:35:43): `edit_files` - ✅ **SUCCESS**
   ```
   {'operations': '[{"type": "modify", "path": "fibonacci.sh", "content": "#!/bin/bash\\n\\n..."}]'}
   ```
   - No normalization warning (passed validation)
   - Tool execution: ✓ Success

3. **Tool Call 3** (23:35:51): `edit_files` - ❌ **FAILED NORMALIZATION**
   ```
   ERROR - [OllamaProvider] Failed to normalize edit_files arguments after all strategies.
   Original: {'operations': '[{"type": "modify", "path": "fibonacci.sh", "content": "#!/bin/bash\\n\\n..."}]'}
   ```
   - Strategy used: `failed`
   - Arguments have valid JSON structure but double-escaped characters

4. **Subsequent Calls**: Multiple `edit_files` calls, mix of success and partial failures

---

## Root Cause Analysis

### Failure Pattern 1: Escaped Single Quotes
**Example from logs (23:35:35)**:
```python
'[{\'type\': \'modify\', \'path\': \'fibonacci.sh\', \'content\': \'#!/bin/bash\n\n...\'}]'
```

**Analysis**:
- Contains backslash-escaped single quotes: `\'`
- Should be normalized via `_normalize_via_regex` (Layer 3)
- **OUR TEST**: ✅ Passed using `regex_quotes` strategy
- **REAL INTEGRATION**: ❌ Failed (all strategies)

**Hypothesis**: The string representation in logs vs. actual Python object may differ. When JSON parses the Ollama response, escape sequences are interpreted differently than in our test.

### Failure Pattern 2: Double-Escaped Characters
**Example from logs (23:35:51)**:
```python
'[{"type": "modify", "path": "fibonacci.sh", "content": "#!/bin/bash\\n\\n# Function"}]'
```

**Analysis**:
- Has valid JSON double quotes
- Contains double-escaped newlines: `\\n` (literal backslash-n) instead of `\n` (newline escape)
- Contains double-escaped quotes: `\\"` instead of `\"`
- **OUR TEST**: ✅ Passed (considered valid JSON by `json.loads()`)
- **REAL INTEGRATION**: ❌ Failed normalization

**Hypothesis**: The `edit_files` tool's JSON parser is stricter than Python's `json.loads()`. It may reject `\\n` as invalid JSON escapesequence.

---

## Critical Discovery: Discrepancy Between Test and Reality

### Why Our Tests Pass But Integration Fails

1. **Test Environment**:
   ```python
   failure1 = {
       "operations": "[{\\'type\\': \\'modify\\'...}]"
   }
   ```
   - We create a Python string object with escaped quotes
   - `json.loads()` successfully parses it
   - Normalizer marks it as valid

2. **Real Integration**:
   - Ollama returns JSON: `{"arguments": {"operations": "[{\'type\': \'modify\'}]"}}`
   - When Python's `json.loads()` parses this, it converts `\'` to actual single-quote character
   - The resulting Python string contains literal single quotes, NOT escaped quotes
   - Our normalizer receives: `"[{'type': 'modify'}]"` (not `"[{\\'type\\': \\'modify\\'}]"`)

3. **The Validation Gap**:
   - Our `_is_valid_json_dict()` only validates that the dict can be JSON-serialized
   - It validates string values that LOOK like JSON (start with `[` or `{`)
   - But `edit_files` tool does ADDITIONAL validation that we're not catching

---

## Action Items

### ✅ Completed
- [x] Create ArgumentNormalizer with multi-layer pipeline
- [x] Integrate with orchestrator
- [x] Unit tests (15 test cases)
- [x] Quick validation tests
- [x] Identified failure patterns from real logs

### 🔄 In Progress
- [ ] Debug why integration tests fail while unit tests pass
- [ ] Understand the exact format Ollama returns
- [ ] Identify what validation `edit_files` tool performs

### 📋 To Do
- [ ] Add debug logging to show exact bytes received from Ollama
- [ ] Enhance validation to match `edit_files` tool's requirements
- [ ] Add integration test that calls actual Ollama API
- [ ] Fix the validation gap

---

## Potential Fixes

### Fix 1: Enhanced Deep Validation
Add validation that matches `edit_files` tool's requirements:

```python
def _is_valid_json_dict(self, obj: Any) -> bool:
    try:
        # Serialize to JSON
        json_str = json.dumps(obj)

        # Additionally, check string values that look like JSON
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith(('[', '{')):
                        # Try parsing with STRICT=True
                        parsed = json.loads(value, strict=True)
                        # For edit_files, also validate it can be re-serialized
                        json.dumps(parsed)
        return True
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
```

### Fix 2: Pre-Processing for Double-Escapes
Add a preprocessing step before validation:

```python
def _preprocess_double_escapes(self, value: str) -> str:
    """Handle double-escaped characters (e.g., \\n → \n)."""
    # Replace \\n with \n, \\t with \t, etc.
    replacements = {
        '\\\\n': '\\n',
        '\\\\t': '\\t',
        '\\\\r': '\\r',
        '\\\\"': '\\"',
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value
```

### Fix 3: Add Debug Logging
Log the exact representation:

```python
logger.debug(f"Argument type: {type(arguments)}")
logger.debug(f"Argument repr: {repr(arguments)}")
logger.debug(f"Argument bytes: {arguments.encode('unicode_escape')}")
```

---

## Test Plan

1. **Add Ollama API Integration Test**:
   - Call real Ollama API with qwen3-coder:30b
   - Capture exact arguments received
   - Test normalization on real data

2. **Add Stricter Validation**:
   - Match `edit_files` tool's validation
   - Test with actual tool execution

3. **Compare with Successful Cases**:
   - Log arguments from successful `edit_files` calls
   - Compare with failed calls
   - Identify exact differences

---

## Conclusion

**Current Status**: The normalizer's LOGIC is correct (unit tests pass), but there's a VALIDATION GAP:
- Our validation: "Can this be JSON-serialized?"
- Tool's validation: "Can this be parsed by `json.loads()` AND used by the tool?"

**Next Steps**:
1. Debug the exact format received from Ollama
2. Enhance validation to match tool requirements
3. Add integration tests with real Ollama API

**Priority**: **HIGH** - This is blocking execution workflows with Qwen/Ollama models.
