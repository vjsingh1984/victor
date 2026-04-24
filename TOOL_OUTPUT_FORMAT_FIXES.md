# Tool Output Format Fixes

## Issues Identified

### 1. **Legacy XML Format Bypassing Strategy Pattern** 🔴 CRITICAL

**Problem:** Lines 264-274 in `tool_output_formatter.py` intercept specific tools and use legacy XML format methods, bypassing the new provider-specific strategy pattern.

```python
# Tool-specific formatting (backward compatibility)
if tool_name in ("read_file", "read"):
    return self._format_read_file(args, output, output_str, original_len, truncated)  # ← XML FORMAT

if tool_name == "list_directory":
    return self._format_list_directory(args, output_str)  # ← XML FORMAT

if tool_name in ("code_search", "semantic_code_search"):
    return self._format_code_search(tool_name, args, output_str, output)  # ← XML FORMAT

if tool_name == "execute_bash":
    return self._format_bash(args, output_str)  # ← XML FORMAT
```

**Impact:**
- ❌ Z.AI and other cloud providers receive XML format instead of Plain JSON
- ❌ Token optimization not working (10-15% extra cost)
- ❌ New strategy pattern architecture not being used

**Fix:** Remove special-case handlers and use strategy pattern for all tools.

---

### 2. **Variable Scope Error in Exception Handler** 🔴 CRITICAL

**Problem:** Line 2008 in `chat.py` references `provider` variable that doesn't exist in exception handler scope.

```python
except Exception as e:
    # ... exception handling ...
    if os.getenv("VICTOR_DEBUG"):  # Line 2008
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'os' where it is not associated with a value
```

**Fix:** Add `import os` at the top of the exception handler.

---

### 3. **Code Search Timeout** 🔴 HIGH

**Problem:** `code_search` tool timing out after 60 seconds.

**Potential Causes:**
- Large codebase indexing
- Network latency to embedding service
- Inefficient vector search

**Fix:** Increase timeout or optimize search query.

---

## Fix #1: Remove Legacy XML Format Handlers

**File:** `victor/agent/tool_output_formatter.py`

**Before:**
```python
# Tool-specific formatting (backward compatibility)
if tool_name in ("read_file", "read"):
    return self._format_read_file(args, output, output_str, original_len, truncated)

if tool_name == "list_directory":
    return self._format_list_directory(args, output_str)

if tool_name in ("code_search", "semantic_code_search"):
    return self._format_code_search(tool_name, args, output_str, output)

if tool_name == "execute_bash":
    return self._format_bash(args, output_str)

# Use strategy pattern for generic formatting
return self._format_with_strategy(tool_name, args, output_str, truncated, format_hint, context)
```

**After:**
```python
# Use strategy pattern for ALL tools (provider-specific optimization)
return self._format_with_strategy(tool_name, args, output_str, truncated, format_hint, context)
```

**Benefits:**
- ✅ All tools use provider-specific format (Plain for cloud, XML for local)
- ✅ Token optimization works for all tools
- ✅ Consistent behavior across all tools
- ✅ Architecture simplified (single code path)

**Trade-off:**
- Loses some helpful formatting like "ACTUAL FILE CONTENT" headers
- Loses pagination guidance for large files
- Loses follow-up suggestions for code search

**Mitigation:**
- Move helpful formatting to UI layer (renderer) instead of tool output
- Keep guidance in preview/tooltip, not in LLM context
- Use strategy pattern with custom hints if needed

---

## Fix #2: Add Missing Import

**File:** `victor/ui/commands/chat.py`

**Line 2008** (in exception handler):

**Before:**
```python
except Exception as e:
    logger.exception(f"Unexpected error in CLI REPL: {e}")
    if os.getenv("VICTOR_DEBUG"):  # ← UnboundLocalError
        import traceback
        traceback.print_exc()
```

**After:**
```python
except Exception as e:
    import os  # ← Add import
    logger.exception(f"Unexpected error in CLI REPL: {e}")
    if os.getenv("VICTOR_DEBUG"):
        import traceback
        traceback.print_exc()
```

---

## Fix #3: Optimize Code Search Timeout

**Options:**

### Option A: Increase Timeout
```python
# In tool configuration or settings
code_search_timeout = 120  # Increase from 60 to 120 seconds
```

### Option B: Add Async Cancellation
```python
# Allow user to cancel long-running searches with Ctrl+C
async def code_search_with_cancellation(query):
    try:
        return await code_search(query)
    except asyncio.CancelledError:
        logger.info("Code search cancelled by user")
        return None
```

### Option C: Optimize Search
```python
# Use approximate search for large codebases
if codebase_size > 10000_files:
    use_approximate_search = True
    max_results = 20
```

---

## Implementation Priority

1. **Fix #1 (Legacy XML)** - CRITICAL - Breaks token optimization
2. **Fix #2 (Import Error)** - CRITICAL - Crashes on exception
3. **Fix #3 (Timeout)** - HIGH - User experience issue

---

## Testing

After fixes, verify:

```bash
# Test 1: Verify Plain format for cloud providers
python -c "
from victor.providers.zai_provider import ZAIProvider
provider = ZAIProvider(api_key='test')
format_spec = provider.get_tool_output_format()
assert format_spec.style == 'plain', f'Expected plain, got {format_spec.style}'
print('✅ Z.AI uses Plain format')
"

# Test 2: Verify XML format for local providers
python -c "
from victor.providers.ollama_provider import OllamaProvider
provider = OllamaProvider(base_url='http://localhost:11434')
format_spec = provider.get_tool_output_format()
assert format_spec.style == 'xml', f'Expected xml, got {format_spec.style}'
print('✅ Ollama uses XML format')
"

# Test 3: Run format tests
pytest tests/unit/agent/test_provider_format_integration.py -v
```

---

## Rollout Plan

1. **Phase 1:** Fix import error (5 minutes, low risk)
2. **Phase 2:** Remove legacy XML handlers (15 minutes, medium risk)
3. **Phase 3:** Test with cloud provider (10 minutes)
4. **Phase 4:** Test with local provider (10 minutes)
5. **Phase 5:** Address code search timeout (20 minutes)

**Total time:** ~1 hour

**Risk assessment:**
- Import fix: Low risk (pure addition)
- XML handler removal: Medium risk (behavior change)
  - Mitigation: Feature flag to re-enable if needed
- Timeout increase: Low risk (pure configuration)

---

## Success Criteria

✅ Z.AI provider uses Plain JSON format (not XML)
✅ Token optimization works (10-15% savings)
✅ No more UnboundLocalError on exceptions
✅ Code search completes within timeout or can be cancelled
✅ All tests passing (73 format tests)
