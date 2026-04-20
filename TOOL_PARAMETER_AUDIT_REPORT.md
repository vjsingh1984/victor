# Tool Parameter Audit Report

**Date**: 2026-04-20
**Scope**: Victor framework tools + victor-* plugins
**Total Tools Analyzed**: 46 tools
**Status**: ✅ Complete

---

## Executive Summary

**Key Findings**:
- **Average parameters per tool**: 5.7
- **Maximum parameters in a tool**: 23 (graph tool)
- **Tools with excessive parameters (>8)**: 8 tools
- **Tools with excessive optional parameters (>6)**: 11 tools
- **Tools with very long docstrings (>2000 chars)**: 2 tools
- **Critical metadata explosion risk**: 1 tool (graph: 23 params, 76 char docstring)

**Recommendation Priority**:
1. **CRITICAL**: graph tool (23 params) - Immediate simplification needed
2. **HIGH**: database tool (13 params) - Consolidate connection params
3. **HIGH**: code_search tool (11 params) - Remove redundant params
4. **MEDIUM**: http, git, jira, batch, web_search (8-11 params each)

---

## Detailed Findings by Category

### 🔴 CRITICAL: Metadata Explosion Risk

#### 1. graph tool (23 parameters) - HIGHEST PRIORITY
**Source**: `victor/tools/graph_tool.py`
**Parameters**: 23 total, 0 required, 23 optional
**Docstring**: 76 characters

**Current Parameters**:
```python
mode, path, node, source, target, file, query, depth, top_k,
direction, edge_types, reindex, only_runtime, files_only,
modules_only, structured, include_modules, include_symbols,
include_calls, include_refs, include_callsites, max_callsites
```

**Problems**:
- ❌ **CRITICAL**: 23 optional parameters with only 76-char docstring
- ❌ Massive metadata explosion in system prompts
- ❌ Many parameters are mode-specific (should be conditional)
- ❌ Multiple boolean flags that could be combined
- ❌ Poor param-to-docstring ratio (3.3 chars per param)

**Recommendations**:
1. **Create mode-specific functions** instead of one monolithic function:
   - `graph_neighbors(node, depth=2, direction="out")`
   - `graph_path(source, target, depth=5)`
   - `graph_search(query, top_k=10)`
   - `graph_stats()`
   - `graph_pagerank(top_k=10)`

2. **Consolidate boolean flags** into a single config object:
   ```python
   # Before: files_only, modules_only, include_modules, include_symbols, include_calls
   # After:
   filter_options = {
       "node_types": ["file", "module", "symbol"],  # What to include
       "edge_types": ["calls", "refs"]              # What to include
   }
   ```

3. **Remove redundant parameters**:
   - `source` and `target` can replace `node` for most modes
   - `query` and `node` and `file` are redundant (search query)

4. **Impact**: Reduce from 23 → 5-7 parameters per mode-specific function

**Estimated Effort**: 4-6 hours (breaking change, requires migration)

---

### 🟠 HIGH: Excessive Parameters

#### 2. database tool (13 parameters)
**Source**: `victor/tools/database_tool.py`
**Parameters**: 13 total, 1 required, 12 optional
**Docstring**: 1968 characters

**Current Parameters**:
```python
action, database, db_type, host, port, username, password,
connection_id, sql, table, limit, allow_modifications
```

**Problems**:
- ❌ Connection params (host, port, username, password) could be a single config
- ❌ `action` parameter makes it a swiss-army knife tool
- ❌ Multiple actions (connect, query, schema, exec) could be separate tools

**Recommendations**:
1. **Create connection config object**:
   ```python
   # Before: host, port, username, password, db_type
   # After:
   connection_config = {
       "host": "localhost",
       "port": 5432,
       "username": "user",
       "password": "pass",
       "db_type": "postgresql"
   }
   ```

2. **Split into mode-specific tools**:
   - `database_connect(config)` - Establish connection
   - `database_query(sql, limit)` - Execute query
   - `database_schema(table)` - Get schema info
   - `database_exec(sql, allow_modifications)` - Execute modifications

**Impact**: Reduce from 13 → 4-6 parameters per tool

**Estimated Effort**: 3-4 hours (breaking change)

---

#### 3. code_search tool (11 parameters)
**Source**: `victor/tools/code_search_tool.py`
**Parameters**: 11 total, 1 required, 10 optional
**Docstring**: 2113 characters

**Current Parameters**:
```python
query, path, k, mode, reindex, file, symbol, lang, test, exts
```

**Problems**:
- ❌ `file`, `symbol`, `lang`, `test`, `exts` are filter params (could be single object)
- ❌ `mode` parameter has multiple values (semantic, text, filename)
- ❌ Very long docstring (2113 chars) contributes to metadata explosion

**Recommendations**:
1. **Consolidate filter params**:
   ```python
   # Before: file, symbol, lang, test, exts
   # After:
   filters = {
       "file_pattern": "*.py",
       "symbol": " MyClass",
       "language": "python",
       "test_only": True,
       "extensions": [".py", ".pyx"]
   }
   ```

2. **Remove redundant params**:
   - `mode` could be inferred from query format (semantic vs text vs filename)

**Impact**: Reduce from 11 → 5-6 parameters

**Estimated Effort**: 2-3 hours (backward compatible with deprecation)

---

### 🟡 MEDIUM: Moderate Parameter Count

#### 4. http tool (11 parameters)
**Source**: `victor/tools/http_tool.py`
**Parameters**: 11 total, 2 required, 9 optional
**Docstring**: 1413 characters

**Current Parameters**:
```python
method, url, mode, headers, params, json, data, auth,
follow_redirects, timeout, expected_status
```

**Problems**:
- ⚠️ `json` and `data` are redundant (use single `body` param)
- ⚠️ `mode` parameter (request vs test) could be separate tools
- ⚠️ Auth could be simplified (basic auth vs token)

**Recommendations**:
1. **Consolidate body params**: `json` and `data` → `body` (auto-detect content-type)
2. **Split test mode**: Create separate `http_test` tool
3. **Simplify auth**: Single `auth` param (string or dict)

**Impact**: Reduce from 11 → 6-7 parameters

**Estimated Effort**: 2 hours (backward compatible)

---

#### 5. git tool (10 parameters)
**Source**: `victor/tools/git_tool.py`
**Parameters**: 10 total, 1 required, 9 optional
**Docstring**: 260 characters

**Current Parameters**:
```python
operation, files, message, branch, staged, limit,
options, author_name, author_email, context
```

**Problems**:
- ⚠️ `operation` is a swiss-army knife (status, commit, push, pull, etc.)
- ⚠️ `author_name` and `author_email` could be a single `author` dict
- ⚠️ `options` is a catch-all (could be more specific)

**Recommendations**:
1. **Keep as-is** for now - git operations are inherently complex
2. **Future**: Consider splitting into `git_commit`, `git_push`, `git_status`, etc.

**Impact**: Low priority - acceptable complexity for git operations

**Estimated Effort**: 4-6 hours if splitting (future work)

---

#### 6. jira tool (10 parameters)
**Source**: `victor/tools/jira_tool.py`
**Parameters**: 10 total, 1 required, 9 optional
**Docstring**: 880 characters

**Current Parameters**:
```python
operation, jql, issue_key, summary, project, issue_type,
description, comment, max_results, context
```

**Problems**:
- ⚠️ `operation` is swiss-army knife (search, create, update, comment)
- ⚠️ Issue creation params could be a single object

**Recommendations**:
1. **Keep as-is** - Jira integration is inherently complex
2. **Future**: Consider splitting into `jira_search`, `jira_create`, `jira_update`

**Impact**: Low priority - acceptable for external service integration

**Estimated Effort**: 3-4 hours if splitting (future work)

---

#### 7. batch tool (10 parameters)
**Source**: `victor/tools/batch_processor_tool.py`
**Parameters**: 10 total, 2 required, 8 optional
**Docstring**: 1755 characters

**Current Parameters**:
```python
operation, path, file_pattern, pattern, find, replace,
regex, dry_run, max_files, options
```

**Problems**:
- ⚠️ Multiple operation modes (find, replace, delete)
- ⚠️ `pattern` and `find` are redundant
- ⚠️ `options` is a catch-all

**Recommendations**:
1. **Simplify find/replace**: Use single `pattern` param
2. **Remove `options`**: Make params explicit

**Impact**: Reduce from 10 → 7-8 parameters

**Estimated Effort**: 1-2 hours

---

#### 8. web_search tool (9 parameters)
**Source**: `victor/tools/web_search_tool.py`
**Parameters**: 9 total, 1 required, 8 optional
**Docstring**: 1296 characters

**Current Parameters**:
```python
query, max_results, region, safe_search, ai_summarize,
fetch_top, fetch_pool, max_content_length, _exec_ctx
```

**Problems**:
- ⚠️ `fetch_top`, `fetch_pool`, `max_content_length` are related (content fetching)
- ⚠️ `ai_summarize` is a separate feature (could be separate tool)

**Recommendations**:
1. **Consolidate fetch params**:
   ```python
   # Before: fetch_top, fetch_pool, max_content_length
   # After:
   fetch_options = {
       "top_k": 5,
       "pool_size": 3,
       "max_length": 10000
   }
   ```

2. **Consider splitting**: `web_search` vs `web_search_and_summarize`

**Impact**: Reduce from 9 → 6 parameters

**Estimated Effort**: 1-2 hours

---

### 🟢 GOOD: Well-Designed Tools

#### Examples of Good Design (< 5 parameters):
- `write`: 2 params (file, content) - Perfect
- `web_fetch`: 1-2 params - Excellent
- `workflow`: 3 params - Good
- `symbol`: 2 params - Perfect
- `cache`: 3 params - Good

**Common Patterns**:
- Single responsibility
- Clear required vs optional params
- No swiss-army knife `operation` param
- Concise docstrings

---

## Summary Statistics

### Parameter Distribution
```
1-3 params:  15 tools (33%)  ✅ Excellent
4-5 params:  13 tools (28%)  ✅ Good
6-8 params:  10 tools (22%)  ⚠️  Acceptable
9+ params:   8 tools (17%)   ❌ Needs improvement
```

### Metadata Explosion Risk
```
High risk (>10 params):     3 tools (7%)   ❌ Critical
Medium risk (7-10 params): 11 tools (24%)  ⚠️  Monitor
Low risk (<7 params):      32 tools (70%)  ✅ Good
```

### Docstring Length
```
> 2000 chars: 2 tools  ⚠️  Consider summarizing
1000-2000 chars: 11 tools ⚠️  Acceptable
< 1000 chars: 33 tools   ✅  Good
```

---

## Recommendations by Priority

### P0 - CRITICAL (Immediate Action Required)
1. **graph tool** (23 params) - Break into mode-specific functions
   - **Impact**: Highest metadata explosion risk
   - **Effort**: 4-6 hours
   - **Breaking**: Yes

### P1 - HIGH (Next Sprint)
2. **database tool** (13 params) - Split connection config, create mode-specific tools
   - **Impact**: High metadata, complex API
   - **Effort**: 3-4 hours
   - **Breaking**: Yes

3. **code_search tool** (11 params) - Consolidate filter params
   - **Impact**: Long docstring, many params
   - **Effort**: 2-3 hours
   - **Breaking**: No (backward compatible)

### P2 - MEDIUM (Backlog)
4. **http tool** (11 params) - Consolidate body params, split test mode
   - **Effort**: 2 hours
   - **Breaking**: No

5. **batch tool** (10 params) - Simplify find/replace, remove options
   - **Effort**: 1-2 hours
   - **Breaking**: No

6. **web_search tool** (9 params) - Consolidate fetch params
   - **Effort**: 1-2 hours
   - **Breaking**: No

### P3 - LOW (Future Consideration)
7. **git tool** (10 params) - Acceptable as-is, consider splitting later
8. **jira tool** (10 params) - Acceptable as-is for external service

---

## Design Principles for Tool Parameters

### ✅ DO:
1. **Single responsibility** - One tool, one job
2. **Few parameters** - Aim for 3-5 parameters max
3. **Explicit over implicit** - Avoid `operation` or `mode` params when possible
4. **Consolidate related params** - Use config objects for groups
5. **Concise docstrings** - Keep under 1000 chars when possible

### ❌ DON'T:
1. **Swiss-army knife tools** - Avoid tools with many modes/operations
2. **Redundant parameters** - `limit`, `limit_stdout`, `limit_stderr`, `unlimited`
3. **Catch-all params** - `options` dicts that hide complexity
4. **Long parameter lists** - More than 8 parameters is a code smell
5. **Poor param-to-docstring ratio** - At least 20 chars per parameter

---

## Estimated Total Effort

- **P0 (Critical)**: 4-6 hours
- **P1 (High)**: 5-7 hours
- **P2 (Medium)**: 4-6 hours
- **P3 (Low)**: 7-10 hours (if done)

**Total**: 20-29 hours to address all high and medium priority issues

---

## Next Steps

1. **Immediate**: Address P0 (graph tool) - highest metadata explosion risk
2. **This sprint**: Address P1 (database, code_search)
3. **Next sprint**: Address P2 (http, batch, web_search)
4. **Monitor**: P3 tools for future refactoring opportunities

---

## Appendix: Full Tool Table

| Tool | Params | Required | Optional | Doc Length | Priority |
|------|--------|----------|----------|------------|----------|
| graph | 23 | 0 | 23 | 76 | P0 |
| database | 13 | 1 | 12 | 1968 | P1 |
| code_search | 11 | 1 | 10 | 2113 | P1 |
| http | 11 | 2 | 9 | 1413 | P2 |
| git | 10 | 1 | 9 | 260 | P3 |
| jira | 10 | 1 | 9 | 880 | P3 |
| batch | 10 | 2 | 8 | 1755 | P2 |
| web_search | 9 | 1 | 8 | 1296 | P2 |
| read | 8 | 1 | 7 | 1344 | - |
| teams | 8 | 1 | 7 | 693 | - |
| patch | 8 | 0 | 8 | 666 | - |
| lsp | 7 | 1 | 6 | 217 | - |
| cicd | 7 | 1 | 6 | 1628 | - |
| scaffold | 7 | 1 | 6 | 2064 | - |
| scan | 7 | 1 | 6 | 1422 | - |
| dependency | 6 | 1 | 5 | 241 | - |
| shell | 6 | 1 | 5 | 535 | ✅ |
| rename | 6 | 2 | 4 | 1705 | - |
| slack | 6 | 1 | 5 | 545 | - |
| ... | ... | ... | ... | ... | ... |

(See detailed output from `analyze_tools.py` for complete table)

---

**Report Generated**: 2026-04-20
**Tool Version**: 1.0
**Analysis Scope**: victor/tools + victor-* plugins
