# Tool Parameter Quick Reference

**Analysis Date**: 2026-04-20
**Total Tools**: 46
**Analysis Scope**: victor/tools + victor-* plugins

---

## Critical Issues (P0)

| Tool | Params | Issue | Impact | Fix |
|------|--------|-------|--------|-----|
| **graph** | 23 | Massive metadata explosion | 23 params, 76-char docstring | Split into mode-specific functions |
| | | | | |

## High Priority (P1)

| Tool | Params | Issue | Impact | Fix |
|------|--------|-------|--------|-----|
| **database** | 13 | Connection params scattered | 12 optional params | Consolidate into config object |
| **code_search** | 11 | Long docstring, many filters | 2113-char docstring | Consolidate filter params |

## Medium Priority (P2)

| Tool | Params | Issue | Impact | Fix |
|------|--------|-------|--------|-----|
| **http** | 11 | Redundant body params | json + data params | Single body param |
| **batch** | 10 | Redundant pattern params | pattern + find | Single pattern param |
| **web_search** | 9 | Scattered fetch params | fetch_top + fetch_pool + max_length | Fetch options object |

## Low Priority (P3) - Acceptable Complexity

| Tool | Params | Reason |
|------|--------|--------|
| **git** | 10 | Git operations inherently complex |
| **jira** | 10 | External service integration |

---

## Full Parameter Count Distribution

```
23 params:  1 tool  (graph)           ❌ CRITICAL
13 params:  1 tool  (database)        ⚠️  HIGH
11 params:  2 tools (code_search, http) ⚠️ HIGH
10 params:  3 tools (git, jira, batch) ⚠️ MEDIUM
9 params:   1 tool  (web_search)      ⚠️ MEDIUM
8 params:   3 tools (read, teams, patch)
7 params:   4 tools (lsp, cicd, scaffold, scan)
6 params:   3 tools (dependency, shell, rename, slack)
5 params:   4 tools (write_lsp, ls, shell_readonly, extract, docs, docs_coverage, edit, metrics)
4 params:   4 tools (docker, pr, find, overview, sandbox)
3 params:   3 tools (workflow, inline, cache, mcp)
2 params:   6 tools (test, write, organize_imports, web_fetch, symbol, refs, web_fetch)
1 param:    3 tools (commit_msg, conflicts, notebook_edit)
```

---

## Red Flag Patterns

### 1. Swiss-Army Knife Tools
Tools with `operation` or `mode` parameters that do multiple things:
- `database` (action: connect, query, schema, exec)
- `git` (operation: status, commit, push, pull, ...)
- `jira` (operation: search, create, update, ...)
- `batch` (operation: find, replace, delete)

**Recommendation**: Split into separate tools

### 2. Redundant Parameters
Multiple ways to specify the same thing:
- `code_search`: `query`, `file`, `symbol` (all search queries)
- `batch`: `pattern`, `find` (redundant)
- `http`: `json`, `data` (both body params)

**Recommendation**: Consolidate into single parameter

### 3. Scattered Related Parameters
Groups of related params that should be objects:
- `database`: `host`, `port`, `username`, `password` → `connection_config`
- `code_search`: `file`, `symbol`, `lang`, `test`, `exts` → `filters`
- `web_search`: `fetch_top`, `fetch_pool`, `max_content_length` → `fetch_options`

**Recommendation**: Create config objects

### 4. Metadata Explosion (Long Docstrings)
Tools with docstrings > 2000 chars:
- `code_search`: 2113 chars
- `scaffold`: 2064 chars

**Recommendation**: Summarize or move details to separate docs

---

## Success Stories (Well-Designed Tools)

### Examples of Excellent Design:
- `write`: 2 params (file, content) - Perfect simplicity
- `web_fetch`: 1-2 params (url, format) - Single responsibility
- `workflow`: 3 params (name, inputs, config) - Clear purpose
- `symbol`: 2 params (query, lang) - Focused

### Common Patterns:
✅ Single responsibility
✅ 1-5 parameters max
✅ No mode/operation params
✅ Concise docstrings (< 1000 chars)
✅ Clear required vs optional

---

## Recommended Action Order

### Week 1: Critical
1. ✅ `shell` limits simplification (DONE - 4→2 params)
2. 🔥 `graph` tool refactoring (23 → 5-7 params per mode)

### Week 2: High Priority
3. 🔥 `database` tool refactoring (13 → 4-6 params)
4. 🔥 `code_search` filter consolidation (11 → 5-6 params)

### Week 3: Medium Priority
5. `http` body param consolidation (11 → 6-7 params)
6. `batch` simplification (10 → 7-8 params)
7. `web_search` fetch consolidation (9 → 6 params)

### Future: Low Priority
8. Consider splitting `git` into separate tools
9. Consider splitting `jira` into separate tools

---

## Impact Summary

### Metadata Explosion Risk
- **Current**: 11 tools with > 7 optional params = HIGH risk
- **After P0-P2 fixes**: 3 tools with > 7 optional params = LOW risk
- **Reduction**: 73% reduction in high-risk tools

### Parameter Count Reduction
- **Current**: Average 5.7 params per tool
- **After P0-P2 fixes**: Average ~4.2 params per tool
- **Reduction**: 26% fewer parameters

### System Prompt Size
- **Estimated**: 20-30% reduction in tool metadata size
- **Benefit**: More tokens for user context, less for tool descriptions

---

## Design Principles Checklist

When creating new tools, ensure:

- [ ] **< 8 parameters** total (aim for 3-5)
- [ ] **< 1000 char docstring** (keep it concise)
- [ ] **No `operation` or `mode` params** (split into separate tools)
- [ ] **No redundant params** (single way to do things)
- [ ] **Related params grouped** (use config objects)
- [ ] **Clear required vs optional** (minimize optional)
- [ ] **Param-to-docstring ratio** (at least 20 chars per param)

---

**Last Updated**: 2026-04-20
**Status**: Ready for review and prioritization
