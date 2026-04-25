# Rich Formatting Candidates - Tool Output Enhancement Analysis

## Overview

This document analyzes all tools in the Victor codebase to identify candidates for Rich formatting enhancements (similar to the diff preview implemented for the edit tool). The goal is to improve UX by providing color-coded, syntax-highlighted output for tools that return structured data.

## Priority Matrix

| Priority | Tool | Category | Impact | Complexity | Status |
|----------|------|----------|--------|------------|--------|
| **P0** | `git_tool.py` | Git ops | High | Medium | ✅ Done |
| **P0** | `testing_tool.py` | Testing | High | Low | 🔄 Todo |
| **P0** | `code_search_tool.py` | Search | High | Medium | 🔄 Todo |
| **P1** | `http_tool.py` | HTTP/API | Medium | Low | 🔄 Todo |
| **P1** | `database_tool.py` | Database | Medium | Medium | 🔄 Todo |
| **P1** | `refactor_tool.py` | Refactoring | High | High | 🔄 Todo |
| **P2** | `docker_tool.py` | Docker | Medium | Medium | 🔄 Todo |
| **P2** | `security_scanner_tool.py` | Security | High | High | 🔄 Todo |
| **P2** | `documentation_tool.py` | Docs | Low | Medium | 🔄 Todo |
| **P3** | `metrics_tool.py` | Metrics | Low | Low | 🔄 Todo |
| **P3** | `graph_tool.py` | Visualization | Medium | High | 🔄 Todo |
| **P3** | `bash.py` | Shell | Low | Low | ✅ Done (basic) |

---

## P0 Candidates (High Impact, Critical UX)

### 1. ✅ `edit` (file_editor_tool.py) - **COMPLETED**
**Current Status:** ✅ Implemented with Rich diff formatting

**What was done:**
- Added `_format_diff_for_console()` to convert unified diff to Rich markup
- Returns both `diff` (raw) and `diff_formatted` (Rich-formatted)
- Green for additions, Red for deletions, Cyan for file paths, Dim for context
- `_DiffPreviewStrategy` updated to use formatted diff from result

**Output Example:**
```
│ +1 -1
│ [dim]@@ -1 +1 @@[/]
│ [red]-x = 1[/]
│ [green]+x = 99[/]
```

---

### 2. 🔄 `testing_tool.py` - Test Results
**Current Output:** Plain JSON with test counts and error messages
```python
{
    "summary": {"total_tests": 10, "passed": 8, "failed": 2, "skipped": 0},
    "failures": [
        {"test_name": "test_func", "error_message": "AssertionError", "full_error": "..."}
    ]
}
```

**Proposed Rich Enhancement:**
```python
{
    "summary": {...},
    "failures": [...],
    "formatted_summary": """
[green]✓ 8 passed[/] [dim]•[/] [red]✗ 2 failed[/] [dim]•[/] [yellow]○ 0 skipped[/]

[red bold]Failed Tests:[/]
  [red]✗[/] [bold]test_func[/]
    [dim]AssertionError: Expected 5 but got 3[/]
    [dim cyan]path/to/test.py:42[/]

  [red]✗[/] [bold]test_another[/]
    [dim]ValueError: Invalid input[/]
    [dim cyan]path/to/test.py:78[/]
"""
}
```

**Benefits:**
- **Immediate visual feedback** on test health (green/red/yellow)
- **Failure hierarchy** with test names and locations
- **Scannable error messages** with color-coded severity
- **Faster debugging** with file:line highlighting

**Implementation:**
- Add `_format_test_summary()` function
- Color-code: ✓ green (passed), ✗ red (failed), ○ yellow (skipped)
- Dim error messages, bold test names, cyan file paths
- Preview strategy shows summary + first 2 failures

---

### 3. 🔄 `code_search_tool.py` - Code Search Results
**Current Output:** Plain list of matches with scores
```python
{
    "results": [
        {"path": "foo.py", "line": 42, "score": 10, "snippet": "def foo():"}
    ]
}
```

**Proposed Rich Enhancement:**
```python
{
    "results": [...],
    "formatted_results": """
[bold cyan]3 matches[/] [dim]in 2 files[/]

  [bold]foo.py[/] [dim]• score: 10[/]
  [dim]42:[/][green]def foo[/]():
         [dim]^ match here[/]

  [bold]bar.py[/] [dim]• score: 8[/]
  [dim]15:[/][green]def foo[/]():
         [dim]^ match here[/]

  [bold]baz.py[/] [dim]• score: 6[/]
  [dim]99:[/][green]def foo[/]():
         [dim]^ match here[/]
"""
}
```

**Benefits:**
- **Visual hierarchy** with file paths (bold cyan) and scores (dim)
- **Match highlighting** with green/cyan for found terms
- **Line numbers** in dim for reference
- **Scannable results** with consistent formatting

**Implementation:**
- Add `_format_search_results()` function
- Bold cyan for file paths, green for matches, dim for line numbers
- Preview strategy shows match count + top 3 results
- Support for semantic, literal, and regex modes

---

### 4. 🔄 `git_tool.py` - Git Operations
**Current Output:** Plain text from git commands
```python
{
    "output": "* main\n  feature-branch\n"
}
```

**Proposed Rich Enhancement:**
```python
{
    "output": "...",
    "formatted_output": """
[bold green]*[/] [bold]main[/] [dim]•[/] [green dim]↑3 ↓2[/]
  [bold]feature-branch[/] [dim]•[/] [yellow dim]↑1 ↓0[/]

[dim]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─[/]
[bold]Recent commits:[/]
  [cyan]a1b2c3d[/] [dim]Fix bug in parser[/] [dim gray](2h ago)[/]
  [cyan]e5f6g7h[/] [dim]Add new feature[/] [dim gray](1d ago)[/]
"""
}
```

**Benefits:**
- **Branch highlighting** with current branch (green bold)
- **Status indicators** with ↑ additions ↓ deletions
- **Commit graph** with colors for hashes (cyan)
- **Relative timestamps** in dim gray

**Implementation:**
- Add `_format_git_status()`, `_format_git_log()`, `_format_git_diff()`
- Green bold for current branch, cyan for commit hashes
- Diff output reuses existing `_DiffPreviewStrategy`
- Preview strategies for each operation type

---

## P1 Candidates (Medium Impact, High Value)

### 5. 🔄 `http_tool.py` - HTTP/API Requests
**Current Output:** Plain JSON with status code and body
```python
{
    "status_code": 200,
    "status": "OK",
    "headers": {"content-type": "application/json"},
    "body": {...},
    "duration_ms": 123
}
```

**Proposed Rich Enhancement:**
```python
{
    "status_code": 200,
    "formatted_response: """
[green bold]200 OK[/] [dim]• 123ms[/]

[dim]Headers:[/]
  [cyan]content-type:[/] [dim]application/json[/]
  [cyan]content-length:[/] [dim]1024[/]

[dim]Body:[/]
  {
    [cyan]"key":[/] [green]"value"[/]
  }
"""
}
```

**Benefits:**
- **Status code colors** (green 2xx, yellow 3xx, red 4xx/5xx)
- **Header highlighting** with cyan keys
- **JSON syntax highlighting** in body
- **Performance metrics** visible (duration)

**Implementation:**
- Add `_format_http_response()` function
- Color status codes: 2xx green, 3xx yellow, 4xx/5xx red
- Cyan for header keys, dim for values
- Reuse existing JSON syntax highlighting
- Preview strategy shows status + duration

---

### 6. 🔄 `database_tool.py` - Database Query Results
**Current Output:** Plain list of rows
```python
{
    "rows": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"}
    ],
    "row_count": 2
}
```

**Proposed Rich Enhancement:**
```python
{
    "rows": [...],
    "formatted_table": """
[dim]┌───┬───────┬─────────────────────┐[/]
[dim]│[/] [bold]id[/] [dim]│[/] [bold] name[/] [dim]│[/] [bold]email              [/] [dim]│[/]
[dim]├───┼───────┼─────────────────────┤[/]
[dim]│[/] 1   [dim]│[/] Alice  [dim]│[/] alice@example.com [dim]│[/]
[dim]│[/] 2   [dim]│[/] Bob    [dim]│[/] bob@example.com   [dim]│[/]
[dim]└───┴───────┴─────────────────────┘[/]

[dim]2 rows in 0.123s[/]
"""
}
```

**Benefits:**
- **Table formatting** with borders and alignment
- **Column headers** in bold
- **Row highlighting** for readability
- **Query performance** visible (duration)

**Implementation:**
- Add `_format_table()` function using Rich's Table
- Bold headers, aligned columns
- Alternating row colors (optional)
- Preview shows first 5 rows + row count

---

### 7. 🔄 `refactor_tool.py` - Refactoring Operations
**Current Output:** Plain text summary of changes
```python
{
    "operations": ["rename: foo -> bar", "extract method: baz()"]
}
```

**Proposed Rich Enhancement:**
```python
{
    "operations": [...],
    "formatted_summary": """
[bold cyan]Refactoring Plan:[/]

  [green]✓[/] [bold]Rename:[/] [dim]foo[/] [dim]→[/] [dim]bar[/]
    [dim]2 occurrences updated[/]

  [green]✓[/] [bold]Extract method:[/] [dim]baz()[/]
    [dim]From: MyClass.method() (lines 42-50)[/]
    [dim]To: MyClass.baz() (new method)[/]

[dim]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─[/]
[green]2 operations planned[/] [dim]•[/] [yellow]0 conflicts detected[/]
"""
}
```

**Benefits:**
- **Operation types** color-coded (rename, extract, inline)
- **Before/after** visualization with dim formatting
- **Conflict detection** with yellow/red highlighting
- **Change summary** with counts

**Implementation:**
- Add `_format_refactor_plan()` function
- Green for safe ops, yellow for warnings, red for conflicts
- Dim for before/after comparisons
- Preview strategy shows operation count + first 3 ops

---

## P2 Candidates (Medium Impact, Specialized Use Cases)

### 8. 🔄 `docker_tool.py` - Docker Operations
**Current Output:** Plain text from docker commands
```python
{
    "output": "CONTAINER ID   IMAGE     STATUS   ...\nabc123      nginx     Up       ..."
}
```

**Proposed Rich Enhancement:**
```python
{
    "formatted_output: """
[bold cyan]Containers:[/]

  [green]↑[/] [bold]abc123[/] [dim]•[/] [bold]nginx[/] [dim]•[/] [green]Up[/] [dim]2 hours[/]
    [dim]Ports: 80:80, 443:443[/]

  [red]↓[/] [bold]def456[/] [dim]•[/] [bold]redis[/] [dim]•[/] [red]Stopped[/]
    [dim]Exit code: 1[/]

[dim]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─[/]
[dim]2 containers • 1 running • 1 stopped[/]
"""
}
```

**Benefits:**
- **Status indicators** with ↑/↓ arrows and colors
- **Container health** visible (green Up, red Stopped)
- **Port mappings** clearly shown
- **Resource usage** (if available)

**Implementation:**
- Add `_format_docker_containers()`, `_format_docker_images()`
- Green ↑ for running, red ↓ for stopped
- Bold container names/IDs, dim for details
- Preview strategies for each docker subcommand

---

### 9. 🔄 `security_scanner_tool.py` - Security Scan Results
**Current Output:** Plain list of vulnerabilities
```python
{
    "vulnerabilities": [
        {"severity": "HIGH", "file": "foo.py", "line": 42, "issue": "SQL injection"}
    ]
}
```

**Proposed Rich Enhancement:**
```python
{
    "vulnerabilities": [...],
    "formatted_report: """
[red bold]🔒 Security Scan Report[/]

  [red bold]HIGH[/] [dim]•[/] [bold]SQL injection[/]
    [dim cyan]foo.py:42[/]
    [dim]User input not sanitized before query[/]

  [yellow bold]MEDIUM[/] [dim]•[/] [bold]Hardcoded secret[/]
    [dim cyan]config.py:15[/]
    [dim]API key exposed in source code[/]

[dim]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─[/]
[red]1 HIGH[/] [dim]•[/] [yellow]2 MEDIUM[/] [dim]•[/] [green]0 LOW[/]
"""
}
```

**Benefits:**
- **Severity levels** color-coded (red HIGH, yellow MEDIUM, green LOW)
- **Issue types** in bold for scanning
- **File locations** in cyan for navigation
- **Summary counts** for quick assessment

**Implementation:**
- Add `_format_security_report()` function
- Red for HIGH, yellow for MEDIUM, green for LOW
- Bold issue types, cyan file:line locations
- Preview strategy shows severity summary + top 3 issues

---

## P3 Candidates (Lower Priority, Nice to Have)

### 10. 🔄 `metrics_tool.py` - Metrics/Analytics
**Proposed Enhancement:**
```python
{
    "formatted_metrics: """
[bold cyan]Performance Metrics:[/]

  [bold]Latency:[/] [green]123ms[/] [dim](p50), [yellow]456ms[/] [dim](p99)[/]
  [bold]Throughput:[/] [green]1,234 req/s[/]
  [bold]Errors:[/] [red]2.3%[/] [dim]error rate[/]

[dim]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─[/]
[dim]Last 5 minutes • 1,234 requests[/]
"""
}
```

**Benefits:**
- **Metric types** in bold (Latency, Throughput, Errors)
- **Value coloring** (green good, yellow warning, red bad)
- **Percentiles** clearly shown
- **Time window** in dim

---

### 11. 🔄 `graph_tool.py` - Codebase Visualization
**Proposed Enhancement:**
```python
{
    "formatted_graph: """
[bold cyan]Codebase Structure:[/]

  [bold]src/[/]
    [dim]├──[/] [bold]main.py[/] [dim]• 234 lines[/]
    [dim]├──[/] [bold]utils.py[/] [dim]• 567 lines[/]
    [dim]└──[/] [bold]tests/[/]
      [dim]├──[/] [bold]test_main.py[/]
      [dim]└──[/] [bold]test_utils.py[/]

[dim]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─[/]
[dim]4 files • 801 lines total[/]
"""
}
```

**Benefits:**
- **Tree structure** with box-drawing characters
- **File sizes** shown in dim
- **Directory hierarchy** clearly visualized
- **Summary statistics** at bottom

---

### 12. ✅ `bash` (bash.py) - Shell Commands
**Current Status:** ✅ Already has basic preview with exit code

**Current Output:**
```python
{
    "return_code": 0,
    "stdout": "hello\nworld",
    "stderr": ""
}
```

**Current Preview:**
```
│ [exit 0]
│ hello
│ world
```

**Potential Enhancement:**
```python
{
    "formatted_output: """
[green bold]✓[/] [bold]exit 0[/] [dim]• 123ms[/]

  hello
  world
"""
}
```

**Benefits:**
- **Exit code color** (green 0, red non-zero)
- **Duration** shown
- **Stderr** in red if present

**Implementation:** Low effort, already has preview strategy

---

## Implementation Strategy

### Phase 1: P0 Candidates (Week 1-2)
1. ✅ `edit` tool - **DONE**
2. `testing_tool.py` - Add `_format_test_summary()`
3. `code_search_tool.py` - Add `_format_search_results()`
4. `git_tool.py` - Add `_format_git_status()`, `_format_git_log()`

### Phase 2: P1 Candidates (Week 3-4)
1. `http_tool.py` - Add `_format_http_response()`
2. `database_tool.py` - Add `_format_table()` with Rich Table
3. `refactor_tool.py` - Add `_format_refactor_plan()`

### Phase 3: P2 Candidates (Week 5-6)
1. `docker_tool.py` - Add docker formatting functions
2. `security_scanner_tool.py` - Add `_format_security_report()`

### Phase 4: P3 Candidates (Week 7-8)
1. `metrics_tool.py` - Add metrics formatting
2. `graph_tool.py` - Add tree visualization
3. `bash.py` - Enhance existing preview

---

## Common Patterns

### 1. **Preview Strategy Pattern**
Each tool needs a custom preview strategy in `tool_preview.py`:

```python
class _TestPreviewStrategy(_ToolPreviewStrategy):
    def render(self, tool_name, arguments, raw_result, max_lines):
        parsed = _try_parse(raw_result)
        if isinstance(parsed, dict) and "formatted_summary" in parsed:
            # Use pre-formatted Rich output
            return RenderedPreview(
                lines=parsed["formatted_summary"].splitlines(),
                header=parsed.get("header"),
                total_line_count=len(parsed["formatted_summary"].splitlines()),
                contains_rich_markup=True,
            )
        # Fallback to generic
        return _GenericPreviewStrategy().render(...)
```

### 2. **Formatting Function Pattern**
Each tool adds a formatting function:

```python
def _format_<output_type>(data: Dict[str, Any]) -> str:
    """Format output with Rich markup."""
    lines = []
    for item in data["items"]:
        lines.append(f"[green]✓[/] {item['name']}")
    return "\n".join(lines)
```

### 3. **Return Value Enhancement**
Tools return both raw and formatted data:

```python
return {
    "raw_data": {...},  # For LLM consumption
    "formatted_output": _format_<type>(data),  # For user display
    "summary": "2 items",  # For preview header
}
```

---

## Testing Considerations

### Unit Tests
- Test formatting functions with various input types
- Verify Rich markup is well-formed
- Test preview strategies with mock data

### Integration Tests
- Test end-to-end tool execution with formatted output
- Verify preview rendering in live console
- Test edge cases (empty results, errors, etc.)

### Visual Regression Tests
- Capture formatted output for known inputs
- Compare against expected output
- Alert on formatting changes

---

## Performance Considerations

### Caching
- Cache formatted output for expensive operations
- Invalidate cache on data changes

### Lazy Formatting
- Only format when preview is enabled
- Skip formatting for non-interactive mode

### Streaming
- For large outputs, format incrementally
- Show partial results while processing

---

## Future Enhancements

### 1. **Syntax Highlighting**
- Add Pygments-based syntax highlighting for code blocks
- Support for multiple languages (Python, JS, SQL, etc.)

### 2. **Interactive Elements**
- Add clickable links for file paths (open in editor)
- Add expand/collapse for large outputs

### 3. **Theming**
- Support for custom color schemes
- Dark/light mode variations

### 4. **Accessibility**
- Ensure colorblind-friendly palettes
- Provide alternative text formatting

---

## Conclusion

This analysis identifies **12 high-value candidates** for Rich formatting enhancements across the Victor codebase. The implementation is prioritized by impact and complexity, with **P0 candidates** (testing, search, git) offering the highest UX improvements.

The edit tool serves as a reference implementation, demonstrating the pattern of:
1. Adding formatting functions to tools
2. Returning both raw and formatted data
3. Creating preview strategies in `tool_preview.py`
4. Rendering formatted output in `live_renderer.py`

Following this pattern consistently will provide a cohesive, visually rich experience across all tool outputs.
