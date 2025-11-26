# Tool Consolidation Plan

## Current State: 59 Tools (was 86) → Target: 41 Tools (52% reduction)

**Progress:** Phase 1 COMPLETE + Phase 2 Docker (31% reduction achieved)
- ✅ Phase 1: Removed duplicate + consolidated 16 tools → 3 tools
- ✅ Phase 2.1: Consolidated Docker tools (15 → 1)

### Problem Statement

Victor currently registers **86 tools**, which:
- Overwhelms small models (qwen2.5-coder:1.5b can't handle this)
- Creates massive context overhead (~60KB per request)
- Requires intelligent filtering just to be usable
- Makes maintenance difficult
- Confuses models with overlapping functionality

### Consolidation Strategy

## Priority 1: Critical Consolidations (48 tools → 6 tools)

### 1. File Editor: 10 tools → 1 tool
**Current (10 tools):**
```
file_editor_start_transaction
file_editor_add_create
file_editor_add_modify
file_editor_add_delete
file_editor_add_rename
file_editor_preview
file_editor_commit
file_editor_rollback
file_editor_abort
file_editor_status
```

**Consolidated (1 tool):**
```python
edit_files(
    operations: List[FileOperation],
    preview: bool = False,
    auto_commit: bool = True
) -> ToolResult

# Example usage:
edit_files(operations=[
    {"type": "create", "path": "foo.py", "content": "..."},
    {"type": "modify", "path": "bar.py", "changes": [...]},
    {"type": "delete", "path": "old.py"}
], auto_commit=True)
```

**Benefits:**
- Single tool call instead of 7+ calls
- Built-in transaction support
- Clearer semantics
- Easier for models to understand

---

### 2. Docker: 15 tools → 1 tool
**Current (15 tools):**
```
docker_ps, docker_images, docker_pull, docker_run, docker_stop,
docker_start, docker_restart, docker_rm, docker_rmi, docker_logs,
docker_stats, docker_inspect, docker_networks, docker_volumes, docker_exec
```

**Consolidated (1 tool):**
```python
docker(
    operation: str,  # ps, images, pull, run, stop, start, logs, etc.
    resource_type: str = "container",  # container, image, network, volume
    resource_id: Optional[str] = None,
    options: Dict[str, Any] = {}
) -> ToolResult

# Example usage:
docker("ps", resource_type="container")
docker("pull", resource_type="image", resource_id="nginx:latest")
docker("logs", resource_id="my-container", options={"tail": 100})
```

**Benefits:**
- Mirrors standard CLI patterns
- Single interface for all Docker operations
- Reduces context by 14 tools

---

### 3. Security: 5 tools → 1 tool
**Current (5 tools):**
```
security_scan_secrets
security_scan_dependencies
security_scan_config
security_scan_all
security_check_file
```

**Consolidated (1 tool):**
```python
security_scan(
    path: str,
    scan_types: List[str] = ["all"],  # secrets, dependencies, config, all
    severity_threshold: str = "medium",
    exclude_patterns: List[str] = []
) -> ToolResult

# Example usage:
security_scan("./src", scan_types=["secrets", "dependencies"])
security_scan("config.yaml", scan_types=["config"])
```

**Benefits:**
- Single scan invocation
- Configurable scan types
- Clearer output format

---

### 4. Code Review: 5 tools → 1 tool
**Current (5 tools):**
```
code_review_file
code_review_directory
code_review_security
code_review_complexity
code_review_best_practices
```

**Consolidated (1 tool):**
```python
code_review(
    path: str,
    aspects: List[str] = ["all"],  # security, complexity, best_practices, style
    recursive: bool = False,
    max_issues: int = 50
) -> ToolResult

# Example usage:
code_review("src/auth.py", aspects=["security", "complexity"])
code_review("src/", aspects=["all"], recursive=True)
```

**Benefits:**
- Single review interface
- Flexible aspect selection
- Better structured output

---

### 5. Metrics: 6 tools → 1 tool
**Current (6 tools):**
```
metrics_complexity
metrics_maintainability
metrics_debt
metrics_profile
metrics_analyze
metrics_report
```

**Consolidated (1 tool):**
```python
analyze_metrics(
    path: str,
    metrics: List[str] = ["all"],  # complexity, maintainability, debt
    format: str = "summary",  # summary, detailed, json
    recursive: bool = False
) -> ToolResult

# Example usage:
analyze_metrics("src/", metrics=["complexity", "maintainability"])
analyze_metrics("app.py", metrics=["all"], format="detailed")
```

**Benefits:**
- Unified metrics interface
- Consistent output formats
- Single analysis pass

---

### 6. Documentation: 5 tools → 2 tools
**Current (5 tools):**
```
docs_generate_docstrings
docs_generate_api
docs_generate_readme
docs_add_type_hints
docs_analyze_coverage
```

**Consolidated (2 tools):**
```python
generate_docs(
    path: str,
    doc_types: List[str] = ["docstrings"],  # docstrings, api, readme, type_hints
    format: str = "google",  # google, numpy, sphinx
    recursive: bool = False
) -> ToolResult

analyze_docs(
    path: str,
    check_coverage: bool = True,
    check_quality: bool = True
) -> ToolResult

# Example usage:
generate_docs("src/", doc_types=["docstrings", "type_hints"], recursive=True)
analyze_docs("src/", check_coverage=True)
```

**Benefits:**
- Generation vs analysis separated
- Clearer purpose
- Still flexible

---

## Priority 2: Remove Duplicates (1 tool removed)

### 7. Duplicate: rename_symbol

**Problem:** `rename_symbol` exists in TWO places:
- `refactor_tool.refactor_rename_symbol`
- `code_intelligence_tool.rename_symbol`

**Solution:** Remove `refactor_rename_symbol`, keep code_intelligence version

---

## Priority 3: Partial Consolidations

### 8. Git: 9 tools → 3-4 tools

**Keep Separate (genuinely distinct):**
- `git_suggest_commit` - AI-powered, unique functionality
- `git_create_pr` - Complex workflow, worth separate tool
- `git_analyze_conflicts` - Complex analysis, worth separate tool

**Consolidate (6 tools → 1):**
```python
git(
    operation: str,  # status, diff, stage, commit, log, branch
    files: Optional[List[str]] = None,
    message: Optional[str] = None,
    options: Dict[str, Any] = {}
) -> ToolResult

# Example usage:
git("status")
git("stage", files=["src/main.py"])
git("commit", message="Fix bug")
git("diff", files=["src/"])
```

**Result:** 9 tools → 4 tools (git, git_suggest_commit, git_create_pr, git_analyze_conflicts)

---

## Priority 4: Keep As-Is (legitimately different)

### Categories to Keep Unchanged

**Batch Processor (5 tools)** - Each is genuinely different:
- batch_search - Search across files
- batch_replace - Replace across files
- batch_analyze - Analyze multiple files
- batch_list_files - List files matching criteria
- batch_transform - Transform file contents

**CI/CD (4 tools)** - Template-based, keep separate:
- cicd_generate - Generate from template
- cicd_validate - Validate syntax
- cicd_list_templates - List available templates
- cicd_create_workflow - Create new workflow

**Scaffold (4 tools)** - Different lifecycle stages:
- scaffold_create - Create new project
- scaffold_list_templates - List templates
- scaffold_add_file - Add file to project
- scaffold_init_git - Initialize git

**Code Intelligence (3 tools)** - Core AST operations:
- find_symbol - Find definitions
- find_references - Find usages
- rename_symbol - Rename across codebase

**Refactor (3 tools after removing duplicate):**
- refactor_extract_function
- refactor_inline_variable
- refactor_organize_imports

**Core Tools (8 tools)** - Essential, keep all:
- read_file, write_file, list_directory
- execute_bash
- execute_python_in_sandbox, upload_files_to_sandbox
- run_workflow
- run_tests

**Web (3 tools)** - Keep separate (optional, air-gapped mode):
- web_search - Search engines
- web_fetch - Fetch specific URLs
- web_summarize - AI-powered summarization

---

## Final Tool Count

| Category | Before | After | Reduction |
|----------|--------|-------|-----------|
| File Editor | 10 | 1 | -9 |
| Docker | 15 | 1 | -14 |
| Security | 5 | 1 | -4 |
| Code Review | 5 | 1 | -4 |
| Metrics | 6 | 1 | -5 |
| Documentation | 5 | 2 | -3 |
| Git | 9 | 4 | -5 |
| Refactor | 4 | 3 | -1 (duplicate) |
| Batch | 5 | 5 | 0 |
| CI/CD | 4 | 4 | 0 |
| Scaffold | 4 | 4 | 0 |
| Code Intelligence | 3 | 3 | 0 |
| Core | 8 | 8 | 0 |
| Web | 3 | 3 | 0 |
| **TOTAL** | **86** | **41** | **-45 (52%)** |

---

## Implementation Plan

### Phase 1: Quick Wins ✅ COMPLETE
1. ✅ Remove duplicate: `refactor_rename_symbol` (86 → 85 tools)
2. ✅ Consolidate Security: 5 → 1 (85 → 81 tools)
3. ✅ Consolidate Code Review: 5 → 1 (81 → 77 tools)
4. ✅ Consolidate Metrics: 6 → 1 (77 → 73 tools)

**Target Impact:** 86 → 73 tools (15% reduction)
**Achieved:** 86 → 73 tools (15% reduction, 4/4 tasks complete)

### Phase 2: Major Consolidations
1. ✅ Consolidate Docker: 15 → 1 (73 → 59 tools)
2. ⏳ Consolidate File Editor: 10 → 1
3. ⏳ Consolidate Documentation: 5 → 2

**Target Impact:** 73 → 48 tools (44% reduction)
**Current Progress:** 73 → 59 tools (31% reduction, 1/3 tasks complete)

### Phase 3: Git Consolidation (Week 4)
1. Consolidate Git: 9 → 4

**Impact:** 48 → 43 tools (50% reduction)

### Phase 4: Testing & Refinement (Week 5)
1. Update intelligent tool selection logic
2. Test with small models (qwen2.5-coder:1.5b)
3. Update documentation
4. Create migration guide

**Final Impact:** 86 → 41 tools (52% reduction)

---

## Benefits

### Performance
- **Context reduction**: ~60KB → ~25KB per request
- **Inference speed**: 30-50% faster for small models
- **Tool calls**: Single call vs 7+ calls for complex operations

### Usability
- **Clearer semantics**: One tool, multiple options vs many specialized tools
- **Fewer choices**: Models make better decisions with fewer options
- **Better discoverability**: Easier to find the right tool

### Maintenance
- **Less code duplication**: Shared implementation logic
- **Easier testing**: Test one tool with multiple configurations
- **Simpler updates**: Change one tool vs updating many

### Model Compatibility
- **Small models work**: qwen2.5-coder:1.5b can handle 15-20 tools comfortably
- **Better accuracy**: Clearer tool purposes reduce confusion
- **Faster learning**: Models understand consolidated tools better

---

## Migration Guide

### For Existing Code

**Before:**
```python
# Old file editor (7 tool calls)
file_editor_start_transaction()
file_editor_add_create(path="foo.py", content="...")
file_editor_add_modify(path="bar.py", old_text="...", new_text="...")
file_editor_preview()
# Review...
file_editor_commit()
```

**After:**
```python
# New file editor (1 tool call)
edit_files(operations=[
    {"type": "create", "path": "foo.py", "content": "..."},
    {"type": "modify", "path": "bar.py", "old_text": "...", "new_text": "..."}
], preview=True, auto_commit=True)
```

### For Tool Selection

**Before:**
```python
# Select from 86 tools
selected_tool_names = {
    "file_editor_start_transaction",
    "file_editor_add_create",
    "file_editor_add_modify",
    "file_editor_commit",
    # ... 82 more
}
```

**After:**
```python
# Select from 41 tools
selected_tool_names = {
    "edit_files",  # Consolidated file editor
    "docker",      # Consolidated docker
    "security_scan",  # Consolidated security
    # ... 38 more
}
```

---

## Success Metrics

1. **Tool count**: 86 → 41 (52% reduction)
2. **Context size**: ~60KB → ~25KB (58% reduction)
3. **Small model performance**: Timeout → 15-30s responses
4. **Tool call efficiency**: 7+ calls → 1-2 calls for common operations
5. **Model accuracy**: Improved tool selection and usage

---

## Next Steps

1. ✅ **Analysis complete** - This document
2. ⏳ **Implementation** - Follow phased plan
3. ⏳ **Testing** - Verify with multiple models
4. ⏳ **Documentation** - Update all tool docs
5. ⏳ **Release** - v0.2.0 with consolidated tools

