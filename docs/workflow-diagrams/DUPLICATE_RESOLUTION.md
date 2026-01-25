# Workflow Diagram Duplicate Resolution

## Issue Identified: `coding_bug_fix.svg` vs `coding_bugfix.svg`

### Analysis

**Duplicate Files Found:**
- `coding_bug_fix.svg` (19KB, 258 lines) - **KEPT**
- `coding_bugfix.svg` (7.4KB, 100 lines) - **REMOVED**

### Investigation Results

1. **Workflow YAML File**: `victor/coding/workflows/bugfix.yaml`
   - Filename uses `bugfix` (no underscore)
   - Contains three workflows: `bug_fix`, `quick_fix`, `debug_investigation`

2. **SVG Diagram Analysis**:
   - `coding_bug_fix.svg` contains workflow title: "bug_fix" (matches YAML workflow name)
   - `coding_bugfix.svg` contains workflow title: "bugfix" (matches YAML filename)
   - `coding_bug_fix.svg` is 2.5x larger and more comprehensive (258 lines vs 100 lines)
   - `coding_bug_fix.svg` includes all stages: investigate, diagnose, approve_fix, apply_fix, run_tests, check_tests, analyze_failure, should_retry, escalate_to_human, check_quality, assess_quality, improve_quality, commit_fix, complete
   - `coding_bugfix.svg` is a simplified version with only: investigate, confirm_cause, implement_fix, verify_fix, check_verification, complete

3. **Decision Criteria**:
   - ✅ **Consistency**: Both files use snake_case naming convention
   - ✅ **Completeness**: `coding_bug_fix.svg` represents the full workflow
   - ✅ **Workflow Alignment**: `coding_bug_fix.svg` matches the `bug_fix` workflow name in the YAML
   - ✅ **Documentation References**: Archive documents reference the duplicate issue but don't indicate which to keep

### Resolution

**Action Taken**: Removed `coding_bugfix.svg` (deprecated as `coding_bugfix.svg.deprecated`)

**Rationale**:
1. The full `bug_fix` workflow is the primary workflow in the YAML file
2. The larger, more detailed diagram (19KB vs 7.4KB) should be kept
3. The diagram title "bug_fix" matches the workflow definition in YAML
4. The simplified version may have been an earlier iteration or partial diagram

### Naming Convention Established

Based on analysis of all workflow diagrams in the repository:

**Pattern**: `{vertical}_{workflow_name}.svg`

Examples:
- `coding_bug_fix.svg` - Full bug fix workflow
- `coding_quick_fix.svg` - Quick fix workflow
- `coding_code_review.svg` - Code review workflow
- `coding_refactor.svg` - Refactor workflow
- `coding_tdd.svg` - TDD workflow
- `coding_tdd_quick.svg` - Quick TDD workflow

**Naming Rules**:
1. Use **snake_case** for all workflow names (e.g., `bug_fix`, `quick_fix`, `code_review`)
2. Use **underscores** to separate words, not hyphens
3. Prefix with vertical name: `coding_`, `devops_`, `research_`, `dataanalysis_`, `benchmark_`, `rag_`, `core_`
4. For quick/simplified variants, append `_quick` suffix (e.g., `coding_quick_fix.svg`)

### Related Diagrams

The following diagrams correctly follow the naming convention:
- `coding_quick_fix.svg` - Quick fix workflow (separate from full bug_fix)
- `coding_debug_investigation.svg` - Debug investigation workflow (third workflow in bugfix.yaml)

### Recommendations for Future

1. **Prevent Duplicates**: When generating new diagrams, check for existing similar names
2. **Naming Validation**: Implement a script to validate diagram names against workflow YAML files
3. **Documentation**: Keep this file updated with any future duplicate resolutions
4. **Clean Up**: Remove `.deprecated` files after confirmed working state

### Verification

After removing the duplicate:
- ✅ No broken references in documentation
- ✅ All three workflows from `bugfix.yaml` have corresponding diagrams:
  - `bug_fix` → `coding_bug_fix.svg`
  - `quick_fix` → `coding_quick_fix.svg`
  - `debug_investigation` → `coding_debug_investigation.svg`

### Files Modified

- **Removed**: `docs/workflow-diagrams/coding_bugfix.svg` → `docs/workflow-diagrams/coding_bugfix.svg.deprecated`
- **Created**: `docs/workflow-diagrams/DUPLICATE_RESOLUTION.md` (this file)

### Date Resolved

2025-01-18

### Related Issues

- Historical reports (archived separately)
- `OSS_READNESS_REPORT.md` - OSS preparation checklist
