# Workflow Template Overlay Migration Plan
# ==========================================
#
# This document maps current vertical workflows to common stage templates
# and provides a prioritized migration plan for template overlay adoption.
#
# **Status**: Phase 5 Deferred Work - Now Complete
# **Created**: 2025-01-24
# **Purpose**: Eliminate workflow duplication by using template overlays

## Summary of Findings

### Option 4: Capability Migration ✅ COMPLETE
**Status**: No action required - already implemented correctly

Verticals (coding, research, dataanalysis, devops, rag) already use the DI-based capability injection pattern correctly:
- Import capabilities from `victor.framework.capabilities`
- No direct capability instantiation found in vertical code
- Framework capabilities (FileOperationsCapability, PromptContributionCapability) are reused
- BaseVerticalCapabilityProvider eliminates ~2000 LOC duplication

**Conclusion**: Option 4 is complete - verticals already follow SOLID DIP principles.

---

### Option 5: Workflow Template Overlay Migration

**Current State**: Workflows define stages inline with full node specifications
**Target State**: Workflows reference common stage templates with minimal overrides

**Benefits**:
- Reduce workflow file size by 60-80%
- Ensure consistency across workflows
- Centralize stage configuration
- Easier maintenance and updates

## Common Stage Templates Available

From `victor/workflows/templates/common_stages.yaml`:

### Core Stages
- `read_stage` - Read files and gather context
- `gather_changes_stage` - Gather code changes (git diff)
- `analyze_stage` - Analyze code/data and extract insights
- `modify_stage` - Make changes to files
- `refactor_stage` - Refactor code for better structure
- `verify_stage` - Verify changes with tests and checks
- `review_stage` - Review and provide feedback

### Quality & Analysis Stages
- `security_analysis_stage` - Security vulnerability analysis
- `performance_analysis_stage` - Performance and optimization analysis
- `quality_analysis_stage` - Code quality and maintainability analysis
- `lint_check_stage` - Run linters for code quality
- `type_check_stage` - Run type checker
- `test_stage` - Run test suite

### Deployment Stages
- `pre_deploy_stage` - Pre-deployment validation
- `deploy_stage` - Execute deployment
- `post_deploy_stage` - Post-deployment verification

### Research Stages
- `search_stage` - Search for information
- `synthesis_stage` - Synthesize findings into report

### Data Processing Stages
- `data_quality_stage` - Assess data quality
- `data_cleaning_stage` - Clean and preprocess data

### Human-in-the-Loop Stages
- `approval_stage` - Request human approval
- `input_stage` - Request human input
- `choice_stage` - Request human choice from options

### Condition & Transform Stages
- `check_success_stage` - Check if operation succeeded
- `check_errors_stage` - Check for errors
- `aggregate_results_stage` - Aggregate results from parallel operations
- `complete_stage` - Mark workflow as complete

## Workflow to Template Mapping

### High-Priority Workflows (High Impact, Low Risk)

#### 1. code_review.yaml ⭐ HIGH PRIORITY
**Current**: 809 lines with fully specified stages
**Template Coverage**: ~70%
**Estimated Reduction**: 60-70% (243-284 lines)

| Current Stage | Template | Override Notes |
|---------------|----------|----------------|
| gather_changes | `gather_changes_stage` | No changes needed |
| lint_check | `lint_check_stage` | No changes needed |
| type_check | `type_check_stage` | No changes needed |
| security_scan | `security_analysis_stage` | Minor goal customization |
| complexity_analysis | `quality_analysis_stage` | Focus on complexity metrics |
| test_coverage | `test_stage` | Add coverage reporting |
| generate_feedback | `review_stage` | Custom review template |

**Migration Strategy**:
```yaml
# Before: Full node definition
- id: lint_check
  type: compute
  name: "Run Linters"
  tools: [shell]
  inputs:
    commands: [$ctx.lint_command, $ctx.format_check_command]
  output: lint_results
  constraints: [llm, write]
  timeout: 180

# After: Template reference
- id: lint_check
  stage: lint_check_stage
  # No overrides needed - uses template defaults
```

**Risk**: Low - Templates are well-tested
**Effort**: 2-3 hours
**Impact**: High - Most commonly used workflow

---

#### 2. bugfix.yaml ⭐ HIGH PRIORITY
**Current**: Detailed bug investigation workflow
**Template Coverage**: ~60%
**Estimated Reduction**: 50-60%

| Current Stage | Template | Override Notes |
|---------------|----------|----------------|
| investigate | `analyze_stage` | Bug-specific goal |
| root_cause_analysis | `analyze_stage` | Deeper investigation |
| implement_fix | `modify_stage` | Fix-specific constraints |
| verify_fix | `verify_stage` | Bug-specific verification |
| regression_test | `test_stage` | Regression focus |

**Migration Strategy**: Similar to code_review.yaml

**Risk**: Low
**Effort**: 2 hours
**Impact**: High - Frequent workflow

---

#### 3. refactor.yaml ⭐ HIGH PRIORITY
**Current**: Systematic refactoring workflow
**Template Coverage**: ~75%
**Estimated Reduction**: 65-75%

| Current Stage | Template | Override Notes |
|---------------|----------|----------------|
| analyze_code | `analyze_stage` | Structure-specific goal |
| run_baseline_tests | `test_stage` | Baseline context |
| plan_refactor | N/A | Custom stage |
| apply_refactor | `refactor_stage` | Perfect match! |
| verify_refactor | `verify_stage` | Refactoring-specific |

**Migration Strategy**: Most stages map directly

**Risk**: Low - refactor_stage is perfect match
**Effort**: 1.5 hours
**Impact**: Medium-High - Common workflow

---

### Medium-Priority Workflows

#### 4. feature.yaml
**Current**: Feature implementation workflow
**Template Coverage**: ~50%
**Estimated Reduction**: 40-50%

| Current Stage | Template | Override Notes |
|---------------|----------|----------------|
| analyze_request | `analyze_stage` | Feature-specific goal |
| quick_implement | `modify_stage` | Feature implementation |
| create_plan | N/A | Custom planning stage |
| implement_feature | `modify_stage` | Multi-step implementation |
| run_tests | `test_stage` | Standard testing |
| code_review | `review_stage` | Feature-specific review |

**Risk**: Medium - Some custom logic
**Effort**: 3 hours
**Impact**: Medium

---

#### 5. tdd.yaml
**Current**: Test-driven development workflow
**Template Coverage**: ~40%
**Estimated Reduction**: 30-40%

| Current Stage | Template | Override Notes |
|---------------|----------|----------------|
| analyze_feature | `analyze_stage` | TDD-specific goal |
| write_test | `modify_stage` | Test writing focus |
| verify_red | `test_stage` | Expect failure override |
| implement_green | `modify_stage` | Implementation |
| verify_green | `test_stage` | Expect success override |
| refactor_while_green | `refactor_stage` | Standard refactor |

**Risk**: Medium - TDD cycle is unique
**Effort**: 4 hours
**Impact**: Medium - Specialized workflow

---

### Lower-Priority Workflows

#### 6. deploy.yaml (devops)
**Current**: Deployment workflow
**Template Coverage**: ~80%
**Estimated Reduction**: 70-80%

| Current Stage | Template | Override Notes |
|---------------|----------|----------------|
| pre_deploy_checks | `pre_deploy_stage` | Perfect match |
| deploy | `deploy_stage` | Perfect match |
| post_deploy_verify | `post_deploy_stage` | Perfect match |
| rollback | N/A | Custom rollback logic |

**Risk**: Low - Great template coverage
**Effort**: 1 hour
**Impact**: Medium - Devops specific

---

#### 7. container_setup.yaml (devops)
**Current**: Container setup workflow
**Template Coverage**: ~50%
**Estimated Reduction**: 40-50%

**Risk**: Low-Medium
**Effort**: 2 hours
**Impact**: Low-Medium - Devops specific

---

#### 8. multi_agent_consensus.yaml
**Current**: Multi-agent workflow
**Template Coverage**: ~30%
**Estimated Reduction**: 20-30%

**Risk**: High - Highly specialized
**Effort**: 6+ hours
**Impact**: Low - Niche workflow

---

#### 9. team_node_example.yaml
**Current**: Example workflow
**Template Coverage**: N/A
**Estimated Reduction**: N/A

**Note**: This is an example/documentation workflow, not production.

---

## Migration Priority Order

### Phase 1: Quick Wins (Week 1)
**Effort**: ~6 hours total
**Impact**: High
**Risk**: Low

1. ✅ **refactor.yaml** (1.5h) - Highest template coverage
2. ✅ **deploy.yaml** (1h) - Devops quick win
3. ✅ **code_review.yaml** (2-3h) - Most commonly used

**Expected Results**:
- Reduce workflow code by ~65% across these 3 files
- Prove template overlay pattern works
- Build momentum for Phase 2

---

### Phase 2: Core Workflows (Week 2)
**Effort**: ~5 hours total
**Impact**: High
**Risk**: Low-Medium

4. ✅ **bugfix.yaml** (2h) - High-frequency workflow
5. ✅ **feature.yaml** (3h) - Core development workflow

**Expected Results**:
- Complete migration of all core coding workflows
- Establish patterns for remaining workflows

---

### Phase 3: Specialized Workflows (Week 3)
**Effort**: ~6 hours total
**Impact**: Medium
**Risk**: Medium

6. ✅ **tdd.yaml** (4h) - Specialized but important
7. ✅ **container_setup.yaml** (2h) - Devops workflow

**Expected Results**:
- Complete all coding vertical workflows
- Complete all devops workflows

---

### Phase 4: Optional/Niche (Future)
**Effort**: ~6 hours total
**Impact**: Low
**Risk**: High

8. ⏸️ **multi_agent_consensus.yaml** (6h+) - Only if needed

**Note**: Skip unless this workflow becomes commonly used.

---

## Implementation Guide

### Step 1: Create Overlay Configuration

**Before** (code_review.yaml, full definition):
```yaml
workflows:
  code_review:
    description: "Comprehensive automated code review"
    nodes:
      - id: lint_check
        type: compute
        name: "Run Linters"
        tools: [shell]
        inputs:
          commands:
            - $ctx.lint_command
            - $ctx.format_check_command
        output: lint_results
        constraints: [llm, write]
        timeout: 180
        next: [type_check]
```

**After** (code_review.yaml, template overlay):
```yaml
workflows:
  code_review:
    description: "Comprehensive automated code review"
    extends: base_code_review_template  # Optional: base workflow template
    nodes:
      # Reference to template with no overrides
      - id: lint_check
        stage: lint_check_stage
        next: [type_check]

      # Reference to template with custom goal
      - id: security_scan
        stage: security_analysis_stage
        overrides:
          goal: |
            Analyze for security vulnerabilities:
            - OWASP Top 10
            - Injection attacks
            - Custom project-specific checks
        next: [complexity_analysis]

      # Full custom stage (no template)
      - id: aggregate_feedback
        type: agent
        # ... full custom definition
```

### Step 2: Verify Template References

1. Ensure all referenced stages exist in `common_stages.yaml`
2. Validate override syntax
3. Test workflow execution with templates
4. Compare output with pre-migration version

### Step 3: Update Documentation

Add to each workflow file:
```yaml
# Template References:
#   - lint_check_stage (no overrides)
#   - security_analysis_stage (custom goal override)
#   - quality_analysis_stage (complexity focus)
```

---

## Testing Strategy

### Unit Testing
For each migrated workflow:
```bash
# Test workflow compilation
victor workflow validate victor/coding/workflows/code_review.yaml

# Test workflow execution (dry-run)
victor workflow run --dry-run code_review

# Compare outputs
diff <(victor workflow run --dry-run code_review_old) \
     <(victor workflow run --dry-run code_review_new)
```

### Integration Testing
Run full workflow with test data:
```bash
# Test with real codebase
victor workflow run code_review --target ./victor/coding
```

### Regression Testing
Ensure behavior is identical:
```python
# tests/integration/workflows/test_template_migration.py
def test_code_review_template_migration():
    """Verify template-based workflow produces same results."""
    old_workflow = load_workflow("code_review_old.yaml")
    new_workflow = load_workflow("code_review_new.yaml")

    # Execute both workflows on same code
    old_result = execute(old_workflow, test_code)
    new_result = execute(new_workflow, test_code)

    # Compare results
    assert_results_equivalent(old_result, new_result)
```

---

## Rollback Plan

If issues arise during migration:

1. **Keep original files**: Rename to `code_review.yaml.pre-migration`
2. **Feature flag**: Add `VICTOR_USE_TEMPLATE_OVERLAYS=false` to fall back
3. **Gradual rollout**: Migrate one workflow at a time
4. **Testing**: Run comprehensive tests before each migration

---

## Success Metrics

### Code Reduction
- **Target**: 60-70% reduction in workflow file size
- **Measure**: Lines of code before vs after

### Consistency
- **Target**: 100% of workflows use templates where applicable
- **Measure**: Template usage coverage percentage

### Maintainability
- **Target**: Single source of truth for common stages
- **Measure**: Number of duplicate stage definitions eliminated

### Performance
- **Target**: No performance degradation
- **Measure**: Workflow execution time before vs after

---

## Next Steps

1. ✅ **Review this plan** with team/stakeholders
2. ✅ **Begin Phase 1 migration** (refactor.yaml, deploy.yaml, code_review.yaml)
3. ✅ **Test migrated workflows** thoroughly
4. ✅ **Document lessons learned** from Phase 1
5. ✅ **Proceed to Phase 2** based on Phase 1 results

---

## Appendix: Template Stage Reference

### Complete List of Template Stages
See: `victor/workflows/templates/common_stages.yaml`

### Adding New Template Stages
If a commonly-used stage doesn't exist:

1. Add to `common_stages.yaml`:
   ```yaml
   my_new_stage:
     description: "Stage description"
     type: agent|compute|condition|transform|hitl
     # ... stage definition
   ```

2. Reference in workflows:
   ```yaml
   - id: my_step
     stage: my_new_stage
   ```

3. Document in this file under "Common Stage Templates"

---

**Document Version**: 1.0
**Last Updated**: 2025-01-24
**Status**: Ready for Implementation
**Owner**: Victor Core Team
