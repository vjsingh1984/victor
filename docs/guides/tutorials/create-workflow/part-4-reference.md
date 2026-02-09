# Creating Workflows - Part 4

**Part 4 of 4:** Reference and Next Steps

---

## Navigation

- [Part 1: Basics & Parallel](part-1-basics-conditions-parallel.md)
- [Part 2: HITL & Escape Hatches](part-2-hitl-escape.md)
- [Part 3: Testing & Examples](part-3-testing-examples.md)
- **[Part 4: Reference](#)** (Current)
- [**Complete Guide**](../create-workflow.md)

---

Here is a complete, production-ready code review workflow combining all concepts:

```yaml
# workflows/complete_code_review.yaml
#
# A comprehensive code review workflow that:
# 1. Analyzes code for issues
# 2. Runs parallel security and style checks
# 3. Routes based on quality assessment
# 4. Requires human approval for critical issues
# 5. Generates a detailed report

workflows:
  complete_code_review:
    description: "Comprehensive code review with parallel analysis and human gates"
    metadata:
      category: code_quality
      version: "2.0"
      author: "victor_team"
      requires_approval: true

    # Workflow-level settings
    max_execution_timeout_seconds: 1800    # 30 minute overall timeout
    default_node_timeout_seconds: 300      # 5 minute default per node
    max_iterations: 10                     # Max loop iterations

    nodes:
      # =====================================================
      # Phase 1: Initial Analysis
      # =====================================================
      - id: gather_context
        type: agent
        role: researcher
        goal: "Gather context about the code changes: files modified, scope of changes, and related components"
        tool_budget: 10
        allowed_tools:
          - read
          - grep
          - git
          - code_search
        output_key: change_context
        next:
          - parallel_analysis

      # =====================================================
      # Phase 2: Parallel Analysis
      # =====================================================
      - id: parallel_analysis
        type: parallel
        parallel_nodes:
          - security_scan
          - style_analysis
          - complexity_check
          - test_coverage_check
        join_strategy: all
        next:
          - merge_analysis

      - id: security_scan
        type: agent
        role: security_analyst
        goal: "Scan code for security vulnerabilities: SQL injection, XSS, hardcoded secrets, unsafe operations"
        tool_budget: 15
        allowed_tools:
          - read
          - grep
          - code_search
        output_key: security_findings

      - id: style_analysis
        type: agent
        role: style_reviewer
        goal: "Check code style, naming conventions, documentation, and PEP8 compliance"
        tool_budget: 10
        allowed_tools:
          - read
          - shell
        output_key: style_findings

      - id: complexity_check
        type: agent
        role: analyst
        goal: "Analyze cyclomatic complexity, function length, and code maintainability"
        tool_budget: 10
        allowed_tools:
          - read
          - symbols
        output_key: complexity_findings

      - id: test_coverage_check
        type: agent
        role: tester
        goal: "Verify test coverage for changed code and identify missing tests"
        tool_budget: 10
        allowed_tools:
          - read
          - grep
          - shell
        output_key: coverage_findings

      # =====================================================
      # Phase 3: Merge and Assess
      # =====================================================
      - id: merge_analysis
        type: transform
        transform: "merge_code_analysis"
        next:
          - assess_quality

      - id: assess_quality
        type: condition
        condition: "code_quality_check"
        branches:
          excellent: prepare_approval
          good: prepare_approval
          acceptable: minor_fixes_needed
          needs_improvement: major_fixes_needed

      # =====================================================
      # Phase 4: Fix Routes
      # =====================================================
      - id: minor_fixes_needed
        type: agent
        role: fixer
        goal: "Apply minor fixes for style issues and small improvements"
        tool_budget: 15
        allowed_tools:
          - read
          - edit
          - shell
        output_key: minor_fix_results
        next:
          - verify_fixes

      - id: major_fixes_needed
        type: agent
        role: fixer
        goal: "Address major code quality issues: security vulnerabilities, complexity problems, missing tests"
        tool_budget: 30
        allowed_tools:
          - read
          - write
          - edit
          - shell
        output_key: major_fix_results
        next:
          - verify_fixes

      - id: verify_fixes
        type: agent
        role: verifier
        goal: "Verify that applied fixes resolved the identified issues"
        tool_budget: 10
        allowed_tools:
          - read
          - shell
          - grep
        output_key: verification_results
        next:
          - check_verification

      - id: check_verification
        type: condition
        condition: "tests_passed"
        branches:
          "true": prepare_approval
          "false": escalate_issues

      - id: escalate_issues
        type: hitl
        hitl_type: review
        prompt: "Automated fixes could not resolve all issues. Please review and provide guidance."
        context_keys:
          - verification_results
          - major_fix_results
          - security_findings
        timeout: 900
        fallback: abort
        next:
          - prepare_approval

      # =====================================================
      # Phase 5: Human Approval
      # =====================================================
      - id: prepare_approval
        type: transform
        transform: "prepare_report_data"
        next:
          - human_approval

      - id: human_approval
        type: hitl
        hitl_type: approval
        prompt: |
          Code Review Summary
          ===================
          Please review the analysis results and approve or reject the changes.

          Critical issues will require explicit approval to proceed.
        context_keys:
          - report_data
          - change_context
          - security_findings
        timeout: 600
        fallback: abort
        next:
          - generate_final_report

      # =====================================================
      # Phase 6: Final Report
      # =====================================================
      - id: generate_final_report
        type: agent
        role: reporter
        goal: "Generate a comprehensive code review report with all findings, fixes applied, and recommendations"
        tool_budget: 5
        allowed_tools:
          - read
        output_key: final_report
```text

### Supporting escape_hatches.py

```python
# escape_hatches.py for complete_code_review workflow
"""Escape hatches for the complete code review workflow."""

from typing import Any, Dict


def code_quality_check(ctx: Dict[str, Any]) -> str:
    """Assess overall code quality.

    Returns: "excellent", "good", "acceptable", or "needs_improvement"
    """
    # Get merged analysis results
    merged = ctx.get("merged_analysis", {})

    security = merged.get("security_findings", {})
    style = merged.get("style_findings", {})
    complexity = merged.get("complexity_findings", {})

    # Count issues by severity
    critical_issues = security.get("critical", 0)
    high_issues = security.get("high", 0)
    style_errors = style.get("errors", 0)
    complexity_score = complexity.get("average_complexity", 5)

    # Scoring logic
    if critical_issues > 0:
        return "needs_improvement"

    if high_issues > 0 or style_errors > 5 or complexity_score > 15:
        return "needs_improvement"

    if high_issues == 0 and style_errors <= 2 and complexity_score <= 8:
        return "excellent"

    if style_errors <= 3 and complexity_score <= 10:
        return "good"

    return "acceptable"


def tests_passed(ctx: Dict[str, Any]) -> str:
    """Check if verification tests passed."""
    results = ctx.get("verification_results", {})
    success = results.get("success", False)
    return "true" if success else "false"


def merge_code_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge all analysis results into unified structure."""
    return {
        "merged_analysis": {
            "security_findings": ctx.get("security_findings", {}),
            "style_findings": ctx.get("style_findings", {}),
            "complexity_findings": ctx.get("complexity_findings", {}),
            "coverage_findings": ctx.get("coverage_findings", {}),
        }
    }


def prepare_report_data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for the final report."""
    merged = ctx.get("merged_analysis", {})

    security = merged.get("security_findings", {})
    style = merged.get("style_findings", {})

    return {
        "report_data": {
            "security_issues": security.get("issues", []),
            "style_issues": style.get("issues", []),
            "overall_score": _calculate_score(merged),
            "recommendation": _get_recommendation(merged),
        }
    }


def _calculate_score(merged: Dict[str, Any]) -> int:
    """Calculate overall score out of 100."""
    security = merged.get("security_findings", {})
    style = merged.get("style_findings", {})
    complexity = merged.get("complexity_findings", {})

    # Start at 100, deduct for issues
    score = 100
    score -= security.get("critical", 0) * 25
    score -= security.get("high", 0) * 10
    score -= style.get("errors", 0) * 3
    score -= max(0, complexity.get("average_complexity", 5) - 10)

    return max(0, score)


def _get_recommendation(merged: Dict[str, Any]) -> str:
    """Generate recommendation based on analysis."""
    score = _calculate_score(merged)

    if score >= 90:
        return "Ready to merge"
    if score >= 70:
        return "Minor changes recommended before merge"
    if score >= 50:
        return "Significant changes required"
    return "Major rework needed"


CONDITIONS = {
    "code_quality_check": code_quality_check,
    "tests_passed": tests_passed,
}

TRANSFORMS = {
    "merge_code_analysis": merge_code_analysis,
    "prepare_report_data": prepare_report_data,
}
```

### Running the Complete Workflow

```bash
# Validate
victor workflow validate workflows/complete_code_review.yaml \
  --escape-hatches escape_hatches.py

# Run
victor workflow run workflows/complete_code_review.yaml \
  --workflow complete_code_review \
  --input '{"target_files": ["src/api/routes.py", "src/api/handlers.py"]}'
```text

---

## Next Steps

Now that you understand Victor workflows, explore:

- **[Vertical Development Guide](../reference/internals/VERTICAL_DEVELOPMENT_GUIDE.md)**: Create custom verticals with workflows
- **[StateGraph API](../reference/internals/workflows-api.md#stategraph-api)**: Build workflows programmatically in Python
- **[Multi-Agent Teams](../guides/MULTI_AGENT_TEAMS.md)**: Coordinate multiple agents within workflows
- **[Workflow Scheduling](../guides/workflow-development/scheduling.md)**: Run workflows on schedules and triggers

## Reference

---

**Reading Time:** 5 min
**Last Updated:** February 08, 2026**
