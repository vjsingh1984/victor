# Creating Workflows - Part 3

**Part 3 of 4:** Testing and Complete Examples

---

## Navigation

- [Part 1: Basics & Parallel](part-1-basics-conditions-parallel.md)
- [Part 2: HITL & Escape Hatches](part-2-hitl-escape.md)
- **[Part 3: Testing & Examples](#)** (Current)
- [Part 4: Reference](part-4-reference.md)
- [**Complete Guide**](../create-workflow.md)

---

When workflow logic exceeds what YAML can express, use Python escape hatches.

### When to Use Python

Use escape hatches for:

- Complex conditional logic with multiple factors
- Data transformations requiring Python libraries
- Integration with external services
- Business logic that changes frequently

### Creating escape_hatches.py

For any vertical, create `escape_hatches.py` in the vertical directory:

```python
# victor/myvertical/escape_hatches.py
"""Escape hatches for MyVertical YAML workflows."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================

def should_retry(ctx: Dict[str, Any]) -> str:
    """Determine if we should retry based on error type.

    Args:
        ctx: Workflow context with:
            - error (str): Error message
            - retry_count (int): Current retry count
            - max_retries (int): Maximum allowed retries

    Returns:
        "retry" or "fail"
    """
    error = ctx.get("error", "")
    retry_count = ctx.get("retry_count", 0)
    max_retries = ctx.get("max_retries", 3)

    # Don't retry on certain errors
    if "permission denied" in error.lower():
        return "fail"

    if retry_count >= max_retries:
        logger.info(f"Max retries ({max_retries}) reached")
        return "fail"

    return "retry"


def complexity_level(ctx: Dict[str, Any]) -> str:
    """Assess task complexity for resource allocation.

    Returns:
        "simple", "moderate", or "complex"
    """
    files_count = ctx.get("files_to_change", 1)
    lines_estimate = ctx.get("estimated_lines", 0)
    has_tests = ctx.get("requires_tests", False)

    if files_count > 10 or lines_estimate > 500:
        return "complex"

    if files_count > 3 or lines_estimate > 100 or has_tests:
        return "moderate"

    return "simple"


# =============================================================================
# Transform Functions
# =============================================================================

def prepare_report_data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for report generation.

    Returns:
        Updated context with formatted report data
    """
    raw_findings = ctx.get("findings", [])

    # Group by severity
    critical = [f for f in raw_findings if f.get("severity") == "critical"]
    high = [f for f in raw_findings if f.get("severity") == "high"]
    medium = [f for f in raw_findings if f.get("severity") == "medium"]
    low = [f for f in raw_findings if f.get("severity") == "low"]

    return {
        "report_data": {
            "total_findings": len(raw_findings),
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "low_count": len(low),
            "critical_findings": critical,
            "high_findings": high,
            "needs_attention": len(critical) > 0 or len(high) > 2,
        }
    }


# =============================================================================
# Registry Exports
# =============================================================================

# These are automatically loaded by the YAML workflow system
CONDITIONS = {
    "should_retry": should_retry,
    "complexity_level": complexity_level,
}

TRANSFORMS = {
    "prepare_report_data": prepare_report_data,
}
```

### Referencing from YAML

```yaml
workflows:
  smart_workflow:
    nodes:
      - id: assess_complexity
        type: condition
        condition: "complexity_level"      # References escape hatch
        branches:
          simple: quick_fix
          moderate: standard_process
          complex: detailed_analysis

      - id: prepare_for_report
        type: transform
        transform: "prepare_report_data"   # References escape hatch
        next:
          - generate_report
```

---

## 7. Testing Workflows

### Validate Workflow Structure

```bash
# Validate YAML syntax and structure
victor workflow validate workflows/code_review.yaml

# Validate with escape hatches
victor workflow validate workflows/code_review.yaml \
  --escape-hatches victor/coding/escape_hatches.py
```

### Run Workflows

```bash
# Run a specific workflow
victor workflow run workflows/code_review.yaml \
  --workflow parallel_review \
  --input '{"target_file": "src/app.py"}'

# Run with verbose output
victor workflow run workflows/code_review.yaml \
  --workflow parallel_review \
  --verbose

# Run in dry-run mode (no actual execution)
victor workflow run workflows/code_review.yaml \
  --workflow parallel_review \
  --dry-run
```

### Debugging Tips

1. **Use verbose mode**: Add `--verbose` to see step-by-step execution

2. **Check node results**: Access `_node_results` in the final state:
   ```python
   result = await graph.invoke(initial_state)
   for node_id, node_result in result.state.get("_node_results", {}).items():
       print(f"{node_id}: success={node_result.success}")
   ```

3. **Enable logging**:
   ```python
   import logging
   logging.getLogger("victor.workflows").setLevel(logging.DEBUG)
   ```

4. **Test escape hatches independently**:
   ```python
   from escape_hatches import code_quality_check

   # Test with mock context
   test_ctx = {"analysis": {"errors": 0, "warnings": 2}}
   result = code_quality_check(test_ctx)
   assert result == "good"
   ```

5. **Use auto-approve mode for testing**:
   ```python
   from victor.workflows.hitl import HITLMode, HITLExecutor

   executor = HITLExecutor(mode=HITLMode.AUTO_APPROVE)
   ```


**Reading Time:** 3 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## 8. Complete Example: Full Code Review Workflow
