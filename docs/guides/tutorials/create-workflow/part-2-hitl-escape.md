# Creating Workflows - Part 2

**Part 2 of 4:** Human-in-the-Loop and Escape Hatches

---

## Navigation

- [Part 1: Basics & Parallel](part-1-basics-conditions-parallel.md)
- **[Part 2: HITL & Escape Hatches](#)** (Current)
- [Part 3: Testing & Examples](part-3-testing-examples.md)
- [Part 4: Reference](part-4-reference.md)
- [**Complete Guide**](../create-workflow.md)

---

Run multiple analyses simultaneously for faster workflows.

### Parallel Nodes

```yaml
workflows:
  parallel_review:
    description: "Code review with parallel checks"
    nodes:
      - id: parallel_checks
        type: parallel
        parallel_nodes:
          - security_scan
          - style_check
          - complexity_analysis
        join_strategy: all        # Wait for all to complete
        next:
          - merge_results

      - id: security_scan
        type: agent
        role: security_analyst
        goal: "Scan code for security vulnerabilities"
        tool_budget: 15
        allowed_tools:
          - read
          - grep
          - code_search
        output_key: security_results

      - id: style_check
        type: agent
        role: style_reviewer
        goal: "Check code style and formatting"
        tool_budget: 10
        allowed_tools:
          - read
          - shell
        output_key: style_results

      - id: complexity_analysis
        type: agent
        role: analyst
        goal: "Analyze code complexity metrics"
        tool_budget: 10
        allowed_tools:
          - read
          - code_search
        output_key: complexity_results

      - id: merge_results
        type: transform
        transform: "merge_review_results"
        next:
          - final_report

      - id: final_report
        type: agent
        role: reporter
        goal: "Compile all findings into a comprehensive report"
        tool_budget: 5
```

### Join Strategies

| Strategy | Behavior |
|----------|----------|
| `all` | Wait for all parallel nodes to complete |
| `any` | Continue when any one node completes |
| `majority` | Continue when majority complete (>50%) |
| `merge` | Merge all results into combined output |

### Transform for Merging Results

Add to your `escape_hatches.py`:

```python
def merge_review_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results from parallel review checks.

    Args:
        ctx: Context with parallel results

    Returns:
        Merged results dictionary
    """
    security = ctx.get("security_results", {})
    style = ctx.get("style_results", {})
    complexity = ctx.get("complexity_results", {})

    # Combine all issues
    all_issues = []
    all_issues.extend(security.get("issues", []))
    all_issues.extend(style.get("issues", []))
    all_issues.extend(complexity.get("issues", []))

    return {
        "merged_results": {
            "total_issues": len(all_issues),
            "issues": all_issues,
            "security_score": security.get("score", 100),
            "style_score": style.get("score", 100),
            "complexity_score": complexity.get("score", 50),
        }
    }


TRANSFORMS = {
    "merge_review_results": merge_review_results,
}
```

---

## 5. Human-in-the-Loop

Add human approval gates for critical workflow decisions.

### HITL Approval Nodes

```yaml
workflows:
  reviewed_merge:
    description: "Code review with human approval"
    nodes:
      - id: analyze
        type: agent
        role: code_reviewer
        goal: "Analyze code changes thoroughly"
        tool_budget: 20
        output_key: analysis
        next:
          - human_review

      - id: human_review
        type: hitl
        hitl_type: approval
        prompt: "Review the following code analysis. Approve to proceed with merge."
        context_keys:
          - analysis
          - files_changed
        timeout: 600              # 10 minutes
        fallback: abort           # abort, continue, skip, retry
        next:
          - merge_code

      - id: merge_code
        type: agent
        role: executor
        goal: "Merge the approved changes"
        tool_budget: 5
```

### HITL Node Types

Victor supports several HITL interaction types:

```yaml
# Binary approval
- id: approve_changes
  type: hitl
  hitl_type: approval
  prompt: "Approve these changes?"
  timeout: 300
  fallback: abort

# Multiple choice
- id: select_action
  type: hitl
  hitl_type: choice
  prompt: "How should we proceed?"
  choices:
    - "Deploy to staging"
    - "Deploy to production"
    - "Cancel deployment"
  timeout: 300
  fallback: continue
  default_value: "Deploy to staging"

# Free-form input
- id: get_feedback
  type: hitl
  hitl_type: input
  prompt: "Enter additional review comments:"
  timeout: 600
  fallback: skip

# Review with modification
- id: review_plan
  type: hitl
  hitl_type: review
  prompt: "Review and optionally modify the implementation plan:"
  context_keys:
    - implementation_plan
    - files_to_modify
  timeout: 900
  fallback: abort

# Simple confirmation
- id: confirm_proceed
  type: hitl
  hitl_type: confirmation
  prompt: "Press Enter to continue..."
  timeout: 60
  fallback: continue
```

### Fallback Behaviors

| Fallback | Behavior on Timeout |
|----------|---------------------|
| `abort` | Stop workflow execution |
| `continue` | Continue with default value |
| `skip` | Skip this node entirely |
| `retry` | Retry the request |

---

## 6. Escape Hatches
