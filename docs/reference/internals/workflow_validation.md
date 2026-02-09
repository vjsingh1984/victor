# Workflow Validation and Linting

This document describes the comprehensive workflow validation and linting system for Victor YAML workflows.

## Overview

The workflow validation system provides:

- **Syntax Validation**: Checks YAML syntax and structure
- **Schema Validation**: Validates node schemas and required fields
- **Connection Validation**: Ensures all node references are valid
- **Circular Dependency Detection**: Detects cycles in workflow graphs
- **Team Node Validation**: Validates team-specific configuration
- **Best Practices Checking**: Enforces workflow design best practices
- **Complexity Analysis**: Analyzes workflow complexity metrics

## Quick Start

### Command Line

```bash
# Lint a single workflow file
victor workflow lint workflow.yaml

# Lint all workflows in a directory
victor workflow lint workflows/ --recursive

# Generate JSON report
victor workflow lint workflow.yaml --format json -o report.json

# Enable strict mode (all rules)
victor workflow lint workflow.yaml --strict

# Disable specific rules
victor workflow lint workflow.yaml --disable-rule complexity_analysis

# Generate markdown report
victor workflow lint . --format markdown -o lint_report.md
```

### Python API

```python
from victor.workflows.linter import WorkflowLinter

# Create linter
linter = WorkflowLinter()

# Lint a file
result = linter.lint_file("workflow.yaml")

# Check if valid
if result.is_valid:
    print("Workflow is valid!")
else:
    print(f"Found {result.error_count} errors")

# Generate report
report = result.generate_report(format="text")
print(report)
```

## Validation Rules

### Available Rules

| Rule ID | Category | Severity | Description |
|---------|----------|----------|-------------|
| `node_id_format` | SYNTAX | WARNING | Ensures node IDs follow naming conventions |
| `required_fields` | SCHEMA | ERROR | Checks required fields for each node type |
| `connection_references` | CONNECTIONS | ERROR | Validates all node references |
| `circular_dependency` | CONNECTIONS | ERROR | Detects circular dependencies |
| `team_formation` | TEAM | ERROR | Validates team node configuration |
| `goal_quality` | BEST_PRACTICES | WARNING | Checks goal description quality |
| `tool_budget` | BEST_PRACTICES | WARNING | Validates tool budget configuration |
| `disconnected_nodes` | CONNECTIONS | WARNING | Detects disconnected nodes |
| `duplicate_node_ids` | SYNTAX | ERROR | Checks for duplicate node IDs |
| `complexity_analysis` | COMPLEXITY | INFO | Analyzes workflow complexity |

### Rule Categories

- **SYNTAX**: YAML syntax and structure issues
- **SCHEMA**: Node schema validation
- **CONNECTIONS**: Node reference validation
- **BEST_PRACTICES**: Workflow design best practices
- **SECURITY**: Security and safety checks
- **COMPLEXITY**: Workflow complexity analysis
- **TEAM**: Team node specific validation

### Severity Levels

- **ERROR**: Must be fixed (blocks execution)
- **WARNING**: Should be fixed (may cause issues)
- **INFO**: Informational (no action required)
- **SUGGESTION**: Improvement suggestions

## Validation Checks

### YAML Syntax Validation

```yaml
# ✓ Valid
nodes:
  - id: start_node
    type: agent
    role: researcher
    goal: "Research topic"

# ✗ Invalid (uppercase, hyphens)
nodes:
  - id: Start-Node
    type: agent
    role: researcher
    goal: "Research topic"
```

### Schema Validation

#### Agent Nodes

Required fields:
- `role`: Agent role (researcher, planner, executor, reviewer, writer, analyst, coder)
- `goal`: Task description

```yaml
# ✓ Valid agent node
- id: research
  type: agent
  role: researcher
  goal: "Conduct comprehensive research on AI trends"
  tool_budget: 25

# ✗ Missing goal
- id: research
  type: agent
  role: researcher
```

#### Compute Nodes

Required fields:
- `handler`: Registered compute handler name

```yaml
# ✓ Valid compute node
- id: process
  type: compute
  handler: data_transformer
  inputs:
    data: $ctx.raw_data

# ✗ Missing handler
- id: process
  type: compute
```

#### Condition Nodes

Required fields:
- `condition`: Condition expression
- `branches`: Branch mappings

```yaml
# ✓ Valid condition node
- id: check_quality
  type: condition
  condition: "quality_score >= 0.7"
  branches:
    "true": proceed
    "false": retry

# ✗ Missing branches
- id: check_quality
  type: condition
  condition: "quality_score >= 0.7"
```

#### Team Nodes

Required fields:
- `team_formation`: Formation type (sequential, parallel, hierarchical, pipeline, consensus, round_robin, dynamic, expert)
- `members`: List of team members
- `goal`: Team goal

Member requirements:
- `role`: Member role
- `goal`: Member goal

```yaml
# ✓ Valid team node
- id: research_team
  type: team
  team_formation: parallel
  goal: "Conduct comprehensive research"
  members:
    - id: researcher
      role: researcher
      goal: "Find information"
    - id: writer
      role: writer
      goal: "Write report"
  max_iterations: 3
  total_tool_budget: 50

# ✗ Invalid formation type
- id: research_team
  type: team
  team_formation: invalid_formation
```

### Connection Validation

```yaml
# ✓ Valid references
nodes:
  - id: node1
    type: agent
    role: researcher
    goal: "Research"
    next: [node2]
  - id: node2
    type: agent
    role: writer
    goal: "Write"

# ✗ Invalid reference
nodes:
  - id: node1
    type: agent
    role: researcher
    goal: "Research"
    next: [nonexistent_node]  # Error: node not found
```

### Circular Dependency Detection

```yaml
# ✗ Circular dependency
nodes:
  - id: node1
    next: [node2]
  - id: node2
    next: [node3]
  - id: node3
    next: [node1]  # Creates cycle
```

### Best Practices

#### Goal Quality

Goals should:
- Be at least 20 characters (agent nodes) or 30 characters (team nodes)
- Describe what to accomplish
- Not be too generic

```yaml
# ✓ Good goal
goal: "Conduct comprehensive research on AI trends and summarize findings"

# ✗ Too short
goal: "Do task"

# ✗ Too generic
goal: "Execute"
```

#### Tool Budget

```yaml
# ✓ Reasonable budget
tool_budget: 25

# ⚠ Very high budget
tool_budget: 150

# ✗ Invalid budget
tool_budget: -5
```

### Complexity Analysis

The linter analyzes:
- Total node count
- Maximum nesting depth
- Cyclomatic complexity (branches + 1)

Warnings are issued for:
- More than 50 nodes (consider splitting)
- Nesting depth > 10 (consider flattening)

```yaml
# ℹ Complexity info
Workflow complexity: 5 nodes, depth 2, cyclomatic complexity 1
```

## Report Formats

### Text Format

```
============================================================
Workflow Linting Report
============================================================
Files checked: 1
Workflows validated: 1
Duration: 0.15s

Summary:
  Errors: 0
  Warnings: 2
  Info: 1
  Suggestions: 0

Issues:

⚠ WARNING (2):
  [goal_quality] Goal description too short (8 < 20 chars)
    Location: workflow.yaml:research
    Suggestion: Provide more detailed goal description

  [tool_budget] Tool budget very high (150 > 100)
    Location: workflow.yaml:research
    Suggestion: Consider reducing tool budget to prevent excessive API calls

ℹ INFO (1):
  [complexity_analysis] Workflow complexity: 5 nodes, depth 2, cyclomatic complexity 1
    Location: workflow.yaml
```

### JSON Format

```json
{
  "summary": {
    "files_checked": 1,
    "workflow_count": 1,
    "duration_seconds": 0.15,
    "error_count": 0,
    "warning_count": 2,
    "info_count": 1,
    "suggestion_count": 0
  },
  "issues": [
    {
      "rule_id": "goal_quality",
      "severity": "warning",
      "category": "best_practices",
      "message": "Goal description too short (8 < 20 chars)",
      "location": "workflow.yaml:research",
      "suggestion": "Provide more detailed goal description",
      "context": {}
    }
  ]
}
```

### Markdown Format

```markdown
# Workflow Linting Report

- **Files checked:** 1
- **Workflows validated:** 1
- **Duration:** 0.15s

## Summary

- **Errors:** 0
- **Warnings:** 2
- **Info:** 1
- **Suggestions:** 0

## Issues

### ⚠️ WARNING (2)

**[goal_quality]** Goal description too short (8 < 20 chars)

- **Location:** `workflow.yaml:research`
- **Category:** best_practices
- **Suggestion:** Provide more detailed goal description
```

## Python API

### Basic Usage

```python
from victor.workflows.linter import WorkflowLinter, LinterResult

# Create linter with default rules
linter = WorkflowLinter()

# Lint a file
result: LinterResult = linter.lint_file("workflow.yaml")

# Check results
if result.is_valid:
    print("✓ Valid workflow")
else:
    print(f"✗ Found {result.error_count} errors")

# Access issues
for issue in result.issues:
    print(f"[{issue.severity.value}] {issue.message}")
    print(f"  Location: {issue.location}")
    if issue.suggestion:
        print(f"  Suggestion: {issue.suggestion}")
```

### Custom Rules

```python
from victor.workflows.validation_rules import ValidationRule, Severity, RuleCategory

class CustomRule(ValidationRule):
    def __init__(self):
        super().__init__(
            rule_id="custom_rule",
            category=RuleCategory.BEST_PRACTICES,
            severity=Severity.WARNING,
        )

    def check(self, workflow):
        issues = []
        # Custom validation logic
        for wf_name, wf_def in workflow.get("workflows", {}).items():
            for node in wf_def.get("nodes", []):
                if node.get("type") == "agent":
                    goal = node.get("goal", "")
                    if "TODO" in goal:
                        issues.append(
                            self.create_issue(
                                message="Goal contains TODO marker",
                                location=f"{wf_name}:{node.get('id')}",
                                suggestion="Replace TODO with actual goal",
                            )
                        )
        return issues

# Add custom rule
linter = WorkflowLinter()
linter.add_rule(CustomRule())
```

### Rule Management

```python
# Disable specific rules
linter.disable_rule("complexity_analysis")
linter.disable_rule("goal_quality")

# Enable specific rules
linter.enable_rule("complexity_analysis")

# Change severity
linter.set_rule_severity("tool_budget", Severity.ERROR)

# Get all rules
rules = linter.get_rules()
for rule in rules:
    print(f"{rule.rule_id}: {rule.enabled}")

# Get enabled rules
enabled = linter.get_enabled_rules()
```

### Filtering Issues

```python
from victor.workflows.validation_rules import Severity, RuleCategory

# Filter by severity
errors = result.get_issues_by_severity(Severity.ERROR)
warnings = result.get_issues_by_severity(Severity.WARNING)

# Filter by category
syntax_issues = result.get_issues_by_category(RuleCategory.SYNTAX)
best_practices = result.get_issues_by_category(RuleCategory.BEST_PRACTICES)

# Filter by location
node_issues = result.get_issues_by_location("workflow.yaml:research_node")
```

### Directory Linting

```python
# Lint all YAML files in directory
result = linter.lint_directory("workflows/")

# Recursive directory linting
result = linter.lint_directory("workflows/", recursive=True)

# Custom pattern
result = linter.lint_directory("workflows/", pattern="*.yaml")
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Workflow Validation

on: [push, pull_request]

jobs:
  validate-workflows:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Victor
        run: pip install victor-ai
      - name: Lint workflows
        run: |
          victor workflow lint workflows/ --recursive --format json -o lint_report.json
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: lint-report
          path: lint_report.json
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Lint all workflow files
victor workflow lint workflows/ --recursive

if [ $? -ne 0 ]; then
    echo "Workflow validation failed. Please fix errors before committing."
    exit 1
fi
```

### GitLab CI

```yaml
workflow_validation:
  stage: test
  script:
    - pip install victor-ai
    - victor workflow lint workflows/ --recursive --format json -o lint_report.json
  artifacts:
    paths:
      - lint_report.json
    when: on_failure
```

## Best Practices

1. **Run linter regularly**: Integrate into CI/CD pipeline
2. **Fix warnings**: Don't ignore warnings, they often indicate real issues
3. **Review complexity**: High complexity workflows should be split
4. **Use descriptive goals**: Goals should clearly state what to accomplish
5. **Validate references**: Ensure all node references point to existing nodes
6. **Check team configuration**: Validate team formations and member roles
7. **Monitor tool budgets**: Excessive tool budgets can lead to high costs

## Troubleshooting

### Common Issues

**Issue**: `Invalid node ID format`
- **Fix**: Use lowercase with underscores: `my_node` instead of `My-Node`

**Issue**: `Missing required field: goal`
- **Fix**: Add goal field to agent nodes

**Issue**: `Invalid 'next' reference`
- **Fix**: Ensure referenced node IDs exist in the workflow

**Issue**: `Circular dependency detected`
- **Fix**: Break the cycle by removing one connection

**Issue**: `Tool budget very high`
- **Fix**: Reduce tool budget or justify high value with comment

### Getting Help

- Check the [Victor documentation](https://victor-ai.readthedocs.io/)
- Review example workflows in `victor/*/workflows/`
- Ask questions on GitHub Discussions
- Report bugs on GitHub Issues

## API Reference

### WorkflowLinter

```python
class WorkflowLinter:
    def __init__(self, rules: Optional[List[ValidationRule]] = None)
    def lint_file(self, file_path: str | Path) -> LinterResult
    def lint_directory(self, directory: str | Path, pattern: str = "*.yaml", recursive: bool = False) -> LinterResult
    def lint_dict(self, workflow: Dict[str, Any]) -> LinterResult
    def add_rule(self, rule: ValidationRule) -> None
    def remove_rule(self, rule_id: str) -> None
    def enable_rule(self, rule_id: str) -> None
    def disable_rule(self, rule_id: str) -> None
    def set_rule_severity(self, rule_id: str, severity: Severity) -> None
    def get_rules(self) -> List[ValidationRule]
    def get_enabled_rules(self) -> List[ValidationRule]
    def get_rule(self, rule_id: str) -> Optional[ValidationRule]
```

### LinterResult

```python
@dataclass
class LinterResult:
    issues: List[ValidationIssue]
    files_checked: int
    workflow_count: int
    duration_seconds: float

    @property
    def error_count(self) -> int
    @property
    def warning_count(self) -> int
    @property
    def info_count(self) -> int
    @property
    def suggestion_count(self) -> int
    @property
    def has_errors(self) -> bool
    @property
    def has_warnings(self) -> bool
    @property
    def is_valid(self) -> bool

    def get_issues_by_severity(self, severity: Severity) -> List[ValidationIssue]
    def get_issues_by_category(self, category: RuleCategory) -> List[ValidationIssue]
    def get_issues_by_location(self, location: str) -> List[ValidationIssue]
    def generate_report(self, format: str = "text", include_suggestions: bool = True, include_context: bool = False) ->
  str
```

### ValidationIssue

```python
@dataclass
class ValidationIssue:
    rule_id: str
    severity: Severity
    category: RuleCategory
    message: str
    location: str
    suggestion: Optional[str]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]
```

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
