# Vertical Creation Guide - Part 2

**Part 2 of 2:** Extracting Templates from Existing Verticals, Template Validation, Best Practices, Examples, Advanced Topics, Troubleshooting, Next Steps, and Related Documentation

---

## Navigation

- [Part 1: Vertical Fundamentals](part-1-vertical-fundamentals.md)
- **[Part 2: Advanced & Examples](#)** (Current)
- [**Complete Guide](../vertical_creation_guide.md)**

---

## Table of Contents

1. [Overview](#overview) *(in Part 1)*
2. [Quick Start](#quick-start) *(in Part 1)*
3. [Template Structure](#template-structure) *(in Part 1)*
4. [Creating a Vertical from Scratch](#creating-a-vertical-from-scratch) *(in Part 1)*
5. [Extracting Templates from Existing Verticals](#extracting-templates-from-existing-verticals)
6. [Template Validation](#template-validation)
7. [Customizing Generated Verticals](#customizing-generated-verticals) *(in Part 1)*
8. [Best Practices](#best-practices)
9. [Examples](#examples)
10. [Advanced Topics](#advanced-topics)
11. [Troubleshooting](#troubleshooting)
12. [Next Steps](#next-steps)
13. [Related Documentation](#related-documentation)

---

## Best Practices

### 1. Follow Naming Conventions

- **Vertical name**: Use lowercase with underscores (e.g., `security_audit`)
- **Class name**: Use PascalCase with "Assistant" suffix (e.g., `SecurityAuditAssistant`)
- **Module name**: Match vertical name (e.g., `victor/security_audit/`)

### 2. Define Clear Stages

Stages define the workflow progression. Ensure:
- Each stage has a clear purpose
- Tools are appropriate for the stage
- Keywords cover common user language
- Transitions form a directed acyclic graph (no cycles)

```yaml
stages:
  INITIAL:
    description: Understanding the request
    tools: [read, ls]
    keywords: [what, how, explain, show]
    next_stages: [PLANNING, READING]

  PLANNING:
    description: Creating a plan
    tools: [read, grep]
    keywords: [plan, design, approach]
    next_stages: [READING, EXECUTION]

  READING:
    description: Gathering information
    tools: [read, grep, code_search]
    keywords: [read, find, search, look]
    next_stages: [ANALYSIS, EXECUTION]
```

### 3. Use Task Type Hints

Define task type hints for better agent behavior:

```yaml
extensions:
  prompt_hints:
    - task_type: create_simple
      hint: "[CREATE] Create the file immediately. One tool call max."
      tool_budget: 2
      priority_tools: [write]

    - task_type: debug
      hint: "[DEBUG] Read error context first, then fix."
      tool_budget: 12
      priority_tools: [read, grep, edit, test]
```

### 4. Define Safety Patterns

Add safety patterns for dangerous operations:

```yaml
extensions:
  safety_patterns:
    - name: production_deletion
      pattern: "rm -rf /prod"
      description: "Deleting production data"
      severity: critical
      category: files

    - name: force_push
      pattern: "git push --force"
      description: "Force pushing to main branch"
      severity: high
      category: git
```

### 5. Specify Tool Dependencies

Define tool dependencies for optimization:

```yaml
extensions:
  tool_dependencies:
    source_code_analysis:
      tools: [code_search, symbol, refs]
      required_tools: [read]
      description: "Analyze source code structure"

    refactoring:
      tools: [rename, extract]
      required_tools: [read, symbol, refs]
      description: "Safely refactor code"
```

### 6. Create Team Formations

Define teams for complex tasks:

```yaml
teams:
  - name: review_team
    display_name: "Code Review Team"
    description: "Multi-agent code review"
    formation: parallel
    roles:
      - name: security_reviewer
        persona: "You are a security expert..."
        capabilities: [security_scan]

      - name: quality_reviewer
        persona: "You are a code quality expert..."
        capabilities: [complexity_analysis, style_check]
```

## Examples

### Example 1: Minimal Vertical

```yaml
metadata:
  name: simple
  description: "A simple minimal vertical"
  version: "0.5.0"

tools:
  - read
  - write

system_prompt: |
  You are a helpful assistant for simple tasks.

stages:
  INITIAL:
    name: INITIAL
    description: Understanding
    tools: [read]
    keywords: [what, how]
    next_stages: [EXECUTION]

  EXECUTION:
    name: EXECUTION
    description: Doing
    tools: [read, write]
    keywords: [do, make]
    next_stages: [COMPLETION]

  COMPLETION:
    name: COMPLETION
    description: Done
    tools: []
    keywords: [done]
    next_stages: []
```

### Example 2: Security Vertical

See the full security vertical example in `templates/security.yaml`:

```yaml
metadata:
  name: security
  description: "Security analysis and vulnerability detection"
  version: "0.5.0"
  category: security
  tags: [security, vulnerability, scanning, audit]

tools:
  - read
  - write
  - grep
  - web_search
  - security_scan
  - cve_lookup
  - dependency_check

system_prompt: |
  You are a security expert specialized in finding
  vulnerabilities, security misconfigurations, and
  compliance issues...

stages:
  INITIAL:
    name: INITIAL
    description: Understanding the security request
    tools: [read, grep]
    keywords: [security, vulnerability, scan]
    next_stages: [SCANNING]

  SCANNING:
    name: SCANNING
    description: Scanning for vulnerabilities
    tools: [security_scan, dependency_check, cve_lookup]
    keywords: [scan, check, find]
    next_stages: [ANALYSIS]

  ANALYSIS:
    name: ANALYSIS
    description: Analyzing findings
    tools: [read, web_search]
    keywords: [analyze, assess, severity]
    next_stages: [REPORTING]

  REPORTING:
    name: REPORTING
    description: Generating security report
    tools: [write]
    keywords: [report, document]
    next_stages: [COMPLETION]

extensions:
  prompt_hints:
    - task_type: vulnerability_scan
      hint: "[SCAN] Use security_scan to find vulnerabilities..."
      tool_budget: 20
      priority_tools: [security_scan, dependency_check]

    - task_type: security_audit
      hint: "[AUDIT] Comprehensive security audit..."
      tool_budget: 50
      priority_tools: [security_scan, cve_lookup, web_search]

  safety_patterns:
    - name: dangerous_command
      pattern: "rm -rf"
      description: "Dangerous file deletion"
      severity: critical
      category: commands

teams:
  - name: security_review_team
    display_name: "Security Review Team"
    description: "Multi-agent security review"
    formation: parallel
    roles:
      - name: vulnerability_scanner
        persona: "You scan for known vulnerabilities..."
        capabilities: [cve_lookup, dependency_check]

      - name: code_auditor
        persona: "You audit code for security issues..."
        capabilities: [security_scan]
```

## Advanced Topics

### Custom File Templates

You can provide custom file templates in your YAML:

```yaml
file_templates:
  "__init__.py": |
    """Custom init for my vertical."""
    from victor.my_vertical.assistant import MyVerticalAssistant

    __all__ = ["MyVerticalAssistant"]

    def custom_function():
        return "custom"
```

### Programmatic Template Creation

```python
from victor.framework import VerticalTemplate, VerticalMetadata
from victor.core.vertical_types import StageDefinition

template = VerticalTemplate(
    metadata=VerticalMetadata(
        name="my_vertical",
        description="My custom vertical",
        version="0.5.0",
    ),
    tools=["read", "write", "grep"],
    system_prompt="You are an expert in...",
    stages={
        "INITIAL": StageDefinition(
            name="INITIAL",
            description="Understanding",
            tools={"read"},
            keywords=["what", "how"],
            next_stages={"EXECUTION"},
        ),
        # ... more stages ...
    },
)

# Register template
from victor.framework import register_template

register_template(template)

# Generate vertical
from victor.framework import VerticalGenerator

generator = VerticalGenerator(template, "victor/my_vertical")
generator.generate()
```

## Troubleshooting

### Template Validation Fails

```
Template validation errors:
  - Template metadata.name is required
  - Template must specify at least one tool
```

**Solution**: Ensure all required fields are present in your template.

### Generated Code Won't Import

```
ImportError: cannot import name 'MyVerticalAssistant'
```

**Solution**:
1. Check that the vertical is in the correct location: `victor/my_vertical/`
2. Ensure `__init__.py` exists and exports the class
3. Register the vertical in `victor/core/verticals/__init__.py`

### Tools Not Available

```
ToolError: Tool 'my_custom_tool' not found
```

**Solution**:
1. Ensure the tool is registered in the tool registry
2. Check that the tool name in `get_tools()` matches the registered name
3. Use canonical tool names from `victor.tools.tool_names.ToolNames`

## Next Steps

- See the [Architecture Guide](../architecture/overview.md) for vertical architecture details
- Check [existing verticals](../verticals/coding.md) for reference implementations
- Review [tool development guide](build-custom-tool.md) to create custom tools
- Read [workflow guide](create-workflow.md) for defining YAML workflows

## Related Documentation

- [Vertical Architecture](../reference/verticals/index.md)
- [Protocol Reference](../architecture/PROTOCOLS_REFERENCE.md)
- [Framework README](../reference/internals/FRAMEWORK_API.md)
- [Configuration Guide](../architecture/BEST_PRACTICES.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
