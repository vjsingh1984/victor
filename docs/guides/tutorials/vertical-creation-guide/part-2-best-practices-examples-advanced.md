# Vertical Creation Guide - Part 2

**Part 2 of 2:** Best Practices, Examples, Advanced Topics, Troubleshooting

---

## Navigation

- [Part 1: Templates & Creation](part-1-overview-customization.md)
- **[Part 2: Best Practices, Examples, Advanced](#)** (Current)
- [**Complete Guide**](../vertical_creation_guide.md)

---

## Best Practices

### 1. Keep Verticals Focused

Each vertical should have a single, well-defined purpose.

**Good:**
- Security Analysis (security, vulnerability scanning)
- Data Analysis (pandas, visualization, statistics)
- DevOps (docker, kubernetes, CI/CD)

**Avoid:**
- "Everything" vertical (mixed concerns)
- Overly narrow verticals (e.g., "python_for_loops")

### 2. Use Declarative Configuration

Define tools, prompts, and settings in YAML rather than code.

**Good (YAML):**
```yaml
tools:
  - read
  - write
  - grep

system_prompt: |
  You are a code analysis assistant...
```

**Avoid (hardcoded):**
```python
def get_tools(cls):
    return ["read", "write", "grep"]
```

### 3. Validate Before Using

Always validate templates before generating verticals.

```bash
victor vertical validate my_vertical.yaml
```

### 4. Document Customizations

Document any custom code you add to generated verticals.

```python
# Custom analysis logic for security vertical
# This extends the base scanning to check for OWASP Top 10
def custom_security_check(code: str) -> List[SecurityIssue]:
    """Custom security check beyond base template."""
    # Implementation...
```

---

## Examples

### Example 1: Simple Analysis Vertical

```yaml
# code_analysis.yaml
metadata:
  name: code_analysis
  description: Static code analysis and quality checks
  version: "0.5.0"
  category: development
  tags: [analysis, quality, code-review]

tools:
  - read
  - write
  - grep
  - pylint
  - mypy

system_prompt: |
  You are a code analysis assistant. Help users:
  - Identify code quality issues
  - Suggest improvements
  - Review code for best practices
  - Analyze complexity and maintainability

modes:
  quick_analysis:
    exploration: 1.5x
    tool_budget_multiplier: 0.8
  deep_review:
    exploration: 3.0x
    tool_budget_multiplier: 1.5
```

### Example 2: Data Science Vertical

```yaml
# data_science.yaml
metadata:
  name: data_science
  description: Data analysis, visualization, and machine learning
  version: "0.5.0"
  category: data
  tags: [data, pandas, visualization, ml]

tools:
  - read
  - write
  - python_execute
  - pandas_query
  - matplotlib_plot

system_prompt: |
  You are a data science assistant. Help users:
  - Analyze datasets using pandas
  - Create visualizations
  - Perform statistical analysis
  - Build and evaluate ML models

workflows:
  analyze_dataset:
    description: "Comprehensive dataset analysis"
    steps:
      - load_data
      - explore_statistics
      - create_visualizations
      - generate_report
```

---

## Advanced Topics

### Custom Tool Implementations

Override default tool implementations:

```python
# In generated vertical/custom_tools.py
from victor.tools.base import BaseTool

class CustomAnalysisTool(BaseTool):
    """Custom analysis for this vertical."""

    name = "custom_analysis"
    description = "Performs vertical-specific analysis"

    def execute(self, **kwargs):
        # Custom implementation
        pass
```

### Custom Prompt Builders

Extend prompt building logic:

```python
# In generated vertical/prompts.py
from victor.agent.coordinators import PromptCoordinator

class CustomPromptCoordinator(PromptCoordinator):
    """Custom prompt building for this vertical."""

    def build_prompt(self, context):
        # Add custom prompt sections
        base_prompt = super().build_prompt(context)
        custom_section = self._build_custom_section(context)
        return base_prompt + custom_section
```

### Mode Configuration

Define custom modes for your vertical:

```yaml
modes:
  conservative:
    exploration: 1.0x
    tool_budget_multiplier: 0.5
    description: "Low exploration, cost-conscious"

  balanced:
    exploration: 2.0x
    tool_budget_multiplier: 1.0
    description: "Balanced exploration and cost"

  aggressive:
    exploration: 4.0x
    tool_budget_multiplier: 2.0
    description: "High exploration, thorough analysis"
```

---

## Troubleshooting

### Template Validation Errors

**Problem**: Template validation fails.

**Solutions**:
1. **Check YAML syntax**: Ensure valid YAML formatting
2. **Verify required fields**: Check all required metadata fields
3. **Validate tool names**: Ensure tools exist in registry

```bash
# Validate template
victor vertical validate my_vertical.yaml

# List available tools
victor vertical list-tools
```

### Generated Code Issues

**Problem**: Generated code has errors.

**Solutions**:
1. **Check template version**: Ensure template matches Victor version
2. **Review customizations**: Check if custom code conflicts
3. **Regenerate**: Regenerate from template

```bash
# Regenerate vertical
victor vertical regenerate my_vertical.yaml --force
```

### Missing Tools

**Problem**: Tools specified in template not found.

**Solutions**:
1. **Check tool names**: Verify tool names match registry
2. **Register custom tools**: Register any custom tools
3. **Use tool aliases**: Use standard tool names

```bash
# List registered tools
victor tools list

# Register custom tool
victor tools register my_tool.py
```

---

## Next Steps

1. **Explore Templates**: Check existing vertical templates
2. **Create Custom Vertical**: Build your first vertical
3. **Extend Vertical**: Add custom features to generated vertical
4. **Contribute**: Share useful verticals with community

---

## Related Documentation

- [Vertical Architecture](../../../architecture/VERTICALS.md)
- [Template Reference](../../reference/TEMPLATE_REFERENCE.md)
- [Creating Tools](../CREATING_TOOLS.md)
- [Advanced Customization](../ADVANCED_VERTICAL_CUSTOMIZATION.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 15 min (all parts)
