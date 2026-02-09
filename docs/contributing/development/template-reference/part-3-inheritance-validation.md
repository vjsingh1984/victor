# Template Reference - Part 3

**Part 3 of 4:** Template Inheritance and Validation

---

## Navigation

- [Part 1: Structure & Extensions](part-1-structure-extensions.md)
- [Part 2: Workflows & Config](part-2-workflows-config.md)
- **[Part 3: Inheritance & Validation](#)** (Current)
- [Part 4: Examples & Migration](part-4-examples-migration.md)
- [**Complete Reference**](../template_reference.md)

---

### File Templates Schema

```yaml
file_templates:
  assistant_py: |                      # Override assistant.py template
    # Custom template for assistant.py
    from typing import List
    from victor.core.verticals.base import VerticalBase

    class {vertical_class_name}(VerticalBase):
        # Custom implementation
        pass

  prompts_py: |                        # Override prompts.py template
    # Custom template for prompts.py
    from typing import Dict, List

    def get_prompt_contributors() -> List[Dict]:
        return []

  safety_py: |                         # Override safety.py template
    # Custom template for safety.py
    from typing import List

    def get_safety_patterns() -> List:
        return []
```

### Template Variables

Available variables in file templates:

| Variable | Description | Example |
|----------|-------------|---------|
| `{vertical_name}` | Vertical name | `my_vertical` |
| `{vertical_class_name}` | Class name | `MyVertical` |
| `{description}` | Description | `My custom vertical` |
| `{version}` | Version | `0.5.0` |
| `{author}` | Author | `Your Name` |
| `{tools_list}` | Tools list | `['tool1', 'tool2']` |
| `{system_prompt}` | System prompt | `You are an expert...` |

## Template Inheritance

### Extending Templates

```yaml
# child_template.yaml
extends: base_vertical

# Override metadata
metadata:
  name: child_vertical
  description: "Child vertical"
  version: "0.5.0"

# Add to inherited tools
tools:
  - custom_tool_1
  - custom_tool_2

# Override system prompt
system_prompt: |
  Custom system prompt...

# Add to inherited extensions
extensions:
  middleware:
    - name: custom_middleware
      class_name: CustomMiddleware
      module: victor.child.middleware
```

### Inheritance Rules

1. **Tools**: Child tools are added to parent tools
2. **System Prompt**: Child prompt replaces parent prompt
3. **Stages**: Child stages replace parent stages
4. **Extensions**: Child extensions are merged with parent extensions
5. **Workflows**: Child workflows are added to parent workflows
6. **Teams**: Child teams are added to parent teams
7. **Capabilities**: Child capabilities are added to parent capabilities

### Override Behavior

| Section | Parent | Child | Result |
|---------|--------|-------|--------|
| `metadata` | - | - | Replaced |
| `tools` | `[a, b]` | `[c]` | `[a, b, c]` |
| `system_prompt` | `"A"` | `"B"` | `"B"` |
| `stages` | `{...}` | `{...}` | Replaced |
| `extensions.middleware` | `[a]` | `[b]` | `[a, b]` |
| `workflows` | `[a]` | `[b]` | `[a, b]` |

## Validation Rules

### Required Fields

Templates must include:

1. **Metadata**: `name`, `description`, `version`
2. **Tools**: At least one tool
3. **System Prompt**: Non-empty string
4. **Stages**: At least one stage

### Protocol Validation

If declaring protocols:

```yaml
metadata:
  protocols:
    - ToolProvider              # Valid
    - PromptContributorProvider # Valid
    - InvalidProtocol           # INVALID - will fail validation
```

Valid protocols:
- `ToolProvider`
- `PromptContributorProvider`
- `MiddlewareProvider`
- `SafetyProvider`
- `WorkflowProvider`
- `HandlerProvider`
- `CapabilityProvider`
- `ModeConfigProvider`
- `ServiceProvider`
- `TieredToolConfigProvider`
- `ToolDependencyProvider`

### Tool Validation

Tools must be registered in the tool registry or defined as custom tools.

### Workflow Reference Validation

Workflow files must exist relative to template location:

```yaml
workflows:
  - name: my_workflow
    file: workflows/my_workflow.yaml  # Must exist!
```

### Stage Validation

Stages must have:
- Unique names
- At least one keyword (or empty list)
- Valid next stage references

### Command-Line Validation

```bash
# Validate template before generation
python scripts/generate_vertical.py --validate template.yaml

# Output:
# ✅ Template is valid
# or
# ❌ Validation errors:
# - Missing required field: metadata.name
# - Invalid protocol: InvalidProtocol
# - Workflow file not found: workflows/missing.yaml
```

## Complete Example

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 2 min
**Last Updated:** February 08, 2026**
