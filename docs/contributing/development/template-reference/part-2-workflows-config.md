# Template Reference - Part 2

**Part 2 of 4:** Workflows, Teams, and Configuration

---

## Navigation

- [Part 1: Structure & Extensions](part-1-structure-extensions.md)
- **[Part 2: Workflows & Config](#)** (Current)
- [Part 3: Inheritance & Validation](part-3-inheritance-validation.md)
- [Part 4: Examples & Migration](part-4-examples-migration.md)
- [**Complete Reference**](../template_reference.md)

---

### Overview

Extensions provide ISP-compliant protocol implementations. Only implement what you need.

### Middleware

```yaml
extensions:
  middleware:
    - name: validation_middleware        # Required: Middleware name
      class_name: ValidationMiddleware  # Required: Python class name
      module: victor.my_vertical.middleware  # Required: Import path
      enabled: true                     # Optional: Enable/disable (default: true)
      config:                           # Optional: Middleware configuration
        strict_mode: true
        max_retries: 3
```

### Safety Patterns

```yaml
extensions:
  safety_patterns:
    - name: dangerous_delete            # Required: Pattern name
      pattern: "rm -rf .*"              # Required: Regex pattern
      description: "Recursive delete"   # Required: Description
      severity: critical                # Optional: low, medium, high, critical (default: medium)
      category: commands                # Optional: Pattern category (default: general)
```

#### Severity Levels

| Level | Description | Action |
|-------|-------------|--------|
| `low` | Minor issues | Warning only |
| `medium` | Moderate issues | Warning |
| `high` | Serious issues | Warning + confirmation |
| `critical` | Dangerous operations | Block or require explicit confirmation |

#### Pattern Categories

- `commands`: Shell commands
- `files`: File operations
- `git`: Git operations
- `network`: Network operations
- `general`: General patterns

### Prompt Hints

```yaml
extensions:
  prompt_hints:
    - task_type: create                 # Required: Task type identifier
      hint: "[CREATE] Description..."   # Required: Prompt hint text
      tool_budget: 15                   # Optional: Suggested budget (default: 10)
      priority_tools:                   # Optional: Priority tools for this task
        - read_file
        - write_file
```

#### Common Task Types

| Task Type | Description |
|-----------|-------------|
| `create` | Creating new content |
| `edit` | Editing existing content |
| `debug` | Debugging issues |
| `refactor` | Refactoring code |
| `analyze` | Analyzing code |
| `review` | Reviewing code |
| `test` | Testing code |
| `deploy` | Deploying changes |
| `general` | General tasks |

### Handlers

```yaml
extensions:
  handlers:
    handler_name:                       # Handler identifier
      class_name: MyHandler            # Required: Python class name
      module: victor.my_vertical.handlers  # Required: Import path
      description: "Description"       # Optional: Handler description
```

### Personas

```yaml
extensions:
  personas:
    expert:
      name: Expert                     # Required: Persona name
      description: "Expert persona"    # Required: Description
      system_prompt_extension: |       # Optional: Additional prompt content
        You are an expert with deep knowledge...
      capabilities:                    # Optional: Associated capabilities
        - capability_1
        - capability_2
```

### Composed Chains

```yaml
extensions:
  composed_chains:
    chain_name:
      name: "Chain Name"               # Required: Chain name
      description: "Description"       # Optional: Description
      tools:                           # Required: List of tools in chain
        - tool_1
        - tool_2
        - tool_3
      config:                          # Optional: Chain configuration
        parallel: false                # Execute tools in parallel?
        stop_on_error: true            # Stop on first error?
```

## Workflows

### Workflow Reference Schema

```yaml
workflows:
  - name: workflow_name                # Required: Workflow identifier
    file: workflows/workflow.yaml      # Required: Path to workflow YAML
    description: "Description"         # Optional: Human-readable description
    enabled: true                      # Optional: Enable/disable (default: true)
```

### Workflow File Structure

```yaml
# workflows/my_workflow.yaml
workflow:
  name: my_workflow
  description: "My workflow"

  nodes:
    - id: start
      type: agent
      role: analyst
      goal: "Analyze the request"
      tool_budget: 5
      next: [process]

    - id: process
      type: compute
      handler: my_handler
      inputs:
        data: $ctx.result
      next: [end]

    - id: end
      type: agent
      role: summarizer
      goal: "Summarize results"
```

### Node Types

| Type | Description |
|------|-------------|
| `agent` | LLM-powered agent node |
| `compute` | Compute handler execution |
| `condition` | Conditional branching |
| `parallel` | Parallel execution |
| `transform` | Data transformation |
| `hitl` | Human-in-the-loop |

## Teams

### Team Definition Schema

```yaml
teams:
  - name: team_name                    # Required: Team identifier
    display_name: "Team Name"          # Required: Human-readable name
    description: "Description"         # Optional: Team description
    formation: parallel                # Required: Formation type
    communication_style: structured    # Optional: Communication style
    max_iterations: 5                  # Optional: Max iterations (default: 5)
    roles:                            # Required: Team roles
      - name: role_name
        display_name: "Role Name"
        description: "Role description"
        persona: "Role persona..."
        tool_categories: [category1]
        capabilities: [capability1]
```

### Formation Types

| Formation | Description |
|-----------|-------------|
| `pipeline` | Sequential execution through stages |
| `parallel` | Concurrent execution with aggregation |
| `sequential` | Step-by-step execution |
| `hierarchical` | Manager-worker coordination |
| `consensus` | Vote-based decision making |

### Communication Styles

| Style | Description |
|-------|-------------|
| `structured` | Structured message passing |
| `freeform` | Free-form communication |

### Role Definition Schema

```yaml
roles:
  - name: reviewer                     # Required: Role identifier
    display_name: "Reviewer"           # Required: Human-readable name
    description: "Reviews code"        # Required: Role description
    persona: |                        # Required: Agent persona
      You are a code reviewer focused on...
    tool_categories:                   # Optional: Tool categories
      - analysis
      - metrics
    capabilities:                      # Optional: Required capabilities
      - code_review
      - style_check
```

## Capabilities

### Capability Definition Schema

```yaml
capabilities:
  - name: capability_name              # Required: Capability identifier
    type: workflow                     # Required: Capability type
    description: "Description"         # Optional: Capability description
    enabled: true                      # Optional: Enable/disable (default: true)
    handler: path.to.Handler           # Optional: Import path to handler
    config:                            # Optional: Capability configuration
      setting_1: value_1
```

### Capability Types

| Type | Description |
|------|-------------|
| `tool` | Tool capability |
| `workflow` | Workflow capability |
| `middleware` | Middleware capability |
| `validator` | Validator capability |
| `observer` | Observer capability |

## Custom Configuration

### Custom Config Schema

```yaml
custom_config:
  # Vertical-specific settings
  my_setting: "value"

  # Grounding rules for prompts
  grounding_rules: |
    Base all responses on tool output.
    Never fabricate information.

  # System prompt sections
  system_prompt_section: |
    Follow best practices for this domain.

  # Additional metadata
  metadata:
    key: value
```

### Common Custom Configurations

#### Language Support

```yaml
custom_config:
  supported_languages:
    - python
    - javascript
    - typescript

  language_specific_prompts:
    python: "Follow PEP 8 style guidelines."
    javascript: "Follow Airbnb style guide."
```

#### Tool Budgets

```yaml
custom_config:
  default_tool_budget: 10

  task_specific_budgets:
    create: 15
    edit: 10
    debug: 20
    refactor: 25
```

#### Output Formats

```yaml
custom_config:
  output_formats:
    - markdown
    - code_block
    - list

  default_format: markdown
```

## File Templates

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 4 min
**Last Updated:** February 08, 2026**
