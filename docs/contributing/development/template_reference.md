# Vertical Template Reference

Complete reference for vertical template YAML syntax and structure.

## Table of Contents

- [Template Structure](#template-structure)
- [Metadata](#metadata)
- [Tools](#tools)
- [System Prompt](#system-prompt)
- [Stages](#stages)
- [Extensions](#extensions)
- [Workflows](#workflows)
- [Teams](#teams)
- [Capabilities](#capabilities)
- [Custom Configuration](#custom-configuration)
- [File Templates](#file-templates)
- [Template Inheritance](#template-inheritance)
- [Validation Rules](#validation-rules)

## Template Structure

### Top-Level Schema

```yaml
metadata:               # Required: Vertical metadata
  name: string
  description: string
  version: string
  # ... other metadata fields

tools:                  # Required: List of tool names
  - tool_name_1
  - tool_name_2

system_prompt: string   # Required: Main system prompt

stages:                 # Required: Workflow stages
  STAGE_NAME:
    name: string
    description: string
    tools: []
    keywords: []
    next_stages: []

extensions:             # Optional: ISP-compliant extensions
  middleware: []
  safety_patterns: []
  prompt_hints: []
  handlers: {}
  personas: {}
  composed_chains: {}

workflows: []           # Optional: Workflow references

teams: []               # Optional: Team formations

capabilities: []        # Optional: Capability declarations

custom_config: {}       # Optional: Vertical-specific config

file_templates: {}      # Optional: Custom file templates

extends: string         # Optional: Parent template name
```

## Metadata

### Required Fields

```yaml
metadata:
  name: my_vertical           # Required: Vertical identifier (snake_case)
  description: "Description"  # Required: Human-readable description
  version: "0.5.0"           # Required: Semantic version
```

### Optional Fields

```yaml
metadata:
  author: "Your Name"                    # Optional: Author
  license: "Apache-2.0"                  # Optional: License (default: Apache-2.0)
  category: general                      # Optional: Category
  tags:                                 # Optional: Discovery tags
    - tag1
    - tag2

  provider_hints:                       # Optional: LLM provider hints
    preferred_models:                   # Preferred model names
      - claude-sonnet-4-5
      - gpt-4
    min_context: 128000                 # Minimum context window
    supports_tools: true                # Whether model supports tools
    preferred_providers:                # Preferred providers
      - anthropic
      - openai

  evaluation_criteria:                  # Optional: Evaluation metrics
    - metric_name_1
    - metric_name_2
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Vertical identifier (use snake_case) |
| `description` | string | Yes | Human-readable description |
| `version` | string | Yes | Semantic version (e.g., "0.5.0") |
| `author` | string | No | Author name or email |
| `license` | string | No | License name (default: "Apache-2.0") |
| `category` | string | No | Vertical category |
| `tags` | list | No | Tags for template discovery |
| `provider_hints` | dict | No | LLM provider selection hints |
| `evaluation_criteria` | list | No | Metrics for evaluation |

### Provider Hints Schema

```yaml
provider_hints:
  preferred_models:           # List of preferred model names
    - model_name_1
    - model_name_2

  min_context: integer        # Minimum context window (tokens)

  supports_tools: boolean     # Whether model must support tool calling

  preferred_providers:        # List of preferred providers
    - provider_name_1
    - provider_name_2

  required_features:          # Required model features
    - feature_1
    - feature_2
```

## Tools

### Tool List

```yaml
tools:
  # Basic file operations
  - read_file
  - write_file
  - edit_file
  - grep
  - ls

  # Search tools
  - code_search
  - semantic_code_search

  # Web tools
  - web_search
  - web_fetch

  # Shell tools
  - execute_bash

  # Domain-specific tools
  - custom_tool_1
  - custom_tool_2
```

### Available Tools

#### File Operations

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `read_file` | Read file contents | file_ops |
| `write_file` | Write/create files | file_ops |
| `edit_file` | Edit existing files | file_ops |
| `grep` | Search for patterns | file_ops |
| `ls` | List directory contents | file_ops |

#### Search Tools

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `code_search` | Semantic code search | search |
| `semantic_code_search` | Semantic code search (alias) | search |
| `plan` | Plan files for analysis | search |

#### Web Tools

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `web_search` | Search the web | web |
| `web_fetch` | Fetch URL content | web |

#### Shell Tools

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `execute_bash` | Execute bash commands | shell |

#### LSP Tools

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `lsp` | LSP operations | lsp |
| `symbol` | Find symbols | lsp |
| `refs` | Find references | lsp |
| `rename` | Rename symbols | lsp |

#### Refactoring Tools

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `extract` | Extract functions | refactor |
| `rename` | Rename symbols | refactor |

#### Git Tools

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `git` | Unified git operations | git |

#### Docker Tools

| Tool Name | Description | Category |
|-----------|-------------|----------|
| `docker` | Docker operations | devops |

### Custom Tools

For custom tools not in the standard registry:

```yaml
tools:
  - custom_tool    # Must be registered in tool registry
```

Or define tool metadata:

```yaml
custom_config:
  custom_tools:
    - name: custom_tool
      description: "Custom tool description"
      category: custom
      cost_tier: LOW
```

## System Prompt

### Basic Prompt

```yaml
system_prompt: |
  You are an expert in X.

  Your capabilities:
  - Capability 1
  - Capability 2

  Guidelines:
  1. Guideline 1
  2. Guideline 2
```

### Multi-Line Prompt with Variables

```yaml
system_prompt: |
  You are {vertical_name}, an expert in {domain}.

  Your capabilities:
  {capabilities}

  Guidelines:
  {guidelines}
```

Variables will be substituted during generation:
- `{vertical_name}` - Vertical name
- `{description}` - Vertical description
- `{version}` - Vertical version

### Prompt Best Practices

1. **Be Specific**: Clearly define domain expertise
2. **List Capabilities**: Enumerate what the vertical can do
3. **Provide Guidelines**: Give actionable instructions
4. **Use Examples**: Include few-shot examples if helpful
5. **Keep Concise**: Avoid overly long prompts (target < 1000 tokens)

## Stages

### Stage Definition Schema

```yaml
stages:
  STAGE_NAME:
    name: STAGE_NAME           # Stage identifier (UPPER_CASE)
    description: string        # Human-readable description
    tools:                     # Tools available in this stage
      - tool_name_1
      - tool_name_2
    keywords:                  # Keywords that trigger this stage
      - keyword_1
      - keyword_2
    next_stages:               # Valid next stage names
      - NEXT_STAGE_1
      - NEXT_STAGE_2
```

### Standard 7-Stage Workflow

```yaml
stages:
  INITIAL:
    name: INITIAL
    description: "Understanding the request and gathering initial context"
    tools: [read_file, ls]
    keywords: [what, how, explain, help, show, describe]
    next_stages: [PLANNING, READING]

  PLANNING:
    name: PLANNING
    description: "Designing the approach and creating a strategy"
    tools: [read_file, grep]
    keywords: [plan, approach, strategy, design, outline]
    next_stages: [READING, EXECUTION]

  READING:
    name: READING
    description: "Gathering detailed information and context"
    tools: [read_file, grep, ls]
    keywords: [read, show, find, search, look, examine]
    next_stages: [ANALYSIS, EXECUTION]

  ANALYSIS:
    name: ANALYSIS
    description: "Analyzing information and identifying solutions"
    tools: [read_file, grep]
    keywords: [analyze, review, understand, why, how does]
    next_stages: [EXECUTION, PLANNING]

  EXECUTION:
    name: EXECUTION
    description: "Implementing the planned changes or actions"
    tools: [read_file, write_file, edit_file, grep]
    keywords: [change, modify, create, add, remove, implement]
    next_stages: [VERIFICATION, COMPLETION]

  VERIFICATION:
    name: VERIFICATION
    description: "Validating results and testing outcomes"
    tools: [read_file, grep]
    keywords: [test, verify, check, validate, confirm]
    next_stages: [COMPLETION, EXECUTION]

  COMPLETION:
    name: COMPLETION
    description: "Finalizing, documenting, and wrapping up"
    tools: []
    keywords: [done, finish, complete, summarize]
    next_stages: []
```

### Custom Workflow Example

```yaml
stages:
  DISCOVER:
    name: DISCOVER
    description: "Discovering relevant files"
    tools: [ls, code_search]
    keywords: [find, discover, locate]
    next_stages: [ANALYZE]

  ANALYZE:
    name: ANALYZE
    description: "Analyzing discovered files"
    tools: [read_file, grep]
    keywords: [analyze, understand, review]
    next_stages: [SOLVE]

  SOLVE:
    name: SOLVE
    description: "Solving the problem"
    tools: [read_file, write_file, edit_file]
    keywords: [solve, fix, implement]
    next_stages: [VERIFY]

  VERIFY:
    name: VERIFY
    description: "Verifying the solution"
    tools: [read_file, grep]
    keywords: [verify, test, check]
    next_stages: [DONE]

  DONE:
    name: DONE
    description: "Task complete"
    tools: []
    keywords: [done, complete, finished]
    next_stages: []
```

### Stage Design Guidelines

1. **Clear Names**: Use descriptive UPPER_CASE names
2. **Purposeful Tools**: Only include tools relevant to the stage
3. **Trigger Keywords**: Include words that indicate this stage
4. **Valid Transitions**: Define logical next stages
5. **Terminal Stage**: Ensure at least one stage leads to completion

## Extensions

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

```yaml
# complete_vertical_template.yaml
metadata:
  name: complete_vertical
  description: "Complete example vertical"
  version: "0.5.0"
  author: "Your Name"
  license: "Apache-2.0"
  category: development
  tags:
    - example
    - complete
  provider_hints:
    preferred_models:
      - claude-sonnet-4-5
    min_context: 128000
    supports_tools: true
  evaluation_criteria:
    - success_rate
    - quality_score

tools:
  - read_file
  - write_file
  - edit_file
  - grep
  - ls

system_prompt: |
  You are {vertical_name}, an expert assistant.

  Your capabilities:
  - Read and analyze files
  - Write and edit code
  - Search for information

  Guidelines:
  1. Be thorough in your analysis
  2. Provide clear explanations
  3. Verify your work

stages:
  INITIAL:
    name: INITIAL
    description: "Understanding the request"
    tools: [read_file, ls]
    keywords: [what, how, explain]
    next_stages: [EXECUTION]

  EXECUTION:
    name: EXECUTION
    description: "Performing the task"
    tools: [read_file, write_file, edit_file]
    keywords: [do, create, modify]
    next_stages: [COMPLETION]

  COMPLETION:
    name: COMPLETION
    description: "Task complete"
    tools: []
    keywords: [done, complete]
    next_stages: []

extensions:
  middleware:
    - name: logging_middleware
      class_name: LoggingMiddleware
      module: victor.complete_vertical.middleware
      enabled: true

  safety_patterns:
    - name: dangerous_delete
      pattern: "rm -rf .*"
      description: "Recursive delete"
      severity: critical
      category: commands

  prompt_hints:
    - task_type: create
      hint: "[CREATE] Plan before implementing..."
      tool_budget: 15
      priority_tools: [read_file, write_file]

workflows:
  - name: example_workflow
    file: workflows/example.yaml
    description: "Example workflow"
    enabled: true

teams:
  - name: example_team
    display_name: "Example Team"
    description: "Example team formation"
    formation: parallel
    communication_style: structured
    max_iterations: 3
    roles:
      - name: analyst
        display_name: "Analyst"
        description: "Analyzes requests"
        persona: "You are an analyst..."
        tool_categories: [analysis]
        capabilities: []

capabilities:
  - name: example_capability
    type: workflow
    description: "Example capability"
    enabled: true
    handler: victor.complete_vertical.capabilities:ExampleHandler
    config:
      setting: value

custom_config:
  example_setting: "value"
  grounding_rules: |
    Base responses on tool output.
  system_prompt_section: |
    Follow best practices.

file_templates: {}
```

## Migration from Legacy Templates

If you have templates from older versions:

1. **Update metadata structure**:
   ```yaml
   # Old
   name: my_vertical
   description: "Description"

   # New
   metadata:
     name: my_vertical
     description: "Description"
   ```

2. **Rename extensions**:
   ```yaml
   # Old
   middleware: []

   # New
   extensions:
     middleware: []
   ```

3. **Update stage structure**:
   ```yaml
   # Old
   stages:
     - name: INITIAL
       tools: []

   # New
   stages:
     INITIAL:
       name: INITIAL
       tools: []
   ```

See [Migration Tool Guide](migration_tool_guide.md) for automated migration.

## Best Practices Summary

1. **Use descriptive names** for templates, stages, and roles
2. **Provide clear descriptions** for all components
3. **Start with base template** and extend as needed
4. **Validate templates** before generation
5. **Use template inheritance** to avoid duplication
6. **Document custom configurations** with comments
7. **Keep prompts concise** and focused
8. **Define clear stage transitions**
9. **Specify appropriate tool budgets**
10. **Add safety patterns** for dangerous operations

## Troubleshooting

### Common Errors

#### Missing Required Field

```bash
❌ Validation error: Missing required field: metadata.version
```

**Solution**: Add all required metadata fields.

#### Invalid Protocol

```bash
❌ Validation error: Invalid protocol: NotARealProtocol
```

**Solution**: Use valid protocol names from reference.

#### Workflow File Not Found

```bash
❌ Validation error: Workflow file not found: workflows/missing.yaml
```

**Solution**: Ensure workflow files exist relative to template location.

#### Circular Inheritance

```bash
❌ Validation error: Circular template inheritance detected
```

**Solution**: Check `extends` field for circular references.

### Getting Help

1. Check example templates in `victor/config/templates/`
2. Review this reference documentation
3. Validate templates with `--validate` flag
4. Check generated code for issues
5. Consult Victor community resources
