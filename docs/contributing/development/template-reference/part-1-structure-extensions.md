# Template Reference - Part 1

**Part 1 of 4:** Template Structure through Extensions

---

## Navigation

- **[Part 1: Structure & Extensions](#)** (Current)
- [Part 2: Workflows & Config](part-2-workflows-config.md)
- [Part 3: Inheritance & Validation](part-3-inheritance-validation.md)
- [Part 4: Examples & Migration](part-4-examples-migration.md)
- [**Complete Reference**](../template_reference.md)

---
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

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 6 min
**Last Updated:** February 08, 2026**
