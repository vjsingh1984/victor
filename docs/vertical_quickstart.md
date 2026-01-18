# Vertical Template Quick Start Guide

This quick start guide will get you up and running with Victor's vertical template system in minutes.

## What is the Vertical Template System?

The vertical template system allows you to create new verticals by defining them in YAML files instead of writing hundreds of lines of Python code. This reduces code duplication by 65-70% and allows you to create new verticals in hours instead of days.

## Prerequisites

- Victor installed (`pip install -e .`)
- Basic understanding of YAML
- Familiarity with vertical concepts (tools, stages, prompts)

## Quick Start (5 Minutes)

### Step 1: Use the Base Template

Copy the base template as your starting point:

```bash
cp victor/config/templates/base_vertical_template.yaml my_vertical.yaml
```

### Step 2: Edit Basic Metadata

Open `my_vertical.yaml` and edit the metadata section:

```yaml
metadata:
  name: my_vertical              # Change to your vertical name
  description: "My vertical"     # Add description
  version: "1.0.0"
  category: general              # Change if needed
```

### Step 3: Define Tools

List the tools your vertical needs:

```yaml
tools:
  # File operations (always include these)
  - read_file
  - write_file
  - edit_file
  - grep

  # Add your vertical-specific tools
  - your_tool_name
```

**Tip**: Use canonical tool names from `victor.tools.tool_names`.

### Step 4: Write System Prompt

Describe your vertical's purpose and behavior:

```yaml
system_prompt: |
  You are Victor, an expert in your domain.

  Your capabilities:
  - Capability 1
  - Capability 2

  Guidelines:
  1. Be specific
  2. Explain reasoning
```

### Step 5: Define Stages

Define the workflow stages:

```yaml
stages:
  INITIAL:
    name: INITIAL
    description: "Understanding the request"
    tools: [read_file, grep]
    keywords: [what, how, explain]
    next_stages: [EXECUTION]

  EXECUTION:
    name: EXECUTION
    description: "Performing the task"
    tools: [read_file, write_file, your_tool]
    keywords: [do, perform, create]
    next_stages: [COMPLETION]

  COMPLETION:
    name: COMPLETION
    description: "Task complete"
    tools: []
    next_stages: []
```

### Step 6: Validate Template

Check your template for errors:

```bash
python scripts/generate_vertical.py \
  --validate my_vertical.yaml \
  --output /tmp/test
```

### Step 7: Generate Vertical

Create the vertical from your template:

```bash
python scripts/generate_vertical.py \
  --template my_vertical.yaml \
  --output victor/my_vertical
```

This generates all necessary files:
- `assistant.py` - Main vertical class
- `prompts.py` - Prompt contributions
- `safety.py` - Safety patterns
- `escape_hatches.py` - Workflow escape hatches
- `handlers.py` - Workflow handlers
- `teams.py` - Team formations
- `capabilities.py` - Capability configs
- `__init__.py` - Package init

### Step 8: Use Your Vertical

```python
from victor.my_vertical import MyVerticalAssistant

# Get configuration
config = MyVerticalAssistant.get_config()

# Create agent
agent = await Agent.create(
    tools=config.tools,
    vertical=MyVerticalAssistant,
)
```

## Example: Creating a Testing Vertical

Let's create a simple testing vertical in 5 minutes:

### 1. Create Template (`testing.yaml`):

```yaml
metadata:
  name: testing
  description: "Automated testing and quality assurance"
  version: "1.0.0"
  category: testing

tools:
  - read_file
  - write_file
  - edit_file
  - grep
  - execute_bash
  - run_tests

system_prompt: |
  You are Victor, a testing automation expert.

  Your capabilities:
  - Test discovery and generation
  - Test execution and reporting
  - Code coverage analysis
  - Quality assurance

  Guidelines:
  1. Write comprehensive tests
  2. Ensure good coverage
  3. Report failures clearly
  4. Suggest improvements

stages:
  INITIAL:
    name: INITIAL
    description: "Understanding testing requirements"
    tools: [read_file, grep]
    keywords: [what, test, verify]
    next_stages: [TESTING]

  TESTING:
    name: TESTING
    description: "Running and analyzing tests"
    tools: [execute_bash, run_tests, grep]
    keywords: [test, run, check]
    next_stages: [REPORTING]

  REPORTING:
    name: REPORTING
    description: "Generating test reports"
    tools: [write_file]
    keywords: [report, summary]
    next_stages: [COMPLETION]

  COMPLETION:
    name: COMPLETION
    description: "Testing complete"
    tools: []
    next_stages: []
```

### 2. Generate:

```bash
python scripts/generate_vertical.py \
  --template testing.yaml \
  --output victor/testing
```

### 3. Use:

```python
from victor.testing import TestingAssistant

config = TestingAssistant.get_config()
agent = await Agent.create(
    tools=config.tools,
    vertical=TestingAssistant,
)

# Use the agent
await agent.chat("Write tests for auth.py")
```

## Common Tasks

### Adding Safety Patterns

Protect against dangerous operations:

```yaml
extensions:
  safety_patterns:
    - name: destructive_test
      pattern: "rm.*-rf"
      description: "Destructive operations"
      severity: "high"
      category: "files"
```

### Adding Task Hints

Guide the model for specific tasks:

```yaml
extensions:
  prompt_hints:
    - task_type: test_generation
      hint: "[TESTS] Write comprehensive tests with good coverage"
      tool_budget: 15
      priority_tools:
        - read_file
        - write_file
        - run_tests
```

### Adding Middleware

Add pre/post-processing:

```yaml
extensions:
  middleware:
    - name: test_validation
      class_name: TestValidationMiddleware
      module: victor.testing.middleware
      enabled: true
      config:
        min_coverage: 80
```

## Best Practices

### 1. Start Simple

Begin with minimal tools and stages, then add complexity gradually.

### 2. Use Canonical Names

Always use canonical tool names:
```yaml
tools:
  - read_file    # Correct
  - read         # Wrong
```

### 3. Define Clear Stages

Stages should represent workflow phases:
- INITIAL → Understanding
- EXECUTION → Doing the work
- VERIFICATION → Checking results
- COMPLETION → Done

### 4. Add Safety Patterns

Define patterns for dangerous operations specific to your vertical.

### 5. Validate Frequently

Check your template often:
```bash
python scripts/generate_vertical.py --validate template.yaml --output /tmp/test
```

## Examples Directory

See `victor/config/templates/` for examples:
- `base_vertical_template.yaml` - Minimal starting point
- `example_security_vertical.yaml` - Complete security vertical

## Next Steps

1. **Read the Full Guide**: See `docs/vertical_template_guide.md`
2. **Study Examples**: Check existing verticals in `victor/*/`
3. **Customize Generated Code**: Edit generated files as needed
4. **Add Workflows**: Define YAML workflows for complex tasks
5. **Define Teams**: Create multi-agent teams for collaboration

## Troubleshooting

### Template Won't Validate

Check:
- Required fields present (name, description, version)
- System prompt not empty
- Stages include INITIAL and COMPLETION
- Tool names are canonical

### Generated Code Won't Import

Check:
- Output directory exists
- `__init__.py` was generated
- Class name is valid
- No syntax errors in template

### Tools Not Available

Check:
- Tool names are canonical
- Tools are registered
- Dependencies installed

## Getting Help

1. Check template validation output
2. Review example templates
3. Consult full guide: `docs/vertical_template_guide.md`
4. Check existing vertical implementations

## Summary

In just 5 minutes, you can:
1. Copy the base template
2. Edit metadata, tools, prompt, stages
3. Validate the template
4. Generate a complete vertical
5. Use it in your agent

The template system dramatically reduces the time and effort needed to create new verticals while maintaining consistency and best practices across your codebase.
