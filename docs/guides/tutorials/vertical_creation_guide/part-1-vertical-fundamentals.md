# Vertical Creation Guide - Part 1

**Part 1 of 2:** Overview, Quick Start, Template Structure, Creating a Vertical from Scratch, and Customizing Generated Verticals

---

## Navigation

- **[Part 1: Vertical Fundamentals](#)** (Current)
- [Part 2: Advanced & Examples](part-2-advanced-examples.md)
- [**Complete Guide](../vertical_creation_guide.md)**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Template Structure](#template-structure)
4. [Creating a Vertical from Scratch](#creating-a-vertical-from-scratch)
5. [Extracting Templates from Existing Verticals](#extracting-templates-from-existing-verticals) *(in Part 2)*
6. [Template Validation](#template-validation) *(in Part 2)*
7. [Customizing Generated Verticals](#customizing-generated-verticals)
8. [Best Practices](#best-practices) *(in Part 2)*
9. [Examples](#examples) *(in Part 2)*
10. [Advanced Topics](#advanced-topics) *(in Part 2)*

---

# Vertical Creation Guide

This guide explains how to create new verticals in Victor using the template-based scaffolding system, which reduces code duplication by 65-70%.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Template Structure](#template-structure)
4. [Creating a Vertical from Scratch](#creating-a-vertical-from-scratch)
5. [Extracting Templates from Existing Verticals](#extracting-templates-from-existing-verticals)
6. [Template Validation](#template-validation)
7. [Customizing Generated Verticals](#customizing-generated-verticals)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

## Overview

Victor's vertical template system provides a declarative way to define and generate vertical implementations. Instead of writing 500+ lines of boilerplate code, you can:

1. Define your vertical in a YAML template
2. Generate all necessary files automatically
3. Customize only the parts that need special logic

This approach:
- **Reduces duplication**: 65-70% less code
- **Ensures consistency**: All verticals follow the same structure
- **Speeds up development**: Create a new vertical in minutes
- **Maintains best practices**: Generated code follows Victor's patterns

## Quick Start

### Create a New Vertical from Template

```bash
# 1. Create a template YAML (or use an existing one)
cat > my_vertical.yaml << 'EOF'
metadata:
  name: security
  description: Security analysis and vulnerability detection
  version: "0.5.0"
  category: security
  tags: [security, vulnerability, scanning]

tools:
  - read
  - grep
  - security_scan
  - web_search

system_prompt: |
  You are a security expert specialized in finding
  vulnerabilities and security issues...

stages:
  INITIAL:
    name: INITIAL
    description: Understanding the security request
    tools: [read, grep]
    keywords: [what, how, explain]
    next_stages: [SCANNING]

  SCANNING:
    name: SCANNING
    description: Scanning for vulnerabilities
    tools: [security_scan, grep]
    keywords: [scan, find, search]
    next_stages: [ANALYSIS]
EOF

# 2. Generate the vertical
python scripts/generate_vertical.py \
  --template my_vertical.yaml \
  --output victor/security \
  --overwrite

# 3. Use your new vertical
python -c "
from victor.security import SecurityAssistant
config = SecurityAssistant.get_config()
print(f'Vertical: {config.metadata[\"vertical_name\"]}')
print(f'Tools: {len(config.tools.tools)}')
"
```

### Extract Template from Existing Vertical

```bash
# Extract template from existing vertical
python -m victor.framework.vertical_extractor \
  --directory victor/coding \
  --output templates/coding.yaml \
  --validate

# Use the extracted template to create a similar vertical
python scripts/generate_vertical.py \
  --template templates/coding.yaml \
  --output victor/my_vertical \
  --overwrite
```

## Template Structure

A vertical template is a YAML file that defines:

### 1. Metadata

```yaml
metadata:
  name: security              # Vertical identifier (required)
  description: Security...    # Human-readable description (required)
  version: "0.5.0"            # Semantic version (default: "0.5.0")
  category: security           # Category for organization
  tags:                       # Tags for discovery
    - security
    - vulnerability
```

### 2. Core Configuration

```yaml
tools:
  - read                      # File operations
  - grep                      # Search
  - security_scan             # Vertical-specific tools

system_prompt: |
  You are an expert in security analysis...
  # Multi-line prompt supported

stages:
  INITIAL:
    name: INITIAL
    description: Understanding the request
    tools: [read, grep]
    keywords: [what, how, explain]
    next_stages: [SCANNING, READING]

  SCANNING:
    name: SCANNING
    description: Scanning for issues
    tools: [security_scan, grep]
    keywords: [scan, find, search]
    next_stages: [ANALYSIS]
```

### 3. Extensions

```yaml
extensions:
  middleware:
    - name: security_middleware
      class_name: SecurityMiddleware
      module: victor.security.middleware
      enabled: true
      config:
        block_dangerous: true

  prompt_hints:
    - task_type: vulnerability_scan
      hint: "[SCAN] Use security_scan tool..."
      tool_budget: 15
      priority_tools: [security_scan, grep]

  safety_patterns:
    - name: dangerous_command
      pattern: "rm -rf"
      description: "Dangerous file deletion"
      severity: critical
      category: commands
```

### 4. Workflows and Teams

```yaml
workflows:
  - name: security_audit
    description: Comprehensive security audit workflow
    yaml_path: victor/security/workflows/security_audit.yaml
    handler_module: victor.security.handlers

teams:
  - name: security_review_team
    display_name: "Security Review Team"
    description: "Multi-agent security review"
    formation: parallel
    communication_style: structured
    max_iterations: 5
    roles:
      - name: vulnerability_scanner
        display_name: "Vulnerability Scanner"
        description: "Scans for known vulnerabilities"
        persona: "You are a vulnerability scanning expert..."
        tool_categories: [security, analysis]
        capabilities: [cvss_scoring, cve_lookup]
```

### 5. Capabilities

```yaml
capabilities:
  - name: cvss_scoring
    type: tool
    description: "CVSS score calculation"
    enabled: true
    handler: victor.security.tools:CVSSCalculator
    config:
      version: "3.1"
```

## Creating a Vertical from Scratch

### Step 1: Define Your Template

Create a YAML file with your vertical definition:

```yaml
# templates/my_vertical.yaml
metadata:
  name: my_vertical
  description: "My custom vertical for..."
  version: "0.5.0"

tools:
  - read
  - write
  - grep

system_prompt: |
  You are an expert in...

stages:
  INITIAL:
    name: INITIAL
    description: Understanding the request
    tools: [read]
    keywords: [what, how]
    next_stages: [EXECUTION]

  EXECUTION:
    name: EXECUTION
    description: Performing the task
    tools: [read, write, grep]
    keywords: [do, perform, execute]
    next_stages: [COMPLETION]

  COMPLETION:
    name: COMPLETION
    description: Task complete
    tools: []
    keywords: [done, complete]
    next_stages: []
```

### Step 2: Generate the Vertical

```bash
python scripts/generate_vertical.py \
  --template templates/my_vertical.yaml \
  --output victor/my_vertical \
  --overwrite
```

This generates:
```
victor/my_vertical/
├── __init__.py              # Package initialization
├── assistant.py             # Main vertical class
├── prompts.py               # Task type hints
├── safety.py                # Safety patterns
├── escape_hatches.py        # Workflow escape hatches
├── handlers.py              # Workflow handlers
├── teams.py                 # Team formations
├── capabilities.py          # Capability configs
└── config/
    └── vertical.yaml        # YAML configuration
```

### Step 3: Customize (Optional)

Edit the generated files to add custom logic:

**victor/my_vertical/assistant.py**
```python
class MyVerticalAssistant(VerticalBase):
    # ... generated code ...

    @classmethod
    def get_middleware(cls):
        """Add custom middleware."""
        def _create_middleware():
            return [
                MyCustomMiddleware(enabled=True),
                # ... other middleware ...
            ]
        return cls._get_cached_extension("middleware", _create_middleware)
```

**victor/my_vertical/prompts.py**
```python
MY_VERTICAL_TASK_TYPE_HINTS: Dict[str, TaskTypeHint] = {
    "my_task": TaskTypeHint(
        task_type="my_task",
        hint="[MY_TASK] Special handling for...",
        tool_budget=20,
        priority_tools=["my_custom_tool"],
    ),
    # ... other hints ...
}
```

### Step 4: Register Your Vertical

**victor/core/verticals/__init__.py**
```python
from victor.my_vertical.assistant import MyVerticalAssistant

def _register_builtin_verticals():
    VerticalRegistry.register(MyVerticalAssistant)
    # ... other verticals ...
```

### Step 5: Use Your Vertical

```python
from victor.my_vertical import MyVerticalAssistant

# Get configuration
config = MyVerticalAssistant.get_config()

# Create agent
agent = await Agent.create(
    vertical=MyVerticalAssistant,
    provider="anthropic",
)

# Use the agent
result = await agent.run("Help me with...")
```

## Extracting Templates from Existing Verticals

You can extract templates from existing verticals to:

1. Document the vertical structure
2. Create similar verticals
3. Migrate to template-based system

### Extract from Class

```python
from victor.framework import VerticalExtractor

extractor = VerticalExtractor()
template = extractor.extract_from_class(CodingAssistant)

# Save template
from victor.framework import VerticalTemplateRegistry

registry = VerticalTemplateRegistry()
registry.save_to_yaml(template, "templates/coding.yaml")
```

### Extract from Module

```bash
python -m victor.framework.vertical_extractor \
  --module victor.coding \
  --output templates/coding.yaml
```

### Extract from Directory

```bash
python -m victor.framework.vertical_extractor \
  --directory victor/coding \
  --output templates/coding.yaml \
  --validate
```

## Template Validation

Templates are validated before generation to ensure completeness:

```bash
# Validate a template
python scripts/generate_vertical.py \
  --validate templates/my_vertical.yaml
```

Validation checks:
- Required fields are present (metadata.name, metadata.description, tools, system_prompt)
- Required stages are defined (INITIAL, COMPLETION)
- Stage structure is correct (name, description, tools, keywords, next_stages)
- Middleware has required fields (name, class_name, module)
- Workflow has required fields (name, description)
- Team has valid formation type (pipeline, parallel, sequential, hierarchical, consensus)
- Capability has valid type (tool, workflow, middleware, validator, observer)

## Customizing Generated Verticals

### Add Custom Tools

**victor/my_vertical/tools.py**
```python
from victor.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name = "my_custom_tool"
    description = "Does something special"

    async def execute(self, **kwargs):
        # Implementation
        return result
```

**victor/my_vertical/assistant.py**
```python
@classmethod
def get_tools(cls):
    tools = super().get_tools()
    tools.extend([
        "my_custom_tool",  # Add your custom tool
    ])
    return tools
```

### Add Custom Middleware

**victor/my_vertical/middleware.py**
```python
from victor.core.verticals.protocols import MiddlewareProtocol

class MyCustomMiddleware(MiddlewareProtocol):
    async def before_tool_execution(self, context):
        # Pre-processing logic
        pass

    async def after_tool_execution(self, context, result):
        # Post-processing logic
        pass
```

### Add Custom Workflow Handlers

**victor/my_vertical/handlers.py**
```python
async def my_custom_handler(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Custom workflow handler."""
    # Implementation
    return {"status": "success", "data": ...}

HANDLERS = {
    "my_custom_handler": my_custom_handler,
}
```

### Add Custom Escape Hatches

**victor/my_vertical/escape_hatches.py**
```python
def my_custom_condition(ctx: Dict[str, Any]) -> str:
    """Complex condition for workflow branching."""
    if ctx.get("quality_score", 0) > 0.9:
        return "approve"
    elif ctx.get("quality_score", 0) > 0.7:
        return "review"
    else:
        return "reject"

CONDITIONS = {
    "my_custom_condition": my_custom_condition,
}
```

