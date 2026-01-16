# Team Templates

Comprehensive library of pre-configured multi-agent team templates for Victor.

## Overview

Team templates provide pre-configured, production-ready multi-agent teams for common use cases across all verticals. Templates encapsulate best practices for team composition, formation patterns, and resource allocation.

**Key Features:**

- 20+ production-ready templates
- 5 team formations (Sequential, Parallel, Pipeline, Hierarchical, Consensus)
- 4 verticals covered (Coding, Research, DataAnalysis, DevOps, General)
- Quick complexity levels (Quick, Standard, Complex)
- Validated configurations
- Easy customization

## Template Library

### Coding Templates

| Template | Formation | Complexity | Description |
|----------|-----------|------------|-------------|
| `code_review_parallel` | Parallel | Standard | Parallel code review with security, performance, quality, and documentation specialists |
| `bug_hunt_hierarchical` | Hierarchical | Complex | Manager-coordinated bug investigation and resolution |
| `refactoring_pipeline` | Pipeline | Complex | Stage-wise refactoring with analysis, planning, implementation, and verification |
| `testing_parallel` | Parallel | Standard | Parallel test generation (unit, integration, E2E, property-based) |
| `architecture_review_hierarchical` | Hierarchical | Complex | Architecture review with scalability, security, and maintainability specialists |
| `migration_pipeline` | Pipeline | Complex | Structured migration with assessment, planning, execution, and verification |
| `documentation_pipeline` | Pipeline | Standard | Professional documentation creation workflow |
| `performance_optimization_sequential` | Sequential | Complex | Systematic performance analysis and optimization |

### Research Templates

| Template | Formation | Complexity | Description |
|----------|-----------|------------|-------------|
| `literature_review_pipeline` | Pipeline | Standard | Systematic literature review with search, analysis, and synthesis |
| `fact_check_consensus` | Consensus | Standard | Consensus-based fact verification and checking |
| `competitive_analysis_parallel` | Parallel | Standard | Competitive analysis with product, market, technical, and business analysts |
| `synthesis_parallel` | Parallel | Standard | Research synthesis from multiple sources and perspectives |

### Data Analysis Templates

| Template | Formation | Complexity | Description |
|----------|-----------|------------|-------------|
| `statistical_analysis_parallel` | Parallel | Complex | Comprehensive statistical analysis with descriptive, inferential, visualization, and domain experts |
| `data_cleaning_parallel` | Parallel | Standard | Parallel data cleaning for missing values, outliers, formats, and duplicates |

### DevOps Templates

| Template | Formation | Complexity | Description |
|----------|-----------|------------|-------------|
| `deployment_pipeline` | Pipeline | Complex | Structured deployment with risk assessment, planning, execution, and verification |

### General Templates

| Template | Formation | Complexity | Description |
|----------|-----------|------------|-------------|
| `quick_task_sequential` | Sequential | Quick | Fast sequential execution for simple tasks |
| `problem_solving_hierarchical` | Hierarchical | Complex | Manager-coordinated complex problem solving |
| `decision_making_consensus` | Consensus | Complex | Consensus-based decision making with diverse perspectives |
| `brainstorming_parallel` | Parallel | Standard | Parallel ideation and creative brainstorming |
| `security_audit_parallel` | Parallel | Complex | Comprehensive security audit from multiple perspectives |

## Using Templates

### Python API

```python
from victor.workflows.team_templates import get_template

# Get a template
template = get_template("code_review_parallel")

# Create team config
team_config = template.to_team_config(
    goal="Review PR #123",
    context={"pr_number": 123}
)

# Create workflow node
team_node = template.to_team_node(
    node_id="review_team",
    goal="Review authentication changes",
    output_key="review_results"
)
```

### Template Manager CLI

```bash
# List all templates
python -m scripts.teams.template_manager list

# List coding templates
python -m scripts.teams.template_manager list --vertical coding

# Show template details
python -m scripts.teams.template_manager show code_review_parallel

# Search templates
python -m scripts.teams.template_manager search "code review"

# Validate template file
python -m scripts.teams.template_manager validate my_template.yaml

# Create new template (interactive wizard)
python -m scripts.teams.template_manager create --output my_template.yaml

# Apply template
python -m scripts.teams.template_manager apply code_review_parallel \
  --goal "Review PR #123"
```

## Template Structure

Each template includes:

- **Metadata**: Name, version, author, vertical
- **Description**: What the template does and when to use it
- **Formation**: Team organization pattern
- **Members**: Agent specifications with roles, goals, expertise
- **Configuration**: Resource budgets and constraints
- **Examples**: Usage examples with typical inputs

### Member Specification

Each team member includes:

```yaml
- id: "security_reviewer"
  role: "researcher"
  name: "Security Reviewer"
  goal: "Review code for security vulnerabilities..."
  backstory: "Security specialist with 12 years..."
  expertise: ["security", "vulnerabilities", "owasp"]
  personality: "Thorough and risk-averse"
  tool_budget: 30
  allowed_tools: ["read", "grep", "code_search"]
  can_delegate: false
  memory: true
  cache: true
```

## Team Formations

### Sequential
Members execute one after another, with output chaining. Best for:
- Quick tasks
- Simple workflows
- Linear processes

### Parallel
Members execute simultaneously, working independently. Best for:
- Comprehensive analysis
- Time-sensitive tasks
- Independent work streams

### Pipeline
Stages build on each other with explicit handoffs. Best for:
- Multi-phase processes
- Verification-heavy workflows
- Rollback requirements

### Hierarchical
Manager coordinates and delegates to specialists. Best for:
- Complex problems
- Multi-faceted investigation
- Strategic decisions

### Consensus
Members work toward agreement through discussion. Best for:
- Important decisions
- Fact verification
- Quality-critical tasks

## Creating Custom Templates

### Option 1: YAML File

Create a YAML file:

```yaml
name: my_custom_template
display_name: "My Custom Team"
description: "Does X, Y, Z"
vertical: "coding"
formation: "parallel"
complexity: "standard"
max_iterations: 75
total_tool_budget: 120
timeout_seconds: 900

members:
  - id: "specialist_1"
    role: "researcher"
    name: "Specialist One"
    goal: "Analyze..."
    backstory: "Expert with 10 years..."
    expertise: ["domain1", "domain2"]
    tool_budget: 30
    memory: true
    cache: true
```

Then validate and use:

```bash
python -m scripts.teams.template_manager validate my_template.yaml
```

### Option 2: Python API

```python
from victor.workflows.team_templates import (
    TeamTemplate,
    TeamMemberSpec,
    register_template
)

# Create member specs
members = [
    TeamMemberSpec(
        id="analyst",
        role="researcher",
        name="Analyst",
        goal="Analyze data...",
        tool_budget=30,
    )
]

# Create template
template = TeamTemplate(
    name="my_template",
    display_name="My Template",
    description="Does X, Y, Z",
    formation="parallel",
    members=members,
    vertical="general",
    complexity="standard",
)

# Register
register_template(template)
```

### Option 3: Interactive Wizard

```bash
python -m scripts.teams.template_manager create --output my_template.yaml
```

## Best Practices

### Template Design

1. **Clear Purpose**: Define specific use cases in description and use_cases
2. **Appropriate Formation**: Choose formation based on task characteristics
3. **Realistic Budgets**: Set tool budgets based on task complexity
4. **Rich Personas**: Provide backstory, expertise, and personality for each member
5. **Validation**: Always validate templates before use

### Team Composition

1. **Optimal Size**: 2-4 members for most tasks
2. **Complementary Skills**: Members should have different expertise
3. **Clear Roles**: Each member should have distinct responsibilities
4. **Appropriate Delegation**: Only managers in hierarchical formations should delegate

### Resource Allocation

1. **Total Budget**: 100-150 for complex tasks, 30-80 for simple tasks
2. **Per-Member**: 20-40 tool calls per member typically sufficient
3. **Timeout**: 5-10 minutes for simple tasks, 15-30 for complex
4. **Iterations**: 50-100 for most tasks, 25 for quick tasks

## Template Catalog

See the [Template Catalog](TEMPLATE_CATALOG.md) for detailed information about each template including:
- Complete descriptions
- Use case scenarios
- Member details
- Configuration options
- Examples and best practices

## Migration Guide

### From Manual Team Creation

**Before:**
```python
from victor.teams import TeamMember, TeamConfig, TeamFormation

members = [
    TeamMember(
        id="reviewer",
        role=SubAgentRole.RESEARCHER,
        name="Reviewer",
        goal="Review code...",
        tool_budget=30,
    )
]

config = TeamConfig(
    name="Review Team",
    goal="Review PR #123",
    members=members,
    formation=TeamFormation.PARALLEL,
)
```

**After (with templates):**
```python
from victor.workflows.team_templates import get_template

template = get_template("code_review_parallel")
config = template.to_team_config(goal="Review PR #123")
```

## Advanced Usage

### Template Inheritance

Create specialized templates from base templates:

```python
base = get_template("code_review_parallel")

# Customize for security-focused review
security_review = TeamTemplate(
    name="security_review_parallel",
    display_name="Security-Focused Code Review",
    description=base.description,
    formation=base.formation,
    members=[m for m in base.members if "security" in m.id],
    vertical=base.vertical,
    complexity=base.complexity,
)
```

### Dynamic Template Selection

```python
from victor.workflows.team_templates import get_registry

registry = get_registry()

# Suggest template based on task
template = registry.suggest_template(
    task_description="I need to review a pull request for security issues",
    vertical="coding",
    complexity="standard",
)
```

### Template Validation

```python
from victor.workflows.team_templates import get_registry

registry = get_registry()
errors = registry.validate_template(template)

if errors:
    for error in errors:
        print(f"Error: {error}")
```

## Troubleshooting

### Template Not Found

```bash
# List available templates
python -m scripts.teams.template_manager list

# Search for similar templates
python -m scripts.teams.template_manager search "review"
```

### Validation Errors

Common issues:
- **Invalid formation**: Must be one of: sequential, parallel, hierarchical, pipeline, consensus
- **No members**: Template must have at least one member
- **Duplicate member IDs**: Each member must have unique ID
- **Invalid role**: Must be valid SubAgentRole value

### Performance Issues

- **Too many iterations**: Reduce max_iterations for faster execution
- **High tool budget**: Reduce total_tool_budget to limit tool calls
- **Long timeout**: Reduce timeout_seconds for faster failure detection

## Contributing

To contribute new templates:

1. Create YAML file in `victor/workflows/templates/{vertical}/`
2. Follow template structure and best practices
3. Validate with template manager
4. Add tests
5. Update documentation

See [Contributing Guide](../CONTRIBUTING.md) for details.
