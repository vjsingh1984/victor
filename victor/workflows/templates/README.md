# Team Templates Library

Comprehensive library of **20 production-ready multi-agent team templates** for Victor AI Coding Agent.

## Quick Start

```python
from victor.workflows.team_templates import get_template

# Get a template
template = get_template("code_review_parallel")

# Apply to workflow
team_config = template.to_team_config(goal="Review PR #123")
team_node = template.to_team_node(node_id="review", goal="Review changes")
```

## Template Catalog

### By Formation

#### Parallel Templates (6)
Simultaneous execution by independent specialists.

1. **code_review_parallel** - Parallel code review (security, performance, quality, docs)
2. **testing_parallel** - Parallel test generation (unit, integration, E2E, property)
3. **statistical_analysis_parallel** - Statistical analysis (descriptive, inferential, viz, domain)
4. **data_cleaning_parallel** - Data cleaning (missing values, outliers, formats, duplicates)
5. **competitive_analysis_parallel** - Competitive analysis (product, market, technical, business)
6. **brainstorming_parallel** - Creative ideation (user, technical, business, creative)
7. **synthesis_parallel** - Research synthesis (sources, patterns, critical, context)
8. **security_audit_parallel** - Security audit (application, infrastructure, compliance, data)

#### Pipeline Templates (6)
Stage-wise processing with output chaining.

1. **refactoring_pipeline** - Refactoring (analysis → planning → implementation → verification)
2. **migration_pipeline** - Code migration (assessment → planning → migration → verification)
3. **documentation_pipeline** - Documentation (research → draft → review → publish)
4. **literature_review_pipeline** - Literature review (search → analyze → synthesize)
5. **deployment_pipeline** - Deployment (risk → plan → deploy → verify)
6. **performance_optimization_sequential** - Performance (profile → identify → optimize → verify)

#### Hierarchical Templates (3)
Manager-coordinated with specialist delegation.

1. **bug_hunt_hierarchical** - Bug investigation (manager + reproduction + logs + trace)
2. **architecture_review_hierarchical** - Architecture review (lead + scalability + security + maintainability)
3. **problem_solving_hierarchical** - Problem solving (manager + analytical + creative + pragmatic)

#### Consensus Templates (2)
Agreement-based through discussion and deliberation.

1. **fact_check_consensus** - Fact verification (3 verifiers reach consensus)
2. **decision_making_consensus** - Decision making (technical + business + UX experts)

#### Sequential Templates (2)
Ordered execution with handoffs.

1. **quick_task_sequential** - Quick tasks (analyzer → implementer)
2. **performance_optimization_sequential** - Performance optimization (4-stage pipeline)

### By Vertical

#### Coding Templates (8)
- code_review_parallel
- bug_hunt_hierarchical
- refactoring_pipeline
- testing_parallel
- architecture_review_hierarchical
- migration_pipeline
- documentation_pipeline
- performance_optimization_sequential

#### Research Templates (4)
- literature_review_pipeline
- fact_check_consensus
- competitive_analysis_parallel
- synthesis_parallel

#### Data Analysis Templates (2)
- statistical_analysis_parallel
- data_cleaning_parallel

#### DevOps Templates (1)
- deployment_pipeline

#### General Templates (5)
- quick_task_sequential
- problem_solving_hierarchical
- decision_making_consensus
- brainstorming_parallel
- security_audit_parallel

### By Complexity

#### Quick (2)
- quick_task_sequential

#### Standard (12)
- code_review_parallel
- testing_parallel
- statistical_analysis_parallel
- data_cleaning_parallel
- literature_review_pipeline
- fact_check_consensus
- competitive_analysis_parallel
- synthesis_parallel
- brainstorming_parallel
- documentation_pipeline
- deployment_pipeline
- security_audit_parallel

#### Complex (6)
- bug_hunt_hierarchical
- refactoring_pipeline
- architecture_review_hierarchical
- migration_pipeline
- problem_solving_hierarchical
- decision_making_consensus
- performance_optimization_sequential

## Usage Examples

### Python API

```python
from victor.workflows.team_templates import get_template, list_templates

# List available templates
templates = list_templates(vertical="coding", formation="parallel")

# Get specific template
template = get_template("code_review_parallel")

# Create team configuration
team_config = template.to_team_config(
    goal="Review PR #123",
    context={"pr_number": 123, "files_changed": 15}
)

# Create workflow node
from victor.workflows.team_templates import TeamTemplateRegistry

registry = TeamTemplateRegistry.get_instance()
template = registry.get_template("code_review_parallel")

team_node = template.to_team_node(
    node_id="code_review",
    goal="Review authentication changes",
    output_key="review_results"
)
```

### Workflow Integration

```yaml
# In your workflow YAML
nodes:
  - id: review_pr
    type: team
    name: "Code Review"
    goal: "Review PR {{pr_number}}"
    team_formation: parallel
    members:
      # Use template members
      $ref: "../templates/coding/code_review_parallel.yaml#members"
    timeout_seconds: 600
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
python -m scripts.teams.template_manager search "security audit"

# Validate template
python -m scripts.teams.template_manager validate my_template.yaml

# Create new template (interactive wizard)
python -m scripts.teams.template_manager create --output my_template.yaml

# Apply template
python -m scripts.teams.template_manager apply code_review_parallel \
  --goal "Review PR #123" \
  --workflow-node \
  --node-id review_team
```

## Template Structure

Each template includes:

### Required Fields
- **name**: Unique identifier (snake_case)
- **display_name**: Human-readable name
- **description**: Short description
- **formation**: Team organization pattern
- **members**: List of member specifications

### Optional Fields
- **long_description**: Detailed description
- **version**: Template version
- **author**: Template author
- **vertical**: Primary vertical
- **use_cases**: Applicable use cases
- **tags**: Discoverability tags
- **complexity**: Task complexity (quick/standard/complex)
- **max_iterations**: Iteration limit
- **total_tool_budget**: Total tool call budget
- **timeout_seconds**: Execution timeout
- **config**: Additional configuration
- **metadata**: Template metadata
- **examples**: Usage examples

### Member Specification

Each member includes:

```yaml
- id: "unique_member_id"
  role: "researcher"  # SubAgentRole
  name: "Human-Readable Name"
  goal: "Member's objective"
  backstory: "Rich persona description"
  expertise: ["domain1", "domain2"]
  personality: "Communication style"
  tool_budget: 30
  allowed_tools: ["read", "grep", "code_search"]
  can_delegate: false  # For hierarchical managers
  max_delegation_depth: 0
  memory: true  # Enable persistent memory
  cache: true  # Cache tool results
  max_iterations: null  # Use team default
```

## Creating Custom Templates

### Option 1: YAML File

Create `victor/workflows/templates/my_vertical/my_template.yaml`:

```yaml
name: my_custom_template
display_name: "My Custom Team"
description: "Does X, Y, Z"
long_description: "Detailed description..."
version: "1.0.0"
author: "Your Name"
vertical: "coding"
formation: "parallel"
complexity: "standard"
max_iterations: 75
total_tool_budget: 120
timeout_seconds: 900

tags:
  - "custom"
  - "specialized"

use_cases:
  - "Use case 1"
  - "Use case 2"

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

### Option 2: Python API

```python
from victor.workflows.team_templates import (
    TeamTemplate,
    TeamMemberSpec,
    register_template,
)

# Create member specs
members = [
    TeamMemberSpec(
        id="analyst",
        role="researcher",
        name="Analyst",
        goal="Analyze data patterns",
        backstory="Data scientist with 10 years...",
        expertise=["statistics", "visualization"],
        tool_budget=30,
        memory=True,
        cache=True,
    )
]

# Create template
template = TeamTemplate(
    name="my_template",
    display_name="My Template",
    description="Custom analysis template",
    long_description="Detailed description...",
    formation="parallel",
    members=members,
    vertical="dataanalysis",
    complexity="standard",
    max_iterations=75,
    total_tool_budget=100,
    timeout_seconds=900,
)

# Register
register_template(template)
```

## Template Selection Guide

### Choose by Task Type

**Code Review**: `code_review_parallel`
- Pull request reviews
- Security audits
- Quality assurance

**Bug Investigation**: `bug_hunt_hierarchical`
- Complex bugs
- Root cause analysis
- Production incidents

**Refactoring**: `refactoring_pipeline`
- Large-scale refactoring
- Technical debt reduction
- Architecture evolution

**Testing**: `testing_parallel`
- Test suite generation
- Coverage improvement
- Test maintenance

**Documentation**: `documentation_pipeline`
- API documentation
- User guides
- Technical docs

**Performance**: `performance_optimization_sequential`
- Performance optimization
- Bottleneck resolution
- Efficiency improvement

**Research**: `literature_review_pipeline`
- Literature reviews
- Technology research
- Background research

**Data Analysis**: `statistical_analysis_parallel`
- Statistical analysis
- Data exploration
- Hypothesis testing

**Deployment**: `deployment_pipeline`
- Production deployments
- Infrastructure changes
- Release management

### Choose by Complexity

**Quick Tasks** (5-10 min, 2-3 members):
- `quick_task_sequential`

**Standard Tasks** (15-30 min, 4-6 members):
- Most templates
- Balanced speed and thoroughness

**Complex Tasks** (30-60 min, 7-10 members):
- `bug_hunt_hierarchical`
- `refactoring_pipeline`
- `architecture_review_hierarchical`
- `problem_solving_hierarchical`
- `decision_making_consensus`
- `performance_optimization_sequential`

### Choose by Formation

**Sequential**: Linear processes, quick tasks
**Parallel**: Independent analysis, time-sensitive
**Pipeline**: Multi-phase with verification
**Hierarchical**: Complex coordination needed
**Consensus**: Agreement required, high-stakes decisions

## Best Practices

### Template Selection

1. **Match Use Case**: Choose template designed for your task
2. **Consider Complexity**: Don't over-engineer simple tasks
3. **Check Vertical**: Use specialized templates when available
4. **Review Examples**: See template examples for typical usage

### Template Customization

1. **Override Goal**: Always provide specific goal
2. **Adjust Budgets**: Tune budgets for task complexity
3. **Set Context**: Provide relevant context for team
4. **Monitor Execution**: Adjust iterations and timeout as needed

### Template Creation

1. **Start Simple**: Begin with existing template as base
2. **Validate Early**: Use template manager to validate
3. **Test Thoroughly**: Test with real tasks before deploying
4. **Document Well**: Include clear descriptions and examples

## Validation and Testing

### Validate Template

```bash
python -m scripts.teams.template_manager validate my_template.yaml
```

### Test Template

```python
from victor.workflows.team_templates import get_registry

registry = get_registry()
template = registry.get_template("my_template")

# Validate configuration
errors = registry.validate_template(template)
assert len(errors) == 0

# Test application
config = template.to_team_config(goal="Test")
assert len(config.members) > 0
```

## Contributing

To contribute new templates:

1. **Follow Structure**: Use template structure documented above
2. **Validate**: Run template manager validation
3. **Test**: Add integration tests
4. **Document**: Update this README with template details
5. **Examples**: Include usage examples

See contributing guidelines for details.

## Troubleshooting

### Template Not Found

```bash
# List available templates
python -m scripts.teams.template_manager list

# Search for similar templates
python -m scripts.teams.template_manager search "keyword"
```

### Validation Errors

Common issues:
- Invalid formation (must be: sequential, parallel, hierarchical, pipeline, consensus)
- No members (template must have at least one member)
- Duplicate member IDs (each member must have unique ID)
- Invalid role (must be valid SubAgentRole value)

### Performance Issues

- Reduce `max_iterations` for faster execution
- Reduce `total_tool_budget` to limit tool calls
- Reduce `timeout_seconds` for faster failure detection
- Use simpler templates (quick vs standard vs complex)

## See Also

- [Team Templates Documentation](../../../docs/teams/team_templates.md)
- [Team Coordination](../../../docs/teams/README.md)
- [Workflow Examples](../../../victor/coding/workflows/examples/)
- [Template Manager CLI](../../../scripts/teams/template_manager.py)

## Template Statistics

- **Total Templates**: 20
- **Formations**: 5 (Sequential, Parallel, Pipeline, Hierarchical, Consensus)
- **Verticals**: 5 (Coding, Research, DataAnalysis, DevOps, General)
- **Complexities**: 3 (Quick, Standard, Complex)
- **Average Members**: 3.5 per template
- **Average Tool Budget**: 105 per template
- **Average Timeout**: 14.5 minutes
