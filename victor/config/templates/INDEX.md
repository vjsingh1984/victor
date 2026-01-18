# Victor Vertical Templates

Complete catalog of production-ready vertical templates for creating new verticals in Victor.

## Overview

Vertical templates provide a 70% reduction in code when creating new verticals by providing:
- Pre-configured tools and stages
- Extension patterns (middleware, safety, prompts)
- Workflow definitions
- Team formations
- Capability configurations
- Custom settings

## Available Templates

### Production Templates

#### 1. Coding Vertical Template

**File**: `coding_vertical_template.yaml`

**Description**: Software development assistant for code exploration, writing, and refactoring. This is Victor's primary vertical, optimized for full-stack development.

**Key Features**:
- 45+ tools for software development
- LSP integration for code intelligence
- Git operations and version control
- Test execution and validation
- Multi-language support (Python, TypeScript, Rust, Go, and more)
- Refactoring tools (rename, extract)
- Docker integration for containerized development

**Tools Included**:
- File operations: read, write, edit, grep, ls
- Code search: code_search, overview, plan
- Git: unified git tool
- LSP: lsp, symbol, refs
- Refactoring: rename, extract
- Testing: test
- Shell: shell
- Container: docker
- Web: web_search, web_fetch

**Stages**: INITIAL → PLANNING → READING → ANALYSIS → EXECUTION → VERIFICATION → COMPLETION

**Workflows**:
- code_review: Comprehensive code review
- test_generation: Generate unit tests from code

**Teams**:
- code_review_team: Parallel code review (security, logic, style)
- feature_implementation_team: Pipeline feature development

**Use Cases**:
- Full-stack application development
- Bug fixing and debugging
- Feature implementation
- Code refactoring
- Test generation and validation
- Git workflow automation

**Best For**: Software development teams, individual developers, technical projects

---

#### 2. DevOps Vertical Template

**File**: `devops_vertical_template.yaml`

**Description**: Infrastructure automation, container management, CI/CD, and deployment. Competitive with Docker Desktop AI, Terraform Assistant, Pulumi AI.

**Key Features**:
- Container orchestration (Docker, Kubernetes)
- CI/CD pipeline automation
- Infrastructure as Code (Terraform, Ansible)
- Deployment orchestration
- Monitoring and observability
- Secret management and security

**Tools Included**:
- File operations: read, write, edit, ls
- Shell: shell for infrastructure commands
- Git: version control
- Container: docker
- Testing: test for validation
- Search: grep, code_search, overview
- Web: web_search, web_fetch for documentation

**Stages**: INITIAL → ASSESSMENT → PLANNING → IMPLEMENTATION → VALIDATION → DEPLOYMENT → MONITORING → COMPLETION

**Workflows**:
- container_setup: Docker and Docker Compose setup
- deploy: Multi-environment deployment

**Teams**:
- infrastructure_team: IaC design and implementation
- deployment_team: Safe deployment orchestration

**Use Cases**:
- Docker containerization
- Kubernetes deployment
- CI/CD pipeline creation
- Infrastructure as Code
- Monitoring setup
- Deployment automation

**Best For**: DevOps engineers, SRE teams, infrastructure teams

---

#### 3. RAG Vertical Template

**File**: `rag_vertical_template.yaml`

**Description**: Retrieval-Augmented Generation assistant for document Q&A. Complete RAG implementation with document ingestion, vector search, and query generation.

**Key Features**:
- Document ingestion (PDF, Markdown, Text, Code)
- LanceDB vector storage (embedded, no server)
- Hybrid search (vector + full-text)
- Semantic chunking with overlap
- Source citations and attribution
- Knowledge base management

**Tools Included**:
- RAG-specific: rag_ingest, rag_search, rag_query, rag_list, rag_delete, rag_stats
- Filesystem: read, ls
- Web: web_fetch for web content
- Shell: shell for document processing

**Stages**: INITIAL → INGESTING → SEARCHING → QUERYING → SYNTHESIZING

**Workflows**:
- bulk_ingestion: Batch document processing
- qa_pipeline: Question-answering with retrieval

**Teams**:
- research_team: Parallel document research and synthesis

**Use Cases**:
- Documentation Q&A
- Knowledge base management
- Research assistance
- Technical document search
- Citation and source attribution

**Best For**: Knowledge workers, researchers, support teams, documentation-heavy projects

---

#### 4. Data Analysis Vertical Template

**File**: `dataanalysis_vertical_template.yaml`

**Description**: Data exploration, statistical analysis, visualization, and ML insights. Competitive with ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.

**Key Features**:
- Data loading (CSV, Excel, JSON, Parquet, SQL)
- Exploratory data analysis
- Statistical testing and modeling
- Visualization (matplotlib, seaborn, plotly)
- Machine learning (scikit-learn)
- Privacy protection (PII anonymization)

**Tools Included**:
- File operations: read, write, edit, grep, ls
- Shell: shell for Python execution
- Search: code_search, overview, graph
- Web: web_search, web_fetch for datasets

**Stages**: INITIAL → DATA_LOADING → EXPLORATION → CLEANING → ANALYSIS → VISUALIZATION → REPORTING → COMPLETION

**Workflows**:
- exploratory_analysis: Comprehensive EDA
- statistical_modeling: Hypothesis testing
- ml_pipeline: End-to-end ML workflow

**Teams**:
- analytics_team: Parallel analysis (profiling, statistics, visualization, reporting)

**Use Cases**:
- Exploratory data analysis
- Statistical testing
- Data visualization
- Machine learning
- Business intelligence
- Reporting and insights

**Best For**: Data scientists, analysts, business intelligence teams, researchers

---

### Example Templates

#### 5. Security Vertical Template (Example)

**File**: `example_security_vertical.yaml`

**Description**: Security analysis, vulnerability scanning, and threat detection. Demonstrates complete vertical structure with advanced features.

**Key Features**:
- Static code analysis
- Dependency vulnerability scanning
- Secret and credential detection
- Security compliance checking
- Threat modeling

**Use Cases**:
- Security audits
- Vulnerability assessment
- Compliance checking
- Threat detection

**Best For**: Security teams, DevSecOps, compliance officers

---

#### 6. Documentation Vertical Template (Example)

**File**: `example_documentation_vertical.yaml`

**Description**: Simple documentation specialist vertical. Demonstrates minimal template structure.

**Key Features**:
- Documentation generation
- Markdown formatting
- Code examples
- API documentation

**Use Cases**:
- API documentation
- User guides
- Technical writing

**Best For**: Technical writers, documentation teams

---

#### 7. Base Vertical Template

**File**: `base_vertical_template.yaml`

**Description**: Minimal template for creating simple verticals. Use as a starting point for custom verticals.

**Key Features**:
- Basic file operations
- Simple workflow stages
- Extension hooks

**Use Cases**:
- Custom verticals
- Learning template structure
- Quick prototyping

**Best For**: Template developers, custom vertical creation

---

## Template Structure

All templates follow this YAML structure:

```yaml
metadata:
  name: vertical_name
  description: "Vertical description"
  version: "1.0.0"
  category: category_name
  tags: [tag1, tag2]

tools: []  # List of tool names

system_prompt: |  # Main system prompt
  You are an expert in...

stages:  # Workflow stages
  STAGE_NAME:
    name: STAGE_NAME
    description: "Stage description"
    tools: [tool1, tool2]
    keywords: [keyword1, keyword2]
    next_stages: [NEXT_STAGE]

extensions:  # Extension specifications
  middleware: []
  safety_patterns: []
  prompt_hints: []
  handlers: {}
  personas: {}
  composed_chains: {}

workflows: []  # Workflow definitions

teams: []  # Team formations

capabilities: []  # Dynamic capabilities

custom_config:  # Vertical-specific settings
  setting: value

file_templates: {}  # Code scaffolding templates
```

---

## How to Use Templates

### Option 1: Generate Vertical from Template

```bash
# Generate a new vertical from template
python scripts/generate_vertical.py \
  --template victor/config/templates/coding_vertical_template.yaml \
  --output victor/my_vertical \
  --name MyVertical \
  --description "My custom vertical"
```

### Option 2: Copy and Customize

```bash
# Copy template
cp victor/config/templates/coding_vertical_template.yaml \
   victor/config/templates/my_vertical_template.yaml

# Edit template
vim victor/config/templates/my_vertical_template.yaml

# Generate vertical
python scripts/generate_vertical.py \
  --template victor/config/templates/my_vertical_template.yaml \
  --output victor/my_vertical
```

### Option 3: Manual Vertical Creation

Use templates as reference for manual vertical creation in `victor/your_vertical/assistant.py`.

---

## Template Features

### 1. Tools

Templates specify which tools the vertical uses:

```yaml
tools:
  - read_file
  - write_file
  - edit_file
  - grep
  - shell
```

**Available Tools**: See `victor/tools/tool_names.py` for canonical tool names.

### 2. Stages

Workflow stages define the step-by-step process:

```yaml
stages:
  INITIAL:
    name: INITIAL
    description: "Understanding the request"
    tools: [read_file, ls]
    keywords: [what, how, explain]
    next_stages: [EXECUTION]
```

**Stage Features**:
- **tools**: Tools available at this stage
- **keywords**: Keywords that trigger this stage
- **next_stages**: Possible next stages

### 3. Extensions

#### Middleware

Pre/post-processing for tool calls:

```yaml
extensions:
  middleware:
    - name: git_safety
      class_name: GitSafetyMiddleware
      module: victor.framework.middleware
      enabled: true
      config:
        block_dangerous: true
```

#### Safety Patterns

Dangerous operation patterns to warn about:

```yaml
extensions:
  safety_patterns:
    - name: git_force_push
      pattern: "git.*push.*-f"
      description: "Force push to remote"
      severity: "high"
      category: "git"
```

#### Prompt Hints

Task-type-specific guidance:

```yaml
extensions:
  prompt_hints:
    - task_type: code_generation
      hint: "[GENERATE] Write code directly. No exploration needed."
      tool_budget: 3
      priority_tools: [write_file]
```

#### Handlers

Workflow compute handlers:

```yaml
extensions:
  handlers:
    code_review:
      description: "Review code for quality"
      handler: victor.coding.workflows:code_review_handler
```

#### Personas

Specialized agent personalities:

```yaml
extensions:
  personas:
    code_reviewer:
      name: "Code Reviewer"
      description: "Expert code reviewer"
      system_prompt_extension: |
        You are a meticulous code reviewer...
```

### 4. Workflows

YAML workflow definitions:

```yaml
workflows:
  - name: code_review
    file: workflows/code_review.yaml
    description: "Comprehensive code review workflow"
```

### 5. Teams

Multi-agent team formations:

```yaml
teams:
  - name: code_review_team
    display_name: Code Review Team
    formation: parallel
    max_iterations: 10
    roles:
      - name: security_reviewer
        display_name: Security Reviewer
        persona: Security-focused reviewer...
        tool_categories: [code_search, grep]
```

**Team Formations**:
- `parallel`: Concurrent execution with aggregation
- `pipeline`: Sequential execution through stages
- `sequential`: Step-by-step execution
- `hierarchical`: Manager-worker coordination
- `consensus`: Vote-based decision making

### 6. Capabilities

Dynamic capability loading:

```yaml
capabilities:
  - name: code_review
    type: workflow
    description: "Review code for quality"
    enabled: true
    handler: "victor.coding.workflows:CodeReviewWorkflow"
```

**Capability Types**:
- `tool`: BaseTool implementations
- `workflow`: StateGraph workflows
- `middleware`: Pre/post-processing
- `validator`: Validation logic
- `observer`: Event observers

### 7. Custom Config

Vertical-specific settings:

```yaml
custom_config:
  supports_lsp: true
  max_file_size: 1000000
  test_frameworks: [pytest, jest]
  grounding_rules: |
    Always read existing code before modifying.
```

### 8. File Templates

Code scaffolding templates:

```yaml
file_templates:
  python_test:
    template_path: templates/python_test.j2
    output_pattern: "test_{}.py"
    description: "Python unit test template"
```

---

## Customizing Templates

### Add Custom Tools

Edit the `tools` section:

```yaml
tools:
  - read_file
  - write_file
  - my_custom_tool  # Add your tool
```

### Add Custom Stages

Edit the `stages` section:

```yaml
stages:
  MY_CUSTOM_STAGE:
    name: MY_CUSTOM_STAGE
    description: "My custom stage"
    tools: [tool1, tool2]
    keywords: [custom, keyword]
    next_stages: [EXECUTION]
```

### Add Custom Middleware

Edit the `extensions.middleware` section:

```yaml
extensions:
  middleware:
    - name: my_middleware
      class_name: MyCustomMiddleware
      module: victor.my_vertical.middleware
      enabled: true
      config:
        setting: value
```

### Add Custom Workflows

Edit the `workflows` section:

```yaml
workflows:
  - name: my_workflow
    file: workflows/my_workflow.yaml
    description: "My custom workflow"
```

### Add Custom Teams

Edit the `teams` section:

```yaml
teams:
  - name: my_team
    display_name: My Team
    formation: parallel
    max_iterations: 5
    roles:
      - name: my_role
        display_name: My Role
        persona: Expert in...
        tool_categories: [tool1, tool2]
```

---

## Template Best Practices

### 1. Use Canonical Tool Names

Always use canonical tool names from `victor.tools.tool_names.ToolNames`:

```yaml
# Good
tools:
  - shell  # Canonical name

# Avoid
tools:
  - execute_bash  # Legacy alias
```

### 2. Define Clear Stage Transitions

Ensure stages form a clear workflow:

```yaml
stages:
  INITIAL:
    next_stages: [PLANNING, READING]  # Clear forward progression
  READING:
    next_stages: [ANALYSIS, EXECUTION]  # Multiple paths
  COMPLETION:
    next_stages: []  # Terminal stage
```

### 3. Provide Comprehensive Prompt Hints

Cover common task types:

```yaml
extensions:
  prompt_hints:
    - task_type: simple_task
      hint: "[SIMPLE] Direct approach. No exploration."
      tool_budget: 3
      priority_tools: [write_file]

    - task_type: complex_task
      hint: "[COMPLEX] Thorough analysis. Multiple steps."
      tool_budget: 20
      priority_tools: [read, analyze, write]
```

### 4. Include Safety Patterns

Protect against dangerous operations:

```yaml
extensions:
  safety_patterns:
    - name: dangerous_operation
      pattern: "dangerous.*command"
      description: "Potential data loss"
      severity: "critical"
      category: "data_loss"
```

### 5. Configure Middleware Chains

Compose middleware for common scenarios:

```yaml
extensions:
  composed_chains:
    safe_edit:
      - code_correction
      - git_safety

    full_audit:
      - logging
      - monitoring
      - alerting
```

### 6. Use Team Formations Appropriately

Choose the right formation for your use case:

- **parallel**: Multiple perspectives, independent analysis (e.g., code review)
- **pipeline**: Sequential stages, handoffs (e.g., feature implementation)
- **sequential**: Step-by-step execution (e.g., deployment)
- **hierarchical**: Manager-worker coordination (e.g., large teams)
- **consensus**: Vote-based decisions (e.g., critical decisions)

---

## Validation

Validate templates before generating verticals:

```bash
# Validate template structure
python scripts/generate_vertical.py --validate \
  --template victor/config/templates/coding_vertical_template.yaml

# Check for required fields
python scripts/generate_vertical.py --check \
  --template victor/config/templates/my_template.yaml
```

---

## Testing Generated Verticals

Test verticals after generation:

```bash
# Generate test vertical
python scripts/generate_vertical.py \
  --template victor/config/templates/coding_vertical_template.yaml \
  --output /tmp/test_coding_vertical

# Verify generated code
python -c "import sys; sys.path.insert(0, '/tmp/test_coding_vertical'); from assistant import TestCodingVerticalAssistant; print('✓ Generated successfully')"

# Run tests
pytest tests/unit/test_vertical.py::test_coding_vertical
```

---

## Contributing Templates

To contribute a new template:

1. Create template file in `victor/config/templates/`
2. Follow template structure
3. Include comprehensive documentation
4. Validate template
5. Test generated vertical
6. Update this index
7. Submit PR

### Template Checklist

- [ ] Complete metadata section
- [ ] Comprehensive tools list
- [ ] Clear system prompt
- [ ] Well-defined stages
- [ ] Extension configurations
- [ ] Workflow definitions
- [ ] Team formations (if applicable)
- [ ] Capability configurations
- [ ] Custom settings
- [ ] File templates (if applicable)
- [ ] Usage examples
- [ ] Validation passed
- [ ] Documentation complete

---

## Support

For template-related questions:

- **Documentation**: See `victor/config/templates/README.md`
- **Examples**: See `example_*.yaml` templates
- **Base Template**: See `base_vertical_template.yaml`
- **Issues**: Report template issues on GitHub

---

## Changelog

### Version 1.0.0 (2025-01-18)

**Added**:
- Coding vertical template
- DevOps vertical template
- RAG vertical template
- Data Analysis vertical template
- Template index documentation

**Templates**: 7 total (4 production, 3 examples)

---

**Last Updated**: 2025-01-18
**Template Count**: 7
**Production Templates**: 4
**Example Templates**: 3
