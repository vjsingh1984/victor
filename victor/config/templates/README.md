# Vertical Templates Directory

This directory contains YAML templates for creating new verticals in Victor.

## Available Templates

### 1. `base_vertical_template.yaml`
**Purpose**: Minimal starting point for new verticals

**Use when**: Creating a simple vertical with basic functionality

**Features**:
- Minimal metadata
- Core tool set (file operations, search, web)
- Simple 3-stage workflow (INITIAL, EXECUTION, COMPLETION)
- Extension placeholders

**Example**:
```bash
cp base_vertical_template.yaml my_vertical.yaml
# Edit my_vertical.yaml
python scripts/generate_vertical.py --template my_vertical.yaml --output victor/my_vertical
```

### 2. `example_documentation_vertical.yaml`
**Purpose**: Complete example of a documentation vertical

**Use when**: Learning the template system or creating a documentation-focused vertical

**Features**:
- Documentation-specific tools (code_search, semantic_code_search)
- 5-stage workflow (INITIAL, ANALYSIS, WRITING, REVIEW, COMPLETION)
- Safety patterns for README protection
- Task hints for documentation tasks
- Custom configuration for documentation style

**Example**:
```bash
python scripts/generate_vertical.py \
  --template example_documentation_vertical.yaml \
  --output victor/documentation
```

### 3. `example_security_vertical.yaml`
**Purpose**: Comprehensive example of a security vertical

**Use when**: Learning advanced features or creating a security-focused vertical

**Features**:
- Complete metadata with provider hints
- Security-specific tools (vulnerability_scan, secret_scan, dependency_check)
- 6-stage workflow (INITIAL, SCANNING, ANALYSIS, REPORTING, COMPLETION)
- Multiple middleware specifications
- Safety patterns for dangerous operations
- Task type hints for different security tasks
- Team formations (parallel security team)
- Workflow definitions
- Capability specifications
- Custom configuration with severity weights

**Example**:
```bash
python scripts/generate_vertical.py \
  --template example_security_vertical.yaml \
  --output victor/security
```

## Quick Start

### 1. Create Your Template

Copy the base template:
```bash
cp base_vertical_template.yaml my_vertical.yaml
```

### 2. Edit the Template

Edit `my_vertical.yaml` to customize:
- **Metadata**: name, description, version
- **Tools**: Add your vertical-specific tools
- **System Prompt**: Describe your vertical's purpose
- **Stages**: Define workflow stages
- **Extensions**: Add middleware, safety patterns, prompt hints

### 3. Validate

Check your template for errors:
```bash
python scripts/generate_vertical.py \
  --validate my_vertical.yaml \
  --output /tmp/test
```

### 4. Generate

Create the vertical:
```bash
python scripts/generate_vertical.py \
  --template my_vertical.yaml \
  --output victor/my_vertical
```

This generates:
- `assistant.py` - Main vertical class
- `prompts.py` - Prompt contributions
- `safety.py` - Safety patterns
- `escape_hatches.py` - Workflow escape hatches
- `handlers.py` - Workflow handlers
- `teams.py` - Team formations
- `capabilities.py` - Capability configs
- `__init__.py` - Package initialization
- `config/vertical.yaml` - YAML configuration

### 5. Use

```python
from victor.my_vertical import MyVerticalAssistant

config = MyVerticalAssistant.get_config()
agent = await Agent.create(
    tools=config.tools,
    vertical=MyVerticalAssistant,
)
```

## Template Structure

All templates follow this structure:

```yaml
metadata:
  name: vertical_name
  description: "Description"
  version: "1.0.0"
  author: "Author"
  category: category
  tags: [tags]

tools:
  - tool_name_1
  - tool_name_2

system_prompt: |
  System prompt text

stages:
  STAGE_NAME:
    name: STAGE_NAME
    description: "Stage description"
    tools: [tool_list]
    keywords: [keyword_list]
    next_stages: [next_stage_list]

extensions:
  middleware: []
  safety_patterns: []
  prompt_hints: []
  handlers: {}
  personas: {}
  composed_chains: {}

workflows: []
teams: []
capabilities: []

custom_config: {}
file_templates: {}
```

## Best Practices

1. **Start Simple**: Begin with base template, add complexity gradually
2. **Use Canonical Names**: Always use canonical tool names from `victor.tools.tool_names`
3. **Define Clear Stages**: Stages should represent meaningful workflow phases
4. **Add Safety Patterns**: Protect against dangerous operations
5. **Provide Task Hints**: Help the model choose the right approach
6. **Validate Often**: Check your template frequently during development
7. **Test Generation**: Generate and test your vertical before production use

## Documentation

- **Quick Start**: See `docs/vertical_quickstart.md`
- **Full Guide**: See `docs/vertical_template_guide.md`
- **Summary**: See `docs/vertical_template_system_summary.md`

## Examples

See the existing verticals in `victor/*/` for real-world examples:
- `victor/coding/` - Software development vertical
- `victor/research/` - Research and web search vertical
- `victor/devops/` - DevOps automation vertical
- `victor/rag/` - RAG (Retrieval Augmented Generation) vertical
- `victor/dataanalysis/` - Data analysis vertical

## Validation

All templates in this directory have been validated:

✓ `base_vertical_template.yaml` - Valid
✓ `example_documentation_vertical.yaml` - Valid
✓ `example_security_vertical.yaml` - Valid

## Contributing

To add a new template:

1. Create a new YAML file in this directory
2. Follow the template structure
3. Validate: `python scripts/generate_vertical.py --validate your_template.yaml --output /tmp/test`
4. Add documentation to this README
5. Test generation: `python scripts/generate_vertical.py --template your_template.yaml --output /tmp/test`

## Support

For help:
1. Check validation output
2. Review example templates
3. Consult documentation in `docs/`
4. Check existing vertical implementations
