# Victor AI - Example Projects

This directory contains complete example projects demonstrating Victor AI's capabilities.

## Available Examples

### 1. Code Analysis Project
**Location:** `code_analysis/`

A complete code analysis and refactoring toolkit.

**Features:**
- Comprehensive codebase analysis
- Security vulnerability scanning
- Performance bottleneck identification
- Automated refactoring recommendations
- Quality metrics and reporting

**Setup:**
```bash
cd examples/projects/code_analysis
pip install -r requirements.txt
victor init
victor chat "Analyze this codebase and provide improvement recommendations"
```

**Learn:** How to build code analysis tools, integrate with AST parsing, generate refactoring recommendations.

### 2. Documentation Generation Project
**Location:** `doc_generation/`

Automated documentation generation system.

**Features:**
- API documentation from code
- README generation
- Architecture diagram creation
- Usage example generation
- Changelog automation

**Setup:**
```bash
cd examples/projects/doc_generation
pip install -r requirements.txt
victor init
victor chat "Generate comprehensive documentation for this project"
```

**Learn:** How to extract docstrings, generate Markdown docs, create architecture diagrams, automate documentation workflows.

### 3. Data Analysis Project
**Location:** `data_analysis/`

Data analysis and visualization assistant.

**Features:**
- Pandas DataFrame analysis
- Statistical analysis
- Visualization generation
- Report generation
- Trend detection

**Setup:**
```bash
cd examples/projects/data_analysis
pip install -r requirements.txt
victor init
victor chat "Analyze the data in data/sample.csv and generate insights"
```

**Learn:** How to work with Pandas, generate visualizations, perform statistical analysis, create data reports.

### 4. Research Assistant Project
**Location:** `research_assistant/`

AI-powered research and synthesis assistant.

**Features:**
- Web search integration
- Source citation
- Research synthesis
- Literature review
- Fact checking

**Setup:**
```bash
cd examples/projects/research_assistant
pip install -r requirements.txt
victor init
victor chat "Research the latest developments in LLM technology and summarize findings"
```

**Learn:** How to integrate web search, manage citations, synthesize research, create literature reviews.

## Using the Examples

### Quick Start

```bash
# Navigate to example directory
cd examples/projects/code_analysis

# Initialize Victor
victor init

# Run example
victor chat "Explain how this project works"
```

### Understanding the Structure

Each example project follows this structure:

```
project_name/
├── README.md              # Project overview
├── requirements.txt       # Python dependencies
├── .victor/              # Victor configuration
│   ├── config.yaml       # Victor settings
│   └── project_context.md  # Project context
├── src/                  # Source code
│   └── main.py          # Example implementation
├── data/                # Sample data (if applicable)
├── tests/               # Example tests
└── docs/                # Generated documentation
```

### Customizing Examples

```bash
# Modify project context
vim .victor/project_context.md

# Adjust Victor configuration
vim .victor/config.yaml

# Add your own code
vim src/your_feature.py

# Run with your changes
victor chat "Review my changes"
```

## Learning Path

### Beginner
1. Start with **code_analysis** - Understand code structure
2. Try **doc_generation** - Learn documentation patterns
3. Explore **data_analysis** - See data processing
4. Review **research_assistant** - Understand research workflows

### Intermediate
1. Modify existing examples
2. Combine features from multiple examples
3. Create custom workflows
4. Integrate your own tools

### Advanced
1. Build your own project from scratch
2. Create custom verticals
3. Implement complex workflows
4. Contribute your examples back

## Key Concepts Demonstrated

### Code Analysis Example
- AST parsing and analysis
- Code quality metrics
- Security scanning
- Refactoring recommendations
- Multi-file analysis

### Documentation Generation Example
- Docstring extraction
- Markdown generation
- API documentation
- Architecture diagrams
- Automated changelogs

### Data Analysis Example
- Pandas integration
- Statistical analysis
- Visualization
- Data cleaning
- Report generation

### Research Assistant Example
- Web search
- Citation management
- Source synthesis
- Literature review
- Fact checking

## Common Patterns

### Pattern 1: Project Initialization

```bash
cd examples/projects/code_analysis
victor init
```

This creates `.victor/` directory with configuration.

### Pattern 2: Context Setting

```bash
# Create project context
cat > .victor/project_context.md << EOF
# Project Overview
This project demonstrates code analysis capabilities.

# Key Features
- AST parsing
- Security scanning
- Quality metrics

# Usage
victor chat "Analyze src/ for security issues"
EOF
```

### Pattern 3: Workflow Execution

```bash
# Define workflow in .victor/workflows/
# Run workflow
victor workflow run analysis --path src/
```

### Pattern 4: Custom Tools

```python
# Define custom tool in src/tools/
# Register in .victor/config.yaml
# Use in Victor
victor chat "Use my_custom_tool on this file"
```

## Troubleshooting

### Issue: Module not found

```bash
# Install dependencies
pip install -r requirements.txt
```

### Issue: Victor not initialized

```bash
# Initialize Victor
victor init
```

### Issue: Configuration not loading

```bash
# Check .victor/config.yaml syntax
cat .victor/config.yaml
```

## Extending Examples

### Add New Features

```bash
# Add new source file
vim src/new_feature.py

# Update documentation
vim README.md

# Test with Victor
victor chat "Review new_feature.py and suggest improvements"
```

### Create Custom Workflow

```bash
# Create workflow file
vim .victor/workflows/my_workflow.yaml

# Run workflow
victor workflow run my_workflow
```

### Add Custom Tools

```python
# Create tool in src/tools/
vim src/tools/my_tool.py

# Register in config
vim .victor/config.yaml

# Use tool
victor chat "Use my_tool to analyze this code"
```

## Best Practices

1. **Start Simple**: Begin with basic examples, gradually add complexity
2. **Read the Code**: Understand how each example works
3. **Experiment**: Modify examples and see what happens
4. **Ask Questions**: Use Victor to explain code you don't understand
5. **Build Incrementally**: Add features one at a time
6. **Test Changes**: Use Victor to review your modifications
7. **Document**: Keep README.md updated with your changes

## Contributing

Have an idea for a new example project?

1. Follow the existing structure
2. Include comprehensive README
3. Add requirements.txt
4. Provide sample data if needed
5. Include usage examples
6. Document learning objectives

Submit a pull request with your example!

## Resources

- **Victor Documentation:** `docs/`
- **API Reference:** `docs/api-reference/`
- **Tutorials:** `docs/tutorials/`
- **Community Examples:** `examples/`

## Support

- **GitHub Issues:** Report bugs or request features
- **Discussions:** Ask questions and share ideas
- **Documentation:** Comprehensive guides available

Happy learning! 🎓
