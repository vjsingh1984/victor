# Code Review Assistant

A comprehensive code review tool powered by Victor AI that demonstrates advanced code analysis capabilities including AST parsing, security scanning, quality metrics, and style checking.

## Features

- **Automated Code Analysis**: Deep analysis using Abstract Syntax Tree (AST) parsing
- **Security Scanning**: Detects common security vulnerabilities (SQL injection, XSS, hardcoded secrets)
- **Quality Metrics**: Cyclomatic complexity, code duplication, maintainability index
- **Style Checking**: Enforces PEP 8 and project-specific style guidelines
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Rust, Java
- **Interactive Mode**: Real-time feedback with suggestions for improvements
- **Report Generation**: HTML, JSON, and Markdown output formats

## Installation

```bash
# Clone Victor AI repository
git clone https://github.com/your-org/victor-ai.git
cd victor-ai/examples/code_review_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Review a single file:

```bash
python src/main.py review path/to/file.py
```

Review an entire directory:

```bash
python src/main.py review path/to/project/ --recursive
```

### Interactive Mode

Start an interactive review session:

```bash
python src/main.py interactive
```

This will:
1. Load the Victor AI orchestrator
2. Analyze your code in real-time
3. Provide suggestions and improvements
4. Allow you to ask questions about your code

### Generate Reports

Generate an HTML report:

```bash
python src/main.py report path/to/project/ --format html --output report.html
```

Generate a JSON report:

```bash
python src/main.py report path/to/project/ --format json --output report.json
```

### Advanced Options

```bash
# Filter by severity
python src/main.py review path/to/ --severity high,critical

# Ignore specific rules
python src/main.py review path/to/ --ignore E501,W293

# Set custom complexity threshold
python src/main.py review path/to/ --max-complexity 10

# Include specific checks
python src/main.py review path/to/ --checks security,style,complexity
```

## Examples

### Example 1: Review Python Code

```bash
# Review sample Python code
python src/main.py review sample_code/example.py --verbose

# Sample output:
# ✓ Security: No issues found
# ⚠ Style: Line too long (85 > 79 characters) at line 12
# ⚠ Complexity: Function 'process_data' has complexity 12 (threshold: 10)
# ℹ Quality: Consider adding docstring to 'helper_function'
```

### Example 2: Security Scan

```bash
# Perform security-focused review
python src/main.py review sample_code/ --checks security --severity high,critical

# Sample output:
# 🔴 CRITICAL: Hardcoded API key detected at line 15
# 🔴 HIGH: Potential SQL injection at line 32
```

### Example 3: Team Review Workflow

```bash
# Run multi-agent team review (different reviewers)
python src/main.py team-review path/to/project/

# This will:
# - Assign security reviewer agent
# - Assign quality reviewer agent
# - Assign performance reviewer agent
# - Aggregate findings and prioritize
```

## Configuration

Create a `.victor-review.yaml` file in your project root:

```yaml
# Review configuration
review:
  # Max complexity threshold
  max_complexity: 10

  # Max line length
  max_line_length: 100

  # Severity levels to report
  severity:
    - high
    - medium
    - low

  # Checks to enable
  checks:
    - security
    - style
    - complexity
    - quality
    - duplication

  # Files/patterns to ignore
  ignore:
    - "**/tests/**"
    - "**/migrations/**"
    - "**/__pycache__/**"
    - "*.pyc"

  # Custom rules
  rules:
    - id: no-print-statements
      pattern: "print\\("
      message: "Use logger instead of print"
      severity: low

# Provider configuration
provider:
  name: anthropic
  model: claude-sonnet-4-5
  temperature: 0.0

# Team configuration (for multi-agent reviews)
team:
  formation: parallel
  roles:
    - name: security_reviewer
      persona: "Security expert focusing on vulnerabilities"
      capabilities: [security_scan, vulnerability_detection]

    - name: quality_reviewer
      persona: "Code quality expert focusing on maintainability"
      capabilities: [complexity_analysis, style_check]
```

## Integration with Victor AI

This demo showcases several Victor AI capabilities:

### 1. Agent Orchestrator
```python
from victor.agent.orchestrator_factory import create_orchestrator

orchestrator = create_orchestrator(settings)
result = await orchestrator.process_request(
    "Review this code for security issues",
    context={"file_path": file_path}
)
```

### 2. Coding Tools
```python
from victor.coding.ast import Parser
from victor.coding.review import CodeReviewer

parser = Parser(language="python")
ast = parser.parse(source_code)

reviewer = CodeReviewer(orchestrator)
issues = await reviewer.analyze(ast, file_path)
```

### 3. LSP Integration
```python
from victor.coding.lsp import LSPClient

lsp_client = LSPClient()
diagnostics = await lsp_client.get_diagnostics(file_path)
```

### 4. Multi-Agent Teams
```python
from victor.teams import create_coordinator

coordinator = create_coordinator(
    formation=TeamFormation.PARALLEL,
    roles=["security_reviewer", "quality_reviewer", "performance_reviewer"]
)

team_result = await coordinator.review_code(file_path)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 CLI Interface                        │
│         (main.py, commands/)                         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            Review Engine                             │
│  - orchestrates review pipeline                      │
│  - manages multi-agent teams                         │
│  - aggregates results                                │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   AST       │ │   Security  │ │   Quality   │
│   Analysis  │ │   Scanner   │ │   Metrics   │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  Victor AI      │
              │  Orchestrator   │
              └─────────────────┘
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_review.py::test_security_scan
```

## Sample Code

The `sample_code/` directory contains example code files with various issues for testing:

- `example.py` - Python code with style and complexity issues
- `security_issues.py` - Code with security vulnerabilities
- `good_code.py` - Well-written code as baseline
- `javascript_example.js` - JavaScript code analysis

## Contributing

This is a demo application for Victor AI. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

- **Documentation**: https://victor-ai.readthedocs.io
- **Issues**: https://github.com/your-org/victor-ai/issues
- **Discord**: https://discord.gg/victor-ai
