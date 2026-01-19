# Code Analysis Example Project

A comprehensive code analysis toolkit demonstrating Victor AI's capabilities for analyzing, understanding, and improving codebases.

## Project Overview

This example project shows how to use Victor AI to:
- Analyze code structure and architecture
- Identify security vulnerabilities
- Detect performance bottlenecks
- Suggest refactoring improvements
- Generate quality metrics

## Features

### 1. Codebase Analysis
- Parse and understand code structure
- Identify components and dependencies
- Map architecture and relationships
- Detect code patterns and anti-patterns

### 2. Security Scanning
- SQL injection detection
- XSS vulnerability detection
- Hardcoded secret detection
- Dependency vulnerability checking

### 3. Performance Analysis
- Identify slow operations
- Detect inefficient algorithms
- Find N+1 query problems
- Suggest optimizations

### 4. Quality Metrics
- Cyclomatic complexity
- Code duplication
- Test coverage analysis
- Documentation coverage

## Setup

### Installation

```bash
# Navigate to project directory
cd examples/projects/code_analysis

# Install dependencies
pip install -r requirements.txt

# Initialize Victor
victor init
```

### Configuration

Create `.victor/config.yaml`:

```yaml
provider: ollama
model: qwen2.5-coder:7b
temperature: 0.7

tools:
  enabled:
    - read_file
    - search_code
    - run_command
    - list_files

mode: analyze
vertical: coding
```

### Project Context

Create `.victor/project_context.md`:

```markdown
# Code Analysis Toolkit

## Purpose
Demonstrate Victor AI's code analysis capabilities.

## Key Components
- **Analyzer**: Core analysis engine
- **Scanner**: Security vulnerability scanner
- **Metrics**: Quality metrics calculator
- **Reporter**: Report generator

## Usage Examples
```bash
victor chat "Analyze src/ for security issues"
victor chat "Identify performance bottlenecks"
victor chat "Generate quality metrics report"
```
```

## Usage Examples

### Example 1: Security Analysis

```bash
victor chat "Analyze src/ for security vulnerabilities focusing on:
1. SQL injection risks
2. XSS vulnerabilities
3. Hardcoded secrets
4. Authentication issues"
```

### Example 2: Performance Analysis

```bash
victor chat "Analyze src/ for performance issues:
1. Inefficient algorithms
2. Database query optimization
3. Caching opportunities
4. Resource management"
```

### Example 3: Code Quality Assessment

```bash
victor chat "Assess code quality for src/:
1. Calculate cyclomatic complexity
2. Identify code duplication
3. Check naming conventions
4. Review documentation coverage"
```

### Example 4: Refactoring Recommendations

```bash
victor chat "Review src/analyzer.py and suggest refactoring to:
1. Improve readability
2. Reduce complexity
3. Apply design patterns
4. Enhance maintainability"
```

## Project Structure

```
code_analysis/
├── README.md
├── requirements.txt
├── .victor/
│   ├── config.yaml
│   └── project_context.md
├── src/
│   ├── __init__.py
│   ├── analyzer.py          # Core analyzer
│   ├── scanner.py           # Security scanner
│   ├── metrics.py           # Quality metrics
│   └── reporter.py          # Report generator
├── tests/
│   ├── test_analyzer.py
│   ├── test_scanner.py
│   └── test_metrics.py
└── examples/
    └── sample_code.py       # Sample code for analysis
```

## Sample Code

### src/analyzer.py

```python
"""Code analyzer for understanding code structure."""

from pathlib import Path
from typing import Dict, List, Any
import ast

class CodeAnalyzer:
    """Analyze Python code structure and dependencies."""

    def __init__(self, root_path: str):
        """Initialize analyzer with root path."""
        self.root_path = Path(root_path)
        self.structure = {}

    def analyze(self) -> Dict[str, Any]:
        """Perform complete codebase analysis."""
        return {
            "structure": self._analyze_structure(),
            "dependencies": self._analyze_dependencies(),
            "complexity": self._analyze_complexity(),
            "patterns": self._analyze_patterns()
        }

    def _analyze_structure(self) -> Dict[str, List[str]]:
        """Analyze directory structure and file organization."""
        structure = {"modules": [], "packages": []}

        for item in self.root_path.rglob("*.py"):
            if item.is_file():
                structure["modules"].append(str(item))
            elif item.is_dir():
                structure["packages"].append(str(item))

        return structure

    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze import dependencies between modules."""
        dependencies = {}

        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())

                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)

                dependencies[str(py_file)] = imports
            except Exception:
                continue

        return dependencies

    def _analyze_complexity(self) -> Dict[str, int]:
        """Calculate cyclomatic complexity for each function."""
        complexity = {}

        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Simplified complexity calculation
                        func_complexity = 1  # Base complexity
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.While, ast.For)):
                                func_complexity += 1

                        complexity[f"{py_file}:{node.name}"] = func_complexity
            except Exception:
                continue

        return complexity

    def _analyze_patterns(self) -> Dict[str, List[str]]:
        """Identify common design patterns and anti-patterns."""
        patterns = {
            "design_patterns": [],
            "anti_patterns": []
        }

        # Pattern detection logic here
        # This is a simplified example

        return patterns
```

### src/scanner.py

```python
"""Security vulnerability scanner."""

import re
from typing import Dict, List, Any

class SecurityScanner:
    """Scan code for security vulnerabilities."""

    def __init__(self):
        self.vulnerabilities = []

    def scan(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Scan code for security issues."""
        self.vulnerabilities = []

        # SQL Injection
        self._check_sql_injection(code)

        # Hardcoded secrets
        self._check_hardcoded_secrets(code)

        # XSS (for web languages)
        if language in ["javascript", "typescript"]:
            self._check_xss(code)

        return {
            "vulnerabilities": self.vulnerabilities,
            "summary": self._generate_summary()
        }

    def _check_sql_injection(self, code: str):
        """Check for SQL injection vulnerabilities."""
        # Look for string concatenation in SQL queries
        patterns = [
            r'(execute|exec|query)\s*\(\s*[\'"]\s*SELECT.*\+.*\)',
            r'(execute|exec|query)\s*\(\s*f[\'"][^\'']*\{.*\}[^\'']*\'[^\)]*\)',
        ]

        for i, line in enumerate(code.split('\n'), 1):
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.vulnerabilities.append({
                        "type": "SQL Injection",
                        "severity": "HIGH",
                        "line": i,
                        "code": line.strip(),
                        "recommendation": "Use parameterized queries"
                    })

    def _check_hardcoded_secrets(self, code: str):
        """Check for hardcoded secrets."""
        patterns = {
            r'api[_-]?key\s*=\s*[\'"][^\'"]{20,}[\'"]': "API Key",
            r'password\s*=\s*[\'"][^\'"]{8,}[\'"]': "Password",
            r'secret\s*=\s*[\'"][^\'"]{16,}[\'"]': "Secret",
            r'token\s*=\s*[\'"][^\'"]{20,}[\'"]': "Token"
        }

        for i, line in enumerate(code.split('\n'), 1):
            for pattern, secret_type in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    self.vulnerabilities.append({
                        "type": f"Hardcoded {secret_type}",
                        "severity": "CRITICAL",
                        "line": i,
                        "code": line.strip(),
                        "recommendation": f"Use environment variables for {secret_type}"
                    })

    def _check_xss(self, code: str):
        """Check for XSS vulnerabilities."""
        patterns = [
            r'innerHTML\s*=',
            r'document\.write\s*\(',
            r'eval\s*\('
        ]

        for i, line in enumerate(code.split('\n'), 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    self.vulnerabilities.append({
                        "type": "XSS Vulnerability",
                        "severity": "HIGH",
                        "line": i,
                        "code": line.strip(),
                        "recommendation": "Use textContent or sanitize input"
                    })

    def _generate_summary(self) -> Dict[str, int]:
        """Generate vulnerability summary."""
        return {
            "total": len(self.vulnerabilities),
            "critical": len([v for v in self.vulnerabilities if v["severity"] == "CRITICAL"]),
            "high": len([v for v in self.vulnerabilities if v["severity"] == "HIGH"]),
            "medium": len([v for v in self.vulnerabilities if v["severity"] == "MEDIUM"]),
            "low": len([v for v in self.vulnerabilities if v["severity"] == "LOW"])
        }
```

## Learning Objectives

After working through this example, you will understand:

1. **Code Parsing**: How to parse and understand code structure
2. **AST Analysis**: Using Abstract Syntax Trees for code analysis
3. **Pattern Detection**: Identifying code patterns and anti-patterns
4. **Security Scanning**: Detecting common security vulnerabilities
5. **Metrics Calculation**: Computing code quality metrics
6. **Report Generation**: Creating comprehensive analysis reports

## Extensions

### Add Your Own Analysis

```python
# Create custom analyzer
class CustomAnalyzer(CodeAnalyzer):
    def custom_analysis(self, code: str) -> Dict[str, Any]:
        """Your custom analysis logic."""
        # Implement your analysis
        pass
```

### Integrate with Victor

```bash
# Use Victor to analyze the analyzer
victor chat "Review src/analyzer.py and suggest improvements"

# Generate tests
victor chat "Generate unit tests for src/scanner.py"

# Create documentation
victor chat "Generate API documentation for this project"
```

## Next Steps

1. **Experiment**: Try analyzing different codebases
2. **Extend**: Add new analysis capabilities
3. **Customize**: Adapt for your specific needs
4. **Integrate**: Combine with other examples

## Resources

- **AST Documentation**: https://docs.python.org/3/library/ast.html
- **Security Best Practices**: OWASP Top 10
- **Code Quality Metrics**: Cyclomatic Complexity, Maintainability Index

Happy analyzing! 🔍
