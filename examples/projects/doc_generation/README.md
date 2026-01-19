# Documentation Generation Example Project

Automated documentation generation system using Victor AI.

## Features

- API documentation from docstrings
- README generation
- Architecture diagram creation
- Usage example generation
- Changelog automation

## Quick Start

```bash
cd examples/projects/doc_generation
pip install -r requirements.txt
victor init
victor chat "Generate comprehensive documentation for this project"
```

## Usage Examples

```bash
# Generate README
victor chat "Generate a comprehensive README.md for this project"

# Generate API docs
victor chat "Generate API documentation for all Python modules"

# Create architecture diagram
victor chat "Create a Mermaid architecture diagram for this project"

# Generate usage examples
victor chat "Generate usage examples for the main API"
```

## Sample Code

### src/doc_generator.py

```python
"""Documentation generator using Victor AI."""

from pathlib import Path
from typing import Dict, List
import ast

class DocumentationGenerator:
    """Generate documentation from code."""

    def __init__(self, source_path: str):
        """Initialize documentation generator."""
        self.source_path = Path(source_path)

    def generate_readme(self) -> str:
        """Generate README.md from code analysis."""
        sections = [
            self._generate_title(),
            self._generate_description(),
            self._generate_installation(),
            self._generate_usage(),
            self._generate_api_docs(),
            self._generate_contributing()
        ]
        return "\n\n".join(sections)

    def generate_api_docs(self) -> Dict[str, str]:
        """Generate API documentation for all modules."""
        docs = {}

        for py_file in self.source_path.rglob("*.py"):
            module_name = py_file.stem
            docs[module_name] = self._generate_module_docs(py_file)

        return docs

    def _generate_module_docs(self, module_path: Path) -> str:
        """Generate documentation for a single module."""
        try:
            with open(module_path, 'r') as f:
                tree = ast.parse(f.read())

            docstring = ast.get_docstring(tree)
            classes = []
            functions = []

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(self._document_class(node))
                elif isinstance(node, ast.FunctionDef):
                    functions.append(self._document_function(node))

            return f"""# {module_path.stem}

{docstring or "No module description"}

## Classes
{chr(10).join(classes)}

## Functions
{chr(10).join(functions)}
"""
        except Exception:
            return f"# {module_path.stem}\n\nError generating documentation"

    def _document_class(self, node: ast.ClassDef) -> str:
        """Document a class."""
        docstring = ast.get_docstring(node) or "No description"
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        return f"""
### {node.name}

{docstring}

**Methods:** {', '.join(methods)}
"""

    def _document_function(self, node: ast.FunctionDef) -> str:
        """Document a function."""
        docstring = ast.get_docstring(node) or "No description"
        args = [arg.arg for arg in node.args.args]

        return f"""
#### {node.name}({', '.join(args)})

{docstring}
"""

    def _generate_title(self) -> str:
        """Generate project title."""
        return f"# {self.source_path.name}"

    def _generate_description(self) -> str:
        """Generate project description."""
        return """
## Description

[Auto-generated description based on code analysis]
"""

    def _generate_installation(self) -> str:
        """Generate installation instructions."""
        return """
## Installation

```bash
pip install your-package-name
```
"""

    def _generate_usage(self) -> str:
        """Generate usage examples."""
        return """
## Usage

```python
from your_package import main

main()
```
"""

    def _generate_api_docs(self) -> str:
        """Generate API documentation section."""
        return """
## API Documentation

See detailed API documentation below.
"""

    def _generate_contributing(self) -> str:
        """Generate contributing guidelines."""
        return """
## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
"""
```

## Victor AI Integration

```bash
# Generate documentation with Victor
victor chat --mode build "Generate comprehensive documentation including:
1. README.md with project overview
2. API documentation for all modules
3. Usage examples
4. Architecture diagram in Mermaid format"

# Improve existing documentation
victor chat "Review README.md and suggest improvements"

# Generate docstrings
victor chat "Add comprehensive docstrings to all functions in src/"
```

## Learning Objectives

1. Extract docstrings and code structure
2. Generate structured documentation
3. Create architecture diagrams
4. Automate documentation workflows
5. Maintain documentation with code changes

## Resources

- **Docstring Conventions**: PEP 257
- **Markdown Guide**: https://www.markdownguide.org/
- **Mermaid Diagrams**: https://mermaid-js.github.io/

Happy documenting! 📝
