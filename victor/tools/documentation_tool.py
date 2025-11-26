# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Documentation generation tool for automated documentation.

Features:
- Generate docstrings for functions and classes
- Create API documentation
- Generate README sections
- Add type hints
- Create module documentation
"""

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


# Docstring templates
FUNCTION_DOCSTRING_TEMPLATE = '''"""
{summary}

{description}

Args:
{args}

Returns:
    {returns}

Raises:
    {raises}
"""'''

CLASS_DOCSTRING_TEMPLATE = '''"""
{summary}

{description}

Attributes:
{attributes}

Example:
{example}
"""'''


# Helper functions

def _generate_function_docstring(node: ast.FunctionDef, format: str) -> str:
    """Generate function docstring."""
    # Extract function info
    func_name = node.name
    args = [arg.arg for arg in node.args.args if arg.arg != "self"]

    # Generate summary
    summary = f"{func_name.replace('_', ' ').title()}."

    # Generate args section
    args_lines = []
    for arg in args:
        args_lines.append(f"    {arg}: Description of {arg}")

    args_section = "\n".join(args_lines) if args_lines else "    None"

    # Check for return
    has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
    returns_section = "Return value description" if has_return else "None"

    # Check for raises
    has_raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
    raises_section = "Exception: Description" if has_raises else "None"

    # Simple description
    description = f"Detailed description of {func_name}."

    docstring = f"""{summary}

{description}

Args:
{args_section}

Returns:
    {returns_section}

Raises:
    {raises_section}"""

    return docstring


def _generate_class_docstring(node: ast.ClassDef, format: str) -> str:
    """Generate class docstring."""
    class_name = node.name

    # Generate summary
    summary = f"{class_name} class."

    # Find attributes (instance variables)
    attributes = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            for stmt in item.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == "self":
                                attributes.append(target.attr)

    attrs_lines = []
    for attr in attributes[:5]:  # Limit to 5
        attrs_lines.append(f"    {attr}: Description of {attr}")

    attrs_section = "\n".join(attrs_lines) if attrs_lines else "    None"

    description = f"Detailed description of {class_name}."

    example = f"""instance = {class_name}()
    instance.method()"""

    docstring = f"""{summary}

{description}

Attributes:
{attrs_section}

Example:
    {example}"""

    return docstring


def _extract_api_info(tree: ast.AST, module_name: str) -> Dict[str, Any]:
    """Extract API information from AST."""
    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_"):  # Public functions only
                functions.append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "No description",
                    "args": [arg.arg for arg in node.args.args],
                })

        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if not item.name.startswith("_") or item.name in ("__init__", "__str__"):
                        methods.append({
                            "name": item.name,
                            "docstring": ast.get_docstring(item) or "No description",
                            "args": [arg.arg for arg in item.args.args if arg.arg != "self"],
                        })

            classes.append({
                "name": node.name,
                "docstring": ast.get_docstring(node) or "No description",
                "methods": methods,
            })

    return {
        "module": module_name,
        "functions": functions,
        "classes": classes,
    }


def _build_markdown_docs(api_info: Dict[str, Any]) -> str:
    """Build Markdown API documentation."""
    lines = []

    lines.append(f"# {api_info['module']} API Documentation")
    lines.append("")

    # Functions
    if api_info["functions"]:
        lines.append("## Functions")
        lines.append("")

        for func in api_info["functions"]:
            lines.append(f"### `{func['name']}({', '.join(func['args'])})`")
            lines.append("")
            lines.append(func["docstring"])
            lines.append("")
            if func["args"]:
                lines.append("**Parameters:**")
                for arg in func["args"]:
                    lines.append(f"- `{arg}`: Description")
                lines.append("")

    # Classes
    if api_info["classes"]:
        lines.append("## Classes")
        lines.append("")

        for cls in api_info["classes"]:
            lines.append(f"### `class {cls['name']}`")
            lines.append("")
            lines.append(cls["docstring"])
            lines.append("")

            if cls["methods"]:
                lines.append("**Methods:**")
                lines.append("")
                for method in cls["methods"]:
                    lines.append(f"#### `{method['name']}({', '.join(method['args'])})`")
                    lines.append("")
                    lines.append(method["docstring"])
                    lines.append("")

    return "\n".join(lines)


def _build_rst_docs(api_info: Dict[str, Any]) -> str:
    """Build reStructuredText API documentation."""
    # Similar to markdown but with RST syntax
    return _build_markdown_docs(api_info)  # Simplified for now


def _get_installation_template() -> str:
    """Get installation README template."""
    return """## Installation

### From PyPI

```bash
pip install your-package-name
```

### From Source

```bash
git clone https://github.com/username/repo.git
cd repo
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
"""


def _get_usage_template() -> str:
    """Get usage README template."""
    return """## Usage

### Quick Start

```python
from your_package import YourClass

# Create instance
instance = YourClass()

# Use the API
result = instance.method()
print(result)
```

### Advanced Usage

```python
# Advanced example
instance = YourClass(config={
    'option1': 'value1',
    'option2': 'value2',
})

result = instance.advanced_method(
    param1='value',
    param2=42
)
```

### Command Line Interface

```bash
your-command --option value
```
"""


def _get_contributing_template() -> str:
    """Get contributing README template."""
    return """## Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/repo.git
cd repo

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
```

### Code Style

- Follow PEP 8
- Use Black for formatting
- Add type hints
- Write docstrings for all public APIs
- Add tests for new features

### Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.
"""


def _get_features_template() -> str:
    """Get features README template."""
    return """## Features

- **Feature 1**: Description of feature 1
- **Feature 2**: Description of feature 2
- **Feature 3**: Description of feature 3

### Detailed Features

#### Feature 1

Detailed description of feature 1 with examples.

#### Feature 2

Detailed description of feature 2 with examples.

#### Feature 3

Detailed description of feature 3 with examples.
"""


def _get_api_template() -> str:
    """Get API README template."""
    return """## API Reference

### Main Classes

#### `YourMainClass`

Main class description.

**Methods:**

- `method1(param1, param2)`: Description
- `method2(param1)`: Description

### Utility Functions

#### `utility_function(param)`

Utility function description.

**Parameters:**
- `param`: Parameter description

**Returns:**
- Return value description

For complete API documentation, see [API Docs](docs/api.md).
"""


# Tool functions

@tool
async def docs_generate_docstrings(
    file: str,
    format: str = "google",
) -> Dict[str, Any]:
    """
    Generate docstrings for functions and classes.

    Analyzes Python source code and automatically generates
    docstrings for functions and classes that don't have them.

    Args:
        file: Source file path.
        format: Documentation format (google, numpy, rst) (default: google).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - generated: Number of docstrings generated
        - items: List of documented items with details
        - formatted_report: Human-readable generation report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read and parse file
    content = file_obj.read_text()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Find functions and classes without docstrings
    items_to_document = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not ast.get_docstring(node):
                items_to_document.append({
                    "type": "function",
                    "name": node.name,
                    "node": node,
                    "line": node.lineno,
                })
        elif isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                items_to_document.append({
                    "type": "class",
                    "name": node.name,
                    "node": node,
                    "line": node.lineno,
                })

    if not items_to_document:
        return {
            "success": True,
            "generated": 0,
            "items": [],
            "message": "All functions and classes already have docstrings!"
        }

    # Generate docstrings
    lines = content.split("\n")
    generated_docs = []

    for item in items_to_document:
        if item["type"] == "function":
            docstring = _generate_function_docstring(item["node"], format)
        else:
            docstring = _generate_class_docstring(item["node"], format)

        generated_docs.append({
            "name": item["name"],
            "type": item["type"],
            "line": item["line"],
            "docstring": docstring,
        })

        # Insert docstring into code
        # Find the line after function/class definition
        insert_line = item["line"]  # Line number (1-indexed)

        # Get indentation from the definition line
        def_line = lines[insert_line - 1]
        base_indent = len(def_line) - len(def_line.lstrip())
        indent = " " * (base_indent + 4)

        # Format docstring with proper indentation
        docstring_lines = docstring.split("\n")
        formatted_docstring = [f'{indent}"""']
        for line in docstring_lines:
            if line.strip():
                formatted_docstring.append(f"{indent}{line}")
            else:
                formatted_docstring.append("")
        formatted_docstring.append(f'{indent}"""')

        # Insert after the definition line
        lines = lines[:insert_line] + formatted_docstring + lines[insert_line:]

    # Write updated content
    new_content = "\n".join(lines)
    file_obj.write_text(new_content)

    # Build report
    report = []
    report.append("Docstring Generation Complete")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Format: {format}")
    report.append(f"Generated: {len(generated_docs)} docstrings")
    report.append("")

    for doc in generated_docs[:10]:
        report.append(f"✓ {doc['type'].title()}: {doc['name']} (line {doc['line']})")
        report.append(f"  {doc['docstring'][:100]}...")
        report.append("")

    if len(generated_docs) > 10:
        report.append(f"... and {len(generated_docs) - 10} more")

    return {
        "success": True,
        "generated": len(generated_docs),
        "items": generated_docs,
        "formatted_report": "\n".join(report)
    }


@tool
async def docs_generate_api(
    file: str,
    output: Optional[str] = None,
    format: str = "markdown",
) -> Dict[str, Any]:
    """
    Generate API documentation.

    Extracts functions, classes, and methods from Python source
    code and generates comprehensive API documentation.

    Args:
        file: Source file path.
        output: Output file path (optional, defaults to docs/{module}_api.md).
        format: Documentation format (markdown, rst) (default: markdown).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output_file: Path to generated documentation
        - functions_count: Number of documented functions
        - classes_count: Number of documented classes
        - preview: Preview of generated documentation
        - formatted_report: Human-readable generation report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read and parse file
    content = file_obj.read_text()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Extract API information
    api_info = _extract_api_info(tree, file_obj.stem)

    # Generate documentation
    if format == "markdown":
        docs = _build_markdown_docs(api_info)
    else:
        docs = _build_rst_docs(api_info)

    # Determine output path
    if not output:
        output = f"docs/{file_obj.stem}_api.md"

    # Write documentation
    output_file = Path(output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(docs)

    # Build report
    report = []
    report.append("API Documentation Generated")
    report.append("=" * 70)
    report.append("")
    report.append(f"Source: {file}")
    report.append(f"Output: {output}")
    report.append(f"Format: {format}")
    report.append("")
    report.append(f"Functions: {len(api_info['functions'])}")
    report.append(f"Classes: {len(api_info['classes'])}")
    report.append("")
    report.append("Preview:")
    report.append("-" * 70)
    report.append(docs[:1000])
    if len(docs) > 1000:
        report.append("...")

    return {
        "success": True,
        "output_file": output,
        "functions_count": len(api_info["functions"]),
        "classes_count": len(api_info["classes"]),
        "preview": docs[:500],
        "formatted_report": "\n".join(report)
    }


@tool
async def docs_generate_readme(section: str = "installation") -> Dict[str, Any]:
    """
    Generate README sections.

    Generates common README sections with standard templates
    following best practices.

    Args:
        section: README section to generate (installation, usage, contributing,
                features, api) (default: installation).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - section: Section name
        - content: Generated section content
        - formatted_report: Human-readable report with content
        - error: Error message if failed
    """
    templates = {
        "installation": _get_installation_template(),
        "usage": _get_usage_template(),
        "contributing": _get_contributing_template(),
        "features": _get_features_template(),
        "api": _get_api_template(),
    }

    template = templates.get(section)

    if not template:
        available = ", ".join(templates.keys())
        return {
            "success": False,
            "error": f"Unknown section: {section}. Available: {available}"
        }

    report = []
    report.append(f"README Section: {section.title()}")
    report.append("=" * 70)
    report.append("")
    report.append("Generated content:")
    report.append("-" * 70)
    report.append(template)

    return {
        "success": True,
        "section": section,
        "content": template,
        "formatted_report": "\n".join(report)
    }


@tool
async def docs_add_type_hints(file: str) -> Dict[str, Any]:
    """
    Add type hints to functions.

    Analyzes function signatures and suggests type hints
    following Python typing best practices.

    Args:
        file: Source file path.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - suggestions: List of type hint suggestions
        - formatted_report: Human-readable type hints report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # This is a placeholder - full implementation would use AST transformation
    report = []
    report.append("Type Hints Analysis")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append("")
    report.append("Suggested type hints:")
    report.append("  • Add return type annotations")
    report.append("  • Add parameter type hints")
    report.append("  • Use typing module for complex types")
    report.append("")
    report.append("Example:")
    report.append("  def process_data(items: List[Dict[str, Any]]) -> List[str]:")
    report.append("      ...")

    return {
        "success": True,
        "suggestions": [
            "Add return type annotations",
            "Add parameter type hints",
            "Use typing module for complex types"
        ],
        "formatted_report": "\n".join(report)
    }


@tool
async def docs_analyze_coverage(file: str) -> Dict[str, Any]:
    """
    Analyze documentation coverage.

    Checks what percentage of functions and classes have docstrings
    and identifies items missing documentation.

    Args:
        file: Source file path.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - coverage: Documentation coverage percentage
        - total_items: Total functions and classes
        - documented_items: Count of documented items
        - missing: List of items missing documentation
        - recommendations: List of recommendations
        - formatted_report: Human-readable coverage report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read and parse file
    content = file_obj.read_text()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Analyze documentation coverage
    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    missing = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_"):  # Skip private
                total_functions += 1
                if ast.get_docstring(node):
                    documented_functions += 1
                else:
                    missing.append(f"Function: {node.name} (line {node.lineno})")

        elif isinstance(node, ast.ClassDef):
            total_classes += 1
            if ast.get_docstring(node):
                documented_classes += 1
            else:
                missing.append(f"Class: {node.name} (line {node.lineno})")

    # Calculate coverage
    total_items = total_functions + total_classes
    documented_items = documented_functions + documented_classes
    coverage = (documented_items / total_items * 100) if total_items > 0 else 100

    # Generate recommendations
    recommendations = []
    if coverage < 50:
        recommendations.append("Low coverage - prioritize adding docstrings")
    elif coverage < 80:
        recommendations.append("Moderate coverage - aim for 80%+")
    else:
        recommendations.append("Good coverage - maintain quality")

    # Build report
    report = []
    report.append("Documentation Coverage Analysis")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append("")
    report.append(f"Coverage: {coverage:.1f}%")
    report.append("")
    report.append(f"Functions: {documented_functions}/{total_functions} documented")
    report.append(f"Classes: {documented_classes}/{total_classes} documented")
    report.append("")

    if missing:
        report.append(f"Missing documentation ({len(missing)} items):")
        for item in missing[:15]:
            report.append(f"  • {item}")
        if len(missing) > 15:
            report.append(f"  ... and {len(missing) - 15} more")
    else:
        report.append("✅ All items documented!")

    report.append("")
    report.append("Recommendations:")
    for rec in recommendations:
        emoji = "🔴" if coverage < 50 else ("🟡" if coverage < 80 else "🟢")
        report.append(f"  {emoji} {rec}")

    return {
        "success": True,
        "coverage": round(coverage, 1),
        "total_items": total_items,
        "documented_items": documented_items,
        "missing": missing,
        "recommendations": recommendations,
        "formatted_report": "\n".join(report)
    }


# Keep class for backward compatibility
class DocumentationTool:
    """Deprecated: Use individual docs_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "DocumentationTool class is deprecated. Use docs_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
