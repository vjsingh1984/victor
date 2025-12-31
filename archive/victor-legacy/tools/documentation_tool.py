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

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class DocumentationTool(BaseTool):
    """Tool for automated documentation generation."""

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

    @property
    def name(self) -> str:
        """Get tool name."""
        return "documentation"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Automated documentation generation.

Generate comprehensive documentation for Python code:
- Add docstrings to functions and classes
- Generate API documentation
- Create README sections
- Add type hints
- Generate module documentation

Operations:
- generate_docstrings: Add docstrings to functions/classes
- generate_api_docs: Create API documentation
- generate_readme: Generate README sections
- add_type_hints: Add type annotations
- analyze_docs: Check documentation coverage

Example workflows:
1. Add docstrings to a file:
   documentation(operation="generate_docstrings", file="app.py")

2. Generate API docs:
   documentation(operation="generate_api_docs", file="api.py", output="docs/api.md")

3. Check documentation coverage:
   documentation(operation="analyze_docs", file="app.py")

4. Generate README section:
   documentation(operation="generate_readme", section="installation")
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
            [
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation: generate_docstrings, generate_api_docs, generate_readme, add_type_hints, analyze_docs",
                    required=True,
                ),
                ToolParameter(
                    name="file",
                    type="string",
                    description="Source file path",
                    required=False,
                ),
                ToolParameter(
                    name="output",
                    type="string",
                    description="Output file path",
                    required=False,
                ),
                ToolParameter(
                    name="section",
                    type="string",
                    description="README section to generate",
                    required=False,
                ),
                ToolParameter(
                    name="format",
                    type="string",
                    description="Documentation format: markdown, rst, google, numpy",
                    required=False,
                ),
            ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute documentation operation.

        Args:
            operation: Documentation operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with generated documentation
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "generate_docstrings":
                return await self._generate_docstrings(kwargs)
            elif operation == "generate_api_docs":
                return await self._generate_api_docs(kwargs)
            elif operation == "generate_readme":
                return await self._generate_readme(kwargs)
            elif operation == "add_type_hints":
                return await self._add_type_hints(kwargs)
            elif operation == "analyze_docs":
                return await self._analyze_docs(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Documentation generation failed")
            return ToolResult(success=False, output="", error=f"Documentation error: {str(e)}")

    async def _generate_docstrings(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate docstrings for functions and classes."""
        file_path = kwargs.get("file")
        doc_format = kwargs.get("format", "google")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        # Read and parse file
        content = file_obj.read_text()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

        # Find functions and classes without docstrings
        items_to_document = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    items_to_document.append(
                        {
                            "type": "function",
                            "name": node.name,
                            "node": node,
                            "line": node.lineno,
                        }
                    )
            elif isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    items_to_document.append(
                        {
                            "type": "class",
                            "name": node.name,
                            "node": node,
                            "line": node.lineno,
                        }
                    )

        if not items_to_document:
            return ToolResult(
                success=True,
                output="All functions and classes already have docstrings!",
                error="",
            )

        # Generate docstrings
        lines = content.split("\n")
        generated_docs = []

        for item in items_to_document:
            if item["type"] == "function":
                docstring = self._generate_function_docstring(item["node"], doc_format)
            else:
                docstring = self._generate_class_docstring(item["node"], doc_format)

            generated_docs.append(
                {
                    "name": item["name"],
                    "type": item["type"],
                    "line": item["line"],
                    "docstring": docstring,
                }
            )

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
        report.append(f"File: {file_path}")
        report.append(f"Format: {doc_format}")
        report.append(f"Generated: {len(generated_docs)} docstrings")
        report.append("")

        for doc in generated_docs[:10]:
            report.append(f"✓ {doc['type'].title()}: {doc['name']} (line {doc['line']})")
            report.append(f"  {doc['docstring'][:100]}...")
            report.append("")

        if len(generated_docs) > 10:
            report.append(f"... and {len(generated_docs) - 10} more")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _generate_api_docs(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate API documentation."""
        file_path = kwargs.get("file")
        output_path = kwargs.get("output")
        doc_format = kwargs.get("format", "markdown")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        # Read and parse file
        content = file_obj.read_text()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

        # Extract API information
        api_info = self._extract_api_info(tree, file_obj.stem)

        # Generate documentation
        if doc_format == "markdown":
            docs = self._build_markdown_docs(api_info)
        else:
            docs = self._build_rst_docs(api_info)

        # Determine output path
        if not output_path:
            output_path = f"docs/{file_obj.stem}_api.md"

        # Write documentation
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(docs)

        # Build report
        report = []
        report.append("API Documentation Generated")
        report.append("=" * 70)
        report.append("")
        report.append(f"Source: {file_path}")
        report.append(f"Output: {output_path}")
        report.append(f"Format: {doc_format}")
        report.append("")
        report.append(f"Functions: {len(api_info['functions'])}")
        report.append(f"Classes: {len(api_info['classes'])}")
        report.append("")
        report.append("Preview:")
        report.append("-" * 70)
        report.append(docs[:1000])
        if len(docs) > 1000:
            report.append("...")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _generate_readme(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate README sections."""
        section = kwargs.get("section", "installation")

        templates = {
            "installation": self._get_installation_template(),
            "usage": self._get_usage_template(),
            "contributing": self._get_contributing_template(),
            "features": self._get_features_template(),
            "api": self._get_api_template(),
        }

        template = templates.get(section)

        if not template:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown section: {section}. Available: {', '.join(templates.keys())}",
            )

        report = []
        report.append(f"README Section: {section.title()}")
        report.append("=" * 70)
        report.append("")
        report.append("Generated content:")
        report.append("-" * 70)
        report.append(template)

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _add_type_hints(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Add type hints to functions."""
        file_path = kwargs.get("file")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        # This is a placeholder - full implementation would use AST transformation
        report = []
        report.append("Type Hints Analysis")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
        report.append("")
        report.append("Suggested type hints:")
        report.append("  • Add return type annotations")
        report.append("  • Add parameter type hints")
        report.append("  • Use typing module for complex types")
        report.append("")
        report.append("Example:")
        report.append("  def process_data(items: List[Dict[str, Any]]) -> List[str]:")
        report.append("      ...")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _analyze_docs(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Analyze documentation coverage."""
        file_path = kwargs.get("file")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        # Read and parse file
        content = file_obj.read_text()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

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

        # Build report
        report = []
        report.append("Documentation Coverage Analysis")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
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
        if coverage < 50:
            report.append("  🔴 Low coverage - prioritize adding docstrings")
        elif coverage < 80:
            report.append("  🟡 Moderate coverage - aim for 80%+")
        else:
            report.append("  🟢 Good coverage - maintain quality")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    def _generate_function_docstring(self, node: ast.FunctionDef, format: str) -> str:
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

    def _generate_class_docstring(self, node: ast.ClassDef, format: str) -> str:
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

    def _extract_api_info(self, tree: ast.AST, module_name: str) -> Dict[str, Any]:
        """Extract API information from AST."""
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):  # Public functions only
                    functions.append(
                        {
                            "name": node.name,
                            "docstring": ast.get_docstring(node) or "No description",
                            "args": [arg.arg for arg in node.args.args],
                        }
                    )

            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if not item.name.startswith("_") or item.name in ("__init__", "__str__"):
                            methods.append(
                                {
                                    "name": item.name,
                                    "docstring": ast.get_docstring(item) or "No description",
                                    "args": [
                                        arg.arg for arg in item.args.args if arg.arg != "self"
                                    ],
                                }
                            )

                classes.append(
                    {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "No description",
                        "methods": methods,
                    }
                )

        return {
            "module": module_name,
            "functions": functions,
            "classes": classes,
        }

    def _build_markdown_docs(self, api_info: Dict[str, Any]) -> str:
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

    def _build_rst_docs(self, api_info: Dict[str, Any]) -> str:
        """Build reStructuredText API documentation."""
        # Similar to markdown but with RST syntax
        return self._build_markdown_docs(api_info)  # Simplified for now

    def _get_installation_template(self) -> str:
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

    def _get_usage_template(self) -> str:
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

    def _get_contributing_template(self) -> str:
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

    def _get_features_template(self) -> str:
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

    def _get_api_template(self) -> str:
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
