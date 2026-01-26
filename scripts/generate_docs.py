#!/usr/bin/env python3
"""Documentation Generator for Victor

This tool generates comprehensive documentation from source code, including:
- API documentation from type hints and docstrings
- Protocol documentation
- Coordinator documentation
- Vertical documentation
- Usage examples

Usage:
    python scripts/generate_docs.py --type api --output docs/api/
    python scripts/generate_docs.py --all --output docs/
    python scripts/generate_docs.py --coordinators --format markdown
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, get_type_hints

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class APIDocument:
    """Represents an API documentation entry."""

    name: str
    type: str  # class, function, method, protocol
    signature: str
    docstring: Optional[str]
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    raises: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    line_number: Optional[int] = None


class DocumentationGenerator:
    """Generates documentation from source code."""

    def __init__(self, output_dir: Path):
        """Initialize generator.

        Args:
            output_dir: Directory to save generated documentation
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[APIDocument] = []

    def generate_api_docs(self, module_path: str) -> None:
        """Generate API documentation for a module.

        Args:
            module_path: Python module path (e.g., "victor.protocols")
        """
        print(f"Generating API docs for: {module_path}")

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"Error importing {module_path}: {e}")
            return

        # Get all public members
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue

            # Document classes
            if inspect.isclass(obj):
                self._document_class(obj, module_path)

            # Document functions
            elif inspect.isfunction(obj):
                self._document_function(obj, module_path)

        # Write API documentation
        self._write_api_docs(module_path)

    def _document_class(self, cls: Type, module_path: str) -> None:
        """Document a class.

        Args:
            cls: Class to document
            module_path: Module path
        """
        # Get signature
        try:
            sig = inspect.signature(cls)
            signature = f"{cls.__name__}{sig}"
        except ValueError:
            signature = f"{cls.__name__}()"

        # Get docstring
        docstring = inspect.getdoc(cls)

        # Get source location
        try:
            source_file = inspect.getsourcefile(cls)
            lines, line_number = inspect.getsourcelines(cls)
            source_file = str(Path(source_file).relative_to(project_root))
        except (TypeError, OSError):
            source_file = None
            line_number = None

        # Create document
        doc = APIDocument(
            name=cls.__name__,
            type="protocol" if "Protocol" in cls.__name__ else "class",
            signature=signature,
            docstring=docstring,
            source_file=source_file,
            line_number=line_number,
        )

        # Document methods
        for name, method in inspect.getmembers(cls):
            if name.startswith("_"):
                continue
            if callable(method):
                self._document_method(method, cls.__name__, doc)

        self.documents.append(doc)

    def _document_function(self, func: Type, module_path: str) -> None:
        """Document a function.

        Args:
            func: Function to document
            module_path: Module path
        """
        # Get signature
        try:
            sig = inspect.signature(func)
            signature = f"{func.__name__}{sig}"
        except ValueError:
            signature = f"{func.__name__}()"

        # Get docstring
        docstring = inspect.getdoc(func)

        # Get source location
        try:
            source_file = inspect.getsourcefile(func)
            source_file = str(Path(source_file).relative_to(project_root))
        except (TypeError, OSError):
            source_file = None

        # Get parameters
        parameters = []
        if hasattr(func, "__annotations__"):
            hints = get_type_hints(func)
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": str(hints.get(param_name, "Any")),
                    "default": (
                        str(param.default) if param.default != inspect.Parameter.empty else None
                    ),
                }
                parameters.append(param_info)

        # Get return type
        return_type = None
        if hasattr(func, "__annotations__") and "return" in func.__annotations__:
            return_type = str(func.__annotations__["return"])

        # Create document
        doc = APIDocument(
            name=func.__name__,
            type="function",
            signature=signature,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            source_file=source_file,
        )

        self.documents.append(doc)

    def _document_method(self, method: Type, class_name: str, class_doc: APIDocument) -> None:
        """Document a method.

        Args:
            method: Method to document
            class_name: Name of containing class
            class_doc: Class document to add method to
        """
        # Get signature
        try:
            sig = inspect.signature(method)
            signature = f"{method.__name__}{sig}"
        except ValueError:
            signature = f"{method.__name__}()"

        # Get docstring
        docstring = inspect.getdoc(method)

        # Get parameters
        parameters = []
        try:
            sig = inspect.signature(method)
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": "Any",
                    "default": (
                        str(param.default) if param.default != inspect.Parameter.empty else None
                    ),
                }
                parameters.append(param_info)
        except ValueError:
            pass

        # Get return type
        return_type = None
        if hasattr(method, "__annotations__") and "return" in method.__annotations__:
            return_type = str(method.__annotations__["return"])

        # Add method as separate document
        doc = APIDocument(
            name=f"{class_name}.{method.__name__}",
            type="method",
            signature=signature,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
        )

        self.documents.append(doc)

    def _write_api_docs(self, module_path: str) -> None:
        """Write API documentation to file.

        Args:
            module_path: Module path being documented
        """
        output_file = self.output_dir / f"{module_path.replace('.', '_')}.md"

        with open(output_file, "w") as f:
            f.write(f"# API Documentation: {module_path}\n\n")
            f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")

            # Group documents by type
            protocols = [d for d in self.documents if d.type == "protocol"]
            classes = [d for d in self.documents if d.type == "class"]
            functions = [d for d in self.documents if d.type == "function"]

            # Write protocols
            if protocols:
                f.write("## Protocols\n\n")
                for doc in sorted(protocols, key=lambda x: x.name):
                    self._write_document_entry(f, doc)

            # Write classes
            if classes:
                f.write("## Classes\n\n")
                for doc in sorted(classes, key=lambda x: x.name):
                    self._write_document_entry(f, doc)

            # Write functions
            if functions:
                f.write("## Functions\n\n")
                for doc in sorted(functions, key=lambda x: x.name):
                    self._write_document_entry(f, doc)

        print(f"  Wrote: {output_file}")

    def _write_document_entry(self, f, doc: APIDocument) -> None:
        """Write a single documentation entry.

        Args:
            f: File handle
            doc: Document to write
        """
        f.write(f"### {doc.name}\n\n")

        if doc.signature:
            f.write(f"```python\n{doc.signature}\n```\n\n")

        if doc.source_file:
            f.write(f"**Source:** [{doc.source_file}]({doc.source_file})")
            if doc.line_number:
                f.write(f":{doc.line_number}")
            f.write("\n\n")

        if doc.docstring:
            f.write(f"{doc.docstring}\n\n")

        if doc.parameters:
            f.write("**Parameters:**\n\n")
            for param in doc.parameters:
                default_str = f" = {param['default']}" if param["default"] else ""
                f.write(f"- `{param['name']}` ({param['type']}){default_str}\n")
            f.write("\n")

        if doc.return_type:
            f.write(f"**Returns:** {doc.return_type}\n\n")

        if doc.raises:
            f.write("**Raises:**\n\n")
            for exc in doc.raises:
                f.write(f"- {exc}\n")
            f.write("\n")

        if doc.examples:
            f.write("**Examples:**\n\n")
            for example in doc.examples:
                f.write(f"```python\n{example}\n```\n\n")

        f.write("---\n\n")

    def generate_coordinator_docs(self) -> None:
        """Generate documentation for all coordinators."""
        print("Generating coordinator docs...")

        coordinators = [
            "ToolCoordinator",
            "PromptCoordinator",
            "ContextCoordinator",
            "ConfigCoordinator",
            "AnalyticsCoordinator",
            "ChatCoordinator",
        ]

        output_file = self.output_dir / "coordinators.md"

        with open(output_file, "w") as f:
            f.write("# Coordinator Documentation\n\n")
            f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")
            f.write("## Overview\n\n")
            f.write("Victor uses a coordinator pattern to orchestrate complex operations.\n\n")
            f.write("## Coordinators\n\n")

            for coordinator_name in coordinators:
                f.write(f"### {coordinator_name}\n\n")

                # Try to import and document
                try:
                    module = importlib.import_module(
                        f"victor.agent.coordinators.{coordinator_name.lower()}"
                    )
                    coordinator_class = getattr(module, coordinator_name)

                    # Get docstring
                    docstring = inspect.getdoc(coordinator_class)
                    if docstring:
                        f.write(f"{docstring}\n\n")

                    # Get methods
                    for name, method in inspect.getmembers(coordinator_class):
                        if name.startswith("_"):
                            continue
                        if callable(method):
                            try:
                                sig = inspect.signature(method)
                                f.write(f"#### {name}{sig}\n\n")

                                method_doc = inspect.getdoc(method)
                                if method_doc:
                                    f.write(f"{method_doc}\n\n")
                            except ValueError:
                                pass

                except (ImportError, AttributeError) as e:
                    f.write(f"Warning: Could not load coordinator: {e}\n\n")

                f.write("---\n\n")

        print(f"  Wrote: {output_file}")

    def generate_vertical_docs(self) -> None:
        """Generate documentation for all verticals."""
        print("Generating vertical docs...")

        # Find all verticals
        victor_dir = Path("victor")
        verticals = []

        for item in victor_dir.iterdir():
            if not item.is_dir() or item.name.startswith("_") or item.name == "core":
                continue
            if (item / "assistant.py").exists():
                verticals.append(item.name)

        output_file = self.output_dir / "verticals.md"

        with open(output_file, "w") as f:
            f.write("# Vertical Documentation\n\n")
            f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")
            f.write("## Overview\n\n")
            f.write("Victor organizes functionality into domain-specific verticals.\n\n")
            f.write("## Verticals\n\n")

            for vertical_name in sorted(verticals):
                f.write(f"### {vertical_name.title()}\n\n")

                # Try to import and document
                try:
                    module = importlib.import_module(f"victor.{vertical_name}")

                    # Get assistant class
                    from victor.core.verticals.base import VerticalBase

                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, VerticalBase)
                            and obj != VerticalBase
                        ):
                            # Get docstring
                            docstring = inspect.getdoc(obj)
                            if docstring:
                                f.write(f"{docstring}\n\n")

                            # Get tools
                            if hasattr(obj, "get_tools"):
                                try:
                                    tools = obj.get_tools()
                                    f.write(f"**Tools:** {', '.join(tools)}\n\n")
                                except Exception:
                                    pass

                            # Get system prompt
                            if hasattr(obj, "get_system_prompt"):
                                try:
                                    prompt = obj.get_system_prompt()
                                    f.write(f"**System Prompt:** {prompt[:200]}...\n\n")
                                except Exception:
                                    pass

                            break

                except ImportError as e:
                    f.write(f"Warning: Could not load vertical: {e}\n\n")

                f.write("---\n\n")

        print(f"  Wrote: {output_file}")

    def generate_protocol_docs(self) -> None:
        """Generate documentation for all protocols."""
        print("Generating protocol docs...")

        # Import all protocols
        from victor import protocols

        output_file = self.output_dir / "protocols.md"

        with open(output_file, "w") as f:
            f.write("# Protocol Documentation\n\n")
            f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")
            f.write("## Overview\n\n")
            f.write("Protocols define interfaces for Victor components.\n\n")
            f.write("## Protocols\n\n")

            # Get all protocol classes
            for name, obj in inspect.getmembers(protocols):
                if name.startswith("_"):
                    continue
                if not inspect.isclass(obj):
                    continue
                if "Protocol" not in name:
                    continue

                f.write(f"### {name}\n\n")

                # Get docstring
                docstring = inspect.getdoc(obj)
                if docstring:
                    f.write(f"{docstring}\n\n")

                # Get methods
                methods = []
                for method_name, method in inspect.getmembers(obj):
                    if method_name.startswith("_"):
                        continue
                    if callable(method):
                        try:
                            sig = inspect.signature(method)
                            methods.append(f"  - {method_name}{sig}")
                        except ValueError:
                            pass

                if methods:
                    f.write("**Methods:**\n\n")
                    for method in methods:
                        f.write(f"{method}\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"  Wrote: {output_file}")

    def generate_readme_update(self) -> None:
        """Generate README update with generated docs."""
        print("Generating README update...")

        output_file = self.output_dir / "README_UPDATE.md"

        with open(output_file, "w") as f:
            f.write("# Generated Documentation\n\n")
            f.write("This directory contains auto-generated documentation.\n\n")
            f.write("## Files\n\n")
            f.write("- `protocols.md` - Protocol documentation\n")
            f.write("- `coordinators.md` - Coordinator documentation\n")
            f.write("- `verticals.md` - Vertical documentation\n")
            f.write("- `victor_*.md` - API documentation by module\n\n")
            f.write("## Regenerating\n\n")
            f.write("To regenerate documentation:\n\n")
            f.write("```bash\n")
            f.write("python scripts/generate_docs.py --all --output docs/\n")
            f.write("```\n\n")

        print(f"  Wrote: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate documentation for Victor")
    parser.add_argument(
        "--type",
        type=str,
        choices=["api", "coordinators", "verticals", "protocols"],
        help="Type of documentation to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all documentation",
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Specific module to document (e.g., victor.protocols)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated"),
        help="Output directory (default: docs/generated)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    args = parser.parse_args()

    generator = DocumentationGenerator(args.output)

    if args.all:
        # Generate all documentation
        generator.generate_protocol_docs()
        generator.generate_coordinator_docs()
        generator.generate_vertical_docs()
        generator.generate_api_docs("victor.protocols")
        generator.generate_api_docs("victor.agent.coordinators")
        generator.generate_readme_update()

    elif args.type == "protocols":
        generator.generate_protocol_docs()

    elif args.type == "coordinators":
        generator.generate_coordinator_docs()

    elif args.type == "verticals":
        generator.generate_vertical_docs()

    elif args.type == "api":
        if args.module:
            generator.generate_api_docs(args.module)
        else:
            # Generate for key modules
            generator.generate_api_docs("victor.protocols")
            generator.generate_api_docs("victor.agent.coordinators")

    elif args.module:
        generator.generate_api_docs(args.module)

    else:
        parser.print_help()
        return 2

    print(f"\nDocumentation generated in: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
