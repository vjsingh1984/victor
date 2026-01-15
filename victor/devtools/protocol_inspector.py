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

"""Protocol Inspector - CLI tool to inspect all protocols in the Victor system.

This tool provides:
- List all protocols in the system
- Show protocol methods and their signatures
- Find where protocols are used
- Show which classes implement which protocols

Usage:
    python -m victor.devtools.protocol_inspector
    python -m victor.devtools.protocol_inspector --list
    python -m victor.devtools.protocol_inspector --protocol IToolSelector
    python -m victor.devtools.protocol_inspector --implementations ToolSelector
    python -m victor.devtools.protocol_inspector --usage IToolSelector
"""

from __future__ import annotations

import argparse
import ast
import inspect
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ProtocolInfo:
    """Information about a protocol."""

    name: str
    module: str
    file_path: str
    methods: List[str]
    docstring: Optional[str]
    line_number: int

    def __str__(self) -> str:
        """Return string representation."""
        methods_str = ", ".join(self.methods[:3])
        if len(self.methods) > 3:
            methods_str += f" ... ({len(self.methods)} total)"
        return f"{self.name} [{self.module}] - {methods_str}"


@dataclass
class ImplementationInfo:
    """Information about a class implementing a protocol."""

    class_name: str
    module: str
    file_path: str
    line_number: int
    implements_protocols: List[str]


class ProtocolInspector:
    """Inspector for Victor's protocol system."""

    def __init__(self, victor_root: Optional[Path] = None):
        """Initialize the inspector.

        Args:
            victor_root: Root directory of Victor project. If None, auto-detects.
        """
        if victor_root is None:
            # Auto-detect victor root
            current = Path(__file__).resolve().parent
            victor_root = current.parent.parent

        self.victor_root = Path(victor_root)
        self.protocols_dir = self.victor_root / "victor" / "protocols"
        self.protocols: Dict[str, ProtocolInfo] = {}
        self.implementations: Dict[str, List[ImplementationInfo]] = defaultdict(list)
        self.usage_map: Dict[str, List[str]] = defaultdict(list)

    def scan_protocols(self) -> None:
        """Scan all protocol definitions in victor/protocols."""
        if not self.protocols_dir.exists():
            logger.warning(f"Protocols directory not found: {self.protocols_dir}")
            return

        for py_file in self.protocols_dir.rglob("*.py"):
            if py_file.name == "__init__.py" or py_file.name.startswith("_"):
                continue

            self._scan_file_for_protocols(py_file)

        logger.info(f"Found {len(self.protocols)} protocols")

    def _scan_file_for_protocols(self, file_path: Path) -> None:
        """Scan a single file for protocol definitions."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if this is a Protocol class
                    is_protocol = self._is_protocol_class(node)

                    if is_protocol:
                        protocol_name = node.name
                        methods = self._extract_protocol_methods(node)

                        # Get module name
                        module = self._get_module_name(file_path)

                        # Get docstring
                        docstring = ast.get_docstring(node)

                        protocol_info = ProtocolInfo(
                            name=protocol_name,
                            module=module,
                            file_path=str(file_path.relative_to(self.victor_root)),
                            methods=methods,
                            docstring=docstring,
                            line_number=node.lineno,
                        )

                        self.protocols[protocol_name] = protocol_info

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")

    def _is_protocol_class(self, class_node: ast.ClassDef) -> bool:
        """Check if a class node is a Protocol."""
        # Check base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id == "Protocol":
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr == "Protocol":
                    return True

        # Check for @runtime_checkable decorator which often accompanies Protocol
        for decorator in class_node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == "runtime_checkable":
                    # Still need to check if it inherits from Protocol
                    return self._inherits_from_protocol(class_node)

        return False

    def _inherits_from_protocol(self, class_node: ast.ClassDef) -> bool:
        """Check if class inherits from Protocol."""
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id == "Protocol":
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr == "Protocol":
                    return True
        return False

    def _extract_protocol_methods(self, class_node: ast.ClassDef) -> List[str]:
        """Extract method signatures from a protocol class."""
        methods = []

        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                # Build method signature
                args = [arg.arg for arg in item.args.args]

                # Check for return annotation
                returns = ""
                if item.returns:
                    returns = f" -> {self._get_annotation_string(item.returns)}"

                signature = f"{item.name}({', '.join(args)}){returns}"
                methods.append(signature)

        return methods

    def _get_annotation_string(self, annotation: ast.expr) -> str:
        """Convert annotation AST node to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            value = self._get_annotation_string(annotation.value)
            slice_val = self._get_annotation_string(annotation.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        else:
            return ""

    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path."""
        relative = file_path.relative_to(self.victor_root)
        parts = list(relative.parts[:-1])  # Remove filename
        parts.append(relative.stem)  # Add module name
        return ".".join(parts)

    def find_implementations(self, protocol_name: Optional[str] = None) -> None:
        """Find classes that implement protocols.

        Args:
            protocol_name: Specific protocol to search for. If None, searches all.
        """
        if protocol_name and protocol_name not in self.protocols:
            logger.warning(f"Protocol '{protocol_name}' not found")
            return

        # Scan all Python files in victor
        for py_file in self.victor_root.rglob("*.py"):
            if "test" in str(py_file) or py_file.name.startswith("_"):
                continue

            self._scan_file_for_implementations(py_file, protocol_name)

        total_impls = sum(len(impls) for impls in self.implementations.values())
        logger.info(f"Found {total_impls} implementations")

    def _scan_file_for_implementations(
        self, file_path: Path, protocol_name: Optional[str] = None
    ) -> None:
        """Scan file for classes implementing protocols."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    implemented = self._check_protocol_implementation(
                        node, protocol_name
                    )

                    if implemented:
                        module = self._get_module_name(file_path)

                        impl_info = ImplementationInfo(
                            class_name=class_name,
                            module=module,
                            file_path=str(file_path.relative_to(self.victor_root)),
                            line_number=node.lineno,
                            implements_protocols=implemented,
                        )

                        for proto in implemented:
                            self.implementations[proto].append(impl_info)

        except Exception as e:
            logger.debug(f"Error scanning {file_path} for implementations: {e}")

    def _check_protocol_implementation(
        self, class_node: ast.ClassDef, protocol_name: Optional[str] = None
    ) -> List[str]:
        """Check if a class implements any protocols."""
        implemented = []

        # Check base classes
        for base in class_node.bases:
            base_name = None

            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr

            if base_name and base_name in self.protocols:
                if protocol_name is None or base_name == protocol_name:
                    implemented.append(base_name)

        # Check for Protocol subclasses in annotations
        # (Some classes use Protocol as a type hint without direct inheritance)

        return implemented

    def find_usage(self, protocol_name: str) -> None:
        """Find where a protocol is used in the codebase.

        Args:
            protocol_name: Protocol to search for
        """
        if protocol_name not in self.protocols:
            logger.warning(f"Protocol '{protocol_name}' not found")
            return

        # Scan all Python files
        for py_file in self.victor_root.rglob("*.py"):
            if "test" in str(py_file) or py_file.name.startswith("_"):
                continue

            self._scan_file_for_usage(py_file, protocol_name)

        logger.info(f"Found {len(self.usage_map[protocol_name])} usages")

    def _scan_file_for_usage(self, file_path: Path, protocol_name: str) -> None:
        """Scan file for protocol usage."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
                lines = source.splitlines()

            tree = ast.parse(source, filename=str(file_path))

            for node in ast.walk(tree):
                # Check for type annotations
                if isinstance(node, ast.AnnAssign):
                    if self._contains_protocol_reference(node.annotation, protocol_name):
                        usage_context = self._get_usage_context(
                            lines, node.lineno, file_path
                        )
                        self.usage_map[protocol_name].append(usage_context)

                # Check for function arguments
                elif isinstance(node, ast.arg):
                    if node.annotation and self._contains_protocol_reference(
                        node.annotation, protocol_name
                    ):
                        usage_context = self._get_usage_context(
                            lines, node.lineno, file_path
                        )
                        self.usage_map[protocol_name].append(usage_context)

                # Check for imports
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name == protocol_name:
                            usage_context = self._get_usage_context(
                                lines, node.lineno, file_path
                            )
                            self.usage_map[protocol_name].append(usage_context)

        except Exception as e:
            logger.debug(f"Error scanning {file_path} for usage: {e}")

    def _contains_protocol_reference(self, annotation: ast.expr, protocol_name: str) -> bool:
        """Check if annotation references the protocol."""
        if isinstance(annotation, ast.Name):
            return annotation.id == protocol_name
        elif isinstance(annotation, ast.Subscript):
            return self._contains_protocol_reference(annotation.value, protocol_name)
        elif isinstance(annotation, ast.Attribute):
            return annotation.attr == protocol_name
        return False

    def _get_usage_context(self, lines: List[str], line_no: int, file_path: Path) -> str:
        """Get context string for usage."""
        # Get the line (0-indexed)
        line = lines[line_no - 1] if line_no <= len(lines) else ""
        return f"{file_path.relative_to(self.victor_root)}:{line_no}: {line.strip()}"

    def list_protocols(self, verbose: bool = False) -> None:
        """List all discovered protocols.

        Args:
            verbose: Show detailed information
        """
        if not self.protocols:
            print("No protocols found")
            return

        print(f"\nFound {len(self.protocols)} protocols:\n")

        for name in sorted(self.protocols.keys()):
            protocol = self.protocols[name]

            if verbose:
                print(f"Protocol: {protocol.name}")
                print(f"  Module: {protocol.module}")
                print(f"  File: {protocol.file_path}:{protocol.line_number}")
                print(f"  Methods ({len(protocol.methods)}):")
                for method in protocol.methods:
                    print(f"    - {method}")
                if protocol.docstring:
                    print(f"  Description: {protocol.docstring.splitlines()[0]}")
                print()
            else:
                print(f"  {protocol}")

    def show_protocol_details(self, protocol_name: str) -> None:
        """Show detailed information about a specific protocol.

        Args:
            protocol_name: Name of the protocol
        """
        if protocol_name not in self.protocols:
            print(f"Protocol '{protocol_name}' not found")
            return

        protocol = self.protocols[protocol_name]

        print(f"\n{'=' * 60}")
        print(f"Protocol: {protocol.name}")
        print(f"{'=' * 60}")
        print(f"Module: {protocol.module}")
        print(f"File: {protocol.file_path}:{protocol.line_number}")

        if protocol.docstring:
            print(f"\nDescription:\n{protocol.docstring}")

        print(f"\nMethods ({len(protocol.methods)}):")
        for method in protocol.methods:
            print(f"  - {method}")

        # Show implementations
        if protocol_name in self.implementations:
            print(f"\nImplementations ({len(self.implementations[protocol_name])}):")
            for impl in self.implementations[protocol_name]:
                print(f"  - {impl.class_name} [{impl.module}]")
                print(f"    {impl.file_path}:{impl.line_number}")

        # Show usage
        if protocol_name in self.usage_map:
            print(f"\nUsed in ({len(self.usage_map[protocol_name])} locations):")
            for usage in self.usage_map[protocol_name][:10]:  # Limit to first 10
                print(f"  - {usage}")
            if len(self.usage_map[protocol_name]) > 10:
                print(f"  ... and {len(self.usage_map[protocol_name]) - 10} more")

        print()

    def export_json(self, output_path: Path) -> None:
        """Export protocol information to JSON.

        Args:
            output_path: Path to output JSON file
        """
        import json

        data = {
            "protocols": {
                name: {
                    "module": proto.module,
                    "file": proto.file_path,
                    "line": proto.line_number,
                    "methods": proto.methods,
                    "docstring": proto.docstring,
                }
                for name, proto in self.protocols.items()
            },
            "implementations": {
                name: [
                    {
                        "class": impl.class_name,
                        "module": impl.module,
                        "file": impl.file_path,
                        "line": impl.line_number,
                    }
                    for impl in impls
                ]
                for name, impls in self.implementations.items()
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Exported to {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect Victor's protocol system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all protocols
  python -m victor.devtools.protocol_inspector --list

  # Show detailed protocol information
  python -m victor.devtools.protocol_inspector --protocol IToolSelector --verbose

  # Find implementations of a protocol
  python -m victor.devtools.protocol_inspector --implementations IToolSelector

  # Find where a protocol is used
  python -m victor.devtools.protocol_inspector --usage IToolSelector

  # Export to JSON
  python -m victor.devtools.protocol_inspector --export protocols.json
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all protocols",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        metavar="NAME",
        help="Show details for a specific protocol",
    )

    parser.add_argument(
        "--implementations",
        type=str,
        metavar="NAME",
        help="Find implementations of a protocol",
    )

    parser.add_argument(
        "--usage",
        type=str,
        metavar="NAME",
        help="Find where a protocol is used",
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export protocol information to JSON",
    )

    parser.add_argument(
        "--root",
        type=str,
        metavar="PATH",
        help="Victor project root directory (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.list, args.protocol, args.implementations, args.usage, args.export]):
        parser.print_help()
        return 1

    # Create inspector
    root_path = Path(args.root) if args.root else None
    inspector = ProtocolInspector(victor_root=root_path)

    # Scan protocols
    print("Scanning protocols...")
    inspector.scan_protocols()

    # Perform requested action
    if args.list:
        inspector.list_protocols(verbose=args.verbose)

    if args.protocol:
        # Find implementations and usage first
        inspector.find_implementations(args.protocol)
        inspector.find_usage(args.protocol)
        inspector.show_protocol_details(args.protocol)

    if args.implementations:
        inspector.find_implementations(args.implementations)
        if args.implementations in inspector.implementations:
            impls = inspector.implementations[args.implementations]
            print(f"\nImplementations of {args.implementations} ({len(impls)}):")
            for impl in impls:
                print(f"  - {impl.class_name}")
                print(f"    Module: {impl.module}")
                print(f"    File: {impl.file_path}:{impl.line_number}")
                print()
        else:
            print(f"No implementations found for {args.implementations}")

    if args.usage:
        inspector.find_usage(args.usage)
        if args.usage in inspector.usage_map:
            usages = inspector.usage_map[args.usage]
            print(f"\nUsage of {args.usage} ({len(usages)} locations):")
            for usage in usages:
                print(f"  - {usage}")
        else:
            print(f"No usage found for {args.usage}")

    if args.export:
        inspector.find_implementations()
        inspector.export_json(Path(args.export))

    return 0


if __name__ == "__main__":
    sys.exit(main())
