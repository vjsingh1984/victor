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

"""Dependency Graph Visualizer - Tool to visualize dependency relationships.

This tool provides:
- Show which components depend on which
- Identify circular dependencies
- Output as text-based graph or DOT format
- Analyze import dependencies

Usage:
    python -m victor.devtools.dependency_viz
    python -m victor.devtools.dependency_viz --module victor.agent
    python -m victor.devtools.dependency_viz --format dot > graph.dot
    python -m victor.devtools.dependency_viz --find-cycles
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """Information about a module."""

    name: str
    file_path: str
    imports: Set[str] = field(default_factory=set)
    imported_by: Set[str] = field(default_factory=set)
    is_internal: bool = False

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class CycleInfo:
    """Information about a circular dependency."""

    cycle: List[str]
    length: int

    def __str__(self) -> str:
        """Return string representation."""
        return " -> ".join(self.cycle) + " -> " + self.cycle[0]


class DependencyGraph:
    """Graph of module dependencies."""

    def __init__(self, victor_root: Optional[Path] = None):
        """Initialize the graph.

        Args:
            victor_root: Root directory of Victor project
        """
        if victor_root is None:
            current = Path(__file__).resolve().parent
            victor_root = current.parent.parent

        self.victor_root = Path(victor_root)
        self.modules: Dict[str, ModuleInfo] = {}
        self.adj_list: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adj_list: Dict[str, Set[str]] = defaultdict(set)

    def build_from_directory(self, directory: Path) -> None:
        """Build dependency graph by scanning directory.

        Args:
            directory: Directory to scan
        """
        if not directory.is_absolute():
            directory = self.victor_root / directory

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return

        # Find all Python files
        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            self._process_file(py_file)

        logger.info(f"Built graph with {len(self.modules)} modules")

    def _process_file(self, file_path: Path) -> None:
        """Process a single Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))

            # Get module name
            module_name = self._get_module_name(file_path)

            # Create module info
            if module_name not in self.modules:
                self.modules[module_name] = ModuleInfo(
                    name=module_name,
                    file_path=str(file_path.relative_to(self.victor_root)),
                    is_internal=file_path.as_posix().startswith(self.victor_root.as_posix()),
                )

            # Extract imports
            imports = self._extract_imports(tree, file_path)

            # Update graph
            for imp in imports:
                self.adj_list[module_name].add(imp)
                self.reverse_adj_list[imp].add(module_name)

                # Create module for imported module if it doesn't exist
                if imp not in self.modules:
                    self.modules[imp] = ModuleInfo(
                        name=imp,
                        file_path="<external>",
                        is_internal=False,
                    )

                self.modules[module_name].imports.add(imp)
                self.modules[imp].imported_by.add(module_name)

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path."""
        try:
            relative = file_path.relative_to(self.victor_root)
            parts = list(relative.parts[:-1])  # Remove filename
            if parts[-1] == "__pycache__":
                return None
            parts.append(relative.stem)  # Add module name
            return ".".join(parts)
        except ValueError:
            # File is not under victor_root
            return str(file_path)

    def _extract_imports(self, tree: ast.AST, file_path: Path) -> Set[str]:
        """Extract imports from AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Get the base module
                    base_module = node.module.split(".")[0]
                    imports.add(base_module)

        # Filter out standard library and common third-party packages
        imports = self._filter_imports(imports)

        return imports

    def _filter_imports(self, imports: Set[str]) -> Set[str]:
        """Filter out non-Victor imports."""
        # Standard library modules (partial list)
        stdlib_modules = {
            "os",
            "sys",
            "re",
            "json",
            "pathlib",
            "datetime",
            "typing",
            "dataclasses",
            "collections",
            "asyncio",
            "logging",
            "enum",
            "abc",
            "copy",
            "hashlib",
            "uuid",
            "time",
            "threading",
            "inspect",
            "functools",
            "itertools",
            "math",
            "random",
        }

        # Common third-party packages
        third_party_packages = {
            "pytest",
            "click",
            "rich",
            "typer",
            "pydantic",
            "httpx",
            "aiohttp",
            "fastapi",
            "sqlalchemy",
        }

        filtered = set()
        for imp in imports:
            # Keep victor imports
            if imp.startswith("victor"):
                filtered.add(imp)
            # Keep imports that look like they might be victor modules
            # (without 'victor.' prefix for relative imports)
            elif any(m.name == imp for m in self.modules.values() if m.is_internal):
                filtered.add(imp)
            # Filter out standard library and third-party packages
            elif imp not in stdlib_modules and imp not in third_party_packages:
                filtered.add(imp)

        return filtered

    def find_dependencies(self, module: str) -> Set[str]:
        """Find all dependencies of a module.

        Args:
            module: Module name

        Returns:
            Set of module names that this module depends on
        """
        return self.adj_list.get(module, set())

    def find_dependents(self, module: str) -> Set[str]:
        """Find all modules that depend on this module.

        Args:
            module: Module name

        Returns:
            Set of module names that depend on this module
        """
        return self.reverse_adj_list.get(module, set())

    def find_cycles(self) -> List[CycleInfo]:
        """Find circular dependencies using DFS.

        Returns:
            List of cycles found
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> None:
            """DFS traversal."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.adj_list.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(CycleInfo(cycle=cycle, length=len(cycle)))

            path.pop()
            rec_stack.remove(node)

        for module in self.modules:
            if module not in visited:
                dfs(module)

        return cycles

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        total_modules = len(self.modules)
        total_edges = sum(len(deps) for deps in self.adj_list.values())

        # Find modules with most dependencies
        most_deps = sorted(
            [(m, len(self.adj_list.get(m, set()))) for m in self.modules],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Find most depended-upon modules
        most_depended = sorted(
            [(m, len(self.reverse_adj_list.get(m, set()))) for m in self.modules],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "total_modules": total_modules,
            "total_dependencies": total_edges,
            "avg_dependencies": total_edges / total_modules if total_modules > 0 else 0,
            "most_dependencies": most_deps,
            "most_depended_upon": most_depended,
        }

    def print_text_graph(
        self,
        module: Optional[str] = None,
        max_depth: int = 3,
        show_dependents: bool = False,
    ) -> None:
        """Print text-based graph.

        Args:
            module: Root module (None for all modules)
            max_depth: Maximum depth to traverse
            show_dependents: Show dependents instead of dependencies
        """
        if module:
            if module not in self.modules:
                print(f"Module '{module}' not found")
                return

            print(f"\nDependency graph for {module}:")
            print("=" * 60)

            if show_dependents:
                print(f"\nModules that depend on {module}:")
                self._print_dependents(module, max_depth)
            else:
                print(f"\nModules that {module} depends on:")
                self._print_dependencies(module, max_depth)
        else:
            # Show overview
            stats = self.get_statistics()
            print("\nDependency Graph Overview")
            print("=" * 60)
            print(f"Total modules: {stats['total_modules']}")
            print(f"Total dependencies: {stats['total_dependencies']}")
            print(f"Avg dependencies per module: {stats['avg_dependencies']:.2f}")

            print("\nTop 10 modules with most dependencies:")
            for mod, count in stats["most_dependencies"]:
                print(f"  {mod}: {count}")

            print("\nTop 10 most depended-upon modules:")
            for mod, count in stats["most_depended_upon"]:
                print(f"  {mod}: {count}")

    def _print_dependencies(self, module: str, max_depth: int, depth: int = 0) -> None:
        """Print dependencies recursively."""
        if depth > max_depth:
            return

        deps = self.adj_list.get(module, set())

        for dep in sorted(deps):
            indent = "  " * (depth + 1)
            print(f"{indent}{dep}")
            if depth < max_depth:
                self._print_dependencies(dep, max_depth, depth + 1)

    def _print_dependents(self, module: str, max_depth: int, depth: int = 0) -> None:
        """Print dependents recursively."""
        if depth > max_depth:
            return

        dependents = self.reverse_adj_list.get(module, set())

        for dep in sorted(dependents):
            indent = "  " * (depth + 1)
            print(f"{indent}{dep}")
            if depth < max_depth:
                self._print_dependents(dep, max_depth, depth + 1)

    def export_dot(self, output_path: Path) -> None:
        """Export graph to DOT format.

        Args:
            output_path: Path to output DOT file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("digraph VictorDependencies {\n")
            f.write("  rankdir=LR;\n")
            f.write("  node [shape=box, style=rounded];\n")
            f.write("\n")

            # Group internal vs external modules
            f.write("  // Internal modules\n")
            for module, info in self.modules.items():
                if info.is_internal:
                    f.write(f'  "{module}" [style="filled", fillcolor=lightblue];\n')

            f.write("\n  // Dependencies\n")
            for module, deps in self.adj_list.items():
                for dep in deps:
                    f.write(f'  "{module}" -> "{dep}";\n')

            f.write("}\n")

        print(f"Exported DOT graph to {output_path}")
        print(f"To render: dot -Tpng {output_path} -o graph.png")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize Victor's dependency graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show overview of all dependencies
  python -m victor.devtools.dependency_viz

  # Show dependencies for a specific module
  python -m victor.devtools.dependency_viz --module victor.agent.orchestrator

  # Show dependents (what depends on this module)
  python -m victor.devtools.dependency_viz --module victor.protocols --show-dependents

  # Find circular dependencies
  python -m victor.devtools.dependency_viz --find-cycles

  # Export to DOT format
  python -m victor.devtools.dependency_viz --format dot > graph.dot

  # Analyze specific directory
  python -m victor.devtools.dependency_viz --directory victor/protocols
        """,
    )

    parser.add_argument(
        "-m",
        "--module",
        type=str,
        metavar="NAME",
        help="Show dependencies for a specific module",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        metavar="PATH",
        default="victor",
        help="Directory to analyze (default: victor)",
    )

    parser.add_argument(
        "--show-dependents",
        action="store_true",
        help="Show dependents instead of dependencies",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        metavar="N",
        default=3,
        help="Maximum depth to traverse (default: 3)",
    )

    parser.add_argument(
        "--find-cycles",
        action="store_true",
        help="Find circular dependencies",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "dot"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--root",
        type=str,
        metavar="PATH",
        help="Victor project root directory (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Create graph
    root_path = Path(args.root) if args.root else None
    graph = DependencyGraph(victor_root=root_path)

    # Build graph
    print(f"Building dependency graph from {args.directory}...")
    graph.build_from_directory(Path(args.directory))

    if not graph.modules:
        print("No modules found")
        return 1

    # Find cycles if requested
    if args.find_cycles:
        cycles = graph.find_cycles()
        if cycles:
            print(f"\nFound {len(cycles)} circular dependencies:\n")
            for i, cycle in enumerate(cycles, 1):
                print(f"{i}. {cycle}")
        else:
            print("\nNo circular dependencies found")
        return 0

    # Export or print
    if args.format == "dot":
        graph.export_dot(Path("graph.dot"))
    else:
        graph.print_text_graph(
            module=args.module,
            max_depth=args.max_depth,
            show_dependents=args.show_dependents,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
