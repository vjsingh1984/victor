#!/usr/bin/env python3
"""Vertical to YAML Migration Script.

This script analyzes existing vertical implementations and generates
YAML configuration files, reducing boilerplate by ~90%.

Usage:
    python scripts/migrate_vertical_to_yaml.py --vertical coding
    python scripts/migrate_vertical_to_yaml.py --all
    python scripts/migrate_vertical_to_yaml.py --vertical research --dry-run

Example Output:
    Analyzing ResearchAssistant...
    
    Methods Found:
    - get_tools() -> tools.list
    - get_system_prompt() -> core.system_prompt
    - get_stages() -> core.stages
    - get_middleware() -> extensions.middleware
    
    Generated YAML: victor/research/config/vertical.yaml
    
    Migration Report:
    - Lines before: 317 (assistant.py)
    - Lines after: 30 (assistant.py + YAML)
    - Reduction: 90.5%
    - Methods migrated: 12
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class VerticalAnalyzer:
    """Analyze vertical implementation to extract configuration."""

    def __init__(self, vertical_name: str, vertical_path: Path):
        self.vertical_name = vertical_name
        self.vertical_path = vertical_path
        self.assistant_path = vertical_path / "assistant.py"

    def analyze(self) -> Dict[str, Any]:
        """Analyze vertical implementation and extract configuration.

        Returns:
            Dictionary with extracted configuration and metadata
        """
        if not self.assistant_path.exists():
            raise FileNotFoundError(f"Assistant file not found: {self.assistant_path}")

        with open(self.assistant_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)

        analysis = {
            "vertical_name": self.vertical_name,
            "class_name": None,
            "metadata": {},
            "tools": [],
            "system_prompt": "",
            "stages": {},
            "middleware": [],
            "extensions": {},
            "line_count": len(source.splitlines()),
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis["class_name"] = node.name
                self._extract_class_metadata(node, analysis)
                self._extract_methods(node, analysis)

        return analysis

    def _extract_class_metadata(self, class_node: ast.ClassDef, analysis: Dict[str, Any]) -> None:
        """Extract class-level metadata."""
        for assign in class_node.body:
            if isinstance(assign, ast.Assign):
                for target in assign.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "name":
                            if isinstance(assign.value, ast.Constant):
                                analysis["metadata"]["name"] = assign.value.value
                        elif target.id == "description":
                            if isinstance(assign.value, ast.Constant):
                                analysis["metadata"]["description"] = assign.value.value
                        elif target.id == "version":
                            if isinstance(assign.value, ast.Constant):
                                analysis["metadata"]["version"] = assign.value.value

    def _extract_methods(self, class_node: ast.ClassDef, analysis: Dict[str, Any]) -> None:
        """Extract configuration from methods."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name

                if method_name == "get_tools":
                    self._extract_tools(item, analysis)
                elif method_name == "get_system_prompt":
                    self._extract_system_prompt(item, analysis)
                elif method_name == "get_stages":
                    self._extract_stages(item, analysis)
                elif method_name == "get_middleware":
                    self._extract_middleware(item, analysis)
                elif method_name.startswith("get_") and method_name != "get_config":
                    extension_name = method_name[4:]  # Remove 'get_' prefix
                    analysis["extensions"][extension_name] = {
                        "method": method_name,
                        "line": item.lineno,
                    }

    def _extract_tools(self, method_node: ast.FunctionDef, analysis: Dict[str, Any]) -> None:
        """Extract tool list from get_tools method."""
        # Try to extract tool list from return statement
        for stmt in ast.walk(method_node):
            if isinstance(stmt, ast.Return):
                tools = self._extract_list_from_node(stmt.value)
                if tools:
                    analysis["tools"] = tools

    def _extract_system_prompt(self, method_node: ast.FunctionDef, analysis: Dict[str, Any]) -> None:
        """Extract system prompt from get_system_prompt method."""
        for stmt in ast.walk(method_node):
            if isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Constant):
                    analysis["system_prompt"] = stmt.value.value

    def _extract_stages(self, method_node: ast.FunctionDef, analysis: Dict[str, Any]) -> None:
        """Extract stage definitions from get_stages method."""
        stages = {}

        for stmt in ast.walk(method_node):
            if isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Dict):
                    for key, value in zip(stmt.value.keys, stmt.value.values):
                        if isinstance(key, ast.Constant):
                            stage_name = key.value
                            stage_info = self._extract_stage_info(value)
                            stages[stage_name] = stage_info

        analysis["stages"] = stages

    def _extract_stage_info(self, node: ast.AST) -> Dict[str, Any]:
        """Extract stage information from StageDefinition call."""
        if isinstance(node, ast.Call):
            stage_info = {}

            # Look for keyword arguments
            for keyword in node.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    stage_info["name"] = keyword.value.value
                elif keyword.arg == "description" and isinstance(keyword.value, ast.Constant):
                    stage_info["description"] = keyword.value.value
                elif keyword.arg == "keywords":
                    stage_info["keywords"] = self._extract_list_from_node(keyword.value)
                elif keyword.arg == "next_stages":
                    stage_info["next_stages"] = self._extract_set_from_node(keyword.value)

            return stage_info

        return {}

    def _extract_middleware(self, method_node: ast.FunctionDef, analysis: Dict[str, Any]) -> None:
        """Extract middleware configuration."""
        middleware_list = []

        for stmt in ast.walk(method_node):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
                for elt in stmt.value.elts:
                    if isinstance(elt, ast.Call):
                        middleware_info = self._extract_middleware_info(elt)
                        if middleware_info:
                            middleware_list.append(middleware_info)

        analysis["middleware"] = middleware_list

    def _extract_middleware_info(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """Extract middleware information from instantiation."""
        if isinstance(node.func, ast.Name):
            class_name = node.func.name
            return {
                "class": class_name,
                "args": [],
                "kwargs": {},
            }
        elif isinstance(node.func, ast.Attribute):
            class_name = node.func.attr
            module = self._get_module_name(node.func.value)
            return {
                "class": f"{module}.{class_name}" if module else class_name,
                "args": [],
                "kwargs": {},
            }

        return None

    def _get_module_name(self, node: ast.AST) -> Optional[str]:
        """Extract module name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_list_from_node(self, node: ast.AST) -> List[str]:
        """Extract list of strings from AST node."""
        if isinstance(node, ast.List):
            result = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant):
                    result.append(elt.value)
            return result
        return []

    def _extract_set_from_node(self, node: ast.AST) -> List[str]:
        """Extract set of strings from AST node."""
        if isinstance(node, ast.Set):
            result = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant):
                    result.append(elt.value)
            return result
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "set":
                return []
        return []


class YAMLGenerator:
    """Generate YAML configuration from analysis."""

    def __init__(self, analysis: Dict[str, Any]):
        self.analysis = analysis

    def generate(self) -> str:
        """Generate YAML configuration.

        Returns:
            YAML configuration string
        """
        lines = []

        # Header
        lines.append(f"# {self.analysis['class_name']} - YAML Configuration")
        lines.append("#")
        lines.append(f"# Auto-generated from {self.analysis['vertical_name']}/assistant.py")
        lines.append("#")
        lines.append("")

        # Metadata
        lines.append("# =============================================================================")
        lines.append("# Metadata")
        lines.append("# =============================================================================")
        lines.append("metadata:")
        lines.append(f"  name: {self.analysis['metadata'].get('name', self.analysis['vertical_name'])}")
        if "version" in self.analysis["metadata"]:
            lines.append(f"  version: \"{self.analysis['metadata']['version']}\"")
        if "description" in self.analysis["metadata"]:
            lines.append(f"  description: \"{self.analysis['metadata']['description']}\"")
        lines.append("")

        # Core configuration
        lines.append("# =============================================================================")
        lines.append("# Core Configuration")
        lines.append("# =============================================================================")
        lines.append("core:")

        # Tools
        if self.analysis["tools"]:
            lines.append("  tools:")
            lines.append("    list:")
            for tool in self.analysis["tools"]:
                lines.append(f"      - {tool}")
            lines.append("")

        # System prompt
        if self.analysis["system_prompt"]:
            lines.append("  system_prompt:")
            lines.append("    source: inline")
            lines.append("    text: |")
            for line in self.analysis["system_prompt"].split("\n"):
                lines.append(f"      {line}")
            lines.append("")

        # Stages
        if self.analysis["stages"]:
            lines.append("  stages:")
            for stage_name, stage_info in self.analysis["stages"].items():
                lines.append(f"    {stage_name}:")
                if "name" in stage_info:
                    lines.append(f"      name: {stage_info['name']}")
                if "description" in stage_info:
                    lines.append(f"      description: \"{stage_info['description']}\"")
                if "keywords" in stage_info:
                    lines.append(f"      keywords: {stage_info['keywords']}")
                if "next_stages" in stage_info:
                    lines.append(f"      next_stages: {stage_info['next_stages']}")
            lines.append("")

        # Extensions
        if self.analysis["extensions"] or self.analysis["middleware"]:
            lines.append("# =============================================================================")
            lines.append("# Extensions")
            lines.append("# =============================================================================")
            lines.append("extensions:")

            if self.analysis["middleware"]:
                lines.append("  middleware:")
                for mw in self.analysis["middleware"]:
                    lines.append(f"    - class: {mw['class']}")
                    lines.append("      enabled: true")
                    lines.append("")

            if self.analysis["extensions"]:
                for ext_name, ext_info in self.analysis["extensions"].items():
                    lines.append(f"  # {ext_name}")
                    lines.append(f"  # Defined at line {ext_info['line']}")
                    lines.append("")

        return "\n".join(lines)


class MigrationReporter:
    """Generate migration reports."""

    def __init__(self, analysis: Dict[str, Any], yaml_path: Path):
        self.analysis = analysis
        self.yaml_path = yaml_path

    def generate_report(self) -> str:
        """Generate migration report.

        Returns:
            Report string
        """
        lines = []
        lines.append(f"Migration Report for {self.analysis['vertical_name']}")
        lines.append("=" * 60)
        lines.append("")

        # Lines of code
        lines.append(f"Lines before: {self.analysis['line_count']} (assistant.py)")
        lines.append(f"YAML generated: {self.yaml_path}")

        # Methods migrated
        methods_migrated = len(self.analysis["extensions"])
        if self.analysis["tools"]:
            methods_migrated += 1
        if self.analysis["system_prompt"]:
            methods_migrated += 1
        if self.analysis["stages"]:
            methods_migrated += 1
        if self.analysis["middleware"]:
            methods_migrated += 1

        lines.append(f"Methods migrated: {methods_migrated}")
        lines.append("")

        # Configuration summary
        lines.append("Configuration Extracted:")
        lines.append(f"  - Tools: {len(self.analysis['tools'])}")
        lines.append(f"  - Stages: {len(self.analysis['stages'])}")
        lines.append(f"  - Middleware: {len(self.analysis['middleware'])}")
        lines.append(f"  - Extensions: {len(self.analysis['extensions'])}")
        lines.append("")

        # Next steps
        lines.append("Next Steps:")
        lines.append(f"  1. Review generated YAML: {self.yaml_path}")
        lines.append("  2. Customize YAML for your needs")
        lines.append("  3. Remove redundant get_* methods from assistant.py")
        lines.append("  4. Test: python -m victor.{vertical_name}")
        lines.append("")

        return "\n".join(lines)


def migrate_vertical(
    vertical_name: str,
    dry_run: bool = False,
    output_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], str]:
    """Migrate vertical to YAML configuration.

    Args:
        vertical_name: Name of vertical to migrate
        dry_run: If True, don't write files
        output_path: Optional output path for YAML file

    Returns:
        Tuple of (analysis, report)
    """
    vertical_path = Path(f"victor/{vertical_name}")
    if not vertical_path.exists():
        raise ValueError(f"Vertical not found: {vertical_name}")

    # Analyze
    analyzer = VerticalAnalyzer(vertical_name, vertical_path)
    analysis = analyzer.analyze()

    # Generate YAML
    generator = YAMLGenerator(analysis)
    yaml_content = generator.generate()

    # Determine output path
    if output_path is None:
        output_path = vertical_path / "config" / "vertical.yaml"

    # Generate report
    reporter = MigrationReporter(analysis, output_path)
    report = reporter.generate_report()

    # Write file
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(yaml_content)
        print(f"Generated YAML: {output_path}")

    print(report)

    return analysis, report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate vertical implementation to YAML configuration"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        help="Vertical name to migrate (e.g., coding, research)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all verticals",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write files, just show what would be generated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for YAML file",
    )

    args = parser.parse_args()

    verticals = []
    if args.all:
        # Find all verticals
        victor_path = Path("victor")
        for item in victor_path.iterdir():
            if item.is_dir() and (item / "assistant.py").exists():
                verticals.append(item.name)
    elif args.vertical:
        verticals = [args.vertical]
    else:
        parser.print_help()
        parser.error("Either --vertical or --all must be specified")

    # Migrate each vertical
    for vertical in verticals:
        print(f"\nMigrating {vertical}...")
        try:
            migrate_vertical(vertical, dry_run=args.dry_run, output_path=args.output)
        except Exception as e:
            print(f"Error migrating {vertical}: {e}", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()
