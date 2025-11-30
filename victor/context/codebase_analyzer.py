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

"""Smart codebase analyzer for generating comprehensive .victor.md files.

This module analyzes Python codebases to extract:
- Package structure and layout
- Key classes and their locations (with line numbers)
- Architectural patterns (providers, tools, managers, etc.)
- CLI commands from pyproject.toml
- Configuration files and their purposes
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClassInfo:
    """Information about a discovered class."""

    name: str
    file_path: str
    line_number: int
    base_classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False
    category: Optional[str] = None  # e.g., "provider", "tool", "manager"


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    name: str
    path: str
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class CodebaseAnalysis:
    """Complete analysis of a codebase."""

    project_name: str
    root_path: Path
    main_package: Optional[str] = None
    deprecated_paths: List[str] = field(default_factory=list)
    packages: Dict[str, List[ModuleInfo]] = field(default_factory=dict)
    key_components: List[ClassInfo] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    cli_commands: List[str] = field(default_factory=list)
    architecture_patterns: List[str] = field(default_factory=list)
    config_files: List[Tuple[str, str]] = field(default_factory=list)  # (path, description)


class CodebaseAnalyzer:
    """Analyzes Python codebases to extract structure and architecture."""

    # Patterns that indicate key architectural components
    KEY_CLASS_PATTERNS = {
        "provider": ["Provider", "Backend", "Client", "Connector"],
        "tool": ["Tool", "Command", "Action", "Handler"],
        "manager": ["Manager", "Orchestrator", "Controller", "Coordinator"],
        "model": ["Model", "Schema", "Entity", "Record"],
        "config": ["Config", "Settings", "Options", "Preferences"],
        "base": ["Base", "Abstract", "Interface"],
        "registry": ["Registry", "Repository", "Store", "Cache"],
        "service": ["Service", "Worker", "Processor"],
    }

    # Files to skip during analysis
    SKIP_DIRS = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        "venv",
        "env",
        ".venv",
        "node_modules",
        ".tox",
        "build",
        "dist",
        "egg-info",
    }

    def __init__(self, root_path: Optional[str] = None):
        """Initialize analyzer.

        Args:
            root_path: Root directory to analyze. Defaults to current directory.
        """
        self.root = Path(root_path) if root_path else Path.cwd()
        self.analysis = CodebaseAnalysis(project_name=self.root.name, root_path=self.root)

    def analyze(self) -> CodebaseAnalysis:
        """Perform full codebase analysis.

        Returns:
            Complete CodebaseAnalysis object.
        """
        logger.info(f"Analyzing codebase at {self.root}")

        # Step 1: Detect package layout
        self._detect_package_layout()

        # Step 2: Parse Python files and extract classes
        self._analyze_python_files()

        # Step 3: Identify key components
        self._identify_key_components()

        # Step 4: Extract entry points from pyproject.toml
        self._extract_entry_points()

        # Step 5: Detect architecture patterns
        self._detect_architecture_patterns()

        # Step 6: Find config files
        self._find_config_files()

        return self.analysis

    def _detect_package_layout(self) -> None:
        """Detect the package layout (src vs flat)."""

        def is_python_package(path: Path) -> bool:
            return path.is_dir() and (path / "__init__.py").exists()

        # Find root-level Python packages
        root_packages = []
        for item in self.root.iterdir():
            if item.name in self.SKIP_DIRS or item.name.startswith("."):
                continue
            if is_python_package(item):
                root_packages.append(item.name)

        has_src = (self.root / "src").is_dir()

        if root_packages:
            # Prefer non-test packages
            main_candidates = [p for p in root_packages if not p.startswith("test")]
            self.analysis.main_package = main_candidates[0] if main_candidates else root_packages[0]

            if has_src:
                self.analysis.deprecated_paths.append("src/")
        elif has_src:
            # Check for packages inside src/
            src_packages = [d.name for d in (self.root / "src").iterdir() if is_python_package(d)]
            if src_packages:
                self.analysis.main_package = f"src/{src_packages[0]}"

    def _analyze_python_files(self) -> None:
        """Parse all Python files and extract class/function info."""
        if not self.analysis.main_package:
            return

        package_path = self.root / self.analysis.main_package.replace("/", "/")
        if not package_path.exists():
            return

        for py_file in package_path.rglob("*.py"):
            if any(skip in py_file.parts for skip in self.SKIP_DIRS):
                continue

            rel_path = py_file.relative_to(self.root)
            module_info = self._parse_python_file(py_file, str(rel_path))

            if module_info:
                # Organize by subpackage
                parts = rel_path.parts
                if len(parts) > 1:
                    subpackage = parts[1] if parts[0] == self.analysis.main_package else parts[0]
                else:
                    subpackage = "root"

                if subpackage not in self.analysis.packages:
                    self.analysis.packages[subpackage] = []
                self.analysis.packages[subpackage].append(module_info)

    def _parse_python_file(self, file_path: Path, rel_path: str) -> Optional[ModuleInfo]:
        """Parse a Python file and extract class/function information.

        Args:
            file_path: Absolute path to the file
            rel_path: Relative path from project root

        Returns:
            ModuleInfo or None if parsing fails
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return None

        module_info = ModuleInfo(
            name=file_path.stem,
            path=rel_path,
        )

        # Get module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            module_info.description = tree.body[0].value.value

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, rel_path)
                module_info.classes.append(class_info)
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                # Top-level function
                module_info.functions.append(node.name)

        return module_info

    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract information from a class AST node."""
        # Get base class names
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(base.attr)

        # Get docstring
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            doc = node.body[0].value.value
            if isinstance(doc, str):
                # Get first line only
                docstring = doc.split("\n")[0].strip()

        # Check if abstract
        is_abstract = (
            any(
                isinstance(d, ast.Name) and d.id in ("abstractmethod", "ABC")
                for d in node.decorator_list
            )
            or "ABC" in base_classes
            or "Abstract" in node.name
        )

        # Categorize the class
        category = self._categorize_class(node.name, base_classes)

        return ClassInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            base_classes=base_classes,
            docstring=docstring,
            is_abstract=is_abstract,
            category=category,
        )

    def _categorize_class(self, name: str, base_classes: List[str]) -> Optional[str]:
        """Categorize a class based on its name and base classes."""
        all_names = [name] + base_classes

        for category, patterns in self.KEY_CLASS_PATTERNS.items():
            for pattern in patterns:
                if any(pattern in n for n in all_names):
                    return category
        return None

    def _identify_key_components(self) -> None:
        """Identify the most important classes in the codebase."""
        all_classes: List[ClassInfo] = []

        for modules in self.analysis.packages.values():
            for module in modules:
                all_classes.extend(module.classes)

        # Priority scoring for key components
        def score_class(cls: ClassInfo) -> int:
            score = 0
            # Base/abstract classes are important
            if cls.is_abstract or "Base" in cls.name:
                score += 10
            # Categorized classes are more important
            if cls.category:
                score += 5
                if cls.category in ("manager", "provider", "registry"):
                    score += 5
            # Orchestrator is the main coordinator - highest priority
            if "Orchestrator" in cls.name:
                score += 15
            # Classes with docstrings are better documented
            if cls.docstring:
                score += 2
            return score

        # Sort by score and take top components
        scored = [(score_class(c), c) for c in all_classes if c.category]
        scored.sort(key=lambda x: -x[0])

        # Take top 15 most important
        self.analysis.key_components = [c for _, c in scored[:15]]

    def _extract_entry_points(self) -> None:
        """Extract CLI entry points from pyproject.toml."""
        pyproject = self.root / "pyproject.toml"
        if not pyproject.exists():
            return

        try:
            content = pyproject.read_text(encoding="utf-8")

            # Parse [project.scripts] section
            scripts_match = re.search(r"\[project\.scripts\](.*?)(?=\[|\Z)", content, re.DOTALL)
            if scripts_match:
                scripts_section = scripts_match.group(1)
                for line in scripts_section.strip().split("\n"):
                    if "=" in line:
                        parts = line.split("=", 1)
                        cmd = parts[0].strip().strip('"')
                        target = parts[1].strip().strip('"')
                        self.analysis.entry_points[cmd] = target

            # Extract dev dependencies for CLI commands
            if "[project.optional-dependencies]" in content:
                # Common test/dev commands
                if "pytest" in content:
                    self.analysis.cli_commands.append("pytest")
                if "black" in content:
                    self.analysis.cli_commands.append("black .")
                if "ruff" in content:
                    self.analysis.cli_commands.append("ruff check .")
                if "mypy" in content:
                    self.analysis.cli_commands.append("mypy " + (self.analysis.main_package or "."))

        except Exception as e:
            logger.debug(f"Failed to parse pyproject.toml: {e}")

    def _detect_architecture_patterns(self) -> None:
        """Detect common architectural patterns in the codebase."""
        patterns = []

        # Check for provider pattern
        provider_classes = [c for c in self.analysis.key_components if c.category == "provider"]
        if len(provider_classes) >= 2:
            base = next((c for c in provider_classes if c.is_abstract or "Base" in c.name), None)
            if base:
                patterns.append(
                    f"Provider Pattern: Base class `{base.name}` ({base.file_path}:{base.line_number})"
                )

        # Check for tool/command pattern
        tool_classes = [c for c in self.analysis.key_components if c.category == "tool"]
        if len(tool_classes) >= 2:
            base = next((c for c in tool_classes if c.is_abstract or "Base" in c.name), None)
            if base:
                patterns.append(
                    f"Tool/Command Pattern: Base class `{base.name}` ({base.file_path}:{base.line_number})"
                )

        # Check for registry pattern
        registry_classes = [c for c in self.analysis.key_components if c.category == "registry"]
        if registry_classes:
            patterns.append(f"Registry Pattern: {len(registry_classes)} registries found")

        # Check for manager/orchestrator
        manager_classes = [c for c in self.analysis.key_components if c.category == "manager"]
        if manager_classes:
            main = manager_classes[0]
            patterns.append(
                f"Orchestrator/Manager: `{main.name}` ({main.file_path}:{main.line_number})"
            )

        # Check for config pattern
        config_classes = [c for c in self.analysis.key_components if c.category == "config"]
        if config_classes:
            patterns.append(f"Configuration: {len(config_classes)} config classes (Pydantic-style)")

        self.analysis.architecture_patterns = patterns

    def _find_config_files(self) -> None:
        """Find important configuration files."""
        config_patterns = [
            ("pyproject.toml", "Project configuration, dependencies, and build settings"),
            ("setup.py", "Legacy Python package setup"),
            ("setup.cfg", "Package configuration"),
            (".env.example", "Environment variable template"),
            ("docker-compose.yml", "Docker service definitions"),
            ("Dockerfile", "Container build instructions"),
            ("Makefile", "Build automation"),
            (".github/workflows/*.yml", "GitHub Actions CI/CD"),
            ("requirements.txt", "Python dependencies"),
        ]

        for pattern, description in config_patterns:
            if "*" in pattern:
                # Glob pattern
                matches = list(self.root.glob(pattern))
                if matches:
                    self.analysis.config_files.append((pattern, description))
            else:
                if (self.root / pattern).exists():
                    self.analysis.config_files.append((pattern, description))


def generate_smart_victor_md(root_path: Optional[str] = None) -> str:
    """Generate a comprehensive .victor.md using codebase analysis.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.

    Returns:
        Generated markdown content.
    """
    analyzer = CodebaseAnalyzer(root_path)
    analysis = analyzer.analyze()

    sections = []

    # Header
    sections.append("# .victor.md\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    sections.append("## Project Overview\n")

    # Try to get description from README
    readme_desc = _extract_readme_description(analysis.root_path)
    if readme_desc:
        sections.append(f"**{analysis.project_name}**: {readme_desc}\n")
    else:
        sections.append(f"**{analysis.project_name}**: [Add project description here]\n")

    # Package Layout
    sections.append("## Package Layout\n")
    sections.append("**IMPORTANT**: Use the correct directory paths:\n")

    layout_lines = []
    layout_lines.append("| Path | Status | Description |")
    layout_lines.append("|------|--------|-------------|")

    if analysis.main_package:
        layout_lines.append(
            f"| `{analysis.main_package}/` | **ACTIVE** | Main package - all source code |"
        )

    for deprecated in analysis.deprecated_paths:
        layout_lines.append(f"| `{deprecated}` | **DEPRECATED** | Legacy - DO NOT USE |")

    if (analysis.root_path / "tests").is_dir():
        layout_lines.append("| `tests/` | Active | Unit and integration tests |")

    if (analysis.root_path / "docs").is_dir():
        layout_lines.append("| `docs/` | Active | Documentation |")

    sections.append("\n".join(layout_lines) + "\n")

    # Key Components
    if analysis.key_components:
        sections.append("## Key Components\n")
        sections.append("| Component | Path | Description |")
        sections.append("|-----------|------|-------------|")

        for comp in analysis.key_components[:10]:  # Top 10
            desc = comp.docstring or f"{comp.category.title() if comp.category else 'Class'}"
            path_with_line = f"`{comp.file_path}:{comp.line_number}`"
            sections.append(f"| {comp.name} | {path_with_line} | {desc[:60]} |")

        sections.append("")

    # Common Commands
    sections.append("## Common Commands\n")
    sections.append("```bash")
    sections.append("# Install with dev dependencies")
    sections.append('pip install -e ".[dev]"')

    # Add entry points
    if analysis.entry_points:
        sections.append("")
        sections.append("# Run the application")
        for cmd in list(analysis.entry_points.keys())[:2]:
            sections.append(cmd)

    # Add dev commands
    if analysis.cli_commands:
        sections.append("")
        sections.append("# Development")
        for cmd in analysis.cli_commands:
            sections.append(cmd)

    sections.append("```\n")

    # Architecture Notes
    if analysis.architecture_patterns:
        sections.append("## Architecture\n")
        for i, pattern in enumerate(analysis.architecture_patterns, 1):
            sections.append(f"{i}. {pattern}")
        sections.append("")

    # Package Structure
    if analysis.packages:
        sections.append("## Package Structure\n")
        for pkg_name, modules in sorted(analysis.packages.items()):
            if pkg_name == "root":
                continue
            class_count = sum(len(m.classes) for m in modules)
            sections.append(f"- **{pkg_name}/**: {len(modules)} modules, {class_count} classes")
        sections.append("")

    # Important Notes
    sections.append("## Important Notes\n")
    if analysis.deprecated_paths:
        sections.append(
            f"- **ALWAYS** use `{analysis.main_package}/` not `{analysis.deprecated_paths[0]}`"
        )
    sections.append("- Check component paths above for exact file:line references")

    if analysis.config_files:
        sections.append(
            "- Key config files: " + ", ".join(f"`{f}`" for f, _ in analysis.config_files[:4])
        )

    sections.append("")

    return "\n".join(sections)


def _extract_readme_description(root: Path) -> str:
    """Extract project description from README."""
    readme_files = ["README.md", "README.rst", "README.txt"]

    for readme in readme_files:
        readme_path = root / readme
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8")
                paragraphs = content.split("\n\n")

                for para in paragraphs:
                    stripped = para.strip()
                    # Skip empty, headers, images, HTML, badges
                    if not stripped:
                        continue
                    if stripped.startswith(("#", "![", "<", "[!", "---", "```", "|")):
                        continue
                    if stripped.startswith("[") and stripped.endswith(")"):
                        continue
                    # Found a text paragraph
                    return stripped[:300]
            except Exception:
                pass

    return ""
