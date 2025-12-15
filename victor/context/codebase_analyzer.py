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

"""Smart codebase analyzer for generating comprehensive init.md files.

This module analyzes Python codebases to extract:
- Package structure and layout
- Key classes and their locations (with line numbers)
- Architectural patterns (providers, tools, managers, etc.)
- CLI commands from pyproject.toml
- Configuration files and their purposes

Output location: .victor/init.md (configurable via settings.py)
"""

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.codebase.ignore_patterns import DEFAULT_SKIP_DIRS, is_hidden_path, should_ignore_path
from victor.config.settings import VICTOR_CONTEXT_FILE, get_project_paths

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
    """Analyzes codebases to extract structure and architecture (language-agnostic)."""

    # Patterns that indicate key architectural components (universal across languages)
    KEY_CLASS_PATTERNS = {
        "provider": ["Provider", "Backend", "Client", "Connector", "Adapter"],
        "tool": ["Tool", "Command", "Action", "Handler", "Executor"],
        "manager": ["Manager", "Orchestrator", "Controller", "Coordinator", "Director"],
        "model": ["Model", "Schema", "Entity", "Record", "DTO", "ViewModel"],
        "config": ["Config", "Settings", "Options", "Preferences", "Environment"],
        "base": ["Base", "Abstract", "Interface", "I[A-Z]"],  # IService, IRepository etc.
        "registry": ["Registry", "Repository", "Store", "Cache", "Factory"],
        "service": ["Service", "Worker", "Processor", "UseCase", "Interactor"],
        "component": ["Component", "Widget", "View", "Screen", "Page"],
        "middleware": ["Middleware", "Interceptor", "Filter", "Guard"],
        "router": ["Router", "Route", "Endpoint", "Controller"],
    }

    # Source file extensions by language
    LANGUAGE_EXTENSIONS = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript React",
        ".ts": "TypeScript",
        ".tsx": "TypeScript React",
        ".java": "Java",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".php": "PHP",
        ".cs": "C#",
        ".cpp": "C++",
        ".c": "C",
        ".swift": "Swift",
        ".dart": "Dart",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".vue": "Vue",
        ".svelte": "Svelte",
    }

    # Use shared default skip directories from ignore_patterns module
    # Hidden directories (starting with '.') are excluded automatically
    # by the shared should_ignore_path() utility
    SKIP_DIRS = DEFAULT_SKIP_DIRS

    def __init__(
        self,
        root_path: Optional[str] = None,
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ):
        """Initialize analyzer.

        Args:
            root_path: Root directory to analyze. Defaults to current directory.
            include_dirs: List of directories to include in the analysis.
            exclude_dirs: List of directories to exclude from the analysis.
        """
        self.root = Path(root_path).resolve() if root_path else Path.cwd()
        self.analysis = CodebaseAnalysis(project_name=self.root.name, root_path=self.root)
        self.include_dirs = include_dirs

        # Combine default and user-provided exclude dirs
        self.effective_skip_dirs = self.SKIP_DIRS.copy()
        if exclude_dirs:
            self.effective_skip_dirs.update(exclude_dirs)

    def analyze(self) -> CodebaseAnalysis:
        """Perform full codebase analysis (language-agnostic).

        Returns:
            Complete CodebaseAnalysis object.
        """
        logger.info(f"Analyzing codebase at {self.root}")

        # Step 1: Detect package/source layout (any language)
        self._detect_package_layout()

        # Step 2: Analyze source files (Python AST or regex for other languages)
        self._analyze_source_files()

        # Step 3: Identify key components
        self._identify_key_components()

        # Step 4: Extract entry points from config files
        self._extract_entry_points()

        # Step 5: Detect architecture patterns
        self._detect_architecture_patterns()

        # Step 6: Find config files
        self._find_config_files()

        return self.analysis

    def _detect_package_layout(self) -> None:
        """Detect the package/source layout (language-agnostic)."""

        def is_python_package(path: Path) -> bool:
            return path.is_dir() and (path / "__init__.py").exists()

        def is_source_directory(path: Path) -> bool:
            """Check if directory contains source files."""
            if not path.is_dir():
                return False
            for ext in self.LANGUAGE_EXTENSIONS.keys():
                if list(path.glob(f"*{ext}")):
                    return True
            return False

        # Find root-level packages/source directories
        source_dirs = []
        python_packages = []

        for item in self.root.iterdir():
            # Use shared ignore logic for consistency
            if should_ignore_path(item, skip_dirs=self.effective_skip_dirs):
                continue
            if is_python_package(item):
                python_packages.append(item.name)
            elif is_source_directory(item):
                source_dirs.append(item.name)

        has_src = (self.root / "src").is_dir()
        has_lib = (self.root / "lib").is_dir()
        has_app = (self.root / "app").is_dir()

        # Prioritize Python packages, then common source dirs
        if python_packages:
            main_candidates = [p for p in python_packages if not p.startswith("test")]
            self.analysis.main_package = (
                main_candidates[0] if main_candidates else python_packages[0]
            )
            if has_src and "src" not in python_packages:
                self.analysis.deprecated_paths.append("src/")
        elif has_src:
            # Check for packages inside src/
            src_packages = [d.name for d in (self.root / "src").iterdir() if is_python_package(d)]
            if src_packages:
                self.analysis.main_package = f"src/{src_packages[0]}"
            elif is_source_directory(self.root / "src"):
                self.analysis.main_package = "src"
        elif has_lib and is_source_directory(self.root / "lib"):
            self.analysis.main_package = "lib"
        elif has_app and is_source_directory(self.root / "app"):
            self.analysis.main_package = "app"
        elif source_dirs:
            # Use first source directory found
            main_candidates = [d for d in source_dirs if d not in ("tests", "test", "spec")]
            self.analysis.main_package = main_candidates[0] if main_candidates else source_dirs[0]

    def _analyze_source_files(self) -> None:
        """Analyze source files across all supported languages."""
        search_paths = []
        if self.include_dirs:
            for d in self.include_dirs:
                path = self.root / d
                if path.exists() and path.is_dir():
                    search_paths.append(path)

        if not search_paths:
            # Fallback to original logic if no include_dirs provided or none exist
            if self.analysis.main_package:
                main_path = self.root / self.analysis.main_package.replace("/", "/")
                if main_path.exists():
                    search_paths.append(main_path)
            if not search_paths:
                for common_dir in ["src", "lib", "app", "components", "pages", "api"]:
                    path = self.root / common_dir
                    if path.exists():
                        search_paths.append(path)
            if not search_paths:
                search_paths.append(self.root)

        for search_path in search_paths:
            self._scan_directory_for_sources(search_path)

    def _scan_directory_for_sources(self, directory: Path, max_depth: int = 5) -> None:
        """Scan directory for source files of any language."""
        for ext, lang in self.LANGUAGE_EXTENSIONS.items():
            for source_file in directory.rglob(f"*{ext}"):
                # Use shared ignore logic (handles hidden dirs and skip dirs)
                if should_ignore_path(source_file, skip_dirs=self.effective_skip_dirs):
                    continue

                # Check depth
                try:
                    rel_path = source_file.relative_to(self.root)
                    if len(rel_path.parts) > max_depth:
                        continue
                except ValueError:
                    continue

                # Parse based on language
                if ext == ".py":
                    module_info = self._parse_python_file(source_file, str(rel_path))
                else:
                    module_info = self._parse_generic_file(source_file, str(rel_path), lang)

                if module_info:
                    # Organize by subpackage/directory
                    parts = rel_path.parts
                    if len(parts) > 1:
                        subpackage = parts[0]
                    else:
                        subpackage = "root"

                    if subpackage not in self.analysis.packages:
                        self.analysis.packages[subpackage] = []
                    self.analysis.packages[subpackage].append(module_info)

    def _parse_generic_file(
        self, file_path: Path, rel_path: str, language: str
    ) -> Optional[ModuleInfo]:
        """Parse any source file using regex patterns to extract components.

        This is language-agnostic - detects class/interface/struct/function patterns.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return None

        module_info = ModuleInfo(name=file_path.stem, path=rel_path)

        # Universal patterns for class/component detection
        patterns = [
            # class ClassName, export class, public class, etc.
            r"(?:export\s+)?(?:public\s+|private\s+|abstract\s+)?class\s+([A-Z][a-zA-Z0-9_]*)",
            # interface IName or interface Name
            r"(?:export\s+)?interface\s+([A-Z][a-zA-Z0-9_]*)",
            # struct Name (Go, Rust, C)
            r"(?:pub\s+)?struct\s+([A-Z][a-zA-Z0-9_]*)",
            # type Name = (TypeScript type aliases)
            r"(?:export\s+)?type\s+([A-Z][a-zA-Z0-9_]*)\s*=",
            # enum Name
            r"(?:export\s+)?(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9_]*)",
            # trait Name (Rust)
            r"(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9_]*)",
            # module Name (Ruby, Elixir)
            r"(?:defmodule|module)\s+([A-Z][a-zA-Z0-9_:]*)",
            # React components: function ComponentName or const ComponentName =
            r"(?:export\s+)?(?:const|function)\s+([A-Z][a-zA-Z0-9_]*)\s*[=\(]",
            # Vue/Svelte component detection via filename
        ]

        for line_no, line in enumerate(content.split("\n"), 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    class_name = match.group(1)
                    # Get brief description from comment above or same line
                    desc = self._extract_inline_comment(line, content, line_no)
                    category = self._categorize_class(class_name, [])

                    class_info = ClassInfo(
                        name=class_name,
                        file_path=rel_path,
                        line_number=line_no,
                        base_classes=[],
                        docstring=desc,
                        is_abstract="abstract" in line.lower() or "interface" in line.lower(),
                        category=category,
                    )
                    module_info.classes.append(class_info)
                    break  # One match per line

        return module_info if module_info.classes else None

    def _extract_inline_comment(self, line: str, content: str, line_no: int) -> Optional[str]:
        """Extract comment from line or line above."""
        # Check for inline comment
        for comment_marker in ["//", "#", "--", "/*", "///"]:
            if comment_marker in line:
                idx = line.find(comment_marker)
                comment = line[idx + len(comment_marker) :].strip()
                if comment:
                    return comment[:60]

        # Check line above for doc comment
        lines = content.split("\n")
        if line_no > 1:
            prev_line = lines[line_no - 2].strip()
            for marker in ["///", "/**", "//", "#", '"""', "'''"]:
                if prev_line.startswith(marker):
                    return prev_line.lstrip(marker).strip("*/ ").strip()[:60]

        return None

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


def generate_smart_victor_md(
    root_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate comprehensive project context using codebase analysis.

    Works with Python projects (AST-based analysis) and falls back to
    language-agnostic analysis for non-Python projects.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Generated markdown content for .victor/init.md.
    """
    analyzer = CodebaseAnalyzer(root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs)
    analysis = analyzer.analyze()

    # If no Python package found, use language-agnostic analysis
    if not analysis.main_package and not analysis.key_components:
        return _generate_generic_victor_md(
            root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )

    sections = []

    # Header - use configurable file name
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
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


def _generate_generic_victor_md(
    root_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate init.md for non-Python projects using language-agnostic analysis.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Generated markdown content.
    """
    context = gather_project_context(
        root_path, max_files=100, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )
    _root = Path(root_path).resolve() if root_path else Path.cwd()

    sections = []

    # Header
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    sections.append("## Project Overview\n")
    if context["readme_content"]:
        # Extract first paragraph from README
        paragraphs = context["readme_content"].split("\n\n")
        for para in paragraphs:
            stripped = para.strip()
            if stripped and not stripped.startswith(("#", "![", "<", "[!", "---", "```", "|")):
                sections.append(f"**{context['project_name']}**: {stripped[:300]}\n")
                break
        else:
            sections.append(f"**{context['project_name']}**: [Add project description here]\n")
    else:
        sections.append(f"**{context['project_name']}**: [Add project description here]\n")

    # Languages detected
    if context["detected_languages"]:
        sections.append(f"**Languages**: {', '.join(context['detected_languages'][:5])}\n")

    # Package Layout
    sections.append("## Package Layout\n")
    sections.append("| Path | Description |")
    sections.append("|------|-------------|")

    # Add directories from structure
    for dir_path in context["directory_structure"][:15]:
        if "/" not in dir_path.rstrip("/"):  # Top-level directories only
            desc = _infer_directory_purpose(dir_path.rstrip("/"))
            sections.append(f"| `{dir_path}` | {desc} |")

    sections.append("")

    # Key Files (source files)
    if context["source_files"]:
        sections.append("## Key Files\n")
        for f in context["source_files"][:15]:
            sections.append(f"- `{f}`")
        sections.append("")

    # Common Commands based on detected project type
    sections.append("## Common Commands\n")
    sections.append("```bash")

    if "pyproject.toml" in context["config_files"] or "requirements.txt" in list(
        context["config_files"]
    ):
        sections.append("# Python project")
        sections.append("pip install -r requirements.txt")
        sections.append("python main.py")
    elif "package.json" in context["config_files"]:
        sections.append("# Node.js project")
        sections.append("npm install")
        sections.append("npm start")
    elif "Cargo.toml" in context["config_files"]:
        sections.append("# Rust project")
        sections.append("cargo build")
        sections.append("cargo run")
    elif "go.mod" in context["config_files"]:
        sections.append("# Go project")
        sections.append("go build")
        sections.append("go run .")
    else:
        sections.append("# Add your build/run commands here")

    sections.append("```\n")

    # Config files found
    if context["config_files"]:
        sections.append("## Configuration\n")
        sections.append(
            "Key config files: " + ", ".join(f"`{f}`" for f in context["config_files"][:5])
        )
        sections.append("")

    # Important Notes
    sections.append("## Important Notes\n")
    sections.append("- Review and customize this file based on your project specifics")
    sections.append("- Use `/init --deep` for LLM-powered comprehensive analysis")
    sections.append("")

    return "\n".join(sections)


def _infer_directory_purpose(dirname: str) -> str:
    """Infer the purpose of a directory from its name."""
    purposes = {
        "src": "Source code",
        "lib": "Library code",
        "app": "Application code",
        "api": "API endpoints",
        "components": "UI components",
        "pages": "Page components",
        "views": "View templates",
        "models": "Data models",
        "utils": "Utility functions",
        "helpers": "Helper functions",
        "config": "Configuration",
        "configs": "Configuration",
        "tests": "Test files",
        "test": "Test files",
        "spec": "Test specifications",
        "docs": "Documentation",
        "public": "Public/static assets",
        "static": "Static files",
        "assets": "Asset files",
        "scripts": "Script files",
        "bin": "Executable scripts",
        "data": "Data files",
        "migrations": "Database migrations",
        "styles": "Stylesheets",
        "css": "CSS styles",
    }
    return purposes.get(dirname.lower(), "Project files")


# Supported context file aliases for other AI coding tools
CONTEXT_FILE_ALIASES = {
    "CLAUDE.md": "Claude Code (Anthropic)",
    "GEMINI.md": "Gemini (Google AI Studio)",
    ".cursorrules": "Cursor IDE",
    ".windsurfrules": "Windsurf IDE",
    "AGENTS.md": "Generic AI agents",
}


def create_context_symlinks(
    root_path: Optional[str] = None,
    source_file: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Create symlinks from .victor/init.md to other context file names.

    This allows a single source of truth (.victor/init.md) to work with
    multiple AI coding tools that look for different filenames.

    Args:
        root_path: Root directory. Defaults to current directory.
        source_file: Source file to link from. Defaults to settings-configured path.
        aliases: List of alias names to create. If None, creates all supported aliases.

    Returns:
        Dict mapping alias name to status ('created', 'exists', 'failed', 'skipped')
    """
    root = Path(root_path).resolve() if root_path else Path.cwd()
    # Use settings-driven path by default
    if source_file is None:
        paths = get_project_paths(root)
        source = paths.project_context_file
        source_file = str(source.relative_to(root))
    else:
        source = root / source_file
    results: Dict[str, str] = {}

    if not source.exists():
        logger.warning(f"Source file {source} does not exist")
        return {"error": f"Source file {source_file} not found"}

    # Use all aliases if not specified
    target_aliases = aliases if aliases is not None else list(CONTEXT_FILE_ALIASES.keys())

    for alias in target_aliases:
        target = root / alias
        try:
            if target.exists():
                if target.is_symlink():
                    # Check if it points to our source
                    if target.resolve() == source.resolve():
                        results[alias] = "exists"
                    else:
                        results[alias] = "exists_different"
                else:
                    results[alias] = "exists_file"
            else:
                # Create relative symlink
                target.symlink_to(source_file)
                results[alias] = "created"
                logger.info(f"Created symlink: {alias} -> {source_file}")
        except OSError as e:
            results[alias] = f"failed: {e}"
            logger.warning(f"Failed to create symlink {alias}: {e}")

    return results


def remove_context_symlinks(
    root_path: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Remove symlinks to context files.

    Only removes files that are symlinks pointing to .victor/init.md or similar.
    Does not remove actual files.

    Args:
        root_path: Root directory. Defaults to current directory.
        aliases: List of alias names to remove. If None, checks all supported aliases.

    Returns:
        Dict mapping alias name to status ('removed', 'not_symlink', 'not_found')
    """
    root = Path(root_path).resolve() if root_path else Path.cwd()
    results: Dict[str, str] = {}

    target_aliases = aliases if aliases is not None else list(CONTEXT_FILE_ALIASES.keys())

    for alias in target_aliases:
        target = root / alias
        try:
            if not target.exists() and not target.is_symlink():
                results[alias] = "not_found"
            elif target.is_symlink():
                target.unlink()
                results[alias] = "removed"
                logger.info(f"Removed symlink: {alias}")
            else:
                results[alias] = "not_symlink"
        except OSError as e:
            results[alias] = f"failed: {e}"

    return results


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
                    # Skip lines that are only italics (like "*Any model. Any provider.*")
                    if stripped.startswith("*") and stripped.endswith("*") and "\n" not in stripped:
                        continue
                    # Strip markdown bold/italic markers to avoid formatting conflicts
                    result = stripped[:300]
                    result = result.strip("*_")  # Remove leading/trailing emphasis markers
                    result = result.replace("**", "").replace("__", "")  # Remove bold
                    return result
            except Exception:
                pass

    return ""


# =============================================================================
# LLM-Powered Analysis (Language-Agnostic)
# =============================================================================


def gather_project_context(
    root_path: Optional[str] = None,
    max_files: int = 50,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Gather project context for LLM analysis (works with any language).

    This function collects structural information about any project type
    without parsing language-specific syntax.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        max_files: Maximum number of source files to list.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Dict containing project structure information.
    """
    root = Path(root_path).resolve() if root_path else Path.cwd()

    # Directories to skip
    skip_dirs = {
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
        "target",
        ".next",
        ".nuxt",
        "coverage",
        ".cache",
    }
    if exclude_dirs:
        skip_dirs.update(exclude_dirs)

    # Project type detection by config files
    project_indicators = {
        "pyproject.toml": "Python (modern)",
        "setup.py": "Python (legacy)",
        "package.json": "JavaScript/TypeScript",
        "Cargo.toml": "Rust",
        "go.mod": "Go",
        "pom.xml": "Java (Maven)",
        "build.gradle": "Java/Kotlin (Gradle)",
        "Gemfile": "Ruby",
        "composer.json": "PHP",
        "mix.exs": "Elixir",
        "CMakeLists.txt": "C/C++ (CMake)",
        "Makefile": "Make-based",
        "pubspec.yaml": "Dart/Flutter",
        "Package.swift": "Swift",
        ".csproj": "C# (.NET)",
    }

    # Source file extensions by language
    source_extensions = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript React",
        ".jsx": "JavaScript React",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
        ".kt": "Kotlin",
        ".rb": "Ruby",
        ".php": "PHP",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C/C++ Header",
        ".cs": "C#",
        ".swift": "Swift",
        ".dart": "Dart",
        ".vue": "Vue",
        ".svelte": "Svelte",
    }

    context = {
        "project_name": root.name,
        "root_path": str(root),
        "detected_languages": [],
        "config_files": [],
        "directory_structure": [],
        "source_files": [],
        "readme_content": "",
        "main_config_content": "",
    }

    # Detect project type(s)
    for config_file, lang in project_indicators.items():
        if (root / config_file).exists():
            context["detected_languages"].append(lang)
            context["config_files"].append(config_file)

    # Get README content (first 2000 chars)
    for readme in ["README.md", "README.rst", "README.txt", "readme.md"]:
        readme_path = root / readme
        if readme_path.exists():
            try:
                context["readme_content"] = readme_path.read_text(encoding="utf-8")[:2000]
            except Exception:
                pass
            break

    # Get main config content
    main_configs = ["pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for config in main_configs:
        config_path = root / config
        if config_path.exists():
            try:
                context["main_config_content"] = config_path.read_text(encoding="utf-8")[:3000]
            except Exception:
                pass
            break

    # Collect directory structure (depth 2)
    def walk_dirs(path: Path, depth: int = 0, max_depth: int = 2) -> List[str]:
        dirs = []
        if depth > max_depth:
            return dirs
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith(".") or item.name in skip_dirs:
                    continue
                if item.is_dir():
                    rel_path = str(item.relative_to(root))
                    dirs.append(rel_path + "/")
                    dirs.extend(walk_dirs(item, depth + 1, max_depth))
        except PermissionError:
            pass
        return dirs

    if include_dirs:
        # Walk only included directories
        all_found_dirs = []
        for d in include_dirs:
            dir_path = root / d
            if dir_path.is_dir():
                all_found_dirs.extend(walk_dirs(dir_path))
        context["directory_structure"] = all_found_dirs[:100]
    else:
        context["directory_structure"] = walk_dirs(root)[:100]

    # Collect source files with extensions
    file_count = 0
    lang_counts: Dict[str, int] = {}

    search_paths = [root / d for d in include_dirs] if include_dirs else [root]

    for search_path in search_paths:
        if not search_path.is_dir():
            continue
        for item in search_path.rglob("*"):
            if file_count >= max_files:
                break
            if any(skip in item.parts for skip in skip_dirs):
                continue
            if item.is_file() and item.suffix in source_extensions:
                rel_path = str(item.relative_to(root))
                context["source_files"].append(rel_path)
                lang = source_extensions[item.suffix]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                file_count += 1
        if file_count >= max_files:
            break

    # Add detected languages from file extensions
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        if lang not in context["detected_languages"]:
            context["detected_languages"].append(f"{lang} ({count} files)")

    # Add content of key files
    context["key_files_content"] = {}

    # Prioritize key files
    key_file_candidates = []
    main_files = [
        f
        for f in context["source_files"]
        if "main" in f or "app" in f or "server" in f or "index" in f
    ]
    key_file_candidates.extend(main_files)

    # Add some of the largest files
    large_files = sorted(
        context["source_files"], key=lambda f: (root / f).stat().st_size, reverse=True
    )
    for f in large_files:
        if f not in key_file_candidates:
            key_file_candidates.append(f)

    key_files_to_read = key_file_candidates[:5]  # Read up to 5 key files

    for file_path in key_files_to_read:
        try:
            content = (root / file_path).read_text(encoding="utf-8")
            context["key_files_content"][file_path] = content[:8192]  # Limit content size (fits 10-12 parallel reads)
        except Exception:
            pass

    return context


def build_llm_prompt_for_victor_md(context: Dict[str, any]) -> str:
    """Build the prompt for LLM to generate project context file.

    Args:
        context: Project context from gather_project_context()

    Returns:
        Prompt string for the LLM.
    """

    # Static part of the prompt (the "80%")
    prompt_header = f"""You are an expert software architect tasked with creating a high-level "user manual" for an AI coding assistant named Victor.
This manual, named {VICTOR_CONTEXT_FILE}, will help Victor understand the project's structure, purpose, and conventions.
Your analysis must be comprehensive, distilling the provided information into a clear and actionable guide.

Analyze the following project data and generate the {VICTOR_CONTEXT_FILE} file.

**Output Rules:**
1.  **Start with the Header**: The response MUST begin with `# {VICTOR_CONTEXT_FILE}`.
2.  **Use Markdown**: Format the entire output in clean, readable Markdown. Use tables for structured data.
3.  **Be Factual**: Base your analysis exclusively on the provided context. Do not infer or add information not present in the data.
4.  **Be Concise**: Provide high-level summaries. Focus on the "what" and "why," not implementation details.
5.  **Follow the Structure**: Generate all of the requested sections.

---
"""

    # Dynamic part of the prompt (the "20%")
    dynamic_context = f"""
**Project Name**: {context['project_name']}
**Detected Languages**: {', '.join(context['detected_languages']) or 'Unknown'}

**Configuration Files**:
{chr(10).join('- ' + f for f in context['config_files']) or 'None detected'}

**Directory Structure Overview**:
```
{chr(10).join(context['directory_structure'][:50]) or 'Unable to determine'}
```

**Sample of Source Files**:
```
{chr(10).join(context['source_files'][:30]) or 'No source files found'}
```
"""

    if context.get("key_files_content"):
        dynamic_context += "\n**Content of Key Files**:\n"
        for file_path, content in context["key_files_content"].items():
            dynamic_context += f"--- `{file_path}` ---\n```\n{content}\n```\n\n"

    # Static part of the prompt (continued)
    prompt_footer = f"""
---

**Generation Task**:

Generate the full content for the `{VICTOR_CONTEXT_FILE}` file, adhering to all rules above. Create the following sections:

1.  `## Project Overview`
    -   Write a one-paragraph summary of the project's purpose, based on the README and file structure.

2.  `## Package Layout`
    -   Create a Markdown table with columns: `| Path | Status | Description |`.
    -   List the most important top-level directories.
    -   Infer the purpose of each directory (e.g., source code, tests, docs). Mark the main source directory as `**ACTIVE**`.

3.  `## Key Components`
    -   Identify 5-7 key files or classes from the provided context.
    -   Create a Markdown table with columns: `| Component | Path | Description |`.
    -   Provide a one-sentence description for each component.

4.  `## Common Commands`
    -   Based on the configuration files (e.g., `package.json`, `pyproject.toml`), list the essential commands for building, testing, and running the project inside a `bash` code block.

5.  `## Architecture Notes`
    -   From the file names and structure, infer 2-3 high-level architectural patterns. (e.g., "Provider Pattern for multiple LLMs", "REST API with FastAPI", "CLI application using Typer").

6.  `## Important Notes`
    -   Add 2-3 bullet points for an AI assistant to remember, such as "Always use `victor/` for core source code" or "Check `pyproject.toml` for dependencies".

Remember, output ONLY the generated `{VICTOR_CONTEXT_FILE}` content, starting with the `# {VICTOR_CONTEXT_FILE}` header.
"""

    return prompt_header + dynamic_context + prompt_footer


async def generate_victor_md_with_llm(
    provider,
    model: str,
    root_path: Optional[str] = None,
    max_files: int = 50,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate project context file using an LLM provider.

    This function works with any project type by gathering structural
    information and asking the LLM to analyze and document it.

    Args:
        provider: A Victor provider instance (BaseProvider)
        model: Model identifier to use for generation
        root_path: Root directory to analyze. Defaults to current directory.
        max_files: Maximum source files to include in context.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Generated content for .victor/init.md.
    """
    from victor.providers.base import Message

    # Gather project context
    context = gather_project_context(
        root_path, max_files, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )

    # Build prompt
    prompt = build_llm_prompt_for_victor_md(context)

    # Call the LLM
    messages = [Message(role="user", content=prompt)]

    expected_header = f"# {VICTOR_CONTEXT_FILE}"

    try:
        response = await provider.chat(messages, model=model)
        content = response.content.strip()

        # Ensure it starts with the header
        if not content.startswith(expected_header):
            content = f"{expected_header}\\n\\n" + content

        return content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Fall back to basic generation
        return generate_smart_victor_md(
            root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )


async def generate_victor_md_from_index(
    root_path: Optional[str] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate init.md from the SymbolStore (pre-indexed symbols).

    This uses the SQLite symbol store for fast, accurate init.md generation.
    The symbol store should be indexed first via `/init --index` or automatically
    during first run.

    Args:
        root_path: Root directory. Defaults to current directory.
        force: If True, re-index all files ignoring cache.

    Returns:
        Generated markdown content for .victor/init.md.
    """
    from victor.codebase.symbol_store import SymbolStore

    root = Path(root_path).resolve() if root_path else Path.cwd()
    store = SymbolStore(str(root), include_dirs=include_dirs, exclude_dirs=exclude_dirs)

    # Index if needed (quick operation if already indexed, unless force=True)
    await store.index_codebase(force=force)

    stats = store.get_stats()
    key_components = store.find_key_components(limit=15)
    patterns = store.get_detected_patterns()
    named_impls = store.find_named_implementations()
    perf_hints = store.find_performance_hints()

    sections = []

    # Header
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    sections.append("## Project Overview\n")
    readme_desc = _extract_readme_description(root)
    if readme_desc:
        sections.append(f"**{root.name}**: {readme_desc}\n")
    else:
        sections.append(f"**{root.name}**: [Add project description here]\n")

    # Languages
    if stats.get("files_by_language"):
        langs = [f"{lang} ({count})" for lang, count in stats["files_by_language"].items()]
        sections.append(f"**Languages**: {', '.join(langs)}\n")

    # Package Layout
    sections.append("## Package Layout\n")
    sections.append("| Path | Type | Description |")
    sections.append("|------|------|-------------|")

    # Infer main directories from key components
    dirs_seen = set()
    for comp in key_components:
        dir_parts = Path(comp.file_path).parts
        if len(dir_parts) > 1:
            main_dir = dir_parts[0]
            if main_dir not in dirs_seen:
                dirs_seen.add(main_dir)
                sections.append(f"| `{main_dir}/` | **ACTIVE** | Source code |")

    if (root / "tests").is_dir():
        sections.append("| `tests/` | Active | Unit and integration tests |")
    if (root / "docs").is_dir():
        sections.append("| `docs/` | Active | Documentation |")

    sections.append("")

    # Key Components (from indexed symbols)
    if key_components:
        sections.append("## Key Components\n")
        sections.append("| Component | Type | Path | Description |")
        sections.append("|-----------|------|------|-------------|")

        for comp in key_components[:12]:
            raw_desc = (
                comp.docstring or comp.category.title()
                if comp.category
                else comp.symbol_type.title()
            )
            # Truncate to first sentence or 120 chars, whichever is shorter
            desc = raw_desc.split("\n")[0][:120].strip()
            if len(raw_desc) > 120 and "." in desc:
                # Truncate at last sentence boundary
                desc = desc.rsplit(".", 1)[0] + "."
            path_with_line = f"`{comp.file_path}:{comp.line_number}`"
            sections.append(
                f"| {comp.name} | {comp.symbol_type} | {path_with_line} | {desc} |"
            )

        sections.append("")

    # Named Implementations (grouped by domain)
    if named_impls:
        sections.append("## Named Implementations\n")
        for domain, impls in sorted(named_impls.items()):
            if impls:
                sections.append(f"### {domain}\n")
                sections.append("| Name | Location | Description |")
                sections.append("|------|----------|-------------|")
                for impl in sorted(impls, key=lambda x: x["name"]):
                    desc = impl.get("description", "") or impl.get("primary_symbol", "")
                    # Format: SymbolName (file.py:line) for LLM navigation
                    line_ref = f":{impl['line']}" if impl.get("line") else ""
                    location = f"`{impl['path']}{line_ref}`"
                    sections.append(f"| **{impl['name']}** | {location} | {desc} |")
                sections.append("")

    # Performance Hints (extracted from docstrings)
    if perf_hints:
        sections.append("## Performance Hints\n")
        sections.append("*Extracted from docstrings and comments*\n")
        hint_count = 0
        for file_path, hints in sorted(perf_hints.items())[:10]:
            unique_hints = list({h["value"] for h in hints})[:3]
            if unique_hints:
                sections.append(f"- `{file_path}`: {', '.join(unique_hints)}")
                hint_count += 1
                if hint_count >= 8:
                    break
        sections.append("")

    # Architecture Patterns (from detected patterns)
    if patterns:
        sections.append("## Architecture\n")
        for i, pattern in enumerate(patterns[:8], 1):
            sections.append(f"{i}. **{pattern['name']}**: {pattern['description']}")
        sections.append("")

    # Symbol Summary by Type
    if stats.get("symbols_by_type"):
        sections.append("## Code Structure\n")
        for sym_type, count in stats["symbols_by_type"].items():
            # Proper pluralization
            plural = sym_type + "es" if sym_type.endswith("s") else sym_type + "s"
            sections.append(f"- {count} {plural}")
        sections.append("")

    # Common Commands (inferred from detected languages)
    sections.append("## Common Commands\n")
    sections.append("```bash")

    langs = stats.get("files_by_language", {})
    if "python" in langs:
        sections.append("# Python project")
        sections.append('pip install -e ".[dev]"')
        sections.append("pytest")
    if "typescript" in langs or "javascript" in langs:
        sections.append("# Node.js project")
        sections.append("npm install")
        sections.append("npm test")
    if "go" in langs:
        sections.append("# Go project")
        sections.append("go build")
        sections.append("go test ./...")
    if "rust" in langs:
        sections.append("# Rust project")
        sections.append("cargo build")
        sections.append("cargo test")

    if not any(lang in langs for lang in ["python", "typescript", "javascript", "go", "rust"]):
        sections.append("# Add your build/run commands here")

    sections.append("```\n")

    # Important Notes
    sections.append("## Important Notes\n")
    sections.append(
        f"- Indexed {stats.get('total_files', 0)} files, {stats.get('total_symbols', 0)} symbols"
    )
    sections.append("- Check component paths above for exact file:line references")
    sections.append("- Run `/init --update` to refresh after code changes")
    sections.append("")

    return "\n".join(sections)


async def extract_conversation_insights(root_path: Optional[str] = None) -> Dict[str, Any]:
    """Extract insights from conversation history to enhance init.md.

    Analyzes stored conversations to identify:
    - Frequently asked questions/topics
    - Common file references
    - Learned patterns and hot spots

    Args:
        root_path: Root directory containing .victor/conversation.db

    Returns:
        Dictionary with extracted insights
    """
    import sqlite3
    from collections import Counter

    root = Path(root_path).resolve() if root_path else Path.cwd()
    db_path = root / ".victor" / "conversation.db"

    if not db_path.exists():
        return {"error": "No conversation history found"}

    insights = {
        "common_topics": [],
        "hot_files": [],
        "faq": [],
        "learned_patterns": [],
        "session_count": 0,
        "message_count": 0,
    }

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get session and message counts
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
        insights["session_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        insights["message_count"] = cursor.fetchone()[0]

        # Extract common user queries (deduplicated, excluding benchmarks)
        cursor.execute(
            """
            SELECT content FROM messages
            WHERE role = 'user'
              AND content NOT LIKE '%<TOOL_OUTPUT%'
              AND content NOT LIKE '%Complete this Python function%'
              AND content NOT LIKE '%Complete the following Python%'
              AND length(content) BETWEEN 20 AND 500
        """
        )

        queries = [row[0] for row in cursor.fetchall()]
        query_counter = Counter()

        # Extract key topics from queries
        topic_keywords = [
            "component",
            "architecture",
            "test",
            "bug",
            "fix",
            "add",
            "create",
            "refactor",
            "improve",
            "explain",
            "how",
            "what",
            "why",
            "where",
            "error",
            "issue",
            "implement",
            "feature",
            "config",
            "setup",
        ]

        for query in queries:
            query_lower = query.lower()
            for keyword in topic_keywords:
                if keyword in query_lower:
                    query_counter[keyword] += 1

        insights["common_topics"] = query_counter.most_common(10)

        # Extract frequently referenced files from assistant responses
        cursor.execute(
            """
            SELECT content FROM messages
            WHERE role = 'assistant'
              AND (content LIKE '%.py%' OR content LIKE '%.ts%' OR content LIKE '%.js%')
        """
        )

        file_counter = Counter()
        file_pattern = re.compile(r"`([a-zA-Z_/]+\.(py|ts|js|go|rs))[:`]")

        for row in cursor.fetchall():
            matches = file_pattern.findall(row[0])
            for match in matches:
                file_path = match[0]
                # Normalize and count
                if "/" in file_path and not file_path.startswith("/"):
                    file_counter[file_path] += 1

        insights["hot_files"] = file_counter.most_common(15)

        # Get architectural patterns from patterns table
        cursor.execute(
            """
            SELECT pattern_name, pattern_type, COUNT(*) as count
            FROM patterns
            GROUP BY pattern_type
            ORDER BY count DESC
        """
        )
        insights["learned_patterns"] = [
            {"name": row[0], "type": row[1], "count": row[2]} for row in cursor.fetchall()
        ]

        # Extract FAQ-like questions (questions asked multiple times)
        cursor.execute(
            """
            SELECT content, COUNT(*) as times
            FROM messages
            WHERE role = 'user'
              AND content LIKE '%?%'
              AND content NOT LIKE '%Complete%function%'
              AND length(content) BETWEEN 15 AND 200
            GROUP BY content
            HAVING times > 1
            ORDER BY times DESC
            LIMIT 5
        """
        )
        insights["faq"] = [{"question": row[0], "times_asked": row[1]} for row in cursor.fetchall()]

        conn.close()

    except Exception as e:
        insights["error"] = str(e)

    return insights


async def extract_graph_insights(root_path: Optional[str] = None) -> Dict[str, Any]:
    """Extract insights from the code graph for init.md enrichment.

    Analyzes the code graph to detect:
    - Design patterns (Provider, Factory, Facade, etc.)
    - Most important symbols (PageRank)
    - Hub classes (high centrality)
    - File dependencies
    - Graph statistics

    Args:
        root_path: Root directory containing .victor/graph

    Returns:
        Dictionary with graph insights
    """
    from pathlib import Path
    from victor.tools.graph_tool import GraphAnalyzer, _load_graph
    from victor.codebase.graph.registry import create_graph_store

    root = Path(root_path).resolve() if root_path else Path.cwd()
    graph_dir = root / ".victor" / "graph"
    graph_db_path = graph_dir / "graph.db"

    insights: Dict[str, Any] = {
        "has_graph": False,
        "patterns": [],
        "important_symbols": [],
        "hub_classes": [],
        "stats": {},
        # Module-level insights
        "important_modules": [],
        "module_coupling": [],
    }

    if not graph_db_path.exists():
        return insights

    try:
        # Use direct SQL queries for fast stats instead of loading entire graph
        import sqlite3
        import json

        conn = sqlite3.connect(graph_db_path)
        try:
            # Get basic stats
            cur = conn.execute("SELECT COUNT(*) FROM nodes")
            total_nodes = cur.fetchone()[0]

            if total_nodes == 0:
                return insights

            cur = conn.execute("SELECT COUNT(*) FROM edges")
            total_edges = cur.fetchone()[0]

            # Get node type distribution
            cur = conn.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type")
            node_types = dict(cur.fetchall())

            # Get edge type distribution
            cur = conn.execute("SELECT type, COUNT(*) FROM edges GROUP BY type")
            edge_types = dict(cur.fetchall())

            insights["has_graph"] = True
            insights["stats"] = {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_types": node_types,
                "edge_types": edge_types,
            }

            # Get high-connectivity nodes (hub classes) via SQL
            cur = conn.execute(
                """
                SELECT n.name, n.type, n.file, n.line,
                       (SELECT COUNT(*) FROM edges WHERE src = n.node_id) +
                       (SELECT COUNT(*) FROM edges WHERE dst = n.node_id) as degree
                FROM nodes n
                WHERE n.type IN ('class', 'struct', 'interface')
                ORDER BY degree DESC
                LIMIT 5
            """
            )
            hub_results = cur.fetchall()
            insights["hub_classes"] = [
                {"name": r[0], "type": r[1], "file": r[2], "line": r[3], "degree": r[4]}
                for r in hub_results
                if r[4] >= 5
            ][:3]

            # Get most-called symbols (important functions)
            cur = conn.execute(
                """
                SELECT n.name, n.type, n.file, n.line,
                       (SELECT COUNT(*) FROM edges WHERE dst = n.node_id AND type = 'CALLS') as in_calls,
                       (SELECT COUNT(*) FROM edges WHERE src = n.node_id AND type = 'CALLS') as out_calls
                FROM nodes n
                WHERE n.type IN ('function', 'method', 'class')
                ORDER BY in_calls DESC
                LIMIT 8
            """
            )
            important_results = cur.fetchall()
            insights["important_symbols"] = [
                {
                    "name": r[0],
                    "type": r[1],
                    "file": r[2],
                    "line": r[3],
                    "in_degree": r[4],
                    "out_degree": r[5],
                    "score": r[4] / max(total_edges, 1),  # Simplified score
                }
                for r in important_results
                if r[4] > 0
            ]

            # Module-level analysis: aggregate to file level
            # Use REFERENCES edges (imports/dependencies) for richer module relationships
            # CALLS edges are sparse as they only track explicit function calls
            # Note: Hidden directories (.*) and archive/ are filtered at index time
            cur = conn.execute(
                """
                SELECT
                    src_n.file as src_module,
                    dst_n.file as dst_module,
                    COUNT(*) as ref_count
                FROM edges e
                JOIN nodes src_n ON e.src = src_n.node_id
                JOIN nodes dst_n ON e.dst = dst_n.node_id
                WHERE e.type = 'REFERENCES'
                  AND src_n.file != dst_n.file
                  AND src_n.file IS NOT NULL
                  AND dst_n.file IS NOT NULL
                  AND src_n.file NOT LIKE 'tests/%'
                  AND dst_n.file NOT LIKE 'tests/%'
                GROUP BY src_n.file, dst_n.file
                HAVING ref_count >= 2
                """
            )
            module_edges = cur.fetchall()

            if module_edges:
                # Build module adjacency for PageRank calculation
                module_in_degree: Dict[str, int] = defaultdict(int)
                module_out_degree: Dict[str, int] = defaultdict(int)
                module_weighted_in: Dict[str, int] = defaultdict(int)
                all_modules: Set[str] = set()

                for src_mod, dst_mod, count in module_edges:
                    all_modules.add(src_mod)
                    all_modules.add(dst_mod)
                    module_out_degree[src_mod] += 1
                    module_in_degree[dst_mod] += 1
                    module_weighted_in[dst_mod] += count

                # Sort by weighted in-degree (approximation of module importance)
                module_importance = [
                    (mod, module_weighted_in[mod], module_in_degree[mod], module_out_degree[mod])
                    for mod in all_modules
                ]
                module_importance.sort(key=lambda x: x[1], reverse=True)

                # Classify module roles
                insights["important_modules"] = []
                for mod, weighted_in, in_deg, out_deg in module_importance[:8]:
                    # Determine role
                    if in_deg > out_deg * 2 and in_deg >= 3:
                        role = "service"  # Many callers, few outgoing
                    elif out_deg > in_deg * 2 and out_deg >= 3:
                        role = "orchestrator"  # Calls many modules
                    elif in_deg >= 2 and out_deg >= 2:
                        role = "intermediary"  # Both caller and callee
                    elif in_deg > 0 and out_deg == 0:
                        role = "leaf"  # Terminal module
                    elif out_deg > 0 and in_deg == 0:
                        role = "entry"  # Entry point
                    else:
                        role = "peripheral"

                    insights["important_modules"].append({
                        "module": mod,
                        "weighted_importance": weighted_in,
                        "in_degree": in_deg,
                        "out_degree": out_deg,
                        "role": role,
                    })

                # Module coupling detection (high fan-in/fan-out)
                coupling_issues = []
                for mod, weighted_in, in_deg, out_deg in module_importance:
                    total_degree = in_deg + out_deg
                    if total_degree >= 8:  # High connectivity threshold
                        if in_deg > 5 and out_deg > 5:
                            pattern = "hub"
                        elif in_deg > 5:
                            pattern = "high_fan_in"
                        else:
                            pattern = "high_fan_out"
                        coupling_issues.append({
                            "module": mod,
                            "pattern": pattern,
                            "in_degree": in_deg,
                            "out_degree": out_deg,
                        })

                insights["module_coupling"] = coupling_issues[:5]

        finally:
            conn.close()

    except Exception as e:
        logger.warning(f"Failed to extract graph insights: {e}")

    return insights


async def generate_enhanced_init_md(
    root_path: Optional[str] = None,
    use_llm: bool = False,
    include_conversations: bool = True,
    on_progress: Optional[callable] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate init.md using symbol index, conversation insights, and optional LLM.

    Pipeline: Index → Learn (optional) → LLM enhance (optional)

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        use_llm: Whether to use LLM for enhancement (default: False)
        include_conversations: Whether to include conversation insights (default: True)
        on_progress: Optional callback: fn(stage: str, message: str)
        force: If True, re-index all files ignoring cache.

    Returns:
        Enhanced init.md content. Falls back gracefully if LLM fails.
    """
    import time

    from victor.providers.base import Message

    step_times: dict = {}
    step_start: float = 0

    def progress(stage: str, msg: str, complete: bool = False):
        nonlocal step_start
        if complete and step_start > 0:
            elapsed = time.time() - step_start
            step_times[stage] = elapsed
            if on_progress:
                on_progress(stage, f"✓ {msg} ({elapsed:.1f}s)")
        else:
            step_start = time.time()
            if on_progress:
                on_progress(stage, msg)

    # Step 1: Index - Use SymbolStore for base content
    progress("index", "Building symbol index...")
    base_content = await generate_victor_md_from_index(
        root_path, force=force, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )
    progress("index", "Symbol index built", complete=True)

    # Step 2: Learn - Add conversation insights
    if include_conversations:
        progress("learn", "Extracting conversation insights...")
        insights = await extract_conversation_insights(root_path)
        sessions = insights.get("session_count", 0)
        progress("learn", f"Insights extracted ({sessions} sessions)", complete=True)

        if sessions > 0:
            enhancements = ["\n## Learned from Conversations\n"]
            enhancements.append(
                f"*Based on {insights['session_count']} sessions, {insights['message_count']} messages*\n"
            )

            if insights.get("hot_files"):
                enhancements.append("### Frequently Referenced Files\n")
                for file_path, count in insights["hot_files"][:8]:
                    enhancements.append(f"- `{file_path}` ({count} references)")
                enhancements.append("")

            if insights.get("common_topics"):
                topics = [t[0] for t in insights["common_topics"][:6]]
                enhancements.append("### Common Topics\n")
                enhancements.append(f"Keywords: {', '.join(topics)}\n")

            if insights.get("faq"):
                enhancements.append("### Frequently Asked Questions\n")
                for faq in insights["faq"][:3]:
                    q = (
                        faq["question"][:100] + "..."
                        if len(faq["question"]) > 100
                        else faq["question"]
                    )
                    enhancements.append(f"- {q}")
                enhancements.append("")

            # Insert before Important Notes
            if "## Important Notes" in base_content:
                parts = base_content.split("## Important Notes")
                base_content = (
                    parts[0] + "\n".join(enhancements) + "\n## Important Notes" + parts[1]
                )
            else:
                base_content += "\n" + "\n".join(enhancements)

    # Step 2.5: Graph - Add graph-based insights (design patterns, important symbols)
    progress("graph", "Analyzing code graph...")
    graph_insights = await extract_graph_insights(root_path)
    if graph_insights.get("has_graph"):
        progress(
            "graph",
            f"Graph analyzed ({graph_insights['stats'].get('total_nodes', 0)} nodes)",
            complete=True,
        )

        graph_section = ["\n## Code Graph Insights\n"]
        graph_section.append(
            f"*{graph_insights['stats'].get('total_nodes', 0)} symbols, {graph_insights['stats'].get('total_edges', 0)} relationships*\n"
        )

        # Design patterns detected
        if graph_insights.get("patterns"):
            graph_section.append("### Detected Design Patterns\n")
            for p in graph_insights["patterns"][:5]:
                details = p.get("details", {})
                if p["pattern"] == "provider_strategy":
                    impls = details.get("implementations", [])
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('base_class', '')}` with {len(impls)} implementations"
                    )
                elif p["pattern"] == "facade":
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` ({details.get('incoming_calls', 0)} callers → {details.get('outgoing_calls', 0)} delegates)"
                    )
                elif p["pattern"] == "composition":
                    composed = details.get("composed_of", [])
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` composed of {len(composed)} components"
                    )
                elif p["pattern"] == "factory":
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` creates {details.get('creates', 0)} types"
                    )
                else:
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', details.get('base_class', ''))}`"
                    )
            graph_section.append("")

        # Most important symbols (PageRank)
        if graph_insights.get("important_symbols"):
            graph_section.append("### Most Important Symbols (PageRank)\n")
            graph_section.append("| Symbol | Type | Connections |")
            graph_section.append("|--------|------|-------------|")
            for sym in graph_insights["important_symbols"][:6]:
                conns = f"↓{sym['in_degree']} ↑{sym['out_degree']}"
                # Format: SymbolName (file.py:line) for LLM navigation
                line_ref = f":{sym['line']}" if sym.get("line") else ""
                location = f"({sym['file']}{line_ref})"
                graph_section.append(f"| `{sym['name']}` {location} | {sym['type']} | {conns} |")
            graph_section.append("")

        # Hub classes
        if graph_insights.get("hub_classes"):
            graph_section.append("### Hub Classes (High Connectivity)\n")
            for hub in graph_insights["hub_classes"]:
                # Format: ClassName (file.py:line) for LLM navigation
                line_ref = f":{hub['line']}" if hub.get("line") else ""
                location = f"({hub['file']}{line_ref})"
                graph_section.append(f"- `{hub['name']}` {location} - {hub['degree']} connections")
            graph_section.append("")

        # Module-level architecture (more meaningful than symbol-level for codebase navigation)
        if graph_insights.get("important_modules"):
            graph_section.append("### Key Modules (Architecture)\n")
            graph_section.append("| Module | Role | Connections |")
            graph_section.append("|--------|------|-------------|")
            for mod in graph_insights["important_modules"][:6]:
                role_emoji = {
                    "service": "🔧",
                    "orchestrator": "🎛️",
                    "intermediary": "↔️",
                    "leaf": "🍃",
                    "entry": "🚀",
                    "peripheral": "📦",
                }.get(mod["role"], "")
                conns = f"↓{mod['in_degree']} ↑{mod['out_degree']}"
                graph_section.append(
                    f"| `{mod['module']}` | {role_emoji} {mod['role']} | {conns} |"
                )
            graph_section.append("")

        # Module coupling warnings
        if graph_insights.get("module_coupling"):
            graph_section.append("### Coupling Hotspots\n")
            for coupling in graph_insights["module_coupling"][:3]:
                pattern_desc = {
                    "hub": "⚠️ High fan-in AND fan-out",
                    "high_fan_in": "Many callers",
                    "high_fan_out": "Calls many modules",
                }.get(coupling["pattern"], coupling["pattern"])
                graph_section.append(
                    f"- `{coupling['module']}` - {pattern_desc} "
                    f"(↓{coupling['in_degree']} ↑{coupling['out_degree']})"
                )
            graph_section.append("")

        # Insert before Important Notes
        if "## Important Notes" in base_content:
            parts = base_content.split("## Important Notes")
            base_content = parts[0] + "\n".join(graph_section) + "\n## Important Notes" + parts[1]
        else:
            base_content += "\n" + "\n".join(graph_section)
    else:
        progress("graph", "No graph data (run 'victor index' first)", complete=True)

    # Step 3: Deep - Use LLM to enhance content
    if not use_llm:
        return base_content

    progress("deep", "Enhancing with LLM analysis...")

    try:
        from victor.config.settings import Settings
        from victor.providers.registry import ProviderRegistry

        settings = Settings()
        provider_name = settings.default_provider
        model_name = settings.default_model
        provider_settings = settings.get_provider_settings(provider_name)
        provider = ProviderRegistry.create(provider_name, **provider_settings)

        if not provider:
            logger.warning(f"Could not get provider {provider_name}, skipping LLM")
            return base_content

        enhance_prompt = f"""You are an expert software architect reviewing a project documentation file.

Below is an auto-generated init.md file for a codebase. Your task is to:
1. Improve the descriptions to be more specific and actionable
2. Identify any key architectural patterns that were missed
3. Add meaningful relationships between components
4. Ensure the most important components are highlighted
5. Keep the same markdown structure but enhance the content quality

IMPORTANT RULES:
- Keep all existing sections and their structure
- Do NOT add generic advice - only project-specific insights
- Do NOT remove any existing content, only enhance it
- Keep the file concise - quality over quantity
- Focus on what makes this project unique

Here is the current init.md content:

```markdown
{base_content}
```

Return ONLY the enhanced markdown content, no explanations."""

        messages = [Message(role="user", content=enhance_prompt)]
        response = await provider.chat(messages, model=model_name)
        enhanced = response.content.strip()
        progress("deep", "LLM enhancement complete", complete=True)

        # Validate and clean response
        if enhanced.startswith("#") or enhanced.startswith("```"):
            if enhanced.startswith("```"):
                lines = enhanced.split("\n")
                lines = lines[1:] if lines[0].startswith("```") else lines
                lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
                enhanced = "\n".join(lines)
            await provider.close()
            return enhanced

        logger.warning("LLM response doesn't look like valid markdown")
        await provider.close()
        return base_content

    except Exception as e:
        progress("deep", f"LLM failed: {e}", complete=True)
        logger.warning(f"LLM enhancement failed: {e}, using base content")
        return base_content
