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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        ".next",
        ".nuxt",
        "target",
        "out",
        "coverage",
        ".cache",
    }

    def __init__(self, root_path: Optional[str] = None):
        """Initialize analyzer.

        Args:
            root_path: Root directory to analyze. Defaults to current directory.
        """
        self.root = Path(root_path) if root_path else Path.cwd()
        self.analysis = CodebaseAnalysis(project_name=self.root.name, root_path=self.root)

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
            if item.name in self.SKIP_DIRS or item.name.startswith("."):
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
        # Determine search paths
        search_paths = []
        if self.analysis.main_package:
            main_path = self.root / self.analysis.main_package.replace("/", "/")
            if main_path.exists():
                search_paths.append(main_path)

        # Also search common source directories if no main package
        if not search_paths:
            for common_dir in ["src", "lib", "app", "components", "pages", "api"]:
                path = self.root / common_dir
                if path.exists():
                    search_paths.append(path)

        # If still nothing, search root (but limit depth)
        if not search_paths:
            search_paths.append(self.root)

        for search_path in search_paths:
            self._scan_directory_for_sources(search_path)

    def _scan_directory_for_sources(self, directory: Path, max_depth: int = 5) -> None:
        """Scan directory for source files of any language."""
        for ext, lang in self.LANGUAGE_EXTENSIONS.items():
            for source_file in directory.rglob(f"*{ext}"):
                if any(skip in source_file.parts for skip in self.SKIP_DIRS):
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


def generate_smart_victor_md(root_path: Optional[str] = None) -> str:
    """Generate comprehensive project context using codebase analysis.

    Works with Python projects (AST-based analysis) and falls back to
    language-agnostic analysis for non-Python projects.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.

    Returns:
        Generated markdown content for .victor/init.md.
    """
    analyzer = CodebaseAnalyzer(root_path)
    analysis = analyzer.analyze()

    # If no Python package found, use language-agnostic analysis
    if not analysis.main_package and not analysis.key_components:
        return _generate_generic_victor_md(root_path)

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


def _generate_generic_victor_md(root_path: Optional[str] = None) -> str:
    """Generate init.md for non-Python projects using language-agnostic analysis.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.

    Returns:
        Generated markdown content.
    """
    context = gather_project_context(root_path, max_files=100)
    _root = Path(root_path) if root_path else Path.cwd()

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
    root = Path(root_path) if root_path else Path.cwd()
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
    root = Path(root_path) if root_path else Path.cwd()
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
                    # Found a text paragraph
                    return stripped[:300]
            except Exception:
                pass

    return ""


# =============================================================================
# LLM-Powered Analysis (Language-Agnostic)
# =============================================================================


def gather_project_context(root_path: Optional[str] = None, max_files: int = 50) -> Dict[str, any]:
    """Gather project context for LLM analysis (works with any language).

    This function collects structural information about any project type
    without parsing language-specific syntax.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        max_files: Maximum number of source files to list.

    Returns:
        Dict containing project structure information.
    """
    root = Path(root_path) if root_path else Path.cwd()

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

    context["directory_structure"] = walk_dirs(root)[:100]  # Limit to 100 dirs

    # Collect source files with extensions
    file_count = 0
    lang_counts: Dict[str, int] = {}
    for item in root.rglob("*"):
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

    # Add detected languages from file extensions
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        if lang not in context["detected_languages"]:
            context["detected_languages"].append(f"{lang} ({count} files)")

    return context


def build_llm_prompt_for_victor_md(context: Dict[str, any]) -> str:
    """Build the prompt for LLM to generate project context file.

    Args:
        context: Project context from gather_project_context()

    Returns:
        Prompt string for the LLM.
    """
    prompt = f"""Analyze this project and generate a comprehensive {VICTOR_CONTEXT_FILE} file.

PROJECT: {context['project_name']}
DETECTED LANGUAGES: {', '.join(context['detected_languages']) or 'Unknown'}

CONFIG FILES FOUND:
{chr(10).join('- ' + f for f in context['config_files']) or 'None detected'}

DIRECTORY STRUCTURE:
{chr(10).join(context['directory_structure'][:50]) or 'Unable to determine'}

SOURCE FILES (sample):
{chr(10).join(context['source_files'][:30]) or 'No source files found'}

README CONTENT:
{context['readme_content'][:1500] or '[No README found]'}

MAIN CONFIG CONTENT:
{context['main_config_content'][:2000] or '[No config found]'}

---

Generate a {VICTOR_CONTEXT_FILE} file with these sections:
1. **Project Overview**: Brief description of what the project does
2. **Package Layout**: Table showing important directories (use | Path | Status | Description | format)
3. **Key Components**: Main modules, classes, or files with their purposes
4. **Common Commands**: Build, test, run commands based on the detected build system
5. **Architecture**: High-level architecture notes (if determinable)
6. **Important Notes**: Any special considerations, deprecated paths, etc.

IMPORTANT:
- Be concise but comprehensive
- Include file paths where relevant
- Use markdown tables for structured data
- Don't make up information - only document what's evident from the structure
- Start with "# {VICTOR_CONTEXT_FILE}" header

Output ONLY the {VICTOR_CONTEXT_FILE} content, no explanations."""

    return prompt


async def generate_victor_md_with_llm(
    provider,
    model: str,
    root_path: Optional[str] = None,
    max_files: int = 50,
) -> str:
    """Generate project context file using an LLM provider.

    This function works with any project type by gathering structural
    information and asking the LLM to analyze and document it.

    Args:
        provider: A Victor provider instance (BaseProvider)
        model: Model identifier to use for generation
        root_path: Root directory to analyze. Defaults to current directory.
        max_files: Maximum source files to include in context.

    Returns:
        Generated content for .victor/init.md.
    """
    from victor.providers.base import Message

    # Gather project context
    context = gather_project_context(root_path, max_files)

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
            content = f"{expected_header}\n\n" + content

        return content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Fall back to basic generation
        return generate_smart_victor_md(root_path)


async def generate_victor_md_from_index(root_path: Optional[str] = None) -> str:
    """Generate init.md from the SymbolStore (pre-indexed symbols).

    This uses the SQLite symbol store for fast, accurate init.md generation.
    The symbol store should be indexed first via `/init --index` or automatically
    during first run.

    Args:
        root_path: Root directory. Defaults to current directory.

    Returns:
        Generated markdown content for .victor/init.md.
    """
    from victor.codebase.symbol_store import SymbolStore

    root = Path(root_path) if root_path else Path.cwd()
    store = SymbolStore(str(root))

    # Index if needed (quick operation if already indexed)
    await store.index_codebase()

    stats = store.get_stats()
    key_components = store.find_key_components(limit=15)
    patterns = store.get_detected_patterns()

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
            desc = (
                comp.docstring or comp.category.title()
                if comp.category
                else comp.symbol_type.title()
            )
            path_with_line = f"`{comp.file_path}:{comp.line_number}`"
            sections.append(
                f"| {comp.name} | {comp.symbol_type} | {path_with_line} | {desc[:50]} |"
            )

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

    root = Path(root_path) if root_path else Path.cwd()
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


async def generate_enhanced_init_md(root_path: Optional[str] = None) -> str:
    """Generate init.md enhanced with conversation history insights.

    Combines static codebase analysis with dynamic insights from
    conversation history to create a more useful project context.

    Args:
        root_path: Root directory to analyze

    Returns:
        Enhanced init.md content
    """
    # Get base init.md from symbol store
    base_content = await generate_victor_md_from_index(root_path)

    # Extract conversation insights
    insights = await extract_conversation_insights(root_path)

    if "error" in insights or insights["session_count"] == 0:
        return base_content

    # Build enhancement sections
    enhancements = []

    # Add conversation-derived insights section
    enhancements.append("\n## Learned from Conversations\n")
    enhancements.append(
        f"*Based on {insights['session_count']} sessions, {insights['message_count']} messages*\n"
    )

    # Hot files (frequently discussed)
    if insights.get("hot_files"):
        enhancements.append("### Frequently Referenced Files\n")
        for file_path, count in insights["hot_files"][:8]:
            enhancements.append(f"- `{file_path}` ({count} references)")
        enhancements.append("")

    # Common topics
    if insights.get("common_topics"):
        topics = [t[0] for t in insights["common_topics"][:6]]
        enhancements.append("### Common Topics\n")
        enhancements.append(f"Keywords: {', '.join(topics)}\n")

    # FAQ section
    if insights.get("faq"):
        enhancements.append("### Frequently Asked Questions\n")
        for faq in insights["faq"][:3]:
            q = faq["question"][:100] + "..." if len(faq["question"]) > 100 else faq["question"]
            enhancements.append(f"- {q}")
        enhancements.append("")

    # Insert enhancements before "Important Notes" section
    if "## Important Notes" in base_content:
        parts = base_content.split("## Important Notes")
        return parts[0] + "\n".join(enhancements) + "\n## Important Notes" + parts[1]
    else:
        return base_content + "\n" + "\n".join(enhancements)
