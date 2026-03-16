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

"""Codebase scanner for file discovery, walking, and parsing.

Handles:
- Package layout detection (any language)
- Source file scanning and filtering
- Python AST-based parsing
- Generic regex-based parsing for non-Python files
- Class/component categorization
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.core.utils.ast_helpers import extract_base_classes

from victor.context.codebase_analyzer.models import ClassInfo, ModuleInfo

# Lazy load ignore patterns from external coding vertical package
_ignore_module, _ = import_module_with_fallback("victor.coding.codebase.ignore_patterns")
if (
    _ignore_module is not None
    and hasattr(_ignore_module, "DEFAULT_SKIP_DIRS")
    and hasattr(_ignore_module, "is_hidden_path")
    and hasattr(_ignore_module, "should_ignore_path")
):
    DEFAULT_SKIP_DIRS = _ignore_module.DEFAULT_SKIP_DIRS
    is_hidden_path = _ignore_module.is_hidden_path
    should_ignore_path = _ignore_module.should_ignore_path
    _IGNORE_PATTERNS_AVAILABLE = True
else:
    _IGNORE_PATTERNS_AVAILABLE = False
    DEFAULT_SKIP_DIRS = frozenset(
        {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "*.pyc",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            "*.egg-info",
        }
    )

    def is_hidden_path(path: Path) -> bool:
        return path.name.startswith(".")

    def should_ignore_path(
        path: Path, skip_dirs: frozenset, extra_skip_dirs: Optional[frozenset] = None
    ) -> bool:
        """Fallback implementation when coding ignore-patterns module is unavailable."""
        if path.name.startswith("."):
            return True
        if extra_skip_dirs and any(part in extra_skip_dirs for part in path.parts):
            return True
        return any(skip_dir in path.parts for skip_dir in skip_dirs)


logger = logging.getLogger(__name__)


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

# Config and documentation file extensions (counted for LOC but not parsed)
CONFIG_EXTENSIONS = {
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".ini": "INI",
    ".hocon": "HOCON",
    ".xml": "XML",
    ".md": "Markdown",
    ".txt": "Text",
    ".cfg": "Config",
    ".conf": "Config",
    ".props": "Properties",
}


class CodebaseScanner:
    """Scans codebases to discover and parse source files (language-agnostic)."""

    def __init__(
        self,
        root: Path,
        effective_skip_dirs: FrozenSet[str],
        include_dirs: Optional[List[str]] = None,
    ):
        self.root = root
        self.effective_skip_dirs = effective_skip_dirs
        self.include_dirs = include_dirs

    def detect_package_layout(self, analysis) -> None:
        """Detect the package/source layout (language-agnostic)."""

        def is_python_package(path: Path) -> bool:
            return path.is_dir() and (path / "__init__.py").exists()

        def is_source_directory(path: Path) -> bool:
            """Check if directory contains source files."""
            if not path.is_dir():
                return False
            for ext in LANGUAGE_EXTENSIONS.keys():
                if list(path.glob(f"*{ext}")):
                    return True
            return False

        # Find root-level packages/source directories
        source_dirs = []
        python_packages = []

        for item in self.root.iterdir():
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
            analysis.main_package = (
                main_candidates[0] if main_candidates else python_packages[0]
            )
            if has_src and "src" not in python_packages:
                analysis.deprecated_paths.append("src/")
        elif has_src:
            src_packages = [d.name for d in (self.root / "src").iterdir() if is_python_package(d)]
            if src_packages:
                analysis.main_package = f"src/{src_packages[0]}"
            elif is_source_directory(self.root / "src"):
                analysis.main_package = "src"
        elif has_lib and is_source_directory(self.root / "lib"):
            analysis.main_package = "lib"
        elif has_app and is_source_directory(self.root / "app"):
            analysis.main_package = "app"
        elif source_dirs:
            main_candidates = [d for d in source_dirs if d not in ("tests", "test", "spec")]
            analysis.main_package = main_candidates[0] if main_candidates else source_dirs[0]

    def analyze_source_files(self, analysis) -> None:
        """Analyze source files across all supported languages."""
        search_paths = []
        if self.include_dirs:
            for d in self.include_dirs:
                path = self.root / d
                if path.exists() and path.is_dir():
                    search_paths.append(path)

        if not search_paths:
            if analysis.main_package:
                main_path = self.root / analysis.main_package.replace("/", "/")
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
            self._scan_directory_for_sources(search_path, analysis)

    def _scan_directory_for_sources(self, directory: Path, analysis, max_depth: int = 5) -> None:
        """Scan directory for source files of any language."""
        for ext, lang in LANGUAGE_EXTENSIONS.items():
            for source_file in directory.rglob(f"*{ext}"):
                if should_ignore_path(source_file, skip_dirs=self.effective_skip_dirs):
                    continue

                try:
                    rel_path = source_file.relative_to(self.root)
                    if len(rel_path.parts) > max_depth:
                        continue
                except ValueError:
                    continue

                if ext == ".py":
                    module_info = self._parse_python_file(source_file, str(rel_path))
                else:
                    module_info = self._parse_generic_file(source_file, str(rel_path), lang)

                if module_info:
                    parts = rel_path.parts
                    if len(parts) > 1:
                        subpackage = parts[0]
                    else:
                        subpackage = "root"

                    if subpackage not in analysis.packages:
                        analysis.packages[subpackage] = []
                    analysis.packages[subpackage].append(module_info)

    def _parse_generic_file(
        self, file_path: Path, rel_path: str, language: str
    ) -> Optional[ModuleInfo]:
        """Parse any source file using regex patterns to extract components."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return None

        module_info = ModuleInfo(name=file_path.stem, path=rel_path)

        patterns = [
            r"(?:export\s+)?(?:public\s+|private\s+|abstract\s+)?class\s+([A-Z][a-zA-Z0-9_]*)",
            r"(?:export\s+)?interface\s+([A-Z][a-zA-Z0-9_]*)",
            r"(?:pub\s+)?struct\s+([A-Z][a-zA-Z0-9_]*)",
            r"(?:export\s+)?type\s+([A-Z][a-zA-Z0-9_]*)\s*=",
            r"(?:export\s+)?(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9_]*)",
            r"(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9_]*)",
            r"(?:defmodule|module)\s+([A-Z][a-zA-Z0-9_:]*)",
            r"(?:export\s+)?(?:const|function)\s+([A-Z][a-zA-Z0-9_]*)\s*[=\(]",
        ]

        for line_no, line in enumerate(content.split("\n"), 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    class_name = match.group(1)
                    desc = self._extract_inline_comment(line, content, line_no)
                    category = categorize_class(class_name, [])

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
        for comment_marker in ["//", "#", "--", "/*", "///"]:
            if comment_marker in line:
                idx = line.find(comment_marker)
                comment = line[idx + len(comment_marker) :].strip()
                if comment:
                    return comment[:60]

        lines = content.split("\n")
        if line_no > 1:
            prev_line = lines[line_no - 2].strip()
            for marker in ["///", "/**", "//", "#", '"""', "'''"]:
                if prev_line.startswith(marker):
                    return prev_line.lstrip(marker).strip("*/ ").strip()[:60]

        return None

    def _parse_python_file(self, file_path: Path, rel_path: str) -> Optional[ModuleInfo]:
        """Parse a Python file and extract class/function information."""
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
                module_info.functions.append(node.name)

        return module_info

    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract information from a class AST node."""
        base_classes = [b.rsplit(".", 1)[-1] for b in extract_base_classes(node)]

        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            doc = node.body[0].value.value
            if isinstance(doc, str):
                docstring = doc.split("\n")[0].strip()

        is_abstract = (
            any(
                isinstance(d, ast.Name) and d.id in ("abstractmethod", "ABC")
                for d in node.decorator_list
            )
            or "ABC" in base_classes
            or "Abstract" in node.name
        )

        category = categorize_class(node.name, base_classes)

        return ClassInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            base_classes=base_classes,
            docstring=docstring,
            is_abstract=is_abstract,
            category=category,
        )


def categorize_class(name: str, base_classes: List[str]) -> Optional[str]:
    """Categorize a class based on its name and base classes."""
    all_names = [name] + base_classes

    for category, patterns in KEY_CLASS_PATTERNS.items():
        for pattern in patterns:
            if any(pattern in n for n in all_names):
                return category
    return None
