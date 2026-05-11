"""Symbol extraction from source files.

Extracted from codebase_analyzer.py (SRP decomposition).
Handles file parsing, AST extraction, and class/module info collection.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from victor_coding.codebase_analyzer import ClassInfo, CodebaseAnalysis, ModuleInfo

try:
    from victor_coding.codebase.ignore_patterns import should_ignore_path
except ImportError:
    def should_ignore_path(path, skip_dirs=frozenset()):  # type: ignore[misc]
        return any(part.startswith(".") for part in path.parts)

try:
    from victor_sdk.utils.ast_helpers import extract_base_classes
except ImportError:
    def extract_base_classes(node):  # type: ignore[misc]
        return [b.id if hasattr(b, "id") else "" for b in node.bases]

logger = logging.getLogger(__name__)


class SymbolExtractor:
    """Extracts symbols (classes, functions, modules) from source files.

    Operates on a CodebaseAnalysis data object — does not own state.
    """

    def __init__(
        self,
        root: Path,
        analysis: "CodebaseAnalysis",
        include_dirs: Optional[List[str]],
        effective_skip_dirs: set,
        language_extensions: dict,
        key_class_patterns: dict,
    ):
        self.root = root
        self.analysis = analysis
        self.include_dirs = include_dirs
        self.effective_skip_dirs = effective_skip_dirs
        self.language_extensions = language_extensions
        self.key_class_patterns = key_class_patterns

    def analyze_source_files(self) -> None:
        """Analyze source files across all supported languages."""
        search_paths: List[Path] = []
        if self.include_dirs:
            for d in self.include_dirs:
                path = self.root / d
                if path.exists() and path.is_dir():
                    search_paths.append(path)

        if not search_paths:
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
        for ext, lang in self.language_extensions.items():
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
                    subpackage = parts[0] if len(parts) > 1 else "root"
                    if subpackage not in self.analysis.packages:
                        self.analysis.packages[subpackage] = []
                    self.analysis.packages[subpackage].append(module_info)

    def _parse_generic_file(
        self, file_path: Path, rel_path: str, language: str
    ) -> Optional["ModuleInfo"]:
        """Parse any source file using regex patterns."""
        from victor_coding.codebase_analyzer import ClassInfo, ModuleInfo

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
                    break
        return module_info if module_info.classes else None

    def _extract_inline_comment(
        self, line: str, content: str, line_no: int
    ) -> Optional[str]:
        """Extract comment from line or line above."""
        for comment_marker in ["//", "#", "--", "/*", "///"]:
            if comment_marker in line:
                idx = line.find(comment_marker)
                comment = line[idx + len(comment_marker):].strip()
                if comment:
                    return comment[:60]
        lines = content.split("\n")
        if line_no > 1:
            prev_line = lines[line_no - 2].strip()
            for marker in ["///", "/**", "//", "#", '"""', "'''"]:
                if prev_line.startswith(marker):
                    return prev_line.lstrip(marker).strip("*/ ").strip()[:60]
        return None

    def _parse_python_file(
        self, file_path: Path, rel_path: str
    ) -> Optional["ModuleInfo"]:
        """Parse a Python file and extract class/function information."""
        from victor_coding.codebase_analyzer import ClassInfo, ModuleInfo

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return None

        module_info = ModuleInfo(name=file_path.stem, path=rel_path)
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

    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> "ClassInfo":
        """Extract information from a class AST node."""
        from victor_coding.codebase_analyzer import ClassInfo

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
        for category, patterns in self.key_class_patterns.items():
            for pattern in patterns:
                if any(pattern in n for n in all_names):
                    return category
        return None
