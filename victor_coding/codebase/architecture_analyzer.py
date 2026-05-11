"""Architecture pattern detection and metadata analysis.

Extracted from codebase_analyzer.py (SRP decomposition).
Handles pattern detection, dependency extraction, LOC stats, test coverage.
"""

from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from victor_coding.codebase_analyzer import ClassInfo, CodebaseAnalysis

try:
    from victor_coding.codebase.ignore_patterns import should_ignore_path
except ImportError:
    def should_ignore_path(path, skip_dirs=frozenset()):  # type: ignore[misc]
        return any(part.startswith(".") for part in path.parts)

try:
    from victor_sdk.utils.ast_helpers import extract_imports, is_stdlib_module
except ImportError:
    def extract_imports(tree, top_level_only=True):  # type: ignore[misc]
        return []
    def is_stdlib_module(mod):  # type: ignore[misc]
        return False

logger = logging.getLogger(__name__)


class ArchitectureAnalyzer:
    """Detects architecture patterns, dependencies, LOC stats, test coverage.

    Operates on a CodebaseAnalysis data object — does not own state.
    """

    def __init__(
        self,
        root: Path,
        analysis: "CodebaseAnalysis",
        include_dirs: Optional[List[str]],
        effective_skip_dirs: set,
        language_extensions: dict,
        config_extensions: dict,
    ):
        self.root = root
        self.analysis = analysis
        self.include_dirs = include_dirs
        self.effective_skip_dirs = effective_skip_dirs
        self.language_extensions = language_extensions
        self.config_extensions = config_extensions

    def identify_key_components(self) -> None:
        """Identify the most important classes in the codebase."""
        all_classes: List["ClassInfo"] = []
        method_count = 0
        protocol_count = 0

        for modules in self.analysis.packages.values():
            for module in modules:
                all_classes.extend(module.classes)
                method_count += len(module.functions)
                for cls in module.classes:
                    if (
                        cls.is_abstract
                        or "Protocol" in cls.name
                        or any(base in ("Protocol", "ABC") for base in cls.base_classes)
                    ):
                        protocol_count += 1

        self.analysis.method_count = method_count
        self.analysis.protocol_count = protocol_count

        def score_class(cls: "ClassInfo") -> int:
            score = 0
            if cls.is_abstract or "Base" in cls.name:
                score += 10
            if cls.category:
                score += 5
                if cls.category in ("manager", "provider", "registry"):
                    score += 5
            if "Orchestrator" in cls.name:
                score += 15
            if cls.docstring:
                score += 2
            return score

        scored = [(score_class(c), c) for c in all_classes if c.category]
        scored.sort(key=lambda x: -x[0])
        self.analysis.key_components = [c for _, c in scored[:20]]

    def extract_entry_points(self) -> None:
        """Extract CLI entry points from pyproject.toml."""
        pyproject = self.root / "pyproject.toml"
        if not pyproject.exists():
            return
        try:
            content = pyproject.read_text(encoding="utf-8")
            scripts_match = re.search(
                r"\[project\.scripts\](.*?)(?=\[|\Z)", content, re.DOTALL
            )
            if scripts_match:
                for line in scripts_match.group(1).strip().split("\n"):
                    if "=" in line:
                        parts = line.split("=", 1)
                        cmd = parts[0].strip().strip('"')
                        target = parts[1].strip().strip('"')
                        self.analysis.entry_points[cmd] = target

            if "[project.optional-dependencies]" in content:
                if "pytest" in content:
                    self.analysis.cli_commands.append("pytest")
                if "black" in content:
                    self.analysis.cli_commands.append("black .")
                if "ruff" in content:
                    self.analysis.cli_commands.append("ruff check .")
                if "mypy" in content:
                    self.analysis.cli_commands.append(
                        "mypy " + (self.analysis.main_package or ".")
                    )
        except Exception as e:
            logger.debug(f"Failed to parse pyproject.toml: {e}")

    def detect_architecture_patterns(self) -> None:
        """Detect common architectural patterns in the codebase."""
        patterns = []
        provider_classes = [
            c for c in self.analysis.key_components if c.category == "provider"
        ]
        if len(provider_classes) >= 2:
            base = next(
                (c for c in provider_classes if c.is_abstract or "Base" in c.name), None
            )
            if base:
                patterns.append(
                    f"Provider Pattern: Base class `{base.name}` "
                    f"({base.file_path}:{base.line_number})"
                )

        tool_classes = [
            c for c in self.analysis.key_components if c.category == "tool"
        ]
        if len(tool_classes) >= 2:
            base = next(
                (c for c in tool_classes if c.is_abstract or "Base" in c.name), None
            )
            if base:
                patterns.append(
                    f"Tool/Command Pattern: Base class `{base.name}` "
                    f"({base.file_path}:{base.line_number})"
                )

        registry_classes = [
            c for c in self.analysis.key_components if c.category == "registry"
        ]
        if registry_classes:
            patterns.append(
                f"Registry Pattern: {len(registry_classes)} registries found"
            )

        manager_classes = [
            c for c in self.analysis.key_components if c.category == "manager"
        ]
        if manager_classes:
            main = manager_classes[0]
            patterns.append(
                f"Orchestrator/Manager: `{main.name}` "
                f"({main.file_path}:{main.line_number})"
            )

        config_classes = [
            c for c in self.analysis.key_components if c.category == "config"
        ]
        if config_classes:
            patterns.append(
                f"Configuration: {len(config_classes)} config classes (Pydantic-style)"
            )
        self.analysis.architecture_patterns = patterns

    def find_config_files(self) -> None:
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
                matches = list(self.root.glob(pattern))
                if matches:
                    self.analysis.config_files.append((pattern, description))
            else:
                if (self.root / pattern).exists():
                    self.analysis.config_files.append((pattern, description))

    def extract_dependencies(self) -> None:
        """Extract dependencies from pyproject.toml or package.json."""
        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                deps: Dict[str, List[str]] = {"core": [], "dev": [], "optional": []}
                deps_match = re.search(
                    r"\[project\].*?dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL
                )
                if deps_match:
                    for line in deps_match.group(1).split(","):
                        dep = line.strip().strip("\"'")
                        if dep and not dep.startswith("#"):
                            pkg_name = re.split(r"[<>=~!]", dep)[0].strip()
                            if pkg_name:
                                deps["core"].append(pkg_name)
                opt_match = re.search(
                    r"\[project\.optional-dependencies\](.*?)(?=\[|\Z)",
                    content,
                    re.DOTALL,
                )
                if opt_match:
                    dev_match = re.search(
                        r"dev\s*=\s*\[(.*?)\]", opt_match.group(1), re.DOTALL
                    )
                    if dev_match:
                        for line in dev_match.group(1).split(","):
                            dep = line.strip().strip("\"'")
                            if dep and not dep.startswith("#"):
                                pkg_name = re.split(r"[<>=~!]", dep)[0].strip()
                                if pkg_name:
                                    deps["dev"].append(pkg_name)
                self.analysis.dependencies = deps
            except Exception as e:
                logger.debug(f"Failed to parse pyproject.toml dependencies: {e}")

        package_json = self.root / "package.json"
        if package_json.exists() and not self.analysis.dependencies:
            try:
                import json

                data = json.loads(package_json.read_text(encoding="utf-8"))
                deps = {"core": [], "dev": []}
                if "dependencies" in data:
                    deps["core"] = list(data["dependencies"].keys())[:15]
                if "devDependencies" in data:
                    deps["dev"] = list(data["devDependencies"].keys())[:10]
                self.analysis.dependencies = deps
            except Exception as e:
                logger.debug(f"Failed to parse package.json dependencies: {e}")

    def calculate_loc_stats(self) -> None:
        """Calculate lines of code statistics."""
        source_lines = 0
        source_files = 0
        config_lines = 0
        config_files = 0
        total_lines = 0
        total_files = 0
        largest_file = ""
        largest_file_lines = 0
        file_sizes: List[Tuple[str, int]] = []
        all_extensions = {**self.language_extensions, **self.config_extensions}

        search_dirs = (
            [self.root / d for d in self.include_dirs]
            if self.include_dirs
            else [self.root]
        )
        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for ext in all_extensions:
                for source_file in search_dir.rglob(f"*{ext}"):
                    if should_ignore_path(
                        source_file, skip_dirs=self.effective_skip_dirs
                    ):
                        continue
                    try:
                        content = source_file.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                        lines = len(content.splitlines())
                        is_config = ext in self.config_extensions
                        if is_config:
                            config_lines += lines
                            config_files += 1
                        else:
                            source_lines += lines
                            source_files += 1
                        total_lines += lines
                        total_files += 1
                        rel_path = str(source_file.relative_to(self.root))
                        file_sizes.append((rel_path, lines))
                        if lines > largest_file_lines:
                            largest_file_lines = lines
                            largest_file = rel_path
                    except Exception:
                        pass

        file_sizes.sort(key=lambda x: -x[1])
        self.analysis.loc_stats = {
            "total_lines": total_lines,
            "total_files": total_files,
            "largest_file": largest_file,
            "largest_file_lines": largest_file_lines,
            "top_files": file_sizes[:5],
            "source_lines": source_lines,
            "source_files": source_files,
            "config_lines": config_lines,
            "config_files": config_files,
        }

    def extract_top_imports(self) -> None:
        """Extract the most commonly imported modules (Python only)."""
        import_counts: Dict[str, int] = defaultdict(int)
        search_dirs = (
            [self.root / d for d in self.include_dirs]
            if self.include_dirs
            else [self.root]
        )
        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for source_file in search_dir.rglob("*.py"):
                if should_ignore_path(
                    source_file, skip_dirs=self.effective_skip_dirs
                ):
                    continue
                try:
                    content = source_file.read_text(encoding="utf-8", errors="ignore")
                    try:
                        tree = ast.parse(content)
                        for module in extract_imports(tree, top_level_only=True):
                            import_counts[module] += 1
                    except SyntaxError:
                        for module in self._extract_imports_regex(content):
                            import_counts[module] += 1
                except Exception:
                    pass

        filtered = [
            (mod, count)
            for mod, count in import_counts.items()
            if not is_stdlib_module(mod)
        ]
        filtered.sort(key=lambda x: -x[1])
        self.analysis.top_imports = filtered[:10]

    @staticmethod
    def _extract_imports_regex(content: str) -> List[str]:
        """Extract imports using regex as fallback for AST failures."""
        imports: List[str] = []
        imports.extend(re.findall(r"^import\s+(\S+)", content, re.MULTILINE))
        imports.extend(
            re.findall(r"^from\s+(\S+)\s+import", content, re.MULTILINE)
        )
        run_matches = re.findall(r"%run\s+(\S+)", content)
        for match in run_matches:
            parts = Path(match).parts
            if parts and parts[-1].endswith(".py"):
                parts = parts[:-1]
            if parts:
                imports.append(".".join(parts))
        dynamic = re.findall(
            r'importlib\.import_module\([\'"]([^\'"]+)[\'"]\)', content
        )
        imports.extend(dynamic)
        dynamic2 = re.findall(r'__import__\([\'"]([^\'"]+)[\'"]\)', content)
        imports.extend(dynamic2)
        return imports

    def extract_test_coverage(self) -> None:
        """Try to extract test coverage from coverage reports."""
        coverage_file = self.root / "coverage.xml"
        if coverage_file.exists():
            try:
                content = coverage_file.read_text(encoding="utf-8")
                match = re.search(r'line-rate="([0-9.]+)"', content)
                if match:
                    self.analysis.test_coverage = round(
                        float(match.group(1)) * 100, 1
                    )
                    return
            except Exception:
                pass

        coverage_db = self.root / ".coverage"
        if coverage_db.exists():
            try:
                import sqlite3

                conn = sqlite3.connect(str(coverage_db))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT SUM(num_statements), "
                    "SUM(num_statements - missing_lines) FROM file_summary"
                )
                row = cursor.fetchone()
                if row and row[0]:
                    total, covered = row[0], row[1] or 0
                    self.analysis.test_coverage = round(
                        (covered / total) * 100, 1
                    )
                conn.close()
            except Exception:
                pass

        htmlcov = self.root / "htmlcov" / "index.html"
        if htmlcov.exists():
            try:
                content = htmlcov.read_text(encoding="utf-8")
                match = re.search(r"(\d+)%\s*</span>\s*</h1>", content)
                if match:
                    self.analysis.test_coverage = float(match.group(1))
            except Exception:
                pass
