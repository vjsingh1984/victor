"""Static contract auditing for extracted vertical repositories."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

FORBIDDEN_RUNTIME_IMPORT_PREFIXES = (
    "victor.framework",
    "victor.core",
    "victor.security",
    "victor.agent",
    "victor.workflows",
    "victor.providers",
)
_IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site",
    "tests",
    "venv",
}


@dataclass(frozen=True)
class AuditIssue:
    """A single contract audit finding."""

    level: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line": self.line,
        }


@dataclass
class VerticalContractAuditReport:
    """Structured report for a single vertical repository audit."""

    root_path: Path
    project_name: str = ""
    plugin_entry_points: list[str] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Return the number of error findings."""

        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        """Return the number of warning findings."""

        return sum(1 for issue in self.issues if issue.level == "warning")

    @property
    def passed(self) -> bool:
        """Return True when the audit has no errors."""

        return self.error_count == 0

    def add_issue(
        self,
        level: str,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
    ) -> None:
        """Append a finding to the report."""

        self.issues.append(
            AuditIssue(level=level, code=code, message=message, path=path, line=line)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "root_path": str(self.root_path),
            "project_name": self.project_name,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "plugin_entry_points": list(self.plugin_entry_points),
            "issues": [issue.to_dict() for issue in self.issues],
        }


class VerticalContractAuditor:
    """Audit extracted vertical repositories against core integration rules."""

    def audit_paths(self, paths: Iterable[str | Path]) -> list[VerticalContractAuditReport]:
        """Audit multiple repository paths."""

        return [self.audit_path(path) for path in paths]

    def audit_path(self, path: str | Path) -> VerticalContractAuditReport:
        """Audit a single extracted vertical repository path."""

        root_path = Path(path).expanduser()
        if root_path.is_file() and root_path.name == "pyproject.toml":
            root_path = root_path.parent

        report = VerticalContractAuditReport(root_path=root_path)

        if not root_path.exists():
            report.add_issue(
                "error",
                "missing_path",
                "Path does not exist.",
            )
            return report

        pyproject_path = root_path / "pyproject.toml"
        if not pyproject_path.exists():
            report.add_issue(
                "error",
                "missing_pyproject",
                "Repository does not contain a pyproject.toml file.",
            )
            return report

        try:
            pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.add_issue(
                "error",
                "invalid_pyproject",
                f"Could not parse pyproject.toml: {exc}",
                path="pyproject.toml",
            )
            return report

        project = pyproject_data.get("project", {}) if isinstance(pyproject_data, dict) else {}
        report.project_name = str(project.get("name") or root_path.name)

        entry_points = project.get("entry-points", {}) if isinstance(project, dict) else {}
        plugin_entries = entry_points.get("victor.plugins", {})
        if isinstance(plugin_entries, dict):
            report.plugin_entry_points = sorted(str(key) for key in plugin_entries.keys())
        if not report.plugin_entry_points:
            report.add_issue(
                "error",
                "missing_plugin_entry_point",
                'project.entry-points."victor.plugins" is missing or empty.',
                path="pyproject.toml",
            )

        legacy_entries = entry_points.get("victor.verticals", {})
        if isinstance(legacy_entries, dict) and legacy_entries:
            report.add_issue(
                "warning",
                "legacy_vertical_entry_point",
                "Legacy victor.verticals entry points are still declared; migrate to victor.plugins.",
                path="pyproject.toml",
            )

        dependency_names = self._collect_dependency_names(project)
        if "victor-sdk" not in dependency_names:
            report.add_issue(
                "warning",
                "missing_sdk_dependency",
                "victor-sdk is not declared in project dependencies.",
                path="pyproject.toml",
            )

        for issue in self._scan_python_sources(root_path):
            report.issues.append(issue)

        return report

    def _collect_dependency_names(self, project: dict[str, object]) -> set[str]:
        """Return normalized dependency package names from project metadata."""

        dependencies: list[str] = []
        raw_dependencies = project.get("dependencies", [])
        if isinstance(raw_dependencies, list):
            dependencies.extend(str(dep) for dep in raw_dependencies)

        optional_dependencies = project.get("optional-dependencies", {})
        if isinstance(optional_dependencies, dict):
            for group_dependencies in optional_dependencies.values():
                if isinstance(group_dependencies, list):
                    dependencies.extend(str(dep) for dep in group_dependencies)

        names: set[str] = set()
        for dependency in dependencies:
            normalized = dependency.strip()
            if not normalized:
                continue
            token = normalized.split(";", 1)[0]
            token = token.split("[", 1)[0]
            for separator in ("<", ">", "=", "!", "~", " "):
                token = token.split(separator, 1)[0]
            token = token.strip().lower()
            if token:
                names.add(token)
        return names

    def _scan_python_sources(self, root_path: Path) -> list[AuditIssue]:
        """Scan repository Python sources for forbidden runtime imports."""

        issues: list[AuditIssue] = []
        for source_path in root_path.rglob("*.py"):
            if self._should_skip_path(root_path, source_path):
                continue
            relative_path = source_path.relative_to(root_path)
            try:
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                issues.append(
                    AuditIssue(
                        level="warning",
                        code="invalid_python_source",
                        message=f"Could not parse Python source: {exc.msg}",
                        path=str(relative_path),
                        line=exc.lineno,
                    )
                )
                continue

            for node in ast.walk(tree):
                module_name = self._extract_imported_module(node)
                if module_name and self._is_forbidden_runtime_import(module_name):
                    issues.append(
                        AuditIssue(
                            level="error",
                            code="forbidden_runtime_import",
                            message=(
                                f"Direct runtime import '{module_name}' leaks core internals "
                                "into the extracted vertical boundary."
                            ),
                            path=str(relative_path),
                            line=getattr(node, "lineno", None),
                        )
                    )
        return issues

    def _should_skip_path(self, root_path: Path, source_path: Path) -> bool:
        """Return True for ignored files or directories."""

        relative_parts = source_path.relative_to(root_path).parts
        if any(part in _IGNORED_DIR_NAMES for part in relative_parts[:-1]):
            return True
        filename = relative_parts[-1]
        return filename.startswith("test_")

    def _extract_imported_module(self, node: ast.AST) -> str | None:
        """Return the imported module path for an import node."""

        if isinstance(node, ast.Import):
            for alias in node.names:
                if self._is_forbidden_runtime_import(alias.name):
                    return alias.name
            return None
        if isinstance(node, ast.ImportFrom):
            return node.module
        return None

    def _is_forbidden_runtime_import(self, module_name: str) -> bool:
        """Return True when *module_name* crosses the allowed vertical boundary."""

        return any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_RUNTIME_IMPORT_PREFIXES
        )
