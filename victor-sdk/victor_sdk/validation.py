"""Package-level contract validation for external Victor verticals."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution, version as installed_version
from typing import Any, List

from victor_sdk.core.plugins import VictorPlugin
from victor_sdk.testing import assert_valid_vertical
from victor_sdk.verticals.protocols.base import VerticalBase as SdkVerticalBase


@dataclass(frozen=True)
class ValidationIssue:
    """A single SDK validation finding."""

    code: str
    message: str
    level: str = "error"


@dataclass
class ValidationReport:
    """Structured validation report for a vertical package."""

    package_name: str
    verticals: List[str] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def add_issue(self, code: str, message: str, *, level: str = "error") -> None:
        self.issues.append(ValidationIssue(code=code, message=message, level=level))

    def to_text(self) -> str:
        lines = [f"Package: {self.package_name}"]
        if self.verticals:
            lines.append("Verticals: " + ", ".join(self.verticals))
        if not self.issues:
            lines.append("Status: ok")
            return "\n".join(lines)

        lines.append("Status: failed")
        for issue in self.issues:
            lines.append(f"- [{issue.level}] {issue.code}: {issue.message}")
        return "\n".join(lines)


def validate_vertical_package(package_name: str) -> ValidationReport:
    """Validate Victor vertical entry points published by *package_name*."""

    report = ValidationReport(package_name=package_name)
    try:
        dist = distribution(package_name)
    except PackageNotFoundError:
        report.add_issue("package_not_found", f"Package '{package_name}' is not installed")
        return report

    entry_points = [ep for ep in dist.entry_points if ep.group == "victor.plugins"]
    if not entry_points:
        report.add_issue(
            "missing_entry_points",
            "Package does not publish any victor.plugins entry points",
        )
        return report

    for entry_point in entry_points:
        try:
            candidate = entry_point.load()
        except Exception as exc:
            report.add_issue(
                "entry_point_load_failed",
                f"Failed to load entry point '{entry_point.name}': {exc}",
            )
            continue

        if isinstance(candidate, type) and issubclass(candidate, SdkVerticalBase):
            _validate_vertical_class(candidate, report)
            continue

        if isinstance(candidate, type):
            try:
                plugin = candidate()
            except Exception as exc:
                report.add_issue(
                    "plugin_instantiation_failed",
                    f"Failed to instantiate plugin entry point '{entry_point.name}': {exc}",
                )
                continue
        else:
            plugin = candidate
        if isinstance(plugin, VictorPlugin) or _looks_like_plugin(plugin):
            _validate_plugin(plugin, report)
            continue

        report.add_issue(
            "unsupported_entry_point",
            f"Entry point '{entry_point.name}' must resolve to an SDK VerticalBase subclass "
            "or a VictorPlugin implementation",
        )

    return report


def _validate_vertical_class(vertical_cls: type[Any], report: ValidationReport) -> None:
    """Validate an SDK vertical class and append findings to *report*."""

    manifest = getattr(vertical_cls, "_victor_manifest", None)
    vertical_name = getattr(vertical_cls, "name", None) or vertical_cls.__name__
    report.verticals.append(str(vertical_name))

    if manifest is None:
        report.add_issue(
            "missing_manifest",
            f"Vertical '{vertical_name}' is missing @register_vertical manifest metadata",
        )
        return

    try:
        assert_valid_vertical(vertical_cls)
    except AssertionError as exc:
        report.add_issue("protocol_drift", f"Vertical '{vertical_name}' violates SDK contract: {exc}")
        return

    _check_framework_version_compatibility(vertical_cls, report)


def _validate_plugin(plugin: Any, report: ValidationReport) -> None:
    """Validate a VictorPlugin by observing the verticals it registers."""

    context = _CollectingPluginContext()
    try:
        plugin.register(context)
    except Exception as exc:
        report.add_issue(
            "plugin_register_failed",
            f"Plugin '{getattr(plugin, 'name', type(plugin).__name__)}' failed during register(): {exc}",
        )
        return

    if not context.verticals:
        report.add_issue(
            "plugin_no_verticals",
            f"Plugin '{getattr(plugin, 'name', type(plugin).__name__)}' did not register any verticals",
        )
        return

    for vertical_cls in context.verticals:
        _validate_vertical_class(vertical_cls, report)


def _check_framework_version_compatibility(
    vertical_cls: type[Any],
    report: ValidationReport,
) -> None:
    """Validate min_framework_version against the installed core."""

    manifest = getattr(vertical_cls, "_victor_manifest", None)
    min_framework_version = getattr(manifest, "min_framework_version", None)
    if not min_framework_version:
        return

    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except Exception:
        report.add_issue(
            "packaging_unavailable",
            "PEP 440 compatibility checks require the 'packaging' library",
            level="warning",
        )
        return

    try:
        framework_version = installed_version("victor-ai")
    except Exception as exc:
        report.add_issue(
            "framework_version_unknown",
            f"Could not determine installed victor-ai version: {exc}",
        )
        return

    specifier = (
        min_framework_version
        if any(ch in min_framework_version for ch in "<>=!~")
        else f">={min_framework_version}"
    )

    try:
        if Version(framework_version) not in SpecifierSet(specifier):
            report.add_issue(
                "framework_version_incompatible",
                f"Installed victor-ai {framework_version} does not satisfy "
                f"{specifier} for vertical '{getattr(manifest, 'name', vertical_cls.__name__)}'",
            )
    except Exception as exc:
        report.add_issue(
            "framework_version_invalid",
            f"Could not parse min_framework_version '{min_framework_version}': {exc}",
        )


def _looks_like_plugin(candidate: Any) -> bool:
    """Return True when *candidate* resembles the VictorPlugin protocol."""

    return all(
        hasattr(candidate, attr)
        for attr in ("name", "register", "get_cli_app")
    )


class _CollectingPluginContext:
    """Minimal PluginContext implementation for validation-only plugin registration."""

    def __init__(self) -> None:
        self.verticals: list[type[Any]] = []

    def register_tool(self, tool_instance: Any) -> None:
        return None

    def register_vertical(self, vertical_class: type[Any]) -> None:
        self.verticals.append(vertical_class)

    def register_chunker(self, chunker_instance: Any) -> None:
        return None

    def register_command(self, name: str, app: Any) -> None:
        return None

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        return None

    def get_service(self, service_type: type[Any]) -> None:
        return None

    def get_settings(self) -> None:
        return None
