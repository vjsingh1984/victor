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

"""IaC Scanner Tool - Security scanning for Victor tools.

This tool provides integration with Victor's tool system for
Infrastructure-as-Code security scanning.
"""

import logging
from pathlib import Path
from typing import Any

from victor.iac import (
    IaCManager,
    IaCPlatform,
    ScanPolicy,
    ScanResult,
    Severity,
)
from victor.tools.base import (
    AccessMode,
    BaseTool,
    CostTier,
    DangerLevel,
    Priority,
    ToolMetadata,
    ToolResult,
)
from victor.tools.tool_names import ToolNames

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icon rendering
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


class IaCScannerTool(BaseTool):
    """Tool for scanning Infrastructure-as-Code for security issues."""

    name = ToolNames.IAC
    description = """Scan IaC files (Terraform, Docker, K8s) for security issues.

    Actions: scan, scan_file, summary, detect.
    Detects: secrets, IAM misconfig, missing encryption, network exposure."""

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "scan_file", "summary", "detect"],
                "description": "Action to perform",
            },
            "file_path": {
                "type": "string",
                "description": "Path to specific file to scan (for scan_file action)",
            },
            "platform": {
                "type": "string",
                "enum": ["terraform", "docker", "docker_compose", "kubernetes", "all"],
                "description": "IaC platform to scan (default: all)",
            },
            "min_severity": {
                "type": "string",
                "enum": ["info", "low", "medium", "high", "critical"],
                "description": "Minimum severity to report (default: low)",
            },
        },
        "required": ["action"],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    @property
    def priority(self) -> Priority:
        """Tool priority for selection availability."""
        return Priority.MEDIUM  # Task-specific security scanning

    @property
    def access_mode(self) -> AccessMode:
        """Tool access mode for approval tracking."""
        return AccessMode.READONLY  # Only reads IaC files for analysis

    @property
    def danger_level(self) -> DangerLevel:
        """Danger level for warning/confirmation logic."""
        return DangerLevel.SAFE  # No side effects

    @property
    def metadata(self) -> ToolMetadata:
        """Inline semantic metadata for dynamic tool selection."""
        return ToolMetadata(
            category="iac",
            keywords=[
                "iac",
                "infrastructure as code",
                "terraform",
                "docker",
                "kubernetes",
                "k8s",
                "security scan",
                "secrets detection",
                "misconfiguration",
                "dockerfile",
                "docker-compose",
                "helm",
                "security policy",
            ],
            use_cases=[
                "infrastructure security scanning",
                "Terraform security analysis",
                "Docker security scanning",
                "Kubernetes security scanning",
                "detecting hardcoded secrets",
                "finding security misconfigurations",
            ],
            examples=[
                "scanning Terraform files for security issues",
                "detecting hardcoded credentials in Docker files",
                "checking Kubernetes manifests for best practices",
            ],
            priority_hints=[
                "Use for infrastructure security analysis",
                "Detects secrets, misconfigurations, and compliance issues",
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute IaC scanning action."""
        action = kwargs.get("action", "summary")
        file_path = kwargs.get("file_path")
        platform_str = kwargs.get("platform", "all")
        min_severity_str = kwargs.get("min_severity", "low")

        try:
            manager = IaCManager()

            # Build policy
            policy = ScanPolicy(min_severity=Severity(min_severity_str))

            if action == "detect":
                platforms = await manager.detect_platforms()
                return ToolResult(
                    success=True,
                    output=self._format_platforms(platforms),
                    metadata={"platforms": [p.value for p in platforms]},
                )

            elif action == "summary":
                summary = await manager.get_summary()
                return ToolResult(
                    success=True,
                    output=self._format_summary(summary),
                    metadata=summary,
                )

            elif action == "scan":
                platforms = None
                if platform_str != "all":
                    platforms = [IaCPlatform(platform_str)]

                result = await manager.scan(platforms=platforms, policy=policy)
                return ToolResult(
                    success=True,
                    output=self._format_scan_result(result),
                    metadata=result.to_dict(),
                )

            elif action == "scan_file":
                if not file_path:
                    return ToolResult(
                        success=False,
                        output="file_path is required for scan_file action",
                        error="Missing parameter",
                    )

                findings = await manager.scan_file(file_path, policy)
                return ToolResult(
                    success=True,
                    output=self._format_findings(findings, Path(file_path)),
                    metadata={"findings": [f.to_dict() for f in findings]},
                )

            else:
                return ToolResult(
                    success=False,
                    output=f"Unknown action: {action}",
                    error="Invalid action",
                )

        except Exception as e:
            logger.exception(f"IaC scanning failed: {e}")
            return ToolResult(
                success=False,
                output=f"IaC scanning failed: {e}",
                error=str(e),
            )

    def _format_platforms(self, platforms: list[IaCPlatform]) -> str:
        """Format detected platforms."""
        if not platforms:
            return "No IaC files detected in this project."

        lines = ["**Detected IaC Platforms:**"]
        platform_icons = {
            IaCPlatform.TERRAFORM: _get_icon("terraform"),
            IaCPlatform.DOCKER: _get_icon("docker"),
            IaCPlatform.DOCKER_COMPOSE: _get_icon("docker"),
            IaCPlatform.KUBERNETES: _get_icon("kubernetes"),
        }
        for p in platforms:
            icon = platform_icons.get(p, _get_icon("file"))
            lines.append(f"- {icon} {p.value}")
        return "\n".join(lines)

    def _format_summary(self, summary: dict[str, Any]) -> str:
        """Format security summary."""
        lines = ["**IaC Security Summary**", ""]

        # Risk score
        risk = summary["risk_score"]
        if risk < 20:
            risk_icon = _get_icon("level_low")
            risk_label = "Low Risk"
        elif risk < 50:
            risk_icon = _get_icon("level_medium")
            risk_label = "Medium Risk"
        elif risk < 80:
            risk_icon = _get_icon("level_high")
            risk_label = "High Risk"
        else:
            risk_icon = _get_icon("level_critical")
            risk_label = "Critical Risk"

        lines.append(f"**Risk Score:** {risk_icon} {risk}/100 ({risk_label})")
        lines.append("")

        # Stats
        lines.append(f"**Files Scanned:** {summary['files_scanned']}")
        lines.append(f"**Resources:** {summary['total_resources']}")
        lines.append(f"**Scan Time:** {summary['scan_duration_ms']}ms")
        lines.append("")

        # Findings by severity
        sev = summary["by_severity"]
        lines.append("**Findings by Severity:**")
        lines.append(f"- {_get_icon('level_critical')} Critical: {sev['critical']}")
        lines.append(f"- {_get_icon('level_high')} High: {sev['high']}")
        lines.append(f"- {_get_icon('level_medium')} Medium: {sev['medium']}")
        lines.append(f"- {_get_icon('level_info')} Low: {sev['low']}")
        lines.append(f"- {_get_icon('level_unknown')} Info: {sev['info']}")
        lines.append("")

        # By category
        if summary["by_category"]:
            lines.append("**By Category:**")
            for cat, count in summary["by_category"].items():
                lines.append(f"- {cat}: {count}")
            lines.append("")

        # Top rules
        if summary["top_rules"]:
            lines.append("**Most Common Issues:**")
            for rule, count in summary["top_rules"]:
                lines.append(f"- {rule}: {count}")

        return "\n".join(lines)

    def _format_scan_result(self, result: ScanResult) -> str:
        """Format full scan result."""
        lines = ["**IaC Security Scan Results**", ""]

        # Summary
        lines.append(f"**Files:** {result.files_scanned}")
        lines.append(f"**Resources:** {result.total_resources}")
        lines.append(f"**Total Findings:** {len(result.findings)}")
        lines.append("")

        # Findings by severity
        if result.critical_count > 0:
            lines.append(f"\n{_get_icon('level_critical')} **Critical ({result.critical_count}):**")
            for f in result.findings:
                if f.severity.value == "critical":
                    lines.append(f"  - [{f.rule_id}] {f.message}")
                    lines.append(f"    {_get_icon('folder')} {f.file_path.name}:{f.line_number}")
                    if f.remediation:
                        lines.append(f"    {_get_icon('hint')} {f.remediation}")

        if result.high_count > 0:
            lines.append(f"\n{_get_icon('level_high')} **High ({result.high_count}):**")
            for f in result.findings:
                if f.severity.value == "high":
                    lines.append(f"  - [{f.rule_id}] {f.message}")
                    lines.append(f"    {_get_icon('folder')} {f.file_path.name}:{f.line_number}")

        if result.medium_count > 0:
            lines.append(f"\n{_get_icon('level_medium')} **Medium ({result.medium_count}):**")
            for f in result.findings[:10]:  # Limit medium findings shown
                if f.severity.value == "medium":
                    lines.append(f"  - [{f.rule_id}] {f.message}")

        if result.low_count > 0:
            lines.append(f"\n{_get_icon('info')} **Low ({result.low_count}):** (use summary for details)")

        return "\n".join(lines)

    def _format_findings(self, findings: list, file_path: Path) -> str:
        """Format findings for a specific file."""
        if not findings:
            return f"{_get_icon('success')} No security issues found in {file_path.name}"

        lines = [f"**Security Findings in {file_path.name}**", ""]
        lines.append(f"**Total:** {len(findings)}")
        lines.append("")

        for finding in findings:
            severity_icon = {
                "critical": _get_icon("level_critical"),
                "high": _get_icon("level_high"),
                "medium": _get_icon("level_medium"),
                "low": _get_icon("level_info"),
                "info": _get_icon("level_unknown"),
            }.get(finding.severity.value, _get_icon("level_unknown"))

            lines.append(f"{severity_icon} **{finding.rule_id}**")
            lines.append(f"   {finding.message}")
            if finding.line_number:
                lines.append(f"   Line: {finding.line_number}")
            if finding.remediation:
                lines.append(f"   {_get_icon('hint')} {finding.remediation}")
            lines.append("")

        return "\n".join(lines)
