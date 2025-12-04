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
from victor.tools.base import BaseTool, CostTier, ToolMetadata, ToolResult

logger = logging.getLogger(__name__)


class IaCScannerTool(BaseTool):
    """Tool for scanning Infrastructure-as-Code for security issues."""

    name = "iac_scanner"
    description = """Scan Infrastructure-as-Code files for security issues.

Supported IaC platforms:
- Terraform (.tf, .tfvars)
- Docker (Dockerfile)
- Docker Compose (docker-compose.yml)
- Kubernetes manifests (*.yaml)

Detects:
- Hardcoded secrets (API keys, passwords, tokens)
- Overly permissive IAM/RBAC
- Missing encryption
- Network exposure
- Container security issues
- Best practice violations

Actions:
- scan: Full security scan of all IaC files
- scan_file: Scan a specific file
- summary: Get security summary with risk score
- detect: Detect which IaC platforms are used"""

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
    def metadata(self) -> ToolMetadata:
        """Inline semantic metadata for dynamic tool selection."""
        return ToolMetadata(
            category="iac",
            keywords=[
                "iac", "infrastructure as code", "terraform", "docker", "kubernetes",
                "k8s", "security scan", "secrets detection", "misconfiguration",
                "dockerfile", "docker-compose", "helm", "security policy"
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
            IaCPlatform.TERRAFORM: "🏗️",
            IaCPlatform.DOCKER: "🐳",
            IaCPlatform.DOCKER_COMPOSE: "🐳",
            IaCPlatform.KUBERNETES: "☸️",
        }
        for p in platforms:
            icon = platform_icons.get(p, "📄")
            lines.append(f"- {icon} {p.value}")
        return "\n".join(lines)

    def _format_summary(self, summary: dict[str, Any]) -> str:
        """Format security summary."""
        lines = ["**IaC Security Summary**", ""]

        # Risk score
        risk = summary["risk_score"]
        if risk < 20:
            risk_icon = "🟢"
            risk_label = "Low Risk"
        elif risk < 50:
            risk_icon = "🟡"
            risk_label = "Medium Risk"
        elif risk < 80:
            risk_icon = "🟠"
            risk_label = "High Risk"
        else:
            risk_icon = "🔴"
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
        lines.append(f"- 🔴 Critical: {sev['critical']}")
        lines.append(f"- 🟠 High: {sev['high']}")
        lines.append(f"- 🟡 Medium: {sev['medium']}")
        lines.append(f"- 🔵 Low: {sev['low']}")
        lines.append(f"- ⚪ Info: {sev['info']}")
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
            lines.append(f"\n🔴 **Critical ({result.critical_count}):**")
            for f in result.findings:
                if f.severity.value == "critical":
                    lines.append(f"  - [{f.rule_id}] {f.message}")
                    lines.append(f"    📁 {f.file_path.name}:{f.line_number}")
                    if f.remediation:
                        lines.append(f"    💡 {f.remediation}")

        if result.high_count > 0:
            lines.append(f"\n🟠 **High ({result.high_count}):**")
            for f in result.findings:
                if f.severity.value == "high":
                    lines.append(f"  - [{f.rule_id}] {f.message}")
                    lines.append(f"    📁 {f.file_path.name}:{f.line_number}")

        if result.medium_count > 0:
            lines.append(f"\n🟡 **Medium ({result.medium_count}):**")
            for f in result.findings[:10]:  # Limit medium findings shown
                if f.severity.value == "medium":
                    lines.append(f"  - [{f.rule_id}] {f.message}")

        if result.low_count > 0:
            lines.append(f"\nℹ️ **Low ({result.low_count}):** (use summary for details)")

        return "\n".join(lines)

    def _format_findings(self, findings: list, file_path: Path) -> str:
        """Format findings for a specific file."""
        if not findings:
            return f"✅ No security issues found in {file_path.name}"

        lines = [f"**Security Findings in {file_path.name}**", ""]
        lines.append(f"**Total:** {len(findings)}")
        lines.append("")

        for finding in findings:
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
                "info": "⚪",
            }.get(finding.severity.value, "⚪")

            lines.append(f"{severity_icon} **{finding.rule_id}**")
            lines.append(f"   {finding.message}")
            if finding.line_number:
                lines.append(f"   Line: {finding.line_number}")
            if finding.remediation:
                lines.append(f"   💡 {finding.remediation}")
            lines.append("")

        return "\n".join(lines)
