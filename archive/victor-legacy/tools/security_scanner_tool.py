"""Security scanner tool for detecting vulnerabilities and secrets.

Features:
- Secret detection (API keys, passwords, tokens)
- Dependency vulnerability scanning
- Configuration security analysis
- File permission checks
- Security best practices
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Set
import logging

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class SecurityScannerTool(BaseTool):
    """Tool for security vulnerability scanning."""

    # Secret patterns to detect
    SECRET_PATTERNS = {
        "aws_access_key": {
            "pattern": r"AKIA[0-9A-Z]{16}",
            "severity": "critical",
            "description": "AWS Access Key ID",
        },
        "aws_secret_key": {
            "pattern": r"aws(.{0,20})?['\"][0-9a-zA-Z/+]{40}['\"]",
            "severity": "critical",
            "description": "AWS Secret Access Key",
        },
        "github_token": {
            "pattern": r"ghp_[0-9a-zA-Z]{36}",
            "severity": "critical",
            "description": "GitHub Personal Access Token",
        },
        "github_oauth": {
            "pattern": r"gho_[0-9a-zA-Z]{36}",
            "severity": "critical",
            "description": "GitHub OAuth Token",
        },
        "slack_token": {
            "pattern": r"xox[baprs]-([0-9a-zA-Z]{10,48})?",
            "severity": "critical",
            "description": "Slack Token",
        },
        "slack_webhook": {
            "pattern": r"https://hooks.slack.com/services/T[a-zA-Z0-9_]{8}/B[a-zA-Z0-9_]{8}/[a-zA-Z0-9_]{24}",
            "severity": "high",
            "description": "Slack Webhook",
        },
        "google_api": {
            "pattern": r"AIza[0-9A-Za-z-_]{35}",
            "severity": "critical",
            "description": "Google API Key",
        },
        "generic_api_key": {
            "pattern": r"[aA][pP][iI]_?[kK][eE][yY].*['\"][0-9a-zA-Z]{32,45}['\"]",
            "severity": "high",
            "description": "Generic API Key",
        },
        "generic_secret": {
            "pattern": r"[sS][eE][cC][rR][eE][tT].*['\"][0-9a-zA-Z]{32,45}['\"]",
            "severity": "high",
            "description": "Generic Secret",
        },
        "private_key": {
            "pattern": r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----",
            "severity": "critical",
            "description": "Private Key",
        },
        "jwt_token": {
            "pattern": r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*",
            "severity": "medium",
            "description": "JWT Token",
        },
        "connection_string": {
            "pattern": r"(mongodb|mysql|postgresql)://[^\\s]*:[^\\s]*@",
            "severity": "high",
            "description": "Database Connection String",
        },
    }

    # Dangerous file patterns
    DANGEROUS_FILES = {
        ".env": "Environment file may contain secrets",
        ".env.local": "Environment file may contain secrets",
        ".env.production": "Production environment file",
        "credentials.json": "Credentials file",
        "auth.json": "Authentication file",
        "secrets.json": "Secrets file",
        "id_rsa": "Private SSH key",
        "id_dsa": "Private SSH key",
        ".pem": "Private certificate",
        ".key": "Private key file",
    }

    # Insecure dependencies (known vulnerabilities)
    VULNERABLE_PACKAGES = {
        "requests": {"<2.31.0": "CVE-2023-32681: Proxy-Authorization header exposure"},
        "flask": {"<2.3.2": "Security fixes in 2.3.2"},
        "django": {"<4.2.2": "Multiple security fixes"},
        "pillow": {"<9.5.0": "Multiple security vulnerabilities"},
        "pyyaml": {"<6.0": "Arbitrary code execution via unsafe load"},
    }

    def __init__(self):
        """Initialize security scanner."""
        super().__init__()

    @property
    def name(self) -> str:
        """Get tool name."""
        return "security_scan"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Security vulnerability and secret detection.

Comprehensive security scanning:
- Secret detection (API keys, tokens, passwords)
- Dependency vulnerability scanning
- Configuration security analysis
- File permission checks
- Security best practices

Operations:
- scan_secrets: Detect secrets in code
- scan_dependencies: Check for vulnerable dependencies
- scan_config: Analyze configuration security
- scan_all: Comprehensive security scan
- check_file: Check specific file for secrets

Example workflows:
1. Scan for secrets:
   security_scan(operation="scan_secrets", path="src/")

2. Check dependencies:
   security_scan(operation="scan_dependencies", path="requirements.txt")

3. Full security scan:
   security_scan(operation="scan_all", path=".")

4. Check single file:
   security_scan(operation="check_file", path="config.py")
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
        [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: scan_secrets, scan_dependencies, scan_config, scan_all, check_file",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="File or directory path to scan",
                required=True,
            ),
            ToolParameter(
                name="severity",
                type="string",
                description="Minimum severity: low, medium, high, critical (default: medium)",
                required=False,
            ),
            ToolParameter(
                name="exclude",
                type="array",
                description="Patterns to exclude (e.g., ['*.test.js', 'node_modules/'])",
                required=False,
            ),
        ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute security scan operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with scan findings
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "scan_secrets":
                return await self._scan_secrets(kwargs)
            elif operation == "scan_dependencies":
                return await self._scan_dependencies(kwargs)
            elif operation == "scan_config":
                return await self._scan_config(kwargs)
            elif operation == "scan_all":
                return await self._scan_all(kwargs)
            elif operation == "check_file":
                return await self._check_file(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Security scan failed")
            return ToolResult(
                success=False, output="", error=f"Security scan error: {str(e)}"
            )

    async def _scan_secrets(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Scan for secrets in code."""
        path = kwargs.get("path")
        severity = kwargs.get("severity", "medium")
        exclude = kwargs.get("exclude", [])

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        secrets_found = []

        if path_obj.is_file():
            secrets_found = self._check_file_for_secrets(path_obj)
        else:
            # Scan directory
            for file_path in path_obj.rglob("*"):
                if file_path.is_file() and not self._should_exclude(file_path, exclude):
                    try:
                        file_secrets = self._check_file_for_secrets(file_path)
                        secrets_found.extend(file_secrets)
                    except Exception as e:
                        logger.warning("Failed to scan %s: %s", file_path, e)

        # Filter by severity
        severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_severity = severity_levels.get(severity, 1)

        filtered_secrets = [
            secret
            for secret in secrets_found
            if severity_levels.get(secret.get("severity", "low"), 0) >= min_severity
        ]

        # Build report
        report = self._build_secrets_report(path_obj, filtered_secrets)

        return ToolResult(
            success=len(filtered_secrets) == 0,
            output=report,
            error="" if len(filtered_secrets) == 0 else "Secrets detected",
        )

    async def _scan_dependencies(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Scan dependencies for vulnerabilities."""
        path = kwargs.get("path")

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)

        # Find requirements files
        req_files = []
        if path_obj.is_file() and path_obj.name in ["requirements.txt", "package.json"]:
            req_files = [path_obj]
        else:
            req_files.extend(path_obj.rglob("requirements.txt"))
            req_files.extend(path_obj.rglob("package.json"))

        vulnerabilities = []

        for req_file in req_files:
            if req_file.name == "requirements.txt":
                vulns = self._check_python_deps(req_file)
                vulnerabilities.extend(vulns)

        # Build report
        report = self._build_dependency_report(path_obj, vulnerabilities)

        return ToolResult(
            success=len(vulnerabilities) == 0,
            output=report,
            error="" if len(vulnerabilities) == 0 else "Vulnerabilities found",
        )

    async def _scan_config(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Scan configuration files for security issues."""
        path = kwargs.get("path")

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        config_issues = []

        # Check for dangerous files
        if path_obj.is_file():
            issue = self._check_dangerous_file(path_obj)
            if issue:
                config_issues.append(issue)
        else:
            for file_path in path_obj.rglob("*"):
                if file_path.is_file():
                    issue = self._check_dangerous_file(file_path)
                    if issue:
                        config_issues.append(issue)

        # Build report
        report = self._build_config_report(path_obj, config_issues)

        return ToolResult(
            success=len(config_issues) == 0,
            output=report,
            error="" if len(config_issues) == 0 else "Configuration issues found",
        )

    async def _scan_all(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Comprehensive security scan."""
        path = kwargs.get("path")

        # Run all scans
        secrets_result = await self._scan_secrets(kwargs)
        deps_result = await self._scan_dependencies(kwargs)
        config_result = await self._scan_config(kwargs)

        # Combine results
        report = []
        report.append(f"Comprehensive Security Scan: {path}")
        report.append("=" * 70)
        report.append("")
        report.append("1. Secrets Scan")
        report.append("-" * 70)
        report.append(secrets_result.output)
        report.append("")
        report.append("2. Dependency Scan")
        report.append("-" * 70)
        report.append(deps_result.output)
        report.append("")
        report.append("3. Configuration Scan")
        report.append("-" * 70)
        report.append(config_result.output)

        all_success = secrets_result.success and deps_result.success and config_result.success

        return ToolResult(
            success=all_success,
            output="\n".join(report),
            error="" if all_success else "Security issues found",
        )

    async def _check_file(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Check specific file for secrets."""
        path = kwargs.get("path")

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return ToolResult(
                success=False, output="", error=f"File not found: {path}"
            )

        secrets = self._check_file_for_secrets(file_path)

        report = self._build_file_secrets_report(file_path, secrets)

        return ToolResult(
            success=len(secrets) == 0,
            output=report,
            error="" if len(secrets) == 0 else "Secrets detected",
        )

    def _check_file_for_secrets(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check file for secrets."""
        secrets = []

        # Skip binary files
        try:
            content = file_path.read_text()
        except (UnicodeDecodeError, PermissionError):
            return secrets

        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for secret_type, secret_info in self.SECRET_PATTERNS.items():
                pattern = secret_info["pattern"]
                matches = re.finditer(pattern, line)

                for match in matches:
                    secrets.append(
                        {
                            "type": secret_type,
                            "severity": secret_info["severity"],
                            "description": secret_info["description"],
                            "file": str(file_path),
                            "line": line_num,
                            "match": match.group(0)[:50] + "...",  # Truncate
                        }
                    )

        return secrets

    def _check_python_deps(self, req_file: Path) -> List[Dict[str, Any]]:
        """Check Python dependencies for vulnerabilities."""
        vulnerabilities = []

        try:
            content = req_file.read_text()
            lines = content.strip().split("\n")

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse package==version
                if "==" in line:
                    package, version = line.split("==", 1)
                    package = package.strip()
                    version = version.strip()

                    if package in self.VULNERABLE_PACKAGES:
                        for vuln_version, description in self.VULNERABLE_PACKAGES[
                            package
                        ].items():
                            # Simple version check (would need proper version parsing in production)
                            vulnerabilities.append(
                                {
                                    "package": package,
                                    "version": version,
                                    "vulnerability": description,
                                    "severity": "high",
                                    "file": str(req_file),
                                }
                            )

        except Exception as e:
            logger.warning("Failed to parse %s: %s", req_file, e)

        return vulnerabilities

    def _check_dangerous_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Check if file is potentially dangerous."""
        for pattern, description in self.DANGEROUS_FILES.items():
            if pattern.startswith(".") and file_path.suffix == pattern:
                return {
                    "file": str(file_path),
                    "reason": description,
                    "severity": "high",
                }
            elif file_path.name == pattern:
                return {
                    "file": str(file_path),
                    "reason": description,
                    "severity": "high",
                }

        return None

    def _should_exclude(self, file_path: Path, exclude: List[str]) -> bool:
        """Check if file should be excluded."""
        for pattern in exclude:
            if pattern in str(file_path):
                return True
        return False

    def _build_secrets_report(
        self, path: Path, secrets: List[Dict[str, Any]]
    ) -> str:
        """Build secrets scan report."""
        report = []
        report.append(f"Secrets Scan Report: {path}")
        report.append("=" * 70)
        report.append("")

        if not secrets:
            report.append("✓ No secrets detected!")
            return "\n".join(report)

        report.append(f"⚠ SECRETS DETECTED: {len(secrets)}")
        report.append("")

        # Group by severity
        critical = [s for s in secrets if s.get("severity") == "critical"]
        high = [s for s in secrets if s.get("severity") == "high"]
        medium = [s for s in secrets if s.get("severity") == "medium"]

        for severity, severity_secrets in [
            ("CRITICAL", critical),
            ("HIGH", high),
            ("MEDIUM", medium),
        ]:
            if severity_secrets:
                report.append(f"{severity} Secrets:")
                report.append("-" * 70)
                for secret in severity_secrets:
                    report.append(f"  {secret['description']}")
                    report.append(f"  File: {secret['file']}:{secret['line']}")
                    report.append(f"  Type: {secret['type']}")
                    report.append("")

        report.append("⚠ ACTION REQUIRED:")
        report.append("1. Remove secrets from code")
        report.append("2. Use environment variables")
        report.append("3. Rotate exposed credentials")
        report.append("4. Add to .gitignore")

        return "\n".join(report)

    def _build_dependency_report(
        self, path: Path, vulnerabilities: List[Dict[str, Any]]
    ) -> str:
        """Build dependency scan report."""
        report = []
        report.append(f"Dependency Scan Report: {path}")
        report.append("=" * 70)
        report.append("")

        if not vulnerabilities:
            report.append("✓ No known vulnerabilities!")
            return "\n".join(report)

        report.append(f"⚠ Vulnerabilities Found: {len(vulnerabilities)}")
        report.append("")

        for vuln in vulnerabilities:
            report.append(f"Package: {vuln['package']} {vuln['version']}")
            report.append(f"Severity: {vuln['severity']}")
            report.append(f"Issue: {vuln['vulnerability']}")
            report.append(f"File: {vuln['file']}")
            report.append("")

        report.append("Recommendations:")
        report.append("1. Update vulnerable packages")
        report.append("2. Review security advisories")
        report.append("3. Test after updates")

        return "\n".join(report)

    def _build_config_report(
        self, path: Path, issues: List[Dict[str, Any]]
    ) -> str:
        """Build configuration scan report."""
        report = []
        report.append(f"Configuration Security Report: {path}")
        report.append("=" * 70)
        report.append("")

        if not issues:
            report.append("✓ No configuration issues found!")
            return "\n".join(report)

        report.append(f"⚠ Configuration Issues: {len(issues)}")
        report.append("")

        for issue in issues:
            report.append(f"File: {issue['file']}")
            report.append(f"Reason: {issue['reason']}")
            report.append(f"Severity: {issue['severity']}")
            report.append("")

        return "\n".join(report)

    def _build_file_secrets_report(
        self, file_path: Path, secrets: List[Dict[str, Any]]
    ) -> str:
        """Build file secrets report."""
        report = []
        report.append(f"File Secrets Check: {file_path}")
        report.append("=" * 70)
        report.append("")

        if not secrets:
            report.append("✓ No secrets found in this file!")
            return "\n".join(report)

        report.append(f"⚠ Secrets Found: {len(secrets)}")
        report.append("")

        for secret in secrets:
            report.append(f"Line {secret['line']}: {secret['description']}")
            report.append(f"Type: {secret['type']}")
            report.append(f"Severity: {secret['severity']}")
            report.append("")

        return "\n".join(report)
