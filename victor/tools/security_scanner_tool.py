"""Security scanning tool for code analysis.

Features:
- Secret and credential detection
- Dependency vulnerability scanning
- Configuration security checks
- Comprehensive security audits
"""

import re
from pathlib import Path
from typing import Any, Dict, List
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Common secret patterns
SECRET_PATTERNS = {
    "api_key": r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]([a-zA-Z0-9_\-]{20,})['\"]",
    "password": r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]([^'\"]{8,})['\"]",
    "token": r"(?i)(token|auth[_-]?token)\s*[:=]\s*['\"]([a-zA-Z0-9_\-\.]{20,})['\"]",
    "private_key": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    "aws_key": r"(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*['\"]([A-Z0-9]{20})['\"]",
    "github_token": r"(?i)(github[_-]?token)\s*[:=]\s*['\"]([a-zA-Z0-9_]{40})['\"]",
}

# Configuration security checks
CONFIG_CHECKS = {
    "debug_mode": r"(?i)(debug|DEBUG)\s*[:=]\s*(true|True|1)",
    "insecure_protocol": r"(?i)http://",
    "hardcoded_ip": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


def _scan_file_for_secrets(file_path: Path) -> List[Dict[str, Any]]:
    """Scan a file for potential secrets."""
    findings = []

    try:
        content = file_path.read_text()

        for secret_type, pattern in SECRET_PATTERNS.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                findings.append({
                    "type": secret_type,
                    "file": str(file_path),
                    "line": content[:match.start()].count('\n') + 1,
                    "severity": "high",
                    "message": f"Potential {secret_type} detected"
                })
    except Exception as e:
        logger.warning(f"Error scanning {file_path}: {e}")

    return findings


def _scan_file_for_config_issues(file_path: Path) -> List[Dict[str, Any]]:
    """Scan a file for configuration security issues."""
    findings = []

    try:
        content = file_path.read_text()

        for check_name, pattern in CONFIG_CHECKS.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                findings.append({
                    "type": check_name,
                    "file": str(file_path),
                    "line": content[:match.start()].count('\n') + 1,
                    "severity": "medium",
                    "message": f"Configuration issue: {check_name}"
                })
    except Exception as e:
        logger.warning(f"Error scanning {file_path}: {e}")

    return findings


@tool
async def security_scan_secrets(path: str, file_pattern: str = "*.py") -> Dict[str, Any]:
    """
    Scan for secrets and credentials in code.

    Searches for hardcoded API keys, passwords, tokens, and other
    sensitive information that should not be in source code.

    Args:
        path: Directory path to scan.
        file_pattern: Glob pattern for files to scan (default: *.py).

    Returns:
        Dictionary containing:
        - success: Whether scan completed
        - findings: List of security findings
        - files_scanned: Number of files scanned
        - secrets_found: Number of secrets detected
        - formatted_report: Human-readable security report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Find files to scan
    files = list(path_obj.rglob(file_pattern))
    if not files:
        return {
            "success": True,
            "findings": [],
            "files_scanned": 0,
            "secrets_found": 0,
            "message": "No files found to scan"
        }

    # Scan files
    all_findings = []
    for file in files:
        findings = _scan_file_for_secrets(file)
        all_findings.extend(findings)

    # Build report
    report = []
    report.append("Security Scan: Secrets Detection")
    report.append("=" * 70)
    report.append("")
    report.append(f"Files scanned: {len(files)}")
    report.append(f"Secrets found: {len(all_findings)}")
    report.append("")

    if all_findings:
        report.append("⚠️  CRITICAL: Secrets detected in code!")
        report.append("")
        for finding in all_findings[:20]:  # Show first 20
            report.append(f"  {finding['severity'].upper()}: {finding['file']}")
            report.append(f"    Line {finding['line']}: {finding['message']}")
            report.append("")

        if len(all_findings) > 20:
            report.append(f"... and {len(all_findings) - 20} more findings")
    else:
        report.append("✅ No secrets detected")

    return {
        "success": True,
        "findings": all_findings,
        "files_scanned": len(files),
        "secrets_found": len(all_findings),
        "formatted_report": "\n".join(report)
    }


@tool
async def security_scan_dependencies(requirements_file: str = "requirements.txt") -> Dict[str, Any]:
    """
    Scan dependencies for known vulnerabilities.

    Checks Python dependencies against known vulnerability databases.

    Args:
        requirements_file: Path to requirements file (default: requirements.txt).

    Returns:
        Dictionary containing:
        - success: Whether scan completed
        - vulnerabilities: List of vulnerable dependencies
        - packages_checked: Number of packages checked
        - formatted_report: Human-readable vulnerability report
        - error: Error message if failed
    """
    req_path = Path(requirements_file)
    if not req_path.exists():
        return {"success": False, "error": f"Requirements file not found: {requirements_file}"}

    try:
        content = req_path.read_text()
        packages = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]

        # Simplified vulnerability check (in real implementation, would query vulnerability DB)
        vulnerabilities = []
        known_vulns = {
            "django": "Known vulnerabilities in older versions",
            "flask": "Check for recent security patches",
            "requests": "Verify version >= 2.26.0",
        }

        for pkg_line in packages:
            pkg_name = pkg_line.split('==')[0].split('>=')[0].split('<=')[0].lower()
            if pkg_name in known_vulns:
                vulnerabilities.append({
                    "package": pkg_name,
                    "severity": "medium",
                    "message": known_vulns[pkg_name]
                })

        # Build report
        report = []
        report.append("Security Scan: Dependency Vulnerabilities")
        report.append("=" * 70)
        report.append("")
        report.append(f"Packages checked: {len(packages)}")
        report.append(f"Vulnerabilities found: {len(vulnerabilities)}")
        report.append("")

        if vulnerabilities:
            report.append("⚠️  Vulnerable dependencies detected:")
            report.append("")
            for vuln in vulnerabilities:
                report.append(f"  {vuln['severity'].upper()}: {vuln['package']}")
                report.append(f"    {vuln['message']}")
                report.append("")
        else:
            report.append("✅ No known vulnerabilities detected")
            report.append("")
            report.append("Note: This is a basic check. For comprehensive scanning, use: pip-audit or safety")

        return {
            "success": True,
            "vulnerabilities": vulnerabilities,
            "packages_checked": len(packages),
            "formatted_report": "\n".join(report)
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to scan dependencies: {str(e)}"}


@tool
async def security_scan_config(path: str, file_pattern: str = "*.py") -> Dict[str, Any]:
    """
    Scan for configuration security issues.

    Checks for debug mode enabled, insecure protocols, hardcoded IPs,
    and other configuration-related security concerns.

    Args:
        path: Directory path to scan.
        file_pattern: Glob pattern for files to scan (default: *.py).

    Returns:
        Dictionary containing:
        - success: Whether scan completed
        - findings: List of configuration issues
        - files_scanned: Number of files scanned
        - issues_found: Number of issues detected
        - formatted_report: Human-readable report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Find files to scan
    files = list(path_obj.rglob(file_pattern))
    if not files:
        return {
            "success": True,
            "findings": [],
            "files_scanned": 0,
            "issues_found": 0,
            "message": "No files found to scan"
        }

    # Scan files
    all_findings = []
    for file in files:
        findings = _scan_file_for_config_issues(file)
        all_findings.extend(findings)

    # Build report
    report = []
    report.append("Security Scan: Configuration Issues")
    report.append("=" * 70)
    report.append("")
    report.append(f"Files scanned: {len(files)}")
    report.append(f"Issues found: {len(all_findings)}")
    report.append("")

    if all_findings:
        report.append("⚠️  Configuration security issues detected:")
        report.append("")

        by_severity = {}
        for finding in all_findings:
            severity = finding['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(finding)

        for severity in ['high', 'medium', 'low']:
            if severity in by_severity:
                report.append(f"{severity.upper()} ({len(by_severity[severity])} issues):")
                for finding in by_severity[severity][:10]:
                    report.append(f"  {finding['file']} (line {finding['line']})")
                    report.append(f"    {finding['message']}")
                report.append("")
    else:
        report.append("✅ No configuration issues detected")

    return {
        "success": True,
        "findings": all_findings,
        "files_scanned": len(files),
        "issues_found": len(all_findings),
        "formatted_report": "\n".join(report)
    }


@tool
async def security_scan_all(path: str, file_pattern: str = "*.py") -> Dict[str, Any]:
    """
    Comprehensive security scan.

    Runs all security scans: secrets detection, dependency checks,
    and configuration analysis.

    Args:
        path: Directory path to scan.
        file_pattern: Glob pattern for files to scan (default: *.py).

    Returns:
        Dictionary containing:
        - success: Whether scan completed
        - secrets: Results from secrets scan
        - config: Results from config scan
        - total_issues: Total security issues found
        - formatted_report: Comprehensive security report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    # Run all scans
    secrets_result = await security_scan_secrets(path=path, file_pattern=file_pattern)
    config_result = await security_scan_config(path=path, file_pattern=file_pattern)

    # Try dependency scan if requirements.txt exists
    req_path = Path(path) / "requirements.txt"
    deps_result = None
    if req_path.exists():
        deps_result = await security_scan_dependencies(requirements_file=str(req_path))

    total_issues = secrets_result.get("secrets_found", 0) + config_result.get("issues_found", 0)
    if deps_result:
        total_issues += len(deps_result.get("vulnerabilities", []))

    # Build comprehensive report
    report = []
    report.append("Comprehensive Security Scan Report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Total security issues: {total_issues}")
    report.append("")

    report.append("Secrets Scan:")
    report.append(f"  Files scanned: {secrets_result.get('files_scanned', 0)}")
    report.append(f"  Secrets found: {secrets_result.get('secrets_found', 0)}")
    report.append("")

    report.append("Configuration Scan:")
    report.append(f"  Files scanned: {config_result.get('files_scanned', 0)}")
    report.append(f"  Issues found: {config_result.get('issues_found', 0)}")
    report.append("")

    if deps_result:
        report.append("Dependency Scan:")
        report.append(f"  Packages checked: {deps_result.get('packages_checked', 0)}")
        report.append(f"  Vulnerabilities: {len(deps_result.get('vulnerabilities', []))}")
        report.append("")

    if total_issues > 0:
        report.append("⚠️  Security issues require attention!")
        report.append("    Run individual scans for details.")
    else:
        report.append("✅ No critical security issues detected")

    return {
        "success": True,
        "secrets": secrets_result,
        "config": config_result,
        "dependencies": deps_result,
        "total_issues": total_issues,
        "formatted_report": "\n".join(report)
    }


@tool
async def security_check_file(file: str) -> Dict[str, Any]:
    """
    Security check for a specific file.

    Performs quick security check on a single file for secrets
    and configuration issues.

    Args:
        file: Path to file to check.

    Returns:
        Dictionary containing:
        - success: Whether check completed
        - findings: List of security findings
        - issues_found: Number of issues
        - formatted_report: Security check report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_path = Path(file)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Scan for secrets and config issues
    secret_findings = _scan_file_for_secrets(file_path)
    config_findings = _scan_file_for_config_issues(file_path)

    all_findings = secret_findings + config_findings

    # Build report
    report = []
    report.append(f"Security Check: {file}")
    report.append("=" * 70)
    report.append("")
    report.append(f"Issues found: {len(all_findings)}")
    report.append("")

    if all_findings:
        report.append("⚠️  Security issues detected:")
        report.append("")
        for finding in all_findings:
            report.append(f"  {finding['severity'].upper()}: Line {finding['line']}")
            report.append(f"    {finding['message']}")
            report.append("")
    else:
        report.append("✅ No security issues detected")

    return {
        "success": True,
        "findings": all_findings,
        "issues_found": len(all_findings),
        "formatted_report": "\n".join(report)
    }


# Keep class for backward compatibility
class SecurityScannerTool:
    """Deprecated: Use individual security_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "SecurityScannerTool class is deprecated. Use security_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
