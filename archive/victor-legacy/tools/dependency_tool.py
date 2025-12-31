"""Dependency management tool for Python projects.

Features:
- Analyze project dependencies
- Check for outdated packages
- Find security vulnerabilities
- Generate requirements files
- Update dependencies
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class DependencyTool(BaseTool):
    """Tool for dependency management."""

    # Known vulnerability databases
    VULN_PACKAGES = {
        "django": {"<2.2.28": "CVE-2022-28346", "<3.2.13": "CVE-2022-28347"},
        "pillow": {"<9.0.0": "CVE-2022-22817"},
        "requests": {"<2.26.0": "CVE-2021-33503"},
    }

    @property
    def name(self) -> str:
        """Get tool name."""
        return "dependency"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Dependency management for Python projects.

Manage and analyze project dependencies:
- List installed packages
- Check for outdated dependencies
- Find security vulnerabilities
- Generate requirements files
- Update dependencies safely
- Analyze dependency tree

Operations:
- list: List all installed packages
- outdated: Find outdated packages
- security: Check for security vulnerabilities
- generate: Generate requirements file
- update: Update dependencies
- tree: Show dependency tree
- check: Verify dependencies match requirements

Example workflows:
1. List installed packages:
   dependency(operation="list")

2. Check for outdated:
   dependency(operation="outdated")

3. Security audit:
   dependency(operation="security")

4. Generate requirements:
   dependency(operation="generate", output="requirements.txt")
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
            [
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation: list, outdated, security, generate, update, tree, check",
                    required=True,
                ),
                ToolParameter(
                    name="package",
                    type="string",
                    description="Specific package name",
                    required=False,
                ),
                ToolParameter(
                    name="output",
                    type="string",
                    description="Output file path",
                    required=False,
                ),
                ToolParameter(
                    name="format",
                    type="string",
                    description="Output format: txt, json, freeze",
                    required=False,
                ),
            ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute dependency operation.

        Args:
            operation: Dependency operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with dependency information
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "list":
                return await self._list_packages(kwargs)
            elif operation == "outdated":
                return await self._check_outdated(kwargs)
            elif operation == "security":
                return await self._security_audit(kwargs)
            elif operation == "generate":
                return await self._generate_requirements(kwargs)
            elif operation == "update":
                return await self._update_packages(kwargs)
            elif operation == "tree":
                return await self._show_tree(kwargs)
            elif operation == "check":
                return await self._check_requirements(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Dependency operation failed")
            return ToolResult(success=False, output="", error=f"Dependency error: {str(e)}")

    async def _list_packages(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List installed packages."""
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )

            packages = json.loads(result.stdout)

            # Build report
            report = []
            report.append("Installed Packages")
            report.append("=" * 70)
            report.append("")
            report.append(f"Total: {len(packages)} packages")
            report.append("")

            # Group by first letter
            by_letter: Dict[str, List[Dict]] = {}
            for pkg in packages:
                letter = pkg["name"][0].upper()
                if letter not in by_letter:
                    by_letter[letter] = []
                by_letter[letter].append(pkg)

            # Show packages
            for letter in sorted(by_letter.keys()):
                pkgs = by_letter[letter]
                report.append(f"{letter}:")
                for pkg in sorted(pkgs, key=lambda x: x["name"])[:5]:
                    report.append(f"  {pkg['name']} ({pkg['version']})")
                if len(pkgs) > 5:
                    report.append(f"  ... and {len(pkgs) - 5} more")
                report.append("")

            return ToolResult(
                success=True,
                output="\n".join(report),
                error="",
            )

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list packages: {e.stderr}",
            )

    async def _check_outdated(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Check for outdated packages."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )

            outdated = json.loads(result.stdout)

            if not outdated:
                return ToolResult(
                    success=True,
                    output="✅ All packages are up to date!",
                    error="",
                )

            # Build report
            report = []
            report.append("Outdated Packages")
            report.append("=" * 70)
            report.append("")
            report.append(f"Total: {len(outdated)} packages need updating")
            report.append("")

            # Categorize by severity
            major_updates = []
            minor_updates = []
            patch_updates = []

            for pkg in outdated:
                current = pkg["version"]
                latest = pkg["latest_version"]

                update_type = self._classify_update(current, latest)

                if update_type == "major":
                    major_updates.append(pkg)
                elif update_type == "minor":
                    minor_updates.append(pkg)
                else:
                    patch_updates.append(pkg)

            if major_updates:
                report.append(f"🔴 Major Updates ({len(major_updates)}):")
                for pkg in major_updates[:10]:
                    report.append(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                if len(major_updates) > 10:
                    report.append(f"  ... and {len(major_updates) - 10} more")
                report.append("")

            if minor_updates:
                report.append(f"🟡 Minor Updates ({len(minor_updates)}):")
                for pkg in minor_updates[:10]:
                    report.append(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                if len(minor_updates) > 10:
                    report.append(f"  ... and {len(minor_updates) - 10} more")
                report.append("")

            if patch_updates:
                report.append(f"🟢 Patch Updates ({len(patch_updates)}):")
                for pkg in patch_updates[:10]:
                    report.append(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                if len(patch_updates) > 10:
                    report.append(f"  ... and {len(patch_updates) - 10} more")
                report.append("")

            report.append("Recommendations:")
            report.append("  • Always update patch versions (bug fixes)")
            report.append("  • Review minor updates (new features)")
            report.append("  • Test major updates carefully (breaking changes)")

            return ToolResult(
                success=True,
                output="\n".join(report),
                error="",
            )

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to check outdated: {e.stderr}",
            )

    async def _security_audit(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Check for security vulnerabilities."""
        # Get installed packages
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )

            packages = json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list packages: {e.stderr}",
            )

        # Check against known vulnerabilities
        vulnerabilities = []

        for pkg in packages:
            pkg_name = pkg["name"].lower()
            pkg_version = pkg["version"]

            if pkg_name in self.VULN_PACKAGES:
                vulns = self.VULN_PACKAGES[pkg_name]
                for version_range, cve in vulns.items():
                    if self._version_matches(pkg_version, version_range):
                        vulnerabilities.append(
                            {
                                "package": pkg["name"],
                                "version": pkg_version,
                                "cve": cve,
                                "affected": version_range,
                            }
                        )

        # Build report
        report = []
        report.append("Security Audit")
        report.append("=" * 70)
        report.append("")

        if not vulnerabilities:
            report.append("✅ No known vulnerabilities found!")
            report.append("")
            report.append("Note: This is a basic check against known vulnerabilities.")
            report.append("For comprehensive security auditing, use:")
            report.append("  • pip-audit")
            report.append("  • safety")
            report.append("  • GitHub Dependabot")
        else:
            report.append(f"🔴 Found {len(vulnerabilities)} vulnerabilities:")
            report.append("")

            for vuln in vulnerabilities:
                report.append(f"Package: {vuln['package']} {vuln['version']}")
                report.append(f"  CVE: {vuln['cve']}")
                report.append(f"  Affected: {vuln['affected']}")
                report.append(f"  Action: Update to latest version")
                report.append("")

            report.append("Immediate actions:")
            report.append("  1. Review all vulnerabilities")
            report.append("  2. Update affected packages")
            report.append("  3. Test application thoroughly")
            report.append("  4. Consider using pip-audit for comprehensive scanning")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _generate_requirements(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate requirements file."""
        output_path = kwargs.get("output", "requirements.txt")
        file_format = kwargs.get("format", "freeze")

        try:
            if file_format == "freeze":
                result = subprocess.run(
                    ["pip", "freeze"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                requirements = result.stdout
            else:
                # List format (without versions)
                result = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                packages = json.loads(result.stdout)
                requirements = "\n".join([pkg["name"] for pkg in packages])

            # Write to file
            output_file = Path(output_path)
            output_file.write_text(requirements)

            # Build report
            report = []
            report.append("Requirements Generated")
            report.append("=" * 70)
            report.append("")
            report.append(f"Output: {output_path}")
            report.append(f"Format: {file_format}")
            report.append(f"Packages: {len(requirements.split())}")
            report.append("")
            report.append("Preview:")
            report.append("-" * 70)
            lines = requirements.split("\n")
            report.append("\n".join(lines[:20]))
            if len(lines) > 20:
                report.append(f"... and {len(lines) - 20} more packages")

            return ToolResult(
                success=True,
                output="\n".join(report),
                error="",
            )

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to generate requirements: {e.stderr}",
            )

    async def _update_packages(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Update packages."""
        package = kwargs.get("package")

        if package:
            # Update specific package
            cmd = ["pip", "install", "--upgrade", package]
            target = f"package '{package}'"
        else:
            # Update all (not recommended in production)
            return ToolResult(
                success=False,
                output="",
                error="Updating all packages is not recommended. Specify a package name.",
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            report = []
            report.append(f"Update Complete: {target}")
            report.append("=" * 70)
            report.append("")
            report.append("Output:")
            report.append(result.stdout)

            return ToolResult(
                success=True,
                output="\n".join(report),
                error="",
            )

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to update {target}: {e.stderr}",
            )

    async def _show_tree(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Show dependency tree."""
        package = kwargs.get("package")

        # Try to use pipdeptree if available
        try:
            cmd = ["pipdeptree"]
            if package:
                cmd.extend(["-p", package])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            report = []
            report.append("Dependency Tree")
            report.append("=" * 70)
            report.append("")
            if package:
                report.append(f"Package: {package}")
                report.append("")
            report.append(result.stdout)

            return ToolResult(
                success=True,
                output="\n".join(report),
                error="",
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="pipdeptree not installed. Install with: pip install pipdeptree",
            )
        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to show tree: {e.stderr}",
            )

    async def _check_requirements(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Check if requirements file matches installed packages."""
        req_file = Path("requirements.txt")

        if not req_file.exists():
            return ToolResult(
                success=False,
                output="",
                error="requirements.txt not found",
            )

        # Read requirements
        requirements = {}
        for line in req_file.read_text().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                match = re.match(r"([a-zA-Z0-9\-\_]+)([=<>!]+)?(.*)", line)
                if match:
                    pkg_name = match.group(1)
                    operator = match.group(2) or ""
                    version = match.group(3) or ""
                    requirements[pkg_name.lower()] = (operator, version)

        # Get installed packages
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )

            packages = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list packages: {e.stderr}",
            )

        # Compare
        missing = []
        mismatched = []

        for req_pkg, (operator, req_version) in requirements.items():
            if req_pkg not in packages:
                missing.append(req_pkg)
            elif operator == "==" and packages[req_pkg] != req_version:
                mismatched.append((req_pkg, req_version, packages[req_pkg]))

        # Build report
        report = []
        report.append("Requirements Check")
        report.append("=" * 70)
        report.append("")

        if not missing and not mismatched:
            report.append("✅ All requirements satisfied!")
        else:
            if missing:
                report.append(f"Missing packages ({len(missing)}):")
                for pkg in missing:
                    report.append(f"  • {pkg}")
                report.append("")

            if mismatched:
                report.append(f"Version mismatches ({len(mismatched)}):")
                for pkg, expected, actual in mismatched:
                    report.append(f"  • {pkg}: expected {expected}, got {actual}")
                report.append("")

            report.append("Action:")
            report.append("  pip install -r requirements.txt")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    def _classify_update(self, current: str, latest: str) -> str:
        """Classify update type (major, minor, patch)."""
        try:
            curr_parts = [int(x) for x in current.split(".")[:3]]
            latest_parts = [int(x) for x in latest.split(".")[:3]]

            # Pad to 3 parts
            while len(curr_parts) < 3:
                curr_parts.append(0)
            while len(latest_parts) < 3:
                latest_parts.append(0)

            if latest_parts[0] > curr_parts[0]:
                return "major"
            elif latest_parts[1] > curr_parts[1]:
                return "minor"
            else:
                return "patch"

        except (ValueError, IndexError):
            return "unknown"

    def _version_matches(self, version: str, range_spec: str) -> bool:
        """Check if version matches range specification."""
        # Simple version comparison
        if range_spec.startswith("<"):
            target = range_spec[1:]
            return version < target
        elif range_spec.startswith("<="):
            target = range_spec[2:]
            return version <= target
        elif range_spec.startswith(">"):
            target = range_spec[1:]
            return version > target
        elif range_spec.startswith(">="):
            target = range_spec[2:]
            return version >= target
        else:
            return False
