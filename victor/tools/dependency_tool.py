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
from typing import Any, Dict, List, Optional
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Known vulnerability databases
VULN_PACKAGES = {
    "django": {"<2.2.28": "CVE-2022-28346", "<3.2.13": "CVE-2022-28347"},
    "pillow": {"<9.0.0": "CVE-2022-22817"},
    "requests": {"<2.26.0": "CVE-2021-33503"},
}


def _parse_version(version: str) -> tuple:
    """Parse version string into comparable tuple."""
    try:
        parts = re.findall(r'\d+', version)
        return tuple(int(p) for p in parts)
    except (ValueError, TypeError, AttributeError):
        return (0, 0, 0)


def _version_satisfies(current: str, constraint: str) -> bool:
    """Check if current version satisfies constraint."""
    if constraint.startswith("<"):
        return _parse_version(current) < _parse_version(constraint[1:])
    return False


@tool
async def dependency_list() -> Dict[str, Any]:
    """
    List all installed Python packages.

    Returns a formatted list of all installed packages with their versions,
    grouped by first letter for easy browsing.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - packages: List of package dicts (name, version)
        - count: Total number of packages
        - formatted_report: Human-readable package list
        - error: Error message if failed
    """
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

        return {
            "success": True,
            "packages": packages,
            "count": len(packages),
            "formatted_report": "\n".join(report)
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Failed to list packages: {e.stderr}"
        }


@tool
async def dependency_outdated() -> Dict[str, Any]:
    """
    Check for outdated Python packages.

    Identifies packages that have newer versions available and categorizes
    them by update severity (major, minor, patch).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - outdated: List of outdated package dicts
        - count: Number of outdated packages
        - by_severity: Categorized updates (major, minor, patch)
        - formatted_report: Human-readable outdated list
        - error: Error message if failed
    """
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )

        outdated = json.loads(result.stdout)

        if not outdated:
            return {
                "success": True,
                "outdated": [],
                "count": 0,
                "message": "✅ All packages are up to date!"
            }

        # Categorize by severity
        major_updates = []
        minor_updates = []
        patch_updates = []

        for pkg in outdated:
            current = _parse_version(pkg["version"])
            latest = _parse_version(pkg["latest_version"])

            if latest[0] > current[0]:
                major_updates.append(pkg)
            elif latest[1] > current[1]:
                minor_updates.append(pkg)
            else:
                patch_updates.append(pkg)

        # Build report
        report = []
        report.append("Outdated Packages")
        report.append("=" * 70)
        report.append("")
        report.append(f"Total: {len(outdated)} packages need updating")
        report.append("")

        if major_updates:
            report.append(f"⚠️  Major updates ({len(major_updates)}):")
            for pkg in major_updates[:5]:
                report.append(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            if len(major_updates) > 5:
                report.append(f"  ... and {len(major_updates) - 5} more")
            report.append("")

        if minor_updates:
            report.append(f"📦 Minor updates ({len(minor_updates)}):")
            for pkg in minor_updates[:5]:
                report.append(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            if len(minor_updates) > 5:
                report.append(f"  ... and {len(minor_updates) - 5} more")
            report.append("")

        if patch_updates:
            report.append(f"🔧 Patch updates ({len(patch_updates)}):")
            for pkg in patch_updates[:5]:
                report.append(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            if len(patch_updates) > 5:
                report.append(f"  ... and {len(patch_updates) - 5} more")

        return {
            "success": True,
            "outdated": outdated,
            "count": len(outdated),
            "by_severity": {
                "major": major_updates,
                "minor": minor_updates,
                "patch": patch_updates
            },
            "formatted_report": "\n".join(report)
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Failed to check outdated packages: {e.stderr}"
        }


@tool
async def dependency_security() -> Dict[str, Any]:
    """
    Check for security vulnerabilities in dependencies.

    Scans installed packages against known vulnerability databases
    to identify security issues.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - vulnerabilities: List of vulnerable packages
        - count: Number of vulnerabilities found
        - formatted_report: Human-readable security report
        - error: Error message if failed
    """
    try:
        # Get installed packages
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )

        packages = json.loads(result.stdout)

        # Check for vulnerabilities
        vulnerabilities = []

        for pkg in packages:
            pkg_name = pkg["name"].lower()
            pkg_version = pkg["version"]

            if pkg_name in VULN_PACKAGES:
                for constraint, cve in VULN_PACKAGES[pkg_name].items():
                    if _version_satisfies(pkg_version, constraint):
                        vulnerabilities.append({
                            "package": pkg_name,
                            "version": pkg_version,
                            "cve": cve,
                            "constraint": constraint
                        })

        # Build report
        report = []
        report.append("Security Audit")
        report.append("=" * 70)
        report.append("")

        if not vulnerabilities:
            report.append("✅ No known vulnerabilities found!")
            report.append("")
            report.append("Note: This is a basic check against known CVEs.")
            report.append("For comprehensive security auditing, use: pip-audit or safety")
        else:
            report.append(f"⚠️  Found {len(vulnerabilities)} potential vulnerabilities:")
            report.append("")

            for vuln in vulnerabilities:
                report.append(f"  {vuln['package']} {vuln['version']}")
                report.append(f"    CVE: {vuln['cve']}")
                report.append(f"    Affected: {vuln['constraint']}")
                report.append("")

        return {
            "success": True,
            "vulnerabilities": vulnerabilities,
            "count": len(vulnerabilities),
            "formatted_report": "\n".join(report)
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Security audit failed: {e.stderr}"
        }


@tool
async def dependency_generate(output: str = "requirements.txt") -> Dict[str, Any]:
    """
    Generate a requirements file from installed packages.

    Creates a requirements.txt file listing all installed packages
    with their versions.

    Args:
        output: Output filename (default: 'requirements.txt').

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - file: Path to generated file
        - packages_count: Number of packages written
        - message: Status message
        - error: Error message if failed
    """
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )

        requirements = result.stdout.strip()
        package_count = len([line for line in requirements.split('\n') if line and not line.startswith('#')])

        # Write to file
        output_path = Path(output)
        output_path.write_text(requirements)

        return {
            "success": True,
            "file": str(output_path),
            "packages_count": package_count,
            "message": f"Generated {output} with {package_count} packages"
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Failed to generate requirements: {e.stderr}"
        }
    except IOError as e:
        return {
            "success": False,
            "error": f"Failed to write file: {str(e)}"
        }


@tool
async def dependency_update(packages: List[str], dry_run: bool = True) -> Dict[str, Any]:
    """
    Update Python packages.

    Updates specified packages to their latest versions. Runs in dry-run
    mode by default for safety.

    Args:
        packages: List of package names to update.
        dry_run: If True, show what would be updated without actually updating (default: True).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - updated: List of packages that were/would be updated
        - message: Status message
        - error: Error message if failed
    """
    if not packages:
        return {
            "success": False,
            "error": "No packages specified for update"
        }

    if dry_run:
        return {
            "success": True,
            "would_update": packages,
            "message": f"Dry run: Would update {len(packages)} packages: {', '.join(packages)}"
        }

    try:
        updated = []
        for package in packages:
            result = subprocess.run(
                ["pip", "install", "--upgrade", package],
                capture_output=True,
                text=True,
                check=True,
            )
            updated.append(package)

        return {
            "success": True,
            "updated": updated,
            "message": f"Successfully updated {len(updated)} packages"
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Failed to update packages: {e.stderr}",
            "partially_updated": updated
        }


@tool
async def dependency_tree(package: Optional[str] = None) -> Dict[str, Any]:
    """
    Show dependency tree.

    Displays the dependency tree for a specific package or all packages.
    Requires 'pipdeptree' to be installed.

    Args:
        package: Optional package name to show tree for (shows all if not specified).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - tree: Dependency tree output
        - error: Error message if failed
    """
    try:
        # Check if pipdeptree is installed
        check = subprocess.run(
            ["pip", "show", "pipdeptree"],
            capture_output=True,
            text=True,
        )

        if check.returncode != 0:
            return {
                "success": False,
                "error": "pipdeptree not installed. Install with: pip install pipdeptree"
            }

        # Run pipdeptree
        cmd = ["pipdeptree"]
        if package:
            cmd.extend(["--packages", package])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        return {
            "success": True,
            "tree": result.stdout,
            "package": package
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Failed to show dependency tree: {e.stderr}"
        }


@tool
async def dependency_check(requirements_file: str = "requirements.txt") -> Dict[str, Any]:
    """
    Check if installed packages match requirements file.

    Verifies that all packages specified in the requirements file are
    installed with the correct versions.

    Args:
        requirements_file: Path to requirements file (default: 'requirements.txt').

    Returns:
        Dictionary containing:
        - success: Whether all requirements are satisfied
        - missing: List of missing packages
        - mismatched: List of version mismatches
        - formatted_report: Human-readable check report
        - error: Error message if failed
    """
    try:
        req_path = Path(requirements_file)

        if not req_path.exists():
            return {
                "success": False,
                "error": f"Requirements file not found: {requirements_file}"
            }

        # Get installed packages
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )

        installed = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}

        # Parse requirements
        requirements = req_path.read_text().strip().split('\n')
        missing = []
        mismatched = []
        satisfied = []

        for req in requirements:
            req = req.strip()
            if not req or req.startswith('#'):
                continue

            # Parse requirement
            match = re.match(r'([a-zA-Z0-9_-]+)(==|>=|<=)?([\d.]+)?', req)
            if not match:
                continue

            pkg_name = match.group(1).lower()
            operator = match.group(2)
            version = match.group(3)

            if pkg_name not in installed:
                missing.append(req)
            elif operator == "==" and version and installed[pkg_name] != version:
                mismatched.append({
                    "package": pkg_name,
                    "required": version,
                    "installed": installed[pkg_name]
                })
            else:
                satisfied.append(pkg_name)

        # Build report
        report = []
        report.append(f"Requirements Check: {requirements_file}")
        report.append("=" * 70)
        report.append("")

        if not missing and not mismatched:
            report.append("✅ All requirements satisfied!")
        else:
            if missing:
                report.append(f"❌ Missing packages ({len(missing)}):")
                for pkg in missing:
                    report.append(f"  {pkg}")
                report.append("")

            if mismatched:
                report.append(f"⚠️  Version mismatches ({len(mismatched)}):")
                for mm in mismatched:
                    report.append(f"  {mm['package']}: required {mm['required']}, installed {mm['installed']}")
                report.append("")

            report.append(f"✅ Satisfied: {len(satisfied)} packages")

        return {
            "success": len(missing) == 0 and len(mismatched) == 0,
            "missing": missing,
            "mismatched": mismatched,
            "satisfied_count": len(satisfied),
            "formatted_report": "\n".join(report)
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Failed to check requirements: {e.stderr}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to check requirements: {str(e)}"
        }


# Keep class for backward compatibility
class DependencyTool:
    """Deprecated: Use individual dependency_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "DependencyTool class is deprecated. Use dependency_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
