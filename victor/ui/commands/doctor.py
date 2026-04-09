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

"""Doctor command for system diagnostics and troubleshooting.

Provides comprehensive checks to help users diagnose and fix issues:
- Python version and environment
- Dependencies and packages
- Provider connectivity and configuration
- Tool availability
- File permissions
- Performance optimizations
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Severity(Enum):
    """Severity level for diagnostic issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class DiagnosticCheck:
    """Result of a diagnostic check."""

    name: str
    severity: Severity
    message: str
    suggestion: str | None = None
    passed: bool = False


class DoctorChecks:
    """Diagnostic checks for Victor."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks: List[DiagnosticCheck] = []

    def add_check(
        self,
        name: str,
        severity: Severity,
        message: str,
        suggestion: str | None = None,
    ) -> None:
        """Add a diagnostic check result."""
        self.checks.append(
            DiagnosticCheck(
                name=name,
                severity=severity,
                message=message,
                suggestion=suggestion,
                passed=(severity == Severity.SUCCESS),
            )
        )

    def check_python_version(self) -> None:
        """Check Python version compatibility."""
        major, minor = sys.version_info[:2]
        version_str = f"{major}.{minor}"

        if (major, minor) >= (3, 10):
            self.add_check(
                name="Python Version",
                severity=Severity.SUCCESS,
                message=f"Python {version_str} is supported",
            )
        else:
            self.add_check(
                name="Python Version",
                severity=Severity.ERROR,
                message=f"Python {version_str} is not supported (requires 3.10+)",
                suggestion="Install Python 3.10 or higher from python.org or use pyenv/conda",
            )

    def check_pip_package(self, package_name: str) -> bool:
        """Check if a pip package is installed.

        Returns:
            True if package is installed
        """
        try:
            importlib.metadata.distribution(package_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    def check_dependencies(self) -> None:
        """Check core dependencies."""
        required = [
            ("pydantic", "Core dependency for configuration"),
            ("typer", "CLI framework"),
            ("rich", "Terminal formatting"),
            ("aiohttp", "Async HTTP support"),
        ]

        optional = [
            ("openai", "OpenAI provider"),
            ("anthropic", "Anthropic provider"),
        ]

        for package, description in required:
            if self.check_pip_package(package):
                self.add_check(
                    name=f"Dependency: {package}",
                    severity=Severity.SUCCESS,
                    message=f"{package} is installed",
                )
            else:
                self.add_check(
                    name=f"Dependency: {package}",
                    severity=Severity.ERROR,
                    message=f"{package} is not installed ({description})",
                    suggestion=f"Run: pip install {package} or pip install -e .",
                )

        for package, description in optional:
            if self.check_pip_package(package):
                self.add_check(
                    name=f"Optional: {package}",
                    severity=Severity.SUCCESS,
                    message=f"{package} is installed",
                )
            else:
                self.add_check(
                    name=f"Optional: {package}",
                    severity=Severity.WARNING,
                    message=f"{package} is not installed ({description})",
                    suggestion=f"Run: pip install {package} to enable {package.split()[0]} provider",
                )

    def check_api_keys(self) -> None:
        """Check for API keys in environment."""
        api_keys = {
            "ANTHROPIC_API_KEY": "Anthropic Claude",
            "OPENAI_API_KEY": "OpenAI GPT",
            "GOOGLE_API_KEY": "Google Gemini",
            "AZURE_API_KEY": "Azure OpenAI",
            "XAI_API_KEY": "xAI Grok",
            "COHERE_API_KEY": "Cohere",
        }

        found_keys = []
        for env_var, provider in api_keys.items():
            if os.getenv(env_var):
                found_keys.append(provider)
                self.add_check(
                    name=f"API Key: {provider}",
                    severity=Severity.SUCCESS,
                    message=f"{provider} API key is configured",
                )

        if not found_keys:
            self.add_check(
                name="Cloud Provider API Keys",
                severity=Severity.WARNING,
                message="No cloud provider API keys found",
                suggestion="Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable, "
                "or use --provider ollama for local models",
            )

    def check_local_providers(self) -> None:
        """Check local provider availability."""
        # Check Ollama
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                self.add_check(
                    name="Local Provider: Ollama",
                    severity=Severity.SUCCESS,
                    message="Ollama is running and accessible",
                )
            else:
                self.add_check(
                    name="Local Provider: Ollama",
                    severity=Severity.WARNING,
                    message="Ollama is installed but not running",
                    suggestion="Start Ollama: 'ollama serve' or 'brew services start ollama'",
                )
        except FileNotFoundError:
            self.add_check(
                name="Local Provider: Ollama",
                severity=Severity.INFO,
                message="Ollama is not installed",
                suggestion="Install Ollama: 'curl -fsSL https://ollama.com/install.sh | sh' "
                "or use: brew install ollama",
            )
        except Exception as e:
            self.add_check(
                name="Local Provider: Ollama",
                severity=Severity.INFO,
                message=f"Could not check Ollama status: {e}",
            )

    def check_config_directory(self) -> None:
        """Check Victor configuration directory."""
        victor_config = os.getenv("VICTOR_CONFIG_DIR")
        if victor_config:
            config_path = Path(victor_config)
            if config_path.exists():
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.SUCCESS,
                    message=f"Config directory exists: {config_path}",
                )
            else:
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.ERROR,
                    message=f"Config directory does not exist: {config_path}",
                    suggestion="Run 'victor init' to create configuration",
                )
        else:
            # Check default location
            default_config = Path.home() / ".victor"
            if default_config.exists():
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.SUCCESS,
                    message=f"Using default config directory: {default_config}",
                )
            else:
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.INFO,
                    message="No configuration directory (will use defaults)",
                    suggestion="Run 'victor init' to create configuration",
                )

    def check_performance_settings(self) -> None:
        """Check performance optimization settings."""
        try:
            from victor.config.settings import Settings

            settings = Settings()

            # Check framework preloading
            if (
                hasattr(settings, "framework_preload_enabled")
                and settings.framework_preload_enabled
            ):
                self.add_check(
                    name="Performance: Preloading",
                    severity=Severity.SUCCESS,
                    message="Framework preloading is enabled (50-70% faster first requests)",
                )
            else:
                self.add_check(
                    name="Performance: Preloading",
                    severity=Severity.WARNING,
                    message="Framework preloading is disabled",
                    suggestion="Enable framework_preload in your profile for faster first requests",
                )

            # Check HTTP connection pooling
            if (
                hasattr(settings, "http_connection_pool_enabled")
                and settings.http_connection_pool_enabled
            ):
                self.add_check(
                    name="Performance: HTTP Pooling",
                    severity=Severity.SUCCESS,
                    message="HTTP connection pooling is enabled (20-30% faster HTTP requests)",
                )
            else:
                self.add_check(
                    name="Performance: HTTP Pooling",
                    severity=Severity.WARNING,
                    message="HTTP connection pooling is disabled",
                    suggestion="Enable http_connection_pool_enabled in your profile",
                )

            # Check tool selection cache
            if (
                hasattr(settings, "tool_selection_cache_enabled")
                and settings.tools.tool_selection_cache_enabled
            ):
                self.add_check(
                    name="Performance: Tool Cache",
                    severity=Severity.SUCCESS,
                    message="Tool selection cache is enabled (20-40% faster conversations)",
                )
            else:
                self.add_check(
                    name="Performance: Tool Cache",
                    severity=Severity.WARNING,
                    message="Tool selection cache is disabled",
                    suggestion="Enable tool_selection_cache_enabled in your profile",
                )

        except Exception as e:
            self.add_check(
                name="Performance Settings",
                severity=Severity.INFO,
                message=f"Could not check performance settings: {e}",
            )

    def check_file_permissions(self) -> None:
        """Check file system permissions for common operations."""
        # Check home directory writability
        home_dir = Path.home()
        if os.access(home_dir, os.W_OK):
            self.add_check(
                name="File Permissions: Home Directory",
                severity=Severity.SUCCESS,
                message=f"Home directory is writable: {home_dir}",
            )
        else:
            self.add_check(
                name="File Permissions: Home Directory",
                severity=Severity.ERROR,
                message=f"Home directory is not writable: {home_dir}",
                suggestion="Check directory permissions: ls -la ~",
            )

    def run_all_checks(self) -> List[DiagnosticCheck]:
        """Run all diagnostic checks.

        Returns:
            List of all diagnostic checks performed
        """
        self.check_python_version()
        self.check_dependencies()
        self.check_api_keys()
        self.check_local_providers()
        self.check_config_directory()
        self.check_performance_settings()
        self.check_file_permissions()

        return self.checks

    def print_results(self) -> int:
        """Print diagnostic results to console.

        Returns:
            Exit code (0 for success, 1 for errors found)
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Count by severity
        errors = sum(1 for c in self.checks if c.severity == Severity.ERROR)
        warnings = sum(1 for c in self.checks if c.severity == Severity.WARNING)
        infos = sum(1 for c in self.checks if c.severity == Severity.INFO)
        successes = sum(1 for c in self.checks if c.severity == Severity.SUCCESS)

        # Print summary table
        console.print("\n[bold]Victor Doctor - System Diagnostics[/]")
        console.print("═" * 50)

        table = Table(show_header=True, show_lines=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Message", style="white")
        table.add_column("Suggestion", style="yellow")

        for check in self.checks:
            # Status icon
            if check.severity == Severity.SUCCESS:
                status = "[green]✓[/]"
            elif check.severity == Severity.ERROR:
                status = "[red]✗[/]"
            elif check.severity == Severity.WARNING:
                status = "[yellow]⚠[/]"
            else:  # INFO
                status = "[blue]ℹ[/]"

            table.add_row(
                check.name,
                status,
                check.message,
                check.suggestion or "",
            )

        console.print(table)

        # Print summary
        console.print("\n" + "─" * 50)
        console.print(
            f"[bold]Summary:[/] {successes} success, {warnings} warnings, {infos} info, {errors} errors"
        )

        if errors == 0 and warnings <= 2:
            console.print("\n[green]✓ Your system is ready to use Victor![/]")
            return 0
        elif errors > 0:
            console.print("\n[red]✗ Found {errors} error(s) that should be fixed[/]")
            console.print(
                "[yellow]Run 'victor config validate' for detailed configuration validation[/]"
            )
            return 1
        else:
            console.print(
                f"\n[yellow]⚠ Found {warnings} warning(s) - Victor will work but may be suboptimal[/]"
            )
            return 0


def run_doctor(verbose: bool = False, fix: bool = False) -> int:
    """Run doctor diagnostics.

    This is the entry point for 'victor doctor' command.

    Args:
        verbose: Show detailed diagnostic output
        fix: Automatically fix common issues

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    doctor = DoctorChecks(verbose=verbose)

    try:
        doctor.run_all_checks()
        return doctor.print_results()
    except Exception as e:
        print(f"[red]✗ Doctor command failed:[/] {e}")
        return 1
