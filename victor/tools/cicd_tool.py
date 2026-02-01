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

"""CI/CD integration tool for pipeline management.

Features:
- Generate CI/CD configuration files
- Validate pipeline configurations
- Create common workflows (test, build, deploy)
- Support for GitHub Actions, GitLab CI, CircleCI
- Best practices enforcement
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml

from victor.tools.enums import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icons
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter (lazy initialization)."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


# Workflow templates
GITHUB_ACTIONS_TEMPLATES: dict[str, dict[str, Any]] = {
    "python-test": {
        "name": "Python Tests",
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]},
        },
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "strategy": {"matrix": {"python-version": ["3.10", "3.11", "3.12"]}},
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {
                        "name": "Set up Python ${{ matrix.python-version }}",
                        "uses": "actions/setup-python@v5",
                        "with": {"python-version": "${{ matrix.python-version }}"},
                    },
                    {
                        "name": "Install dependencies",
                        "run": "pip install -e .[dev]",
                    },
                    {"name": "Lint with ruff", "run": "ruff check ."},
                    {"name": "Format check with black", "run": "black --check ."},
                    {
                        "name": "Type check with mypy",
                        "run": "mypy .",
                        "continue-on-error": True,
                    },
                    {
                        "name": "Run tests",
                        "run": "pytest --cov --cov-report=xml",
                    },
                    {
                        "name": "Upload coverage",
                        "uses": "codecov/codecov-action@v4",
                        "with": {"file": "./coverage.xml"},
                    },
                ],
            }
        },
    },
    "python-publish": {
        "name": "Publish to PyPI",
        "on": {"release": {"types": ["created"]}},
        "jobs": {
            "deploy": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {
                        "name": "Set up Python",
                        "uses": "actions/setup-python@v5",
                        "with": {"python-version": "3.12"},
                    },
                    {
                        "name": "Install dependencies",
                        "run": "pip install build twine",
                    },
                    {"name": "Build package", "run": "python -m build"},
                    {
                        "name": "Publish to PyPI",
                        "env": {
                            "TWINE_USERNAME": "__token__",
                            "TWINE_PASSWORD": "${{ secrets.PYPI_TOKEN }}",
                        },
                        "run": "twine upload dist/*",
                    },
                ],
            }
        },
    },
    "docker-build": {
        "name": "Docker Build and Push",
        "on": {"push": {"branches": ["main"]}},
        "jobs": {
            "docker": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {
                        "name": "Set up Docker Buildx",
                        "uses": "docker/setup-buildx-action@v3",
                    },
                    {
                        "name": "Login to DockerHub",
                        "uses": "docker/login-action@v3",
                        "with": {
                            "username": "${{ secrets.DOCKERHUB_USERNAME }}",
                            "password": "${{ secrets.DOCKERHUB_TOKEN }}",
                        },
                    },
                    {
                        "name": "Build and push",
                        "uses": "docker/build-push-action@v5",
                        "with": {
                            "context": ".",
                            "push": True,
                            "tags": "user/app:latest",
                        },
                    },
                ],
            }
        },
    },
}

GITLAB_CI_TEMPLATES = {
    "python-test": {
        "stages": ["test"],
        "test": {
            "stage": "test",
            "image": "python:3.12",
            "before_script": [
                "pip install -e .[dev]",
            ],
            "script": [
                "ruff check .",
                "black --check .",
                "pytest --cov",
            ],
        },
    },
}

CIRCLECI_TEMPLATES = {
    "python-test": {
        "version": 2.1,
        "jobs": {
            "test": {
                "docker": [{"image": "cimg/python:3.12"}],
                "steps": [
                    "checkout",
                    {"run": {"name": "Install dependencies", "command": "pip install -e .[dev]"}},
                    {"run": {"name": "Run tests", "command": "pytest"}},
                ],
            },
        },
        "workflows": {
            "test-workflow": {
                "jobs": ["test"],
            },
        },
    },
}

# Type to template mapping for convenience
TYPE_MAPPING = {
    "test": "python-test",
    "build": "docker-build",
    "deploy": "docker-build",
    "release": "python-publish",
    "publish": "python-publish",
}


@tool(
    category="cicd",
    priority=Priority.MEDIUM,  # Task-specific CI/CD configuration
    access_mode=AccessMode.WRITE,  # Generates configuration files
    danger_level=DangerLevel.LOW,  # Creates new files only
    keywords=["cicd", "pipeline", "github actions", "gitlab ci", "workflow", "deploy"],
)
async def cicd(
    operation: str,
    platform: str = "github",
    workflow: Optional[str] = None,
    type: Optional[str] = None,
    file: Optional[str] = None,
    output: Optional[str] = None,
    validate_command: Optional[str] = None,
) -> dict[str, Any]:
    """
    Unified CI/CD tool for pipeline management.

    Performs CI/CD operations including generating configurations, validating
    pipelines, and listing available templates. Consolidates all CI/CD
    functionality into a single interface.

    Args:
        operation: Operation to perform. Options: "generate", "validate", "list".
        platform: CI/CD platform (github, gitlab, circle). Default: github.
        workflow: Workflow template name (python-test, python-publish, docker-build).
        type: Workflow type shorthand (test, build, deploy, release, publish).
        file: Configuration file path (for validate operation).
        output: Output file path (for generate operation, auto-generated if not provided).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - operation: Operation performed
        - output_file: Path where configuration was written (generate)
        - config: Generated configuration (generate)
        - issues: List of issues (validate)
        - warnings: List of warnings (validate)
        - templates: Available templates (list)
        - formatted_report: Human-readable report
        - error: Error message if failed

    Examples:
        # Generate CI/CD configuration
        cicd(operation="generate", platform="github", workflow="python-test")

        # Generate using type shorthand
        cicd(operation="generate", type="test")

        # Validate configuration file
        cicd(operation="validate", file=".github/workflows/test.yml")

        # List available templates
        cicd(operation="list")

        # Custom output path
        cicd(operation="generate", workflow="docker-build", output=".github/workflows/build.yml")
    """
    if not operation:
        return {"success": False, "error": "Missing required parameter: operation"}

    # Generate operation
    if operation == "generate":
        # Determine workflow from type or use explicit workflow
        if type and not workflow:
            workflow = TYPE_MAPPING.get(type)
            if not workflow:
                available = ", ".join(TYPE_MAPPING.keys())
                return {
                    "success": False,
                    "error": f"Unknown workflow type: {type}. Available: {available}",
                }
        elif not workflow:
            return {
                "success": False,
                "error": "Either 'workflow' or 'type' parameter required for generate operation",
            }

        if platform != "github":
            return {
                "success": False,
                "error": f"Platform '{platform}' not yet supported. Use 'github'.",
            }

        # Get template
        template = GITHUB_ACTIONS_TEMPLATES.get(workflow)

        if not template:
            available = ", ".join(GITHUB_ACTIONS_TEMPLATES.keys())
            return {
                "success": False,
                "error": f"Workflow template '{workflow}' not found. Available: {available}",
            }

        # Generate YAML
        config_yaml = yaml.dump(template, default_flow_style=False, sort_keys=False)

        # Determine output path
        if not output:
            output = f".github/workflows/{workflow}.yml"

        # Write file
        output_file = Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(config_yaml)

        # Build report
        report = []
        report.append("CI/CD Configuration Generated")
        report.append("=" * 70)
        report.append("")
        report.append(f"Platform: {platform}")
        report.append(f"Workflow: {workflow}")
        report.append(f"Output: {output}")
        report.append("")
        report.append("Generated configuration:")
        report.append("-" * 70)
        report.append(config_yaml)
        report.append("")
        report.append("Next steps:")
        report.append("  1. Review the configuration")
        report.append("  2. Customize as needed")
        report.append("  3. Commit to repository")
        report.append("  4. Push to trigger workflow")

        return {
            "success": True,
            "operation": "generate",
            "output_file": output,
            "config": config_yaml,
            "formatted_report": "\n".join(report),
        }

    # Validate operation
    elif operation == "validate":
        if not file:
            return {"success": False, "error": "Validate operation requires 'file' parameter"}

        file_obj = Path(file)
        if not file_obj.exists():
            return {"success": False, "error": f"File not found: {file}"}

        # Read and validate YAML
        try:
            content = file_obj.read_text()
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            return {"success": False, "error": f"Invalid YAML syntax: {e}"}

        # Validate structure (basic checks)
        issues = []
        warnings = []

        # Check for required fields (GitHub Actions)
        if "name" not in config:
            warnings.append("Missing 'name' field (recommended)")

        if "on" not in config:
            issues.append("Missing 'on' field (required for triggers)")

        if "jobs" not in config:
            issues.append("Missing 'jobs' field (required)")

        # Check jobs
        jobs_count = 0
        if "jobs" in config:
            jobs_count = len(config["jobs"])
            for job_name, job in config["jobs"].items():
                if "runs-on" not in job:
                    issues.append(f"Job '{job_name}': Missing 'runs-on' field")

                if "steps" not in job:
                    issues.append(f"Job '{job_name}': Missing 'steps' field")

                # Check for best practices
                if "steps" in job:
                    has_checkout = any(
                        step.get("uses", "").startswith("actions/checkout") for step in job["steps"]
                    )
                    if not has_checkout:
                        warnings.append(f"Job '{job_name}': No checkout step (usually needed)")

        external_output = None
        if validate_command:
            try:
                proc = subprocess.run(
                    validate_command.split(),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                external_output = proc.stdout.strip()
            except subprocess.CalledProcessError as exc:
                issues.append(f"External validator failed: {exc.stderr.strip()}")

        # Build report
        report = []
        report.append("Configuration Validation")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file}")
        report.append("")

        if not issues and not warnings:
            report.append(f"{_get_icon('success')} Configuration is valid!")
            report.append("\nNo issues or warnings found.")
        else:
            if issues:
                report.append(f"{_get_icon('error')} {len(issues)} issue(s) found:")
                for issue in issues:
                    report.append(f"  {_get_icon('bullet')} {issue}")
                report.append("")

            if warnings:
                report.append(f"{_get_icon('warning')}  {len(warnings)} warning(s):")
                for warning in warnings:
                    report.append(f"  {_get_icon('bullet')} {warning}")
                report.append("")

        # Configuration summary
        if "jobs" in config:
            report.append(f"Jobs: {len(config['jobs'])}")
            for job_name, job in config["jobs"].items():
                steps = len(job.get("steps", []))
                report.append(f"  • {job_name}: {steps} steps")

        if external_output:
            report.append("")
            report.append("External validation output:")
            report.append(external_output)

        success = len(issues) == 0

        return {
            "success": success,
            "operation": "validate",
            "issues": issues,
            "warnings": warnings,
            "jobs_count": jobs_count,
            "external_output": external_output,
            "formatted_report": "\n".join(report),
        }

    # List operation
    elif operation == "list":
        templates = []

        report = []
        report.append("Available CI/CD Templates")
        report.append("=" * 70)
        report.append("")
        report.append("GitHub Actions Templates:")
        report.append("")

        for name, template in GITHUB_ACTIONS_TEMPLATES.items():
            template_info = {
                "name": name,
                "platform": "github",
                "display_name": template["name"],
                "jobs": list((template.get("jobs") or {}).keys()),
                "triggers": list((template.get("on") or {}).keys()),
            }
            templates.append(template_info)

            report.append(f"{_get_icon('clipboard')} {name}")
            report.append(f"   Name: {template['name']}")

            # List jobs
            if "jobs" in template:
                jobs = list((template["jobs"] or {}).keys())
                report.append(f"   Jobs: {', '.join(jobs)}")

            # List triggers
            if "on" in template:
                triggers = list((template["on"] or {}).keys())
                report.append(f"   Triggers: {', '.join(triggers)}")

            report.append("")

        report.append("Type Shortcuts:")
        for type_name, workflow_name in TYPE_MAPPING.items():
            report.append(f"  • {type_name} → {workflow_name}")
        report.append("")

        report.append("Usage:")
        report.append("  cicd(operation='generate', workflow='<name>')")
        report.append("  cicd(operation='generate', type='<type>')")

        return {
            "success": True,
            "operation": "list",
            "templates": templates,
            "platforms": ["github"],
            "formatted_report": "\n".join(report),
        }

    else:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Valid operations: generate, validate, list",
        }
