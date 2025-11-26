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

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Workflow templates
GITHUB_ACTIONS_TEMPLATES = {
    "python-test": {
        "name": "Python Tests",
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]},
        },
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "strategy": {
                    "matrix": {"python-version": ["3.10", "3.11", "3.12"]}
                },
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
                        "env": {"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": "${{ secrets.PYPI_TOKEN }}"},
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


@tool
async def cicd_generate(
    platform: str = "github",
    workflow: str = "python-test",
    output: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate CI/CD configuration file.

    Creates a CI/CD pipeline configuration file based on templates.
    Currently supports GitHub Actions with plans for GitLab CI and CircleCI.

    Args:
        platform: CI/CD platform (github, gitlab, circle). Default: github.
        workflow: Workflow template name (python-test, python-publish, docker-build). Default: python-test.
        output: Output file path (optional, auto-generated if not provided).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output_file: Path where configuration was written
        - config: Generated configuration as YAML string
        - formatted_report: Human-readable generation report
        - error: Error message if failed
    """
    if platform != "github":
        return {
            "success": False,
            "error": f"Platform '{platform}' not yet supported. Use 'github'."
        }

    # Get template
    template = GITHUB_ACTIONS_TEMPLATES.get(workflow)

    if not template:
        available = ", ".join(GITHUB_ACTIONS_TEMPLATES.keys())
        return {
            "success": False,
            "error": f"Workflow template '{workflow}' not found. Available: {available}"
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
    report.append(f"CI/CD Configuration Generated")
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
        "output_file": output,
        "config": config_yaml,
        "formatted_report": "\n".join(report)
    }


@tool
async def cicd_validate(
    file: str,
) -> Dict[str, Any]:
    """
    Validate CI/CD configuration file.

    Validates YAML syntax and checks for common issues and best practices
    in CI/CD configuration files. Currently optimized for GitHub Actions.

    Args:
        file: Path to configuration file to validate.

    Returns:
        Dictionary containing:
        - success: Whether configuration is valid (no critical issues)
        - issues: List of critical issues found
        - warnings: List of warnings/recommendations
        - jobs_count: Number of jobs in configuration
        - formatted_report: Human-readable validation report
        - error: Error message if failed to read/parse
    """
    if not file:
        return {
            "success": False,
            "error": "Missing required parameter: file"
        }

    file_obj = Path(file)
    if not file_obj.exists():
        return {
            "success": False,
            "error": f"File not found: {file}"
        }

    # Read and validate YAML
    try:
        content = file_obj.read_text()
        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        return {
            "success": False,
            "error": f"Invalid YAML syntax: {e}"
        }

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
                    step.get("uses", "").startswith("actions/checkout")
                    for step in job["steps"]
                )
                if not has_checkout:
                    warnings.append(
                        f"Job '{job_name}': No checkout step (usually needed)"
                    )

    # Build report
    report = []
    report.append("Configuration Validation")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append("")

    if not issues and not warnings:
        report.append("✅ Configuration is valid!")
        report.append("\nNo issues or warnings found.")
    else:
        if issues:
            report.append(f"❌ {len(issues)} issue(s) found:")
            for issue in issues:
                report.append(f"  • {issue}")
            report.append("")

        if warnings:
            report.append(f"⚠️  {len(warnings)} warning(s):")
            for warning in warnings:
                report.append(f"  • {warning}")
            report.append("")

    # Configuration summary
    if "jobs" in config:
        report.append(f"Jobs: {len(config['jobs'])}")
        for job_name, job in config["jobs"].items():
            steps = len(job.get("steps", []))
            report.append(f"  • {job_name}: {steps} steps")

    success = len(issues) == 0

    return {
        "success": success,
        "issues": issues,
        "warnings": warnings,
        "jobs_count": jobs_count,
        "formatted_report": "\n".join(report)
    }


@tool
async def cicd_list_templates() -> Dict[str, Any]:
    """
    List available CI/CD workflow templates.

    Returns a list of all available workflow templates for different
    CI/CD platforms, along with their descriptions and usage.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - templates: List of template names with details
        - platforms: Available platforms
        - formatted_report: Human-readable template listing
    """
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
            "jobs": list(template.get("jobs", {}).keys()),
            "triggers": list(template.get("on", {}).keys())
        }
        templates.append(template_info)

        report.append(f"📋 {name}")
        report.append(f"   Name: {template['name']}")

        # List jobs
        if "jobs" in template:
            jobs = list(template["jobs"].keys())
            report.append(f"   Jobs: {', '.join(jobs)}")

        # List triggers
        if "on" in template:
            triggers = list(template["on"].keys())
            report.append(f"   Triggers: {', '.join(triggers)}")

        report.append("")

    report.append("Usage:")
    report.append("  cicd_generate(platform='github', workflow='<name>')")

    return {
        "success": True,
        "templates": templates,
        "platforms": ["github"],
        "formatted_report": "\n".join(report)
    }


@tool
async def cicd_create_workflow(
    type: str = "test",
    platform: str = "github",
    output: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a specific workflow type.

    Convenience function to create common workflow types without
    needing to know the exact template name.

    Args:
        type: Workflow type (test, build, deploy, release, publish). Default: test.
        platform: CI/CD platform (github, gitlab, circle). Default: github.
        output: Output file path (optional, auto-generated if not provided).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output_file: Path where configuration was written
        - config: Generated configuration as YAML string
        - formatted_report: Human-readable generation report
        - error: Error message if failed
    """
    # Map type to template
    type_mapping = {
        "test": "python-test",
        "build": "docker-build",
        "deploy": "docker-build",
        "release": "python-publish",
        "publish": "python-publish",
    }

    workflow = type_mapping.get(type)

    if not workflow:
        available = ", ".join(type_mapping.keys())
        return {
            "success": False,
            "error": f"Unknown workflow type: {type}. Available: {available}"
        }

    # Use cicd_generate
    return await cicd_generate(platform=platform, workflow=workflow, output=output)


# Keep class for backward compatibility
class CICDTool:
    """Deprecated: Use individual cicd_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "CICDTool class is deprecated. Use cicd_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
