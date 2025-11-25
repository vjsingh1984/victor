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
from typing import Any, Dict, List, Optional
import logging

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class CICDTool(BaseTool):
    """Tool for CI/CD pipeline management."""

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

    @property
    def name(self) -> str:
        """Get tool name."""
        return "cicd"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """CI/CD pipeline management and configuration.

Generate and manage CI/CD pipelines:
- Create GitHub Actions workflows
- Generate GitLab CI configurations
- Create CircleCI configs
- Validate existing pipelines
- Generate common workflows

Operations:
- generate: Generate CI/CD configuration
- validate: Validate existing configuration
- list_templates: List available workflow templates
- create_workflow: Create specific workflow (test, build, deploy)

Example workflows:
1. Generate GitHub Actions workflow:
   cicd(operation="generate", platform="github", workflow="python-test")

2. Create deployment workflow:
   cicd(operation="create_workflow", type="deploy", platform="github")

3. Validate existing config:
   cicd(operation="validate", file=".github/workflows/test.yml")

4. List available templates:
   cicd(operation="list_templates")
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
        [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: generate, validate, list_templates, create_workflow",
                required=True,
            ),
            ToolParameter(
                name="platform",
                type="string",
                description="CI/CD platform: github, gitlab, circle",
                required=False,
            ),
            ToolParameter(
                name="workflow",
                type="string",
                description="Workflow template name",
                required=False,
            ),
            ToolParameter(
                name="type",
                type="string",
                description="Workflow type: test, build, deploy, release",
                required=False,
            ),
            ToolParameter(
                name="file",
                type="string",
                description="Configuration file path",
                required=False,
            ),
            ToolParameter(
                name="output",
                type="string",
                description="Output file path",
                required=False,
            ),
        ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute CI/CD operation.

        Args:
            operation: CI/CD operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with CI/CD configuration
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "generate":
                return await self._generate_config(kwargs)
            elif operation == "validate":
                return await self._validate_config(kwargs)
            elif operation == "list_templates":
                return await self._list_templates(kwargs)
            elif operation == "create_workflow":
                return await self._create_workflow(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("CI/CD operation failed")
            return ToolResult(
                success=False, output="", error=f"CI/CD error: {str(e)}"
            )

    async def _generate_config(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate CI/CD configuration."""
        platform = kwargs.get("platform", "github")
        workflow = kwargs.get("workflow", "python-test")
        output_path = kwargs.get("output")

        if platform != "github":
            return ToolResult(
                success=False,
                output="",
                error=f"Platform '{platform}' not yet supported. Use 'github'.",
            )

        # Get template
        template = self.GITHUB_ACTIONS_TEMPLATES.get(workflow)

        if not template:
            return ToolResult(
                success=False,
                output="",
                error=f"Workflow template '{workflow}' not found. Use list_templates to see available templates.",
            )

        # Generate YAML
        config_yaml = yaml.dump(template, default_flow_style=False, sort_keys=False)

        # Determine output path
        if not output_path:
            output_path = f".github/workflows/{workflow}.yml"

        # Write file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(config_yaml)

        # Build report
        report = []
        report.append(f"CI/CD Configuration Generated")
        report.append("=" * 70)
        report.append("")
        report.append(f"Platform: {platform}")
        report.append(f"Workflow: {workflow}")
        report.append(f"Output: {output_path}")
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

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _validate_config(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Validate CI/CD configuration."""
        file_path = kwargs.get("file")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {file_path}"
            )

        # Read and validate YAML
        try:
            content = file_obj.read_text()
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid YAML syntax: {e}",
            )

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
        if "jobs" in config:
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
        report.append(f"File: {file_path}")
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

        return ToolResult(
            success=success,
            output="\n".join(report),
            error="",
        )

    async def _list_templates(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List available workflow templates."""
        report = []
        report.append("Available CI/CD Templates")
        report.append("=" * 70)
        report.append("")
        report.append("GitHub Actions Templates:")
        report.append("")

        for name, template in self.GITHUB_ACTIONS_TEMPLATES.items():
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
        report.append("  cicd(operation='generate', platform='github', workflow='<name>')")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _create_workflow(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Create specific workflow type."""
        workflow_type = kwargs.get("type", "test")
        platform = kwargs.get("platform", "github")
        output_path = kwargs.get("output")

        # Map type to template
        type_mapping = {
            "test": "python-test",
            "build": "docker-build",
            "deploy": "docker-build",
            "release": "python-publish",
            "publish": "python-publish",
        }

        workflow = type_mapping.get(workflow_type)

        if not workflow:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown workflow type: {workflow_type}. Available: {', '.join(type_mapping.keys())}",
            )

        # Use generate_config
        return await self._generate_config({
            "platform": platform,
            "workflow": workflow,
            "output": output_path,
        })


    def _get_gitlab_template(self, workflow: str) -> Optional[Dict[str, Any]]:
        """Get GitLab CI template."""
        templates = {
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

        return templates.get(workflow)

    def _get_circle_template(self, workflow: str) -> Optional[Dict[str, Any]]:
        """Get CircleCI template."""
        templates = {
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

        return templates.get(workflow)
