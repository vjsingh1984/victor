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

"""Coding vertical compute handlers.

Domain-specific handlers for coding workflows:
- code_validation: Syntax and lint checking
- test_runner: Test execution and reporting

Usage:
    # Handlers are auto-registered via @handler_decorator
    # Just import this module to register them

    # In YAML workflow:
    - id: validate
      type: compute
      handler: code_validation
      inputs:
        files: $ctx.changed_files
        checks: [lint, type]
      output: validation_results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from victor.framework.workflows.base_handler import BaseHandler
from victor.framework.handler_registry import handler_decorator

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import WorkflowContext

logger = logging.getLogger(__name__)


@handler_decorator("code_validation", description="Validate code syntax and style")
@dataclass
class CodeValidationHandler(BaseHandler):
    """Validate code syntax and style.

    Runs linters and type checkers on code files without LLM.

    Example YAML:
        - id: validate_code
          type: compute
          handler: code_validation
          inputs:
            files: $ctx.changed_files
            checks: [syntax, lint, type]
          output: validation_results
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> tuple[Any, int]:
        """Execute code validation checks."""
        files: Any = node.input_mapping.get("files", [])
        checks = node.input_mapping.get("checks", ["lint"])

        if isinstance(files, str):
            files = context.get(files) or [files]

        results = {}
        all_passed = True
        tool_calls = 0

        for check in checks:
            check_result = await self._run_check(check, files, tool_registry)
            tool_calls += 1
            results[check] = check_result
            if not check_result.get("passed", False):
                all_passed = False

        output = {"checks": results, "all_passed": all_passed}
        return output, tool_calls

    async def _run_check(
        self, check: str, files: list[str], tool_registry: "ToolRegistry"
    ) -> dict[str, Any]:
        """Run a specific check type."""
        try:
            file_args = " ".join(files) if files else "."

            if check == "lint":
                cmd = f"ruff check {file_args}"
            elif check == "type":
                cmd = f"mypy {file_args}"
            elif check == "syntax":
                cmd = f"python -m py_compile {file_args}"
            elif check == "format":
                cmd = f"ruff format --check {file_args}"
            else:
                return {"passed": True, "message": f"Unknown check: {check}"}

            result = await tool_registry.execute("shell", {}, command=cmd)
            return {
                "passed": result.success,
                "output": result.output,
                "error": result.error,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}


@handler_decorator("test_runner", description="Run tests and collect results")
@dataclass
class TestRunnerHandler(BaseHandler):
    """Run tests and collect results.

    Executes test suites and parses results.

    Example YAML:
        - id: run_tests
          type: compute
          handler: test_runner
          inputs:
            test_path: tests/
            framework: pytest
            coverage: true
          output: test_results
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> tuple[Any, int]:
        """Execute test runner."""
        test_path = node.input_mapping.get("test_path", "tests/")
        framework = node.input_mapping.get("framework", "pytest")
        coverage = node.input_mapping.get("coverage", False)

        if framework == "pytest":
            cmd = f"pytest {test_path} -v"
            if coverage:
                cmd += " --cov --cov-report=json"
        elif framework == "unittest":
            cmd = f"python -m unittest discover {test_path}"
        else:
            cmd = f"{framework} {test_path}"

        result = await tool_registry.execute("shell", {}, command=cmd)

        # Raise exception if test execution failed
        if not result.success:
            raise Exception(f"Test execution failed: {result.error}")

        output = {
            "framework": framework,
            "passed": result.success,
            "output": result.output,
        }

        return output, 1


__all__ = [
    "CodeValidationHandler",
    "TestRunnerHandler",
]
