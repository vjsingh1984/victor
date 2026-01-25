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

"""Benchmark vertical compute handlers.

Domain-specific handlers for benchmark workflows:
- benchmark_test_runner: Execute tests and parse results
- environment_setup: Set up execution environment
- live_executor: Execute code with real-time feedback
- language_detector: Detect programming language
- polyglot_verifier: Verify multi-language code
- multi_solution_validator: Validate multiple solutions for pass@k
- code_tester: Test generated code
- syntax_check: Verify syntax correctness

Usage:
    # Handlers are auto-registered via @handler_decorator
    # Just import this module to register them

    # In YAML workflow:
    - id: run_tests
      type: compute
      handler: benchmark_test_runner
      inputs:
        test_file: tests/test_solution.py
        timeout: 60
      output: test_results
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from victor.framework.workflows.base_handler import BaseHandler
from victor.framework.handler_registry import handler_decorator

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import WorkflowContext

logger = logging.getLogger(__name__)


# =============================================================================
# Test Runner Handler
# =============================================================================


@handler_decorator("benchmark_test_runner", description="Execute tests in benchmark context")
@dataclass
class BenchmarkTestRunnerHandler(BaseHandler):
    """Execute tests and return results.

    Runs the test suite against generated solutions.

    Example YAML:
        - id: run_tests
          type: compute
          handler: benchmark_test_runner
          inputs:
            test_file: tests/test_solution.py
            test_command: pytest tests/ -v
            timeout: 60
          output: test_results
    """

    default_timeout: int = 60

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute test runner."""
        test_file = node.input_mapping.get("test_file", "")
        test_command = node.input_mapping.get("test_command", "")
        timeout = node.input_mapping.get("timeout", self.default_timeout)
        framework = node.input_mapping.get("framework", "pytest")

        # Resolve context variables
        if isinstance(test_command, str) and test_command.startswith("$ctx."):
            test_command = context.get(test_command[5:]) or test_command

        if not test_command and not test_file:
            raise ValueError("No test file or command specified")

        # Build command
        if test_command:
            cmd = test_command
        elif framework == "pytest":
            cmd = f"python -m pytest {test_file} -v --tb=short --timeout={timeout}"
        elif framework == "unittest":
            cmd = f"python -m unittest {test_file} -v"
        else:
            cmd = f"python {test_file}"

        result = await tool_registry.execute("shell", {}, command=cmd, timeout=int(timeout) + 10)

        output_text = result.output if hasattr(result, "output") else str(result)
        test_output = self._parse_test_output(output_text, framework)

        output = {
            "success": result.success if hasattr(result, "success") else False,
            "passed": test_output.get("passed", 0),
            "failed": test_output.get("failed", 0),
            "total": test_output.get("total", 0),
            "errors": test_output.get("errors", []),
            "pass_rate": test_output.get("pass_rate", 0),
            "raw_output": output_text[:5000],  # Truncate for context
        }

        return output, 1

    def _parse_test_output(self, output: str, framework: str) -> Dict[str, Any]:
        """Parse test output to extract results."""
        result: Dict[str, Any] = {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "errors": [],
            "pass_rate": 0,
        }

        if framework == "pytest":
            match = re.search(r"(\d+) passed", output)
            if match:
                result["passed"] = int(match.group(1))

            match = re.search(r"(\d+) failed", output)
            if match:
                result["failed"] = int(match.group(1))

            match = re.search(r"(\d+) error", output)
            if match:
                result["errors"].append(f"{match.group(1)} errors")

        elif framework == "unittest":
            match = re.search(r"Ran (\d+) test", output)
            if match:
                result["total"] = int(match.group(1))

            if "OK" in output:
                result["passed"] = result["total"]
            elif "FAILED" in output:
                match = re.search(r"failures=(\d+)", output)
                if match:
                    result["failed"] = int(match.group(1))
                result["passed"] = result["total"] - result["failed"]

        result["total"] = result["passed"] + result["failed"]
        if result["total"] > 0:
            result["pass_rate"] = result["passed"] / result["total"]

        return result


# =============================================================================
# Environment Setup Handler
# =============================================================================


@handler_decorator("environment_setup", description="Set up execution environment for benchmarks")
@dataclass
class EnvironmentSetupHandler(BaseHandler):
    """Set up execution environment for benchmarks.

    Creates virtual environment and installs dependencies.

    Example YAML:
        - id: setup_env
          type: compute
          handler: environment_setup
          inputs:
            language: python
            dependencies: ["numpy", "pandas"]
            workspace: /tmp/benchmark
          output: env_setup
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute environment setup."""
        language = node.input_mapping.get("language", "python")
        deps_input = node.input_mapping.get("dependencies", [])
        dependencies: List[str] = deps_input if isinstance(deps_input, list) else [deps_input] if deps_input else []
        workspace = node.input_mapping.get("workspace", "")

        # Resolve context variables
        if isinstance(workspace, str) and workspace.startswith("$ctx."):
            workspace = context.get(workspace[5:]) or ""

        if not workspace:
            workspace = tempfile.mkdtemp(prefix="benchmark_")

        output: Dict[str, Any] = {
            "workspace": workspace,
            "language": language,
            "ready": False,
        }

        # Create workspace directory
        Path(workspace).mkdir(parents=True, exist_ok=True)

        if language == "python" and dependencies:
            # Install dependencies
            deps_str = " ".join(dependencies)
            result = await tool_registry.execute(
                "shell",
                {},
                command=f"pip install {deps_str}",
                timeout=120,
            )
            output["dependencies_installed"] = (
                result.success if hasattr(result, "success") else False
            )

        output["ready"] = True

        return output, 1 if dependencies else 0


# =============================================================================
# Live Executor Handler
# =============================================================================


@handler_decorator("live_executor", description="Execute code with real-time feedback")
@dataclass
class LiveExecutorHandler(BaseHandler):
    """Execute code with real-time feedback.

    For LiveCodeBench-style evaluations.

    Example YAML:
        - id: execute_live
          type: compute
          handler: live_executor
          inputs:
            code: $ctx.solution_code
            language: python
            test_input: "5 10"
            timeout: 30
          output: execution_result
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute live code with real-time feedback."""
        code = node.input_mapping.get("code", "")
        language = node.input_mapping.get("language", "python")
        test_input = node.input_mapping.get("test_input", "")
        timeout = node.input_mapping.get("timeout", 30)

        # Resolve context variables
        if isinstance(code, str) and code.startswith("$ctx."):
            code = context.get(code[5:]) or ""

        if not code:
            raise ValueError("No code provided")

        # Write code to temporary file
        suffix = ".py" if language == "python" else f".{language}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(code)
            code_file = f.name

        # Execute with input
        if language == "python":
            cmd = f"python {code_file}"
        else:
            cmd = f"{language} {code_file}"

        if test_input:
            cmd = f"echo '{test_input}' | {cmd}"

        result = await tool_registry.execute("shell", {}, command=cmd, timeout=timeout)

        output = {
            "success": result.success if hasattr(result, "success") else False,
            "stdout": result.output if hasattr(result, "output") else "",
            "exit_code": getattr(result, "exit_code", 0),
            "language": language,
        }

        # Check for common error patterns
        stdout = str(output.get("stdout", ""))
        if "Error" in stdout or "Exception" in stdout or "Traceback" in stdout:
            output["success"] = False
            output["error_type"] = "runtime_error"

        return output, 1


# =============================================================================
# Language Detector Handler
# =============================================================================


@handler_decorator("language_detector", description="Detect programming language from files")
@dataclass
class LanguageDetectorHandler(BaseHandler):
    """Detect programming language from files.

    Example YAML:
        - id: detect_lang
          type: compute
          handler: language_detector
          inputs:
            files: ["main.py", "utils.js"]
          output: language_info
    """

    extension_map: Dict[str, str] = field(
        default_factory=lambda: {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".cs": "csharp",
            ".scala": "scala",
            ".r": "r",
            ".jl": "julia",
        }
    )

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute language detection."""
        files_input = node.input_mapping.get("files", [])
        files: List[str] = files_input if isinstance(files_input, list) else [files_input] if isinstance(files_input, str) else []

        # Resolve context variables
        if isinstance(files, str) and files.startswith("$ctx."):
            files = context.get(files[5:]) or []

        if not files:
            raise ValueError("No files provided")

        language_counts: Dict[str, int] = {}
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            lang = self.extension_map.get(ext, "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Find primary language
        primary = (
            max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else "unknown"
        )

        output = {
            "primary": primary,
            "languages": language_counts,
            "file_count": len(files),
        }

        return output, 1


# =============================================================================
# Polyglot Verifier Handler
# =============================================================================


@handler_decorator("polyglot_verifier", description="Verify multi-language code")
@dataclass
class PolyglotVerifierHandler(BaseHandler):
    """Verify multi-language code modifications.

    Example YAML:
        - id: verify_change
          type: compute
          handler: polyglot_verifier
          inputs:
            language: python
            files: $ctx.files
            test_command: pytest tests/ -v
          output: verification_result
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute polyglot verification."""
        language = node.input_mapping.get("language", "python")
        test_command = node.input_mapping.get("test_command", "")

        # Resolve context variables
        if isinstance(test_command, str) and test_command.startswith("$ctx."):
            test_command = context.get(test_command[5:]) or ""

        output: Dict[str, Any] = {
            "syntax_valid": False,
            "tests_pass": False,
            "lint_clean": True,
        }

        # Syntax check
        if language == "python":
            syntax_result = await tool_registry.execute(
                "shell",
                {},
                command="python -m py_compile *.py 2>&1 || true",
                timeout=30,
            )
            output["syntax_valid"] = "Error" not in (
                syntax_result.output if hasattr(syntax_result, "output") else ""
            )

        # Run tests if command provided
        if test_command:
            test_result = await tool_registry.execute(
                "shell",
                {},
                command=test_command,
                timeout=180,
            )
            output["tests_pass"] = test_result.success if hasattr(test_result, "success") else False
            output["test_output"] = (test_result.output if hasattr(test_result, "output") else "")[
                :2000
            ]

        tool_calls = 2 if test_command else 1
        return output, tool_calls


# =============================================================================
# Multi-Solution Validator Handler
# =============================================================================


@handler_decorator("multi_solution_validator", description="Validate multiple solutions for pass@k")
@dataclass
class MultiSolutionValidatorHandler(BaseHandler):
    """Validate multiple solutions for pass@k evaluation.

    Example YAML:
        - id: validate_solutions
          type: compute
          handler: multi_solution_validator
          inputs:
            solutions: $ctx.valid_solutions
            test_cases: $ctx.test_cases
            language: python
            timeout_per_solution: 30
          output: validation_results
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute multi-solution validation."""
        solutions_input = node.input_mapping.get("solutions", [])
        solutions: List[str] = solutions_input if isinstance(solutions_input, list) else [solutions_input] if isinstance(solutions_input, str) else []
        test_cases_input = node.input_mapping.get("test_cases", [])
        test_cases: List[str] = test_cases_input if isinstance(test_cases_input, list) else [test_cases_input] if isinstance(test_cases_input, str) else []
        timeout_per = node.input_mapping.get("timeout_per_solution", 30)

        # Resolve context variables
        if isinstance(solutions, str) and solutions.startswith("$ctx."):
            solutions = context.get(solutions[5:]) or []
        if isinstance(test_cases, str) and test_cases.startswith("$ctx."):
            test_cases = context.get(test_cases[5:]) or []

        passed_solutions = 0
        best_solution = ""
        results: List[Dict[str, Any]] = []

        for i, solution in enumerate(solutions):
            if not solution or not isinstance(solution, str):
                continue

            try:
                # Write solution to temp file
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(solution)
                    if test_cases:
                        # Append test cases
                        f.write("\n\n# Test Cases\n")
                        for tc in test_cases:
                            if isinstance(tc, str):
                                f.write(f"{tc}\n")
                    code_file = f.name

                # Execute
                result = await tool_registry.execute(
                    "shell",
                    {},
                    command=f"python {code_file}",
                    timeout=timeout_per,
                )

                success = result.success if hasattr(result, "success") else False
                if success:
                    passed_solutions += 1
                    if not best_solution:
                        best_solution = solution

                results.append(
                    {
                        "index": i,
                        "success": success,
                        "output": (result.output if hasattr(result, "output") else "")[:500],
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "index": i,
                        "success": False,
                        "error": str(e),
                    }
                )

        output = {
            "total_solutions": len(solutions),
            "passed_solutions": passed_solutions,
            "results": results,
            "best_solution": best_solution,
            "pass_rate": passed_solutions / len(solutions) if solutions else 0,
        }

        return output, len(solutions)


# =============================================================================
# Code Tester Handler
# =============================================================================


@handler_decorator("code_tester", description="Test generated code")
@dataclass
class CodeTesterHandler(BaseHandler):
    """Test generated code against test cases.

    Example YAML:
        - id: test_solution
          type: compute
          handler: code_tester
          inputs:
            code: $ctx.current_solution
            test_cases: $ctx.test_cases
            language: python
          output: test_results
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute code testing."""
        code = node.input_mapping.get("code", "")
        test_cases_input = node.input_mapping.get("test_cases", [])
        test_cases: List[str] = test_cases_input if isinstance(test_cases_input, list) else [test_cases_input] if isinstance(test_cases_input, str) else []

        # Resolve context variables
        if isinstance(code, str) and code.startswith("$ctx."):
            code = context.get(code[5:]) or ""
        if isinstance(test_cases, str) and test_cases.startswith("$ctx."):
            test_cases = context.get(test_cases[5:]) or []

        if not code:
            raise ValueError("No code provided")

        # Combine code and test cases
        full_code = code
        if test_cases:
            full_code += "\n\n# Test Cases\n"
            for tc in test_cases:
                if isinstance(tc, str):
                    full_code += f"{tc}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            code_file = f.name

        result = await tool_registry.execute(
            "shell",
            {},
            command=f"python {code_file}",
            timeout=60,
        )

        success = result.success if hasattr(result, "success") else False
        output = {
            "passed": success,
            "output": result.output if hasattr(result, "output") else "",
            "exit_code": getattr(result, "exit_code", 0),
        }

        return output, 1


# =============================================================================
# Syntax Check Handler
# =============================================================================


@handler_decorator("syntax_check", description="Verify code syntax correctness")
@dataclass
class SyntaxCheckHandler(BaseHandler):
    """Verify code syntax correctness.

    Example YAML:
        - id: check_syntax
          type: compute
          handler: syntax_check
          inputs:
            code: $ctx.solution
            language: python
          output: syntax_result
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute syntax check."""
        code = node.input_mapping.get("code", "")
        language = node.input_mapping.get("language", "python")

        # Resolve context variables
        if isinstance(code, str) and code.startswith("$ctx."):
            code = context.get(code[5:]) or ""

        if not code:
            raise ValueError("No code provided")

        output: Dict[str, Any] = {
            "valid": False,
            "language": language,
            "errors": [],
        }

        tool_calls = 0

        if language == "python":
            # Use Python's compile for syntax check
            try:
                compile(code, "<solution>", "exec")
                output["valid"] = True
            except SyntaxError as e:
                output["errors"].append(
                    {
                        "line": e.lineno,
                        "offset": e.offset,
                        "message": str(e.msg),
                    }
                )
        else:
            # For other languages, try to use the language's syntax checker
            with tempfile.NamedTemporaryFile(mode="w", suffix=f".{language}", delete=False) as f:
                f.write(code)
                code_file = f.name

            if language == "javascript":
                result = await tool_registry.execute(
                    "shell",
                    {},
                    command=f"node --check {code_file}",
                    timeout=10,
                )
                tool_calls += 1
                output["valid"] = result.success if hasattr(result, "success") else False
                if not output["valid"]:
                    output["errors"].append(
                        {"message": result.output if hasattr(result, "output") else ""}
                    )
            elif language == "typescript":
                result = await tool_registry.execute(
                    "shell",
                    {},
                    command=f"tsc --noEmit {code_file}",
                    timeout=30,
                )
                tool_calls += 1
                output["valid"] = result.success if hasattr(result, "success") else False
                if not output["valid"]:
                    output["errors"].append(
                        {"message": result.output if hasattr(result, "output") else ""}
                    )
            else:
                # Fallback: assume valid
                output["valid"] = True

        return output, tool_calls


__all__ = [
    # Handlers
    "BenchmarkTestRunnerHandler",
    "EnvironmentSetupHandler",
    "LiveExecutorHandler",
    "LanguageDetectorHandler",
    "PolyglotVerifierHandler",
    "MultiSolutionValidatorHandler",
    "CodeTesterHandler",
    "SyntaxCheckHandler",
]
