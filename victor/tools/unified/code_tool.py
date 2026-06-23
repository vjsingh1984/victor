import argparse
import sys
from io import StringIO
import json

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import split_command

# Mock imports for the underlying code tools
try:
    from victor.tools.code import run_tests, execute_python, analyze_metrics
except ImportError:

    async def run_tests(runner: str, path: str):
        return {}

    async def execute_python(code: str):
        return ""

    async def analyze_metrics(path: str):
        return {}


class UnifiedCodeParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_code_parser() -> UnifiedCodeParser:
    parser = UnifiedCodeParser(
        prog="code", description="Unified code operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    # `test` subcommand
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("runner", help="Test runner (e.g. pytest)")
    test_parser.add_argument("path", nargs="?", default=".", help="Path to tests")

    # `execute` subcommand
    exec_parser = subparsers.add_parser("execute", help="Execute python code")
    exec_parser.add_argument("code", help="Python code to execute")

    # `metrics` subcommand
    metrics_parser = subparsers.add_parser("metrics", help="Analyze code metrics")
    metrics_parser.add_argument("path", nargs="?", default=".", help="Directory to analyze")

    return parser


@tool(
    name="code",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.HIGH,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def code_tool(command: str) -> str:
    """Unified code tool.
    Example commands:
      code test pytest tests/
      code execute "print('hello')"
      code metrics src/
    """
    parser = create_code_parser()

    try:
        args_list = split_command(command)
        if args_list and args_list[0] == "code":
            args_list = args_list[1:]
        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error: {e}"

    if parsed_args.subcommand == "test":
        try:
            results = await run_tests(runner=parsed_args.runner, path=parsed_args.path)
            if isinstance(results, dict) and "output" in results:
                return str(results["output"])
            return str(results)
        except Exception as e:
            return f"### ❌ ERROR\nTest execution failed: {e}"

    elif parsed_args.subcommand == "execute":
        try:
            return str(await execute_python(parsed_args.code))
        except Exception as e:
            return f"### ❌ ERROR\nCode execution failed: {e}"

    elif parsed_args.subcommand == "metrics":
        try:
            results = await analyze_metrics(parsed_args.path)
            if not isinstance(results, dict):
                return str(results)

            # Format dict as yaml-like markdown
            out = ["### Code Metrics"]
            for k, v in results.items():
                out.append(f"- **{k}**: {v}")
            return "\n".join(out)
        except Exception as e:
            return f"### ❌ ERROR\nMetrics analysis failed: {e}"
    else:
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"
