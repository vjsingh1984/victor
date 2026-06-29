import argparse
from pathlib import Path
import shlex
import sys
from io import StringIO

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import (
    detect_shell_operators,
    shell_operator_rejection,
    split_command,
)


async def run_tests(runner: str, path: str):
    """Run tests through the production shell surface."""
    from victor.tools.bash import shell

    cmd = f"{shlex.quote(runner)} {shlex.quote(path)}"
    return await shell(cmd=cmd, readonly=True, action="read")


async def execute_python(code: str):
    """Execute a short Python snippet through the production shell surface."""
    from victor.tools.bash import shell

    result = await shell(
        cmd=f"python -c {shlex.quote(code)}",
        readonly=False,
        action="execute",
    )
    if isinstance(result, dict):
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        if result.get("success") is False:
            return stderr or result.get("error", "")
        return stdout or stderr
    return str(result)


async def analyze_metrics(path: str):
    """Return lightweight code metrics for Python files under a path."""
    root = Path(path).expanduser()
    files = [root] if root.is_file() else list(root.rglob("*.py"))
    loc = 0
    functions = 0
    classes = 0
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        loc += sum(1 for line in lines if line.strip())
        functions += sum(1 for line in lines if line.lstrip().startswith("def "))
        classes += sum(1 for line in lines if line.lstrip().startswith("class "))
    return {"files": len(files), "loc": loc, "functions": functions, "classes": classes}


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

    # `execute` / `python` subcommands
    exec_parser = subparsers.add_parser("execute", help="Execute python code")
    exec_parser.add_argument("code", help="Python code to execute")

    python_parser = subparsers.add_parser("python", help="Execute python code")
    python_parser.add_argument("code", help="Python code to execute")

    # `metrics` subcommand
    metrics_parser = subparsers.add_parser("metrics", help="Analyze code metrics")
    metrics_parser.add_argument("path", nargs="?", default=".", help="Directory to analyze")

    # `grep` subcommand — literal content search (the canonical grep surface)
    grep_parser = subparsers.add_parser("grep", help="Search file contents (literal/regex)")
    grep_parser.add_argument("query", help="Text or regex to search for")
    grep_parser.add_argument("path", nargs="?", default=".", help="File or directory to search")
    grep_parser.add_argument("--regex", action="store_true", help="Treat query as a regex")
    grep_parser.add_argument(
        "--case-sensitive",
        "-C",
        action="store_true",
        help="Case-sensitive match (default is case-insensitive)",
    )

    # `search` subcommand — semantic/graph code search (delegates to victor-coding)
    search_parser = subparsers.add_parser("search", help="Semantic code search")
    search_parser.add_argument("query", help="Natural-language search query")
    search_parser.add_argument("path", nargs="?", default=".", help="Scope of the search")
    search_parser.add_argument(
        "--mode",
        choices=["semantic", "literal", "hybrid", "graph"],
        default="semantic",
        help="Search mode (default: semantic)",
    )
    search_parser.add_argument("--k", type=int, default=10, help="Max results")

    return parser


@tool(
    name="code",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.HIGH,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def code_tool(cmd: str) -> str:
    """Unified code tool.
    Example commands:
      code test pytest tests/
      code python "print('hello')"
      code python <<'PY'
      print('hello')
      PY
      code execute "print('hello')"
      code metrics src/

    Args:
        cmd: Bash-style grouped code command. Use `code python` for ad hoc
            Python snippets and heredocs; use `code test` for test runners.
    """
    parser = create_code_parser()

    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "code":
            args_list = args_list[1:]
        operator = detect_shell_operators(args_list)
        if operator is not None:
            return shell_operator_rejection("code", operator)
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
            if isinstance(results, dict):
                stdout = results.get("stdout", "").strip()
                stderr = results.get("stderr", "").strip()
                if stdout or stderr:
                    return "\n\n".join(part for part in (stdout, stderr) if part)
                if results.get("success") is False:
                    return f"### ❌ ERROR\n{results.get('error', 'Test execution failed')}"
            return str(results)
        except Exception as e:
            return f"### ❌ ERROR\nTest execution failed: {e}"

    elif parsed_args.subcommand in {"execute", "python"}:
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

    elif parsed_args.subcommand == "grep":
        from victor.tools.unified._search_helpers import format_grep_results, grep_search

        try:
            results = await grep_search(
                query=parsed_args.query,
                path=parsed_args.path,
                regex=parsed_args.regex,
                case_sensitive=parsed_args.case_sensitive,
            )
            return format_grep_results(results)
        except Exception as e:
            return f"### ❌ ERROR\ngrep failed: {e}"

    elif parsed_args.subcommand == "search":
        return await _code_search(parsed_args)

    else:
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"


async def _code_search(parsed_args) -> str:
    """``code search`` — semantic/graph code search with a literal fallback.

    Delegates to ``victor_coding.code_search`` when the package is available;
    otherwise (and for ``--mode literal``) falls back to a literal grep so the
    surface always works, per the graceful-degradation principle.
    """
    from victor.tools.unified._search_helpers import format_grep_results, grep_search

    if parsed_args.mode == "literal":
        results = await grep_search(query=parsed_args.query, path=parsed_args.path)
        return format_grep_results(results)

    from victor.tools.unified._vertical_resolver import resolve_vertical_callable

    search_fn, _src = resolve_vertical_callable(
        "code_search",
        fallback_module="victor_coding.tools.code_search_tool",
        fallback_attr="code_search",
    )
    if search_fn is None:
        # No semantic backend available — degrade to literal with a hint.
        results = await grep_search(query=parsed_args.query, path=parsed_args.path)
        return (
            "### 💡 SYSTEM HINT\nSemantic code search requires the victor-coding "
            "package, which is not installed. Showing literal matches instead.\n\n"
            + format_grep_results(results)
        )

    try:
        result = await search_fn(
            query=parsed_args.query,
            path=parsed_args.path,
            mode=parsed_args.mode,
            k=parsed_args.k,
        )
    except TypeError:
        # Signature drift between versions — retry without the ``k`` kwarg.
        try:
            result = await search_fn(
                query=parsed_args.query, path=parsed_args.path, mode=parsed_args.mode
            )
        except Exception as e:
            return f"### ❌ ERROR\nSemantic search failed: {e}"
    except Exception as e:
        return f"### ❌ ERROR\nSemantic search failed: {e}"

    return _format_code_search_result(result)


def _format_code_search_result(result: object) -> str:
    """Normalize a ``code_search`` result (dict/list/str) into a display string."""
    if isinstance(result, dict):
        if result.get("success") is False:
            return f"### ❌ ERROR\n{result.get('error', 'code search failed')}"
        results = result.get("results") or result.get("matches") or result.get("output")
        if isinstance(results, list):
            out = []
            for item in results:
                if isinstance(item, dict):
                    loc = item.get("file") or item.get("path") or item.get("location", "?")
                    line = item.get("line", "")
                    score = item.get("score")
                    snippet = (item.get("content") or item.get("snippet") or "").strip()
                    head = f"{loc}:{line}" if line else str(loc)
                    if score is not None:
                        head += f" (score: {score})"
                    out.append(f"- {head}\n  {snippet}" if snippet else f"- {head}")
                else:
                    out.append(f"- {item}")
            return "\n".join(out) if out else "No matches found."
        return str(results) if results is not None else "No matches found."
    if isinstance(result, list):
        return "\n".join(f"- {item}" for item in result) or "No matches found."
    return str(result)
