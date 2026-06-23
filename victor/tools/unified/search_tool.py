import argparse
import sys
from io import StringIO
from typing import Any, List, Dict

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import split_command

# We assume these exist or will be imported correctly from the core codebase
try:
    from victor.tools.search import grep_search
except ImportError:
    # Fallback or mock for testing if it doesn't exist
    async def grep_search(query: str, path: str, regex: bool = False, case_sensitive: bool = False):
        return []


try:
    from victor.tools.filesystem import find
except ImportError:

    async def find(pattern: str, path: str):
        return []


class UnifiedSearchParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_search_parser() -> UnifiedSearchParser:
    parser = UnifiedSearchParser(
        prog="search", description="Unified search operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    # `grep` subcommand
    grep_parser = subparsers.add_parser("grep", help="Search file contents")
    grep_parser.add_argument("query", help="Text or regex to search for")
    grep_parser.add_argument("path", nargs="?", default=".", help="Directory or file to search")
    grep_parser.add_argument(
        "--regex", action="store_true", help="Treat query as a regular expression"
    )
    grep_parser.add_argument(
        "--case-sensitive", "-C", action="store_true", help="Case-sensitive search"
    )

    # `files` subcommand
    files_parser = subparsers.add_parser("files", help="Find files by name")
    files_parser.add_argument("pattern", help="Glob pattern (e.g. *.py)")
    files_parser.add_argument("path", nargs="?", default=".", help="Directory to search")

    return parser


@tool(
    name="search",
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def search_tool(command: str) -> str:
    """Unified search tool with bash-like syntax.
    Example commands:
      search grep "def foo" /path
      search grep --regex "^def " /path
      search files "*.py" /path
    """
    parser = create_search_parser()

    try:
        args_list = split_command(command)
        if args_list and args_list[0] == "search":
            args_list = args_list[1:]
        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error: {e}"

    if parsed_args.subcommand == "grep":
        try:
            results = await grep_search(
                query=parsed_args.query,
                path=parsed_args.path,
                regex=parsed_args.regex,
                case_sensitive=parsed_args.case_sensitive,
            )

            if not isinstance(results, list):
                return str(results)

            out = []
            limit = 50
            for i, match in enumerate(results):
                if i >= limit:
                    break
                file_path = match.get("file", "unknown")
                line_number = match.get("line", "?")
                content = match.get("content", "").strip()
                out.append(f"{file_path}:{line_number}: {content}")

            if len(results) > limit:
                out.append(
                    f"\n### 💡 SYSTEM HINT\nToo many matches found ({len(results)}). Results truncated. Please refine your search query or directory."
                )

            if not out:
                return "No matches found."
            return "\n".join(out)
        except Exception as e:
            return f"### ❌ ERROR\nSearch failed: {e}"

    elif parsed_args.subcommand == "files":
        try:
            results = await find(pattern=parsed_args.pattern, path=parsed_args.path)
            if not isinstance(results, list):
                return str(results)

            out = []
            limit = 50
            for i, match in enumerate(results):
                if i >= limit:
                    break
                # find returns list of dicts or list of strings depending on implementation
                if isinstance(match, dict):
                    out.append(match.get("path", str(match)))
                else:
                    out.append(str(match))

            if len(results) > limit:
                out.append(
                    f"\n### 💡 SYSTEM HINT\nToo many files found ({len(results)}). Results truncated."
                )

            if not out:
                return "No files found."
            return "\n".join(out)
        except Exception as e:
            return f"### ❌ ERROR\nFile search failed: {e}"
    else:
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"
