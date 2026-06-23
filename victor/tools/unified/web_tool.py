import argparse
import sys
from io import StringIO

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import split_command

# Mock imports for the underlying web tools
try:
    from victor.tools.web import fetch_url, search_web, render_page
except ImportError:

    async def fetch_url(url: str):
        return ""

    async def search_web(query: str):
        return []

    async def render_page(url: str):
        return ""


class UnifiedWebParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_web_parser() -> UnifiedWebParser:
    parser = UnifiedWebParser(
        prog="web", description="Unified web operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    # `fetch` subcommand
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a URL as markdown")
    fetch_parser.add_argument("url", help="The URL to fetch")

    # `search` subcommand
    search_parser = subparsers.add_parser("search", help="Search the web")
    search_parser.add_argument("query", help="The search query")

    # `render` subcommand
    render_parser = subparsers.add_parser("render", help="Render a page with a browser")
    render_parser.add_argument("url", help="The URL to render")

    return parser


@tool(
    name="web",
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def web_tool(command: str) -> str:
    """Unified web tool.
    Example commands:
      web fetch "https://google.com"
      web search "python async tutorials"
      web render "https://example.com"
    """
    parser = create_web_parser()

    try:
        args_list = split_command(command)
        if args_list and args_list[0] == "web":
            args_list = args_list[1:]
        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error: {e}"

    if parsed_args.subcommand == "fetch":
        try:
            return str(await fetch_url(parsed_args.url))
        except Exception as e:
            return f"### ❌ ERROR\nFetch failed: {e}"

    elif parsed_args.subcommand == "search":
        try:
            results = await search_web(parsed_args.query)
            if not isinstance(results, list):
                return str(results)

            out = []
            for r in results:
                title = r.get("title", "No Title")
                url = r.get("url", "#")
                snippet = r.get("snippet", "")
                out.append(f"### [{title}]({url})\n{snippet}\n")

            if not out:
                return "No results found."
            return "\n".join(out)
        except Exception as e:
            return f"### ❌ ERROR\nWeb search failed: {e}"

    elif parsed_args.subcommand == "render":
        try:
            return str(await render_page(parsed_args.url))
        except Exception as e:
            return f"### ❌ ERROR\nRender failed: {e}"
    else:
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"
