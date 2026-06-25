# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unified Filesystem Tool with bash-like options.

This module provides a unified `fs` tool that exposes granular filesystem
operations (read, write, list) via a single entrypoint using bash-like syntax
(e.g., `fs cat /path`, `fs ls /path`, `fs patch /path --diff="..."`).
"""

import argparse
import json
from pathlib import Path
import sys
from io import StringIO

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.filesystem import find, read, write, ls


class UnifiedFsParser(argparse.ArgumentParser):
    """Custom parser that captures output instead of exiting on error."""

    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_fs_parser() -> UnifiedFsParser:
    """Create the parser for the fs tool."""
    parser = UnifiedFsParser(
        prog="fs", description="Unified filesystem operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    # `cat` / `read` subcommand
    cat_parser = subparsers.add_parser("cat", help="Read file contents")
    cat_parser.add_argument("path", help="Path to the file to read")

    # `ls` / `list` subcommand
    ls_parser = subparsers.add_parser("ls", help="List directory contents")
    ls_parser.add_argument("path", nargs="?", default=".", help="Directory path to list")

    # `write` subcommand
    write_parser = subparsers.add_parser("write", help="Write content to a file")
    write_parser.add_argument("path", help="Path to the file")
    write_parser.add_argument("--content", "-c", required=True, help="Content to write")

    # `patch` subcommand
    patch_parser = subparsers.add_parser("patch", help="Patch file contents")
    patch_parser.add_argument("path", help="Path to the file")
    patch_parser.add_argument("--search", required=True, help="Text to search for")
    patch_parser.add_argument("--replace", required=True, help="Text to replace with")

    # `edit` subcommand — delegates to the structured edit() tool (atomic, undoable)
    edit_parser = subparsers.add_parser("edit", help="Edit files atomically with undo")
    edit_parser.add_argument("path", help="Path to the file")
    edit_parser.add_argument("--old", default=None, help="Text to find (replace op)")
    edit_parser.add_argument("--new", default=None, help="Replacement text (replace op)")
    edit_parser.add_argument(
        "--ops",
        default=None,
        help="Raw JSON ops list for advanced multi-op edits",
    )

    # `search` subcommand — find files by name/metadata (delegates to find())
    search_parser = subparsers.add_parser("search", help="Find files by name pattern")
    search_parser.add_argument("pattern", help="Glob pattern (e.g. *.py, *_tool.py)")
    search_parser.add_argument("path", nargs="?", default=".", help="Root directory")
    search_parser.add_argument(
        "--type",
        choices=["file", "dir", "all"],
        default="all",
        help="Restrict to files or directories (default: all)",
    )
    search_parser.add_argument("--limit", type=int, default=50, help="Max results")

    return parser


@tool(
    name="fs",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.MEDIUM,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def fs_tool(cmd: str) -> str:
    """Unified filesystem tool with bash-like syntax. Use subcommands to
    interact with the file system. Example commands:
      fs ls /path/to/dir
      fs cat /path/to/file
      fs write /path/to/file -c "Hello"
      fs patch /path/to/file --search "old" --replace "new"
      fs edit /path --old "a" --new "b"
      fs search "*.py" /path
    """
    parser = create_fs_parser()

    try:
        from victor.tools.unified.parser import split_command

        args_list = split_command(cmd)
        if args_list and args_list[0] == "fs":
            args_list = args_list[1:]

        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"Error parsing command: {e}"
    except Exception as e:
        return f"Unexpected error parsing command: {e}"

    # Delegate to the underlying granular tools
    if parsed_args.subcommand == "cat":
        try:
            return str(await read(parsed_args.path))
        except Exception as e:
            return f"Error reading file: {e}"

    elif parsed_args.subcommand == "ls":
        try:
            results = await ls(parsed_args.path)
            # Format results into a markdown table instead of raw JSON
            if not isinstance(results, list):
                return str(results)

            out = [f"Directory listing for {parsed_args.path}:"]
            out.append("| Type | Size/Lines | Path |")
            out.append("|---|---|---|")
            for item in results:
                t = item.get("type", "unknown")
                s = item.get("size", "-")
                if "hint" in item:
                    s = f"{s} ({item['hint']})"
                p = item.get("path", item.get("name", ""))
                out.append(f"| {t} | {s} | {p} |")

            if len(results) >= 100:
                out.append(
                    "\n### 💡 SYSTEM HINT\nDirectory listing truncated. Please use specific paths or depth limits to see more."
                )

            return "\n".join(out)
        except Exception as e:
            return f"### ❌ ERROR\nError listing directory: {e}"

    elif parsed_args.subcommand == "write":
        try:
            return str(await write(parsed_args.path, parsed_args.content))
        except Exception as e:
            return f"### ❌ ERROR\nError writing file: {e}"

    elif parsed_args.subcommand == "patch":
        try:
            file_path = Path(parsed_args.path).expanduser()
            content = file_path.read_text(encoding="utf-8")
            if parsed_args.search not in content:
                raise ValueError("String not found")
            updated = content.replace(parsed_args.search, parsed_args.replace, 1)
            file_path.write_text(updated, encoding="utf-8")
            return f"Patched {parsed_args.path}: replaced 1 occurrence."
        except Exception as e:
            return f"### ❌ ERROR\nPatch failed: {e}\n\n### 💡 SYSTEM HINT\nUse `fs cat {parsed_args.path}` to refresh your view of the code before editing."

    elif parsed_args.subcommand == "edit":
        return await _fs_edit(parsed_args)

    elif parsed_args.subcommand == "search":
        return await _fs_search(parsed_args)

    else:
        # Provide help if no subcommand is recognized
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"


async def _fs_edit(parsed_args) -> str:
    """``fs edit`` — delegate to the structured, atomic ``edit()`` tool."""
    from victor.tools.file_editor_tool import edit

    if parsed_args.ops:
        try:
            ops = json.loads(parsed_args.ops)
        except json.JSONDecodeError as e:
            return f"### ❌ ERROR\n--ops must be a valid JSON list: {e}"
    else:
        if parsed_args.old is None or parsed_args.new is None:
            return (
                "### ❌ ERROR\nfs edit requires --old/--new (replace) or --ops (JSON). "
                "Example: fs edit path.py --old 'DEBUG = True' --new 'DEBUG = False'"
            )
        ops = [
            {
                "type": "replace",
                "path": parsed_args.path,
                "old_str": parsed_args.old,
                "new_str": parsed_args.new,
            }
        ]

    try:
        result = await edit(ops=ops)
    except Exception as e:
        return f"### ❌ ERROR\nEdit failed: {e}"

    if isinstance(result, dict):
        if result.get("success") is False:
            return f"### ❌ ERROR\n{result.get('error', 'edit failed')}"
        # Surface a compact summary; the structured tool returns rich details.
        summary = result.get("summary") or result.get("message") or "Edit applied."
        changed = result.get("changed_files") or result.get("files") or []
        if changed:
            return f"{summary}\nChanged: {', '.join(map(str, changed))}"
        return str(summary)
    return str(result)


async def _fs_search(parsed_args) -> str:
    """``fs search`` — find files by name/metadata via the granular ``find()``."""
    try:
        results = await find(
            name=parsed_args.pattern,
            path=parsed_args.path,
            type=parsed_args.type,
            limit=parsed_args.limit,
        )
    except Exception as e:
        return f"### ❌ ERROR\nSearch failed: {e}"

    if not isinstance(results, list):
        return str(results)
    if not results:
        return f"No files matching '{parsed_args.pattern}' under {parsed_args.path}."

    out = [f"Found {len(results)} match(es) for '{parsed_args.pattern}':"]
    for item in results:
        if isinstance(item, dict):
            p = item.get("path") or item.get("name", "")
            t = item.get("type", "")
            out.append(f"- [{t}] {p}" if t else f"- {p}")
        else:
            out.append(f"- {item}")
    return "\n".join(out)
