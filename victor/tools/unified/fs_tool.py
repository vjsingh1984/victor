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
import shlex
import sys
from io import StringIO
from typing import List, Optional

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.filesystem import read, write, ls
from victor.tools.patch_tool import patch as apply_patch


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

    return parser


@tool(
    name="fs",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.MEDIUM,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def fs_tool(command: str) -> str:
    """Unified filesystem tool with bash-like syntax. Use subcommands to
    interact with the file system. Example commands:
      fs ls /path/to/dir
      fs cat /path/to/file
      fs write /path/to/file -c "Hello"
      fs patch /path/to/file --search "old" --replace "new"
    """
    parser = create_fs_parser()

    try:
        from victor.tools.unified.parser import split_command

        args_list = split_command(command)
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
            return str(
                await apply_patch(
                    operation="apply", file_path=parsed_args.path, patch_content=parsed_args.replace
                )
            )
        except Exception as e:
            return f"### ❌ ERROR\nPatch failed: {e}\n\n### 💡 SYSTEM HINT\nUse `fs cat {parsed_args.path}` to refresh your view of the code before editing."

    else:
        # Provide help if no subcommand is recognized
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"
