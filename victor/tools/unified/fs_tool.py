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
    cat_parser.add_argument(
        "--offset", type=int, default=0, help="Start line (1-based) for paging large files"
    )
    cat_parser.add_argument("--limit", type=int, default=0, help="Max lines to read (0 = all/auto)")
    cat_parser.add_argument(
        "--search", default="", help="Only show lines matching this pattern (in-file search)"
    )
    cat_parser.add_argument(
        "--ctx", type=int, default=2, help="Context lines around --search matches"
    )
    cat_parser.add_argument("--regex", action="store_true", help="Treat --search as a regex")

    # `ls` / `list` subcommand
    ls_parser = subparsers.add_parser("ls", help="List directory contents")
    ls_parser.add_argument("path", nargs="?", default=".", help="Directory path to list")
    ls_parser.add_argument("-r", "--recursive", action="store_true", help="List recursively")
    ls_parser.add_argument("--depth", type=int, default=2, help="Directory depth to descend")
    ls_parser.add_argument("--pattern", default="", help="Glob filter (e.g. '*.py', 'test_*')")
    ls_parser.add_argument("--limit", type=int, default=1000, help="Max entries to return")

    # `write` subcommand
    write_parser = subparsers.add_parser("write", help="Write content to a file")
    write_parser.add_argument("path", help="Path to the file")
    write_parser.add_argument("--content", "-c", required=True, help="Content to write")
    write_parser.add_argument("--validate", action="store_true", help="LSP-validate before writing")
    write_parser.add_argument(
        "--format", action="store_true", help="Auto-format with a language formatter"
    )
    write_parser.add_argument(
        "--dry-run", action="store_true", help="Validate/preview without writing"
    )

    # `patch` subcommand
    patch_parser = subparsers.add_parser("patch", help="Patch file contents")
    patch_parser.add_argument("path", help="Path to the file")
    patch_parser.add_argument("--search", required=True, help="Text to search for")
    patch_parser.add_argument("--replace", required=True, help="Text to replace with")

    # `edit` subcommand — delegates to the structured edit() tool (atomic, undoable)
    edit_parser = subparsers.add_parser("edit", help="Edit files atomically with undo")
    edit_parser.add_argument("path", help="Path to the file")
    edit_parser.add_argument("--old", default=None, help="Text to find (replace op)")
    edit_parser.add_argument("--new", default=None, help="Replacement / inserted text")
    edit_parser.add_argument(
        "--new-file",
        default=None,
        help="Read the new/inserted text from this file (robust for multiline/code)",
    )
    edit_parser.add_argument(
        "--insert",
        default=None,
        help="Anchor line — insert --new text immediately AFTER the unique matching line",
    )
    edit_parser.add_argument(
        "--before",
        default=None,
        help="Anchor line — insert --new text immediately BEFORE the unique matching line",
    )
    edit_parser.add_argument(
        "--append",
        action="store_true",
        help="Append --new text to the end of the file",
    )
    edit_parser.add_argument(
        "--ops",
        default=None,
        help="Raw JSON ops list for advanced multi-op edits",
    )
    edit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the diff without writing",
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
      fs ls /path/to/dir [-r] [--depth N] [--pattern '*.py']
      fs cat /path/to/file [--offset N --limit N] [--search PAT [--ctx N] [--regex]]
      fs write /path/to/file -c "Hello" [--validate] [--format] [--dry-run]
      fs patch /path/to/file --search "old" --replace "new"
      fs edit /path --old "a" --new "b"            # replace
      fs edit /path --insert "anchor" --new "..."  # insert after a line
      fs edit /path --append --new "..."           # append to end
      fs edit /path --old "a" --new-file new.txt   # new text from a file
      fs edit /path --old "a" --new "b" --dry-run  # preview diff only
      fs search "*.py" /path

    Note: `fs` is NOT a shell — pipes (`|`), redirects (`>`), `||`, `&&`, `;`
    are not interpreted. For shell pipelines use the `shell` tool.

    Paging: `fs cat big.py --offset 200 --limit 100` reads a slice of a large file.
    In-file search: `fs cat app.py --search "def login" --ctx 3`.
    """
    parser = create_fs_parser()

    try:
        from victor.tools.unified.parser import (
            detect_shell_operators,
            shell_operator_rejection,
            split_command,
        )

        args_list = split_command(cmd)
        if args_list and args_list[0] == "fs":
            args_list = args_list[1:]

        operator = detect_shell_operators(args_list)
        if operator is not None:
            return shell_operator_rejection("fs", operator)

        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"Error parsing command: {e}"
    except Exception as e:
        return f"Unexpected error parsing command: {e}"

    # Delegate to the underlying granular tools
    if parsed_args.subcommand == "cat":
        try:
            return str(
                await read(
                    parsed_args.path,
                    offset=parsed_args.offset,
                    limit=parsed_args.limit,
                    search=parsed_args.search,
                    ctx=parsed_args.ctx,
                    regex=parsed_args.regex,
                )
            )
        except Exception as e:
            return f"Error reading file: {e}"

    elif parsed_args.subcommand == "ls":
        try:
            results = await ls(
                parsed_args.path,
                recursive=parsed_args.recursive,
                depth=parsed_args.depth,
                pattern=parsed_args.pattern,
                limit=parsed_args.limit,
            )
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
            return str(
                await write(
                    parsed_args.path,
                    parsed_args.content,
                    validate=parsed_args.validate,
                    format_code=parsed_args.format,
                    dry_run=parsed_args.dry_run,
                )
            )
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
    """``fs edit`` — delegate to the structured, atomic ``edit()`` tool.

    Supports four modes (plus raw ``--ops`` JSON):

    - **replace**: ``--old <text> --new <text>`` (default; the classic find/replace).
    - **insert**:  ``--insert <anchor> --new <text>`` inserts after the unique line
      containing ``<anchor>``. ``--before <anchor>`` inserts before it.
    - **append**:  ``--append --new <text>`` appends to end of file.
    - **ops**:     ``--ops '<json>'`` for advanced multi-op edits.

    The new/inserted text may come from ``--new`` (inline) or ``--new-file``
    (read from a file — robust for large, multiline, or quote/brace-heavy code).
    Triple-quoted ``--new \"\"\"...\"\"\"`` is also supported by the parser.

    insert/append are resolved locally into a full-file ``modify`` op so the
    underlying ``edit()`` keeps its atomic-write + undo semantics; the
    ``EditorProtocol`` surface is untouched.
    """
    from victor.tools.file_editor_tool import edit

    dry_run = bool(parsed_args.dry_run)

    # --- raw JSON ops path -------------------------------------------------
    if parsed_args.ops:
        try:
            ops = json.loads(parsed_args.ops)
        except json.JSONDecodeError as e:
            return f"### ❌ ERROR\n--ops must be a valid JSON list: {e}"
        return await _dispatch_edit(ops, dry_run=dry_run)

    # --- resolve the new/inserted text source ------------------------------
    new_text = parsed_args.new
    if parsed_args.new_file is not None:
        try:
            new_text = Path(parsed_args.new_file).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            return f"### ❌ ERROR\n--new-file could not be read ({parsed_args.new_file}): {e}"

    path = parsed_args.path

    # --- replace mode ------------------------------------------------------
    if parsed_args.old is not None:
        if new_text is None:
            return (
                "### ❌ ERROR\nfs edit replace needs --new (or --new-file). Example:\n"
                "  fs edit <path> --old '<exact old text>' --new '<new text>'"
            )
        ops = [
            {
                "type": "replace",
                "path": path,
                "old_str": parsed_args.old,
                "new_str": new_text,
            }
        ]
        return await _dispatch_edit(ops, dry_run=dry_run)

    # --- insert / append modes --------------------------------------------
    if parsed_args.insert is not None or parsed_args.before is not None or parsed_args.append:
        if new_text is None:
            mode = "append" if parsed_args.append else "insert"
            return (
                f"### ❌ ERROR\nfs edit {mode} needs --new (or --new-file). Example:\n"
                f"  fs edit <path> {'--append' if parsed_args.append else '--insert <anchor>'} "
                "--new '<new code>'"
            )
        return await _fs_edit_splice(parsed_args, new_text, dry_run=dry_run)

    # --- no usable mode ----------------------------------------------------
    # The most common failure: --new given with no --old/--insert/--append.
    if new_text is not None:
        return _no_mode_error()
    return (
        "### ❌ ERROR\nfs edit needs a mode. Examples:\n"
        "  fs edit <path> --old '<old>' --new '<new>'        # replace\n"
        "  fs edit <path> --insert '<anchor>' --new '<new>'  # insert after line\n"
        "  fs edit <path> --append --new '<new>'             # append\n"
        "  fs edit <path> --ops '<json>'                     # advanced\n"
        "Add --dry-run to preview, --new-file <path> for large/multiline content."
    )


def _no_mode_error() -> str:
    """Short, actionable error for `--new` given without a target mode."""
    return (
        "### ❌ ERROR\nfs edit: `--new` given without a target mode. Pick one:\n"
        "  • replace: fs edit <path> --old '<exact old text>' --new '<new text>'\n"
        "  • insert:  fs edit <path> --insert '<anchor line>' --new '<new code>'\n"
        "  • append:  fs edit <path> --append --new '<new code>'\n"
        "  • create:  use `fs write <path> --content '<full content>'` for new files\n"
        "Tip: for large/multiline content use `--new-file <path>` or triple quotes "
        '("""...""").'
    )


async def _fs_edit_splice(parsed_args, new_text: str, *, dry_run: bool) -> str:
    """Resolve an insert/append into a full-file ``modify`` op and dispatch.

    Reads the target file, splices the new text at the anchor (or end), then
    delegates to :func:`edit` as a single ``modify`` op so the write is atomic
    and undoable. The anchor must match exactly one line (substring match).
    """
    file_path = Path(parsed_args.path).expanduser()
    if not file_path.exists():
        return f"### ❌ ERROR\nCannot insert/append into missing file: {parsed_args.path}"
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"### ❌ ERROR\nCould not read {parsed_args.path}: {e}"

    lines = content.split("\n")
    if parsed_args.append:
        # Append is clean string concatenation (line-splice would add a stray
        # blank line when the file ends with a newline). The caller controls
        # leading newlines via --new.
        full_content = content + new_text
        ops = [{"type": "modify", "path": parsed_args.path, "content": full_content}]
        return await _dispatch_edit(ops, dry_run=dry_run)

    anchor = parsed_args.insert if parsed_args.insert is not None else parsed_args.before
    matches = [i for i, line in enumerate(lines) if anchor in line]
    if not matches:
        return (
            f"### ❌ ERROR\nInsert anchor not found: no line contains {anchor!r}.\n"
            f"Refresh with `fs cat {parsed_args.path}` and copy the exact anchor text."
        )
    if len(matches) > 1:
        return (
            f"### ❌ ERROR\nInsert anchor ambiguous: {len(matches)} lines contain "
            f"{anchor!r}. Add more characters so the anchor matches a single line."
        )
    anchor_idx = matches[0]
    # --insert places text AFTER the anchor line; --before places it BEFORE.
    insert_at = anchor_idx if parsed_args.before is not None else anchor_idx + 1

    new_lines = lines[:insert_at] + new_text.split("\n") + lines[insert_at:]
    full_content = "\n".join(new_lines)

    ops = [{"type": "modify", "path": parsed_args.path, "content": full_content}]
    return await _dispatch_edit(ops, dry_run=dry_run)


async def _dispatch_edit(ops: list, *, dry_run: bool) -> str:
    """Call the structured ``edit()`` tool and format its result for fs."""
    from victor.tools.file_editor_tool import edit

    try:
        result = await edit(ops=ops, preview=dry_run, commit=not dry_run)
    except Exception as e:
        return f"### ❌ ERROR\nEdit failed: {e}"

    if not isinstance(result, dict):
        return str(result)
    if result.get("success") is False:
        return f"### ❌ ERROR\n{result.get('error', 'edit failed')}"

    if dry_run:
        diff = result.get("diff") or result.get("preview_output") or ""
        msg = result.get("message", "Dry-run preview (not written).")
        return f"{msg}\n```diff\n{diff}\n```" if diff else msg

    summary = result.get("message") or result.get("summary") or "Edit applied."
    changed = result.get("changed_files") or result.get("files") or []
    if changed:
        return f"{summary}\nChanged: {', '.join(map(str, changed))}"
    return str(summary)


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
