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

"""Filesystem tools for reading, writing, and listing contents."""

from pathlib import Path
from typing import List, Dict, Any

import aiofiles

from victor.tools.decorators import tool


# Supported text/code file extensions
TEXT_EXTENSIONS = {
    # Code
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",  # Python
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",  # JavaScript/TypeScript
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".groovy",  # JVM
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",  # Systems
    ".cs",
    ".fs",
    ".vb",  # .NET
    ".rb",
    ".php",
    ".pl",
    ".pm",
    ".lua",
    ".r",
    ".R",  # Scripting
    ".swift",
    ".m",
    ".mm",  # Apple
    ".sol",
    ".vy",  # Blockchain
    # Config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".xsl",
    ".xsd",
    ".dtd",
    ".env",
    ".env.example",
    ".env.local",
    ".properties",
    ".plist",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
    ".astro",
    # Data
    ".csv",
    ".tsv",
    ".sql",
    # Docs
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".adoc",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".bat",
    ".cmd",
    # Build
    ".make",
    ".cmake",
    ".gradle",
    ".sbt",
    ".dockerfile",
    ".containerfile",
    # Other
    ".graphql",
    ".proto",
    ".thrift",
    ".avsc",
    ".tf",
    ".tfvars",
    ".hcl",  # Terraform
    ".vim",
    ".el",
    ".clj",
    ".cljs",
    ".edn",
    ".ex",
    ".exs",
    ".erl",
    ".hrl",  # Elixir/Erlang
    ".hs",
    ".lhs",
    ".ml",
    ".mli",
    ".fsi",  # Functional
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
}


@tool(
    category="filesystem",
    keywords=[
        "read",
        "file",
        "open",
        "view",
        "content",
        "source",
        "code",
        "show",
        "display",
        "cat",
        "look",
        "examine",
        "inspect",
    ],
    use_cases=[
        "reading source code files",
        "viewing configuration files",
        "examining text documents",
        "searching within code files",
        "looking at specific lines",
    ],
    examples=[
        "read the file src/main.py",
        "show me the contents of config.yaml",
        "what's in the README",
        "search for 'def calculate' in utils.py",
        "show first 50 lines of main.py",
    ],
    priority_hints=[
        "Use for TEXT and CODE files only (.py, .js, .json, .yaml, .md, etc.)",
        "NOT for binary files (.pdf, .docx, .db, .pyc, images, archives)",
        "Use search parameter for efficient grep-like targeted lookups",
        "Use list_directory first if unsure what files exist",
    ],
)
async def read_file(
    path: str,
    offset: int = 0,
    limit: int = 0,
    search: str = "",
    context_lines: int = 2,
    regex: bool = False,
) -> str:
    """Read TEXT/CODE file contents with optional search and line range.

    SUPPORTED: .py, .js, .ts, .java, .go, .rs, .c, .cpp, .json, .yaml, .toml,
               .xml, .html, .css, .md, .txt, .sh, .sql, .env, config files, etc.

    NOT SUPPORTED: Binary files (.pdf, .docx, .xlsx, .db, .sqlite, .pyc, .pkl,
                   images, videos, archives). These will return an error.

    Use search param for efficient grep-like lookup (returns matches + context_lines).
    Use offset/limit for line ranges. Default: returns full file.
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        if file_path.is_dir():
            raise IsADirectoryError(
                f"Cannot read directory as file: {path}\n"
                f"Suggestion: Use list_directory(path='{path}') to explore its contents, "
                f"or specify a file path within this directory."
            )
        raise IsADirectoryError(f"Path is not a file: {path}")

    # Binary file categories with helpful suggestions
    BINARY_CATEGORIES = {
        # Documents - suggest extraction approach
        "document": {
            "extensions": {
                ".pdf",
                ".doc",
                ".docx",
                ".xls",
                ".xlsx",
                ".ppt",
                ".pptx",
                ".odt",
                ".ods",
                ".odp",
            },
            "suggestion": "For document content, consider: 1) Convert to text externally (e.g., pdftotext), "
            "2) Use code_search to find related text files, 3) Check for a .txt or .md version.",
        },
        # Images - suggest description
        "image": {
            "extensions": {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".ico",
                ".webp",
                ".svg",
                ".tiff",
                ".tif",
            },
            "suggestion": "Images cannot be read as text. If you need to understand image content, "
            "describe what you expect based on filename/context, or check for alt-text in nearby HTML/MD files.",
        },
        # Databases - suggest querying
        "database": {
            "extensions": {".db", ".sqlite", ".sqlite3"},
            "suggestion": "This is a database file. Use execute_bash with 'sqlite3' to query it, "
            "e.g., `sqlite3 file.db '.tables'` or `sqlite3 file.db 'SELECT * FROM table LIMIT 5'`.",
        },
        # Python bytecode/cache
        "python_cache": {
            "extensions": {".pyc", ".pyo", ".pyd"},
            "suggestion": "This is compiled Python bytecode. Read the source .py file instead.",
        },
        # Pickled data
        "pickle": {
            "extensions": {".pkl", ".pickle"},
            "suggestion": "This is serialized Python data. To inspect, use: "
            "`python -c \"import pickle; print(pickle.load(open('file.pkl', 'rb')))\"`",
        },
        # Archives
        "archive": {
            "extensions": {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz"},
            "suggestion": "This is an archive. Use execute_bash to list contents: "
            "`unzip -l file.zip`, `tar -tf file.tar.gz`, etc.",
        },
        # Compiled/binary
        "compiled": {
            "extensions": {
                ".so",
                ".o",
                ".a",
                ".dylib",
                ".dll",
                ".exe",
                ".bin",
                ".whl",
                ".egg",
                ".class",
                ".jar",
                ".war",
            },
            "suggestion": "This is a compiled binary. Check for source code files (.py, .java, .c, etc.) instead.",
        },
        # Media
        "media": {
            "extensions": {
                ".mp3",
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".wav",
                ".flac",
                ".ogg",
                ".webm",
            },
            "suggestion": "Media files cannot be read as text. Check for subtitles (.srt, .vtt) or metadata files.",
        },
        # Lock files
        "lock": {
            "extensions": {".lock"},
            "suggestion": "Lock files are often binary or auto-generated. Check the non-lock version if available.",
        },
        # Data files
        "data": {
            "extensions": {".dat", ".bin"},
            "suggestion": "This appears to be a binary data file. Look for documentation or schema files nearby.",
        },
    }

    # Known binary filenames (no extension)
    BINARY_FILENAMES = {".coverage"}

    def _get_binary_error(ext: str, filename: str) -> str:
        """Generate helpful error message based on file type."""
        ext_lower = ext.lower()
        for category, info in BINARY_CATEGORIES.items():
            if ext_lower in info["extensions"]:
                return (
                    f"Cannot read binary file: {filename}\n"
                    f"Type: {category} ({ext})\n"
                    f"Suggestion: {info['suggestion']}"
                )
        return f"Cannot read binary file: {filename} (extension: {ext}). This is not a text file."

    # Flatten all binary extensions
    ALL_BINARY_EXTENSIONS = set()
    for info in BINARY_CATEGORIES.values():
        ALL_BINARY_EXTENSIONS.update(info["extensions"])

    # Check by extension
    if file_path.suffix.lower() in ALL_BINARY_EXTENSIONS:
        raise ValueError(_get_binary_error(file_path.suffix, path))

    # Check by filename (for dotfiles without extensions like .coverage)
    if file_path.name in BINARY_FILENAMES or file_path.name.startswith(".coverage."):
        raise ValueError(
            f"Cannot read binary file: {path}\n"
            f"Type: coverage database\n"
            f"Suggestion: This is a pytest-cov SQLite database. Use execute_bash with sqlite3 to query, "
            f"or use 'coverage report' to see formatted coverage data."
        )

    # Try to read the file, handling encoding errors gracefully
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
    except UnicodeDecodeError as e:
        # File contains binary content - provide helpful message
        file_size = file_path.stat().st_size
        raise ValueError(
            f"Cannot read file: {path}\n"
            f"Reason: Contains binary/non-UTF-8 content (error at byte {e.start})\n"
            f"Size: {file_size:,} bytes\n"
            f"Suggestion: This file appears to be binary despite its extension. "
            f"Check if it's the correct file, or look for a text-based alternative."
        )

    # Normalize parameters (handle non-int input from model)
    def _to_int(val, default: int) -> int:
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
        return default

    offset = _to_int(offset, 0)
    limit = _to_int(limit, 0)
    context_lines = _to_int(context_lines, 2)

    # TOKEN-EFFICIENT MODE: Search/grep
    if search:
        from victor.tools.output_utils import grep_lines

        result = grep_lines(
            content=content,
            pattern=search,
            context_before=context_lines,
            context_after=context_lines,
            case_sensitive=True,
            is_regex=regex,
            max_matches=50,
            file_path=path,
        )
        return result.to_string(show_line_numbers=True, max_matches=50)

    # If no offset/limit, return full content
    if offset == 0 and limit == 0:
        return content

    # Handle offset and limit
    lines = content.split("\n")
    total_lines = len(lines)

    # Clamp offset to valid range
    offset = max(0, min(offset, total_lines))

    # Select lines
    if limit > 0:
        selected = lines[offset : offset + limit]
        end_line = min(offset + limit, total_lines)
    else:
        selected = lines[offset:]
        end_line = total_lines

    # Add header showing line range
    header = f"[Lines {offset + 1}-{end_line} of {total_lines}]\n"
    return header + "\n".join(selected)


@tool(
    category="filesystem",
    keywords=[
        "write",
        "file",
        "create",
        "save",
        "new",
        "output",
        "overwrite",
        "store",
        "put",
        "generate",
    ],
    use_cases=[
        "creating new files",
        "writing content to files",
        "saving generated code",
        "creating configuration files",
        "overwriting existing files",
    ],
    examples=[
        "create a new file called utils.py",
        "write this code to main.py",
        "save the configuration to config.yaml",
        "create README.md with project description",
    ],
    priority_hints=[
        "Use for creating new files or completely replacing file content",
        "For surgical edits to existing files, use edit_files with 'replace' operation instead",
        "Supports undo via /undo command",
    ],
)
async def write_file(path: str, content: str) -> str:
    """Write/overwrite file with content. Creates parent dirs if needed.

    For surgical edits (replace specific text), use edit_files instead.
    Supports /undo to revert.
    """
    from victor.agent.change_tracker import ChangeType, get_change_tracker

    file_path = Path(path).expanduser().resolve()

    if file_path.exists() and file_path.is_dir():
        raise IsADirectoryError(f"Cannot write to directory: {path}")

    # Track the change for undo/redo
    tracker = get_change_tracker()
    original_content = None
    change_type = ChangeType.CREATE

    if file_path.exists():
        # File exists - this is a modification
        change_type = ChangeType.MODIFY
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            original_content = await f.read()

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Record the change
    tracker.begin_change_group("write_file", f"Write to {path}")
    tracker.record_change(
        file_path=str(file_path),
        change_type=change_type,
        original_content=original_content,
        new_content=content,
        tool_name="write_file",
        tool_args={"path": path},
    )

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(content)

    tracker.commit_change_group()

    action = "created" if change_type == ChangeType.CREATE else "modified"
    return f"Successfully {action} {path} ({len(content)} characters). Use /undo to revert."


@tool(
    category="filesystem",
    keywords=[
        "list",
        "directory",
        "ls",
        "dir",
        "files",
        "folders",
        "find",
        "browse",
        "explore",
        "contents",
        "tree",
    ],
    use_cases=[
        "listing directory contents",
        "finding files by pattern",
        "exploring project structure",
        "listing specific file types",
    ],
    examples=[
        "list files in src directory",
        "show all Python files",
        "what files are in this folder",
        "find all test files",
        "list directories only",
    ],
    priority_hints=[
        "Use for browsing directory contents",
        "Use pattern parameter for efficient filtering",
        "Use extensions parameter to filter by file type",
    ],
)
async def list_directory(
    path: str,
    recursive: bool = False,
    pattern: str = "",
    extensions: str = "",
    dirs_only: bool = False,
    files_only: bool = False,
    max_items: int = 500,
) -> List[Dict[str, Any]]:
    """List directory contents with filtering.

    Filters: pattern (glob), extensions (comma-separated), dirs_only, files_only.
    Use recursive=True to traverse subdirectories. Max 500 items by default.
    """
    import fnmatch

    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Parse extensions if provided
        ext_set = None
        if extensions:
            ext_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions.split(",")}

        # Normalize max_items (handle non-int input from model)
        if not isinstance(max_items, int):
            max_items = (
                int(max_items) if isinstance(max_items, str) and max_items.isdigit() else 500
            )

        items = []
        count = 0

        if recursive:
            iterator = dir_path.rglob("*")
        else:
            iterator = sorted(dir_path.iterdir())

        for p in iterator:
            if count >= max_items:
                break

            is_dir = p.is_dir()
            name = str(p.relative_to(dir_path)) if recursive else p.name

            # Apply filters
            if dirs_only and not is_dir:
                continue
            if files_only and is_dir:
                continue

            # Pattern filter (glob)
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue

            # Extension filter
            if ext_set and not is_dir and p.suffix not in ext_set:
                continue

            items.append(
                {
                    "path" if recursive else "name": name,
                    "type": "directory" if is_dir else "file",
                }
            )
            count += 1

        # Add metadata if filtered
        if pattern or extensions or dirs_only or files_only:
            # Return with metadata header
            return {
                "items": items,
                "count": len(items),
                "truncated": count >= max_items,
                "filters": {
                    "pattern": pattern or None,
                    "extensions": extensions or None,
                    "dirs_only": dirs_only,
                    "files_only": files_only,
                },
            }

        return items

    except Exception as e:
        # Let the decorator handle the exception and format it
        raise e
