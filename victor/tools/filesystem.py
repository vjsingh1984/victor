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

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

import aiofiles

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


# ============================================================================
# FILE TYPE DETECTION SYSTEM
# ============================================================================
# Extensible architecture for detecting file types via magic bytes and extensions.
# Future: Add handlers for PDF, images, archives, etc. via the handler registry.


class FileCategory(Enum):
    """High-level file category for routing to appropriate handlers."""

    TEXT = auto()  # Plain text, code, config files
    DOCUMENT = auto()  # PDF, DOCX, ODT, etc.
    IMAGE = auto()  # PNG, JPEG, GIF, WebP, etc.
    ARCHIVE = auto()  # ZIP, TAR, RAR, 7Z, etc.
    DATABASE = auto()  # SQLite, etc.
    MEDIA = auto()  # Audio/video files
    COMPILED = auto()  # Executables, bytecode, shared libraries
    DATA = auto()  # Generic binary data


@dataclass
class FileTypeInfo:
    """Information about a detected file type."""

    category: FileCategory
    mime_type: str
    description: str
    extensions: Tuple[str, ...]  # Expected extensions for this type
    magic_bytes: Optional[bytes] = None  # Magic signature if detected
    magic_offset: int = 0  # Offset where magic bytes are found
    suggestion: str = ""  # Help message for unsupported types


# Magic bytes signatures for common file types
# Format: (magic_bytes, offset, FileTypeInfo)
# Order matters: more specific signatures should come first
MAGIC_SIGNATURES: List[Tuple[bytes, int, FileTypeInfo]] = [
    # Documents
    (
        b"%PDF",
        0,
        FileTypeInfo(
            category=FileCategory.DOCUMENT,
            mime_type="application/pdf",
            description="PDF document",
            extensions=(".pdf",),
            suggestion="PDF reading support coming soon. For now, use `pdftotext` externally.",
        ),
    ),
    (
        b"PK\x03\x04",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/zip",
            description="ZIP archive (or DOCX/XLSX/PPTX)",
            extensions=(
                ".zip",
                ".docx",
                ".xlsx",
                ".pptx",
                ".odt",
                ".ods",
                ".odp",
                ".jar",
                ".whl",
                ".apk",
            ),
            suggestion="Use `unzip -l` to list contents, or extract and read individual files.",
        ),
    ),
    (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
        0,
        FileTypeInfo(
            category=FileCategory.DOCUMENT,
            mime_type="application/msword",
            description="Microsoft Office legacy format (DOC/XLS/PPT)",
            extensions=(".doc", ".xls", ".ppt"),
            suggestion="Convert to .docx/.xlsx/.pptx or use external tools to extract text.",
        ),
    ),
    # Images
    (
        b"\x89PNG\r\n\x1a\n",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/png",
            description="PNG image",
            extensions=(".png",),
            suggestion="Image viewing support coming soon. Check for related text/alt-text files.",
        ),
    ),
    (
        b"\xff\xd8\xff",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/jpeg",
            description="JPEG image",
            extensions=(".jpg", ".jpeg"),
            suggestion="Image viewing support coming soon. Check for related text/alt-text files.",
        ),
    ),
    (
        b"GIF87a",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/gif",
            description="GIF image (GIF87a)",
            extensions=(".gif",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"GIF89a",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/gif",
            description="GIF image (GIF89a)",
            extensions=(".gif",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"RIFF",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/webp",
            description="WebP image (RIFF container)",
            extensions=(".webp",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"BM",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/bmp",
            description="BMP image",
            extensions=(".bmp",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"II*\x00",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/tiff",
            description="TIFF image (little-endian)",
            extensions=(".tiff", ".tif"),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"MM\x00*",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/tiff",
            description="TIFF image (big-endian)",
            extensions=(".tiff", ".tif"),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"\x00\x00\x01\x00",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/x-icon",
            description="ICO icon file",
            extensions=(".ico",),
            suggestion="Icon files are binary images. Check for SVG alternatives.",
        ),
    ),
    # Archives
    (
        b"Rar!\x1a\x07",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-rar-compressed",
            description="RAR archive",
            extensions=(".rar",),
            suggestion="Use `unrar l` to list contents.",
        ),
    ),
    (
        b"7z\xbc\xaf\x27\x1c",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-7z-compressed",
            description="7-Zip archive",
            extensions=(".7z",),
            suggestion="Use `7z l` to list contents.",
        ),
    ),
    (
        b"\x1f\x8b",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/gzip",
            description="Gzip compressed file",
            extensions=(".gz", ".tgz"),
            suggestion="Use `zcat` to view contents, or `tar -tzf` for tarballs.",
        ),
    ),
    (
        b"BZh",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-bzip2",
            description="Bzip2 compressed file",
            extensions=(".bz2",),
            suggestion="Use `bzcat` to view contents.",
        ),
    ),
    (
        b"\xfd7zXZ\x00",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-xz",
            description="XZ compressed file",
            extensions=(".xz",),
            suggestion="Use `xzcat` to view contents.",
        ),
    ),
    # Database
    (
        b"SQLite format 3",
        0,
        FileTypeInfo(
            category=FileCategory.DATABASE,
            mime_type="application/x-sqlite3",
            description="SQLite database",
            extensions=(".db", ".sqlite", ".sqlite3"),
            suggestion="Use `sqlite3 file.db '.tables'` to explore the database.",
        ),
    ),
    # Media
    (
        b"ID3",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/mpeg",
            description="MP3 audio (ID3 tag)",
            extensions=(".mp3",),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    (
        b"\xff\xfb",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/mpeg",
            description="MP3 audio (frame sync)",
            extensions=(".mp3",),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    (
        b"fLaC",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/flac",
            description="FLAC audio",
            extensions=(".flac",),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    (
        b"OggS",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/ogg",
            description="Ogg container (audio/video)",
            extensions=(".ogg", ".ogv", ".oga"),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    # Compiled/executables
    (
        b"\x7fELF",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-elf",
            description="ELF executable/shared library",
            extensions=(".so", ".o", ""),
            suggestion="This is a compiled binary. Check for source files instead.",
        ),
    ),
    (
        b"MZ",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-msdownload",
            description="Windows executable (PE/MZ)",
            extensions=(".exe", ".dll"),
            suggestion="This is a compiled binary. Check for source files instead.",
        ),
    ),
    (
        b"\xca\xfe\xba\xbe",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/java-archive",
            description="Java class file / Mach-O fat binary",
            extensions=(".class",),
            suggestion="This is compiled Java bytecode. Check for .java source files.",
        ),
    ),
    (
        b"\xcf\xfa\xed\xfe",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-mach-binary",
            description="Mach-O binary (64-bit)",
            extensions=("", ".dylib"),
            suggestion="This is a compiled macOS binary. Check for source files instead.",
        ),
    ),
    (
        b"\xce\xfa\xed\xfe",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-mach-binary",
            description="Mach-O binary (32-bit)",
            extensions=("", ".dylib"),
            suggestion="This is a compiled macOS binary. Check for source files instead.",
        ),
    ),
    # Python bytecode
    # Note: Python bytecode magic changes with versions, so we check extension primarily
]

# Maximum bytes to read for magic detection
MAGIC_BYTES_READ_SIZE = 32


def detect_file_type_by_magic(file_path: Path) -> Optional[FileTypeInfo]:
    """Detect file type by reading magic bytes from file header.

    Args:
        file_path: Path to the file to inspect

    Returns:
        FileTypeInfo if a known signature is found, None otherwise
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(MAGIC_BYTES_READ_SIZE)
    except (IOError, PermissionError):
        return None

    if not header:
        return None

    for magic_bytes, offset, type_info in MAGIC_SIGNATURES:
        if len(header) >= offset + len(magic_bytes):
            if header[offset : offset + len(magic_bytes)] == magic_bytes:
                # Return a copy with the detected magic bytes
                return FileTypeInfo(
                    category=type_info.category,
                    mime_type=type_info.mime_type,
                    description=type_info.description,
                    extensions=type_info.extensions,
                    magic_bytes=magic_bytes,
                    magic_offset=offset,
                    suggestion=type_info.suggestion,
                )

    return None


def check_extension_magic_mismatch(
    file_path: Path, magic_type: Optional[FileTypeInfo]
) -> Optional[str]:
    """Check if file extension matches detected magic bytes.

    Returns a warning message if there's a mismatch, None otherwise.
    """
    if magic_type is None:
        return None

    ext = file_path.suffix.lower()

    # If extension matches expected extensions, no warning
    if ext in magic_type.extensions:
        return None

    # Extension doesn't match magic bytes - potential mismatch
    expected_exts = ", ".join(magic_type.extensions) if magic_type.extensions else "(none)"
    return (
        f"Warning: File extension '{ext}' doesn't match detected content type.\n"
        f"Detected: {magic_type.description} (mime: {magic_type.mime_type})\n"
        f"Expected extensions: {expected_exts}\n"
        f"The file may have been renamed or misidentified."
    )


# ============================================================================
# BINARY FILE HANDLER REGISTRY (Extensible Architecture)
# ============================================================================
# Future handlers for different binary types can be registered here.
# Each handler takes a Path and returns extracted text content.

BinaryFileHandler = Callable[[Path], str]

# Registry for binary file handlers by category
# Future: Add handlers like:
#   FileCategory.DOCUMENT: pdf_handler, docx_handler
#   FileCategory.IMAGE: image_description_handler
#   FileCategory.ARCHIVE: archive_list_handler
_BINARY_HANDLERS: Dict[FileCategory, BinaryFileHandler] = {}


def register_binary_handler(category: FileCategory, handler: BinaryFileHandler) -> None:
    """Register a handler for a binary file category.

    Args:
        category: The file category this handler supports
        handler: Function that takes a Path and returns extracted text content

    Example:
        def pdf_handler(path: Path) -> str:
            # Extract text from PDF
            return extracted_text

        register_binary_handler(FileCategory.DOCUMENT, pdf_handler)
    """
    _BINARY_HANDLERS[category] = handler


def unregister_binary_handler(category: FileCategory) -> None:
    """Unregister a handler for a binary file category.

    Useful for test cleanup to avoid test pollution.

    Args:
        category: The file category to unregister
    """
    _BINARY_HANDLERS.pop(category, None)


def get_binary_handler(category: FileCategory) -> Optional[BinaryFileHandler]:
    """Get the handler for a binary file category, if one is registered."""
    return _BINARY_HANDLERS.get(category)


# ============================================================================
# FILE TYPE CONSTANTS
# ============================================================================

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
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.READONLY,  # Only reads files
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["path", "offset", "limit"],  # Params indicating exploration progress
    stages=["reading", "initial", "analysis", "verification"],  # Conversation stages where relevant
    task_types=["analysis", "search"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.READ_ONLY,  # Safe for parallel execution
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
    mandatory_keywords=[
        "read file",
        "show file",
        "explain this code",
        "what does this",
        # Additional keywords from MANDATORY_TOOL_KEYWORDS
        "explain", "describe", "what does",
    ],  # Force inclusion
    priority_hints=[
        "TRUNCATION: Output limited to ~15,000 chars. Use offset/limit for large files.",
        "PAGINATION: For files >100KB, use limit=200-500 and increment offset to read in chunks.",
        "Use for TEXT and CODE files only (.py, .js, .json, .yaml, .md, etc.)",
        "NOT for binary files (.pdf, .docx, .db, .pyc, images, archives)",
        "Use search parameter for efficient grep-like targeted lookups",
        "Use ls first to check file sizes before reading",
    ],
)
async def read(
    path: str,
    offset: int = 0,
    limit: int = 0,
    search: str = "",
    ctx: int = 2,
    regex: bool = False,
    # Parameter aliases for models that use different names (e.g., gpt-oss)
    line_start: int = None,
    line_end: int = None,
) -> str:
    """Read text/code file. Binary files rejected.

    IMPORTANT: Output is truncated to ~15,000 chars (~500 lines). For large files:
    - Use offset/limit for paginated reads: read(path, offset=0, limit=200), then offset=200, etc.
    - Use search param to find specific content without reading entire file

    Args:
        path: File path
        offset: Start line (0=beginning). Use for pagination of large files.
        limit: Max lines to read (0=all, but truncated at ~500 lines).
               Recommended: Use limit=200-500 for large files and paginate.
        search: Grep pattern - efficient for finding specific content
        ctx: Context lines around matches
        regex: Pattern is regex
        line_start: Alias for offset (some models use this name)
        line_end: Alias for limit (some models use this name)

    Returns:
        File content (truncated if >15,000 chars). Use offset to continue reading.
    """
    # Handle parameter aliases from models that use different names
    if line_start is not None and offset == 0:
        offset = line_start
    if line_end is not None and limit == 0:
        # Convert line_end to limit (line_end is absolute, limit is count)
        if line_start is not None:
            limit = max(0, line_end - line_start)
        else:
            limit = line_end
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

    # =========================================================================
    # MAGIC BYTES DETECTION (Primary check - cannot be spoofed by extension)
    # =========================================================================
    # Detect actual file type via magic bytes first
    magic_type = detect_file_type_by_magic(file_path)

    if magic_type is not None:
        # Check for extension/magic mismatch (potential spoofing or misnamed file)
        mismatch_warning = check_extension_magic_mismatch(file_path, magic_type)
        if mismatch_warning:
            logger.warning(mismatch_warning)

        # If magic bytes indicate binary content, check if we have a handler
        if magic_type.category != FileCategory.TEXT:
            handler = get_binary_handler(magic_type.category)
            if handler is not None:
                # Future: Use registered handler to extract text
                try:
                    return handler(file_path)
                except Exception as e:
                    logger.error(f"Binary handler failed for {path}: {e}")
                    # Fall through to error

            # No handler available - provide helpful error with magic-based info
            raise ValueError(
                f"Cannot read binary file: {path}\n"
                f"Detected type: {magic_type.description}\n"
                f"MIME type: {magic_type.mime_type}\n"
                f"Category: {magic_type.category.name}\n"
                + (f"Suggestion: {magic_type.suggestion}" if magic_type.suggestion else "")
                + (f"\n\n{mismatch_warning}" if mismatch_warning else "")
            )

    # =========================================================================
    # EXTENSION-BASED CHECK (Fallback for files without recognized magic bytes)
    # =========================================================================
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

    # =========================================================================
    # TEXT FILE READING
    # =========================================================================
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
    ctx = _to_int(ctx, 2)

    # TOKEN-EFFICIENT MODE: Search/grep
    if search:
        from victor.tools.output_utils import grep_lines

        result = grep_lines(
            content=content,
            pattern=search,
            context_before=ctx,
            context_after=ctx,
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
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.WRITE,  # Creates/overwrites files
    danger_level=DangerLevel.LOW,  # Minor risk, easily undoable
    # Registry-driven metadata for tool selection and cache invalidation
    progress_params=["path"],  # Same file = loop, regardless of content
    stages=["execution"],  # Conversation stages where relevant
    task_types=["edit", "generation", "action"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.WRITE,  # Cannot run in parallel with conflicting ops
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
        "For surgical edits to existing files, use edit with 'replace' operation instead",
        "Supports undo via /undo command",
    ],
)
async def write(path: str, content: str) -> str:
    """Write file. Creates parent dirs. Use edit_files for partial edits.

    Args:
        path: File path (creates dirs)
        content: Full content (overwrites)

    Returns:
        Success message.
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
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.READONLY,  # Only reads directory contents
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["path", "depth", "pattern"],  # Params indicating exploration progress
    stages=["initial", "planning", "reading", "analysis"],  # Conversation stages where relevant
    task_types=["search", "analysis"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.READ_ONLY,  # Safe for parallel execution
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
    mandatory_keywords=[
        "list files",
        "show files",
        "how many files",
        "count files",
    ],  # Force inclusion
    priority_hints=[
        "Use for browsing directory contents (default depth=2 shows subdirectories)",
        "Use pattern parameter for filtering (e.g., '*.py', 'test_*')",
        "For searching specific files, use find(name='filename') instead",
    ],
)
async def ls(
    path: str,
    recursive: bool = False,
    depth: int = 2,
    pattern: str = "",
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """List directory contents with file sizes.

    Args:
        path: Directory path
        recursive: All levels (ignores depth)
        depth: Levels to explore (default=2 for subdirectory visibility)
        pattern: Glob filter (*.py, test_*)
        limit: Max entries

    Returns:
        List of {name/path, type, depth, size, hint}.
        - size: File size in bytes (files only)
        - hint: For large files (>100KB), pagination instruction

    Tip: Default depth=2 shows contents of immediate subdirectories.
         Use depth=1 for just the top level, or recursive=True for all.
    """
    import fnmatch

    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Normalize limit (handle non-int input from model)
        if not isinstance(limit, int):
            limit = int(limit) if isinstance(limit, str) and limit.isdigit() else 1000

        items = []
        count = 0

        # Normalize depth (handle non-int input from model)
        if not isinstance(depth, int):
            depth = int(depth) if isinstance(depth, str) and depth.isdigit() else 1

        # Determine max depth for traversal
        max_depth = float("inf") if recursive else depth

        def walk_breadth_first(base: Path):
            """Walk directory breadth-first (level by level)."""
            from collections import deque

            queue = deque([(base, 0)])  # (path, current_depth)

            while queue:
                current, current_depth = queue.popleft()
                if current_depth > 0:  # Don't yield the root
                    yield current, current_depth

                if current.is_dir() and current_depth < max_depth:
                    try:
                        children = sorted(current.iterdir())
                        for child in children:
                            queue.append((child, current_depth + 1))
                    except PermissionError:
                        pass

        def walk_depth_first(base: Path, current_depth: int):
            """Walk directory depth-first (recursive)."""
            if current_depth > max_depth:
                return
            try:
                for p in sorted(base.iterdir()):
                    yield p, current_depth
                    if p.is_dir():
                        yield from walk_depth_first(p, current_depth + 1)
            except PermissionError:
                pass

        if depth == 1 and not recursive:
            # Only immediate children (optimized path)
            iterator = ((p, 1) for p in sorted(dir_path.iterdir()))
        else:
            # Breadth-first: see all children at each level first (better coverage when truncated)
            iterator = walk_breadth_first(dir_path)

        # Use relative paths when exploring beyond depth 1
        use_relative_paths = recursive or depth > 1

        for p, entry_depth in iterator:
            if count >= limit:
                break

            is_dir = p.is_dir()
            name = str(p.relative_to(dir_path)) if use_relative_paths else p.name

            # Pattern filter (glob)
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue

            # Compute full path relative to cwd for use in subsequent tool calls
            try:
                full_path = str(p.relative_to(Path.cwd()))
            except ValueError:
                # If path is not relative to cwd, use absolute
                full_path = str(p)

            item = {
                "path" if use_relative_paths else "name": name,
                "full_path": full_path,  # Full path for model to use in subsequent calls
                "type": "directory" if is_dir else "file",
                "depth": entry_depth,
            }

            # Add file size for files (helps LLM plan read operations)
            if not is_dir:
                try:
                    size = p.stat().st_size
                    item["size"] = size
                    # Direct hint for large files - tell LLM exactly what to do
                    if size > 100_000:
                        item["hint"] = "USE read(offset=0,limit=500) to paginate"
                except OSError:
                    pass  # Skip size on permission errors

            items.append(item)
            count += 1

        # Build result with cwd context for better LLM orientation
        cwd = str(Path.cwd())

        # Try to express the target path relative to cwd for clarity
        try:
            relative_target = str(dir_path.relative_to(Path.cwd()))
        except ValueError:
            relative_target = str(dir_path)  # Use absolute if outside cwd

        # Always include cwd context and relative target path in response
        result = {
            "cwd": cwd,
            "target": relative_target if relative_target != "." else ".",
            "items": items,
            "count": len(items),
        }

        # Add optional metadata
        if pattern:
            result["filter"] = pattern
        if count >= limit:
            result["truncated"] = True

        return result

    except Exception as e:
        # Let the decorator handle the exception and format it
        raise e


@tool(
    category="filesystem",
    priority=Priority.HIGH,  # Very useful for file discovery
    access_mode=AccessMode.READONLY,  # Only searches files
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=[
        "find",
        "search",
        "locate",
        "where",
        "which",
        "discover",
        "lookup",
    ],
    use_cases=[
        "finding files by name pattern",
        "locating a specific file",
        "discovering where a file exists",
        "searching for files recursively",
    ],
    examples=[
        "find tool_executor.py",
        "where is the config file",
        "locate all test files",
        "find files named *_tool.py",
    ],
)
async def find(
    name: str,
    path: str = ".",
    type: str = "all",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Find files by name pattern (like Unix find -name).

    Searches recursively through the directory tree to locate files
    matching the given name pattern. Supports glob patterns.

    Args:
        name: File name pattern to find (supports glob: *.py, *test*, tool_*.py)
        path: Root directory to search from (default: current directory)
        type: Filter by type: 'file', 'dir', or 'all' (default: all)
        limit: Maximum results to return (default: 50)

    Returns:
        List of matching files with path, type, and size.

    Examples:
        find("tool_executor.py")  # Find exact filename anywhere
        find("*_tool.py")         # Find files ending in _tool.py
        find("test*", type="dir") # Find directories starting with 'test'
    """
    import fnmatch

    try:
        base_path = Path(path).expanduser().resolve()

        if not base_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        results = []
        count = 0

        # Walk the directory tree
        for root, dirs, files in base_path.walk():
            # Skip hidden and common excluded directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in {"node_modules", "__pycache__", "venv", ".venv", "build", "dist"}
            ]

            # Check directories if type allows
            if type in ("all", "dir"):
                for d in dirs:
                    if fnmatch.fnmatch(d, name) or fnmatch.fnmatch(d.lower(), name.lower()):
                        dir_path = root / d
                        results.append(
                            {
                                "path": str(dir_path.relative_to(base_path)),
                                "type": "directory",
                                "size": 0,
                            }
                        )
                        count += 1
                        if count >= limit:
                            break

            # Check files if type allows
            if type in ("all", "file") and count < limit:
                for f in files:
                    if fnmatch.fnmatch(f, name) or fnmatch.fnmatch(f.lower(), name.lower()):
                        file_path = root / f
                        try:
                            size = file_path.stat().st_size
                        except OSError:
                            size = 0
                        results.append(
                            {
                                "path": str(file_path.relative_to(base_path)),
                                "type": "file",
                                "size": size,
                            }
                        )
                        count += 1
                        if count >= limit:
                            break

            if count >= limit:
                break

        if not results:
            return [
                {
                    "message": f"No files matching '{name}' found in {path}",
                    "suggestion": "Try a broader pattern like '*{name}*' or search from project root with path='.'",
                }
            ]

        return results

    except Exception as e:
        raise e


# Important documentation file patterns (case-insensitive)
IMPORTANT_DOC_PATTERNS = [
    "readme*",
    "architecture*",
    "index*",
    "claude*",
    "contributing*",
    "changelog*",
    "license*",
    "design*",
    "overview*",
    "getting*started*",
    "setup*",
    "install*",
]


@tool(
    category="filesystem",
    priority=Priority.HIGH,  # Useful for initial exploration
    access_mode=AccessMode.READONLY,  # Only reads directory structure
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["path", "max_depth"],  # Params indicating exploration progress
    stages=["initial", "planning", "reading", "analysis"],  # Best used at start of conversation
    task_types=["analysis", "search"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.READ_ONLY,  # Safe for parallel execution
    keywords=[
        "overview",
        "project",
        "structure",
        "explore",
        "summary",
        "codebase",
        "architecture",
    ],
    use_cases=[
        "getting a project overview",
        "understanding codebase structure",
        "finding important files",
        "initial exploration",
    ],
    examples=[
        "show me the project overview",
        "what's in this codebase",
        "explore the project structure",
    ],
)
async def overview(
    path: str = ".",
    max_depth: int = 2,
    top_files_by_size: int = 100,
    top_doc_files: int = 15,
) -> Dict[str, Any]:
    """Get a curated project overview for initial exploration.

    Provides:
    1. Directory structure at max_depth (default: 2 levels)
    2. Top documentation files (README*, ARCHITECTURE*, INDEX*, etc.)
    3. Largest source files by size (helps identify core modules)

    Args:
        path: Root directory to explore (default: current directory)
        max_depth: Directory exploration depth (default: 2)
        top_files_by_size: Number of largest files to include (default: 100)
        top_doc_files: Number of documentation files to include (default: 15)

    Returns:
        Dict with directories, important_docs, and largest_files sections
    """
    import fnmatch

    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not root.is_dir():
            # GAP-14 FIX: If a file path is given, use its parent directory
            # This is a common model mistake - be helpful and auto-correct
            parent = root.parent
            if parent.is_dir():
                root = parent
            else:
                raise NotADirectoryError(f"Path is not a directory: {path}. Use the parent directory or a directory path.")

        # Excluded directories
        exclude_dirs = {
            ".git",
            "node_modules",
            "venv",
            ".venv",
            "__pycache__",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".eggs",
            "htmlcov",
            ".cache",
            ".ruff_cache",
        }

        # Documentation extensions
        doc_extensions = {".md", ".rst", ".txt", ".adoc"}

        # Collect all files with metadata
        all_files: List[Dict[str, Any]] = []
        directories: List[Dict[str, Any]] = []
        important_docs: List[Dict[str, Any]] = []

        def walk_tree(base: Path, current_depth: int):
            """Walk tree collecting files and directories."""
            if current_depth > max_depth:
                return
            try:
                for p in sorted(base.iterdir()):
                    rel_path = str(p.relative_to(root))

                    # Skip excluded directories
                    if p.is_dir() and p.name in exclude_dirs:
                        continue
                    # Skip hidden files/dirs
                    if p.name.startswith("."):
                        continue

                    if p.is_dir():
                        directories.append(
                            {
                                "path": rel_path,
                                "depth": current_depth,
                            }
                        )
                        walk_tree(p, current_depth + 1)
                    else:
                        try:
                            size = p.stat().st_size
                        except OSError:
                            size = 0

                        file_info = {
                            "path": rel_path,
                            "name": p.name,
                            "size": size,
                            "extension": p.suffix.lower(),
                            "depth": current_depth,
                        }
                        all_files.append(file_info)

                        # Check if it's an important doc file
                        name_lower = p.name.lower()
                        is_doc = p.suffix.lower() in doc_extensions

                        # Check against important doc patterns (case-insensitive)
                        for pattern in IMPORTANT_DOC_PATTERNS:
                            if fnmatch.fnmatch(name_lower, pattern):
                                file_info["importance"] = "high"
                                important_docs.append(file_info)
                                break
                        else:
                            # Also include .md files at root level as potentially important
                            if is_doc and current_depth == 1:
                                file_info["importance"] = "medium"
                                important_docs.append(file_info)

            except PermissionError:
                pass

        walk_tree(root, 1)

        # Sort important docs by importance then size
        importance_order = {"high": 0, "medium": 1}
        important_docs.sort(
            key=lambda x: (
                importance_order.get(x.get("importance", "medium"), 1),
                -x.get("size", 0),
            )
        )
        important_docs = important_docs[:top_doc_files]

        # Get largest files (excluding docs already included)
        doc_paths = {d["path"] for d in important_docs}
        source_files = [f for f in all_files if f["path"] not in doc_paths]
        source_files.sort(key=lambda x: -x.get("size", 0))
        largest_files = source_files[:top_files_by_size]

        # Format sizes for readability
        def format_size(size: int) -> str:
            if size < 1024:
                return f"{size}B"
            elif size < 1024 * 1024:
                return f"{size // 1024}KB"
            else:
                return f"{size // (1024 * 1024)}MB"

        for f in important_docs + largest_files:
            f["size_formatted"] = format_size(f.get("size", 0))

        return {
            "success": True,
            "root": str(root),
            "directories": directories,
            "directory_count": len(directories),
            "important_docs": important_docs,
            "largest_files": largest_files,
            "total_files_scanned": len(all_files),
            "hints": [
                "Use read_file to examine important documentation first",
                "Large files often indicate core modules",
                "Check README.md and CLAUDE.md for project-specific guidance",
            ],
        }

    except Exception as e:
        raise e
