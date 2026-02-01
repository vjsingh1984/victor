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
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, TypedDict
from collections.abc import Callable

import aiofiles

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from typing import TypedDict

# PathResolver for centralized path normalization
from victor.protocols.path_resolver import PathResolver, create_path_resolver

logger = logging.getLogger(__name__)


# Type for truncation info
class TruncationInfo(TypedDict):
    lines_returned: int
    bytes_returned: int
    was_truncated: bool
    truncation_reason: str


# Global PathResolver instance (lazy initialized)
_path_resolver: Optional[PathResolver] = None
_path_resolver_lock = threading.Lock()


def get_path_resolver() -> PathResolver:
    """Get or create the global PathResolver instance.

    Uses double-checked locking pattern for thread-safe lazy initialization.
    Multiple threads can safely call this function concurrently.
    """
    global _path_resolver
    if _path_resolver is None:
        with _path_resolver_lock:
            # Double-check: another thread may have initialized while we waited for lock
            if _path_resolver is None:
                _path_resolver = create_path_resolver()
    return _path_resolver


def reset_path_resolver() -> None:
    """Reset the path resolver (useful when cwd changes).

    Thread-safe: Uses lock to prevent race conditions during reset.
    """
    global _path_resolver
    with _path_resolver_lock:
        _path_resolver = None


# ============================================================================
# SESSION-LEVEL FILE CONTENT CACHE (P3-2)
# ============================================================================
# Prevents redundant file reads within a session. Especially helpful for
# providers like DeepSeek that tend to re-read the same files repeatedly.
# Cache is keyed by normalized absolute path and tracks modification time.


@dataclass
class CachedFileContent:
    """Cached file content with metadata for invalidation."""

    content: str
    mtime: float  # File modification time when cached
    cached_at: float  # When the cache entry was created
    size: int  # File size in bytes
    hits: int = 0  # Number of cache hits


@dataclass
class FileContentCacheStats:
    """Statistics for the file content cache."""

    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    evictions: int = 0
    total_bytes_cached: int = 0
    total_bytes_saved: int = 0  # Bytes not re-read due to cache hits


class FileContentCache:
    """Session-level cache for file contents.

    Features:
    - Caches file contents keyed by normalized absolute path
    - Auto-invalidates when file modification time changes
    - Thread-safe operations
    - Memory-bounded with LRU eviction
    - Tracks hit/miss statistics

    Usage:
        cache = FileContentCache(max_entries=100, max_total_bytes=10_000_000)
        content = cache.get("/path/to/file")  # Returns None on miss
        cache.set("/path/to/file", content, mtime, size)
        cache.invalidate("/path/to/file")  # On write
        cache.clear()  # On session end
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_total_bytes: int = 10_000_000,  # 10MB default
        ttl_seconds: int = 300,  # 5 minute TTL
    ):
        """Initialize file content cache.

        Args:
            max_entries: Maximum number of files to cache
            max_total_bytes: Maximum total bytes to cache (soft limit)
            ttl_seconds: Time-to-live for cache entries
        """
        self._cache: dict[str, CachedFileContent] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._ttl_seconds = ttl_seconds
        self._stats = FileContentCacheStats()
        self._access_order: list[str] = []  # For LRU eviction

    def _normalize_path(self, path: str) -> str:
        """Normalize path to absolute resolved form."""
        return str(Path(path).expanduser().resolve())

    def get(self, path: str) -> Optional[str]:
        """Get cached content if valid.

        Validates:
        - File still exists
        - Modification time hasn't changed
        - TTL hasn't expired

        Args:
            path: File path (will be normalized)

        Returns:
            Cached content or None if miss/invalid
        """
        normalized = self._normalize_path(path)

        with self._lock:
            entry = self._cache.get(normalized)
            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL
            if time.time() - entry.cached_at > self._ttl_seconds:
                self._invalidate_entry(normalized, reason="ttl_expired")
                self._stats.misses += 1
                return None

            # Check if file still exists and mtime matches
            try:
                file_path = Path(normalized)
                if not file_path.exists():
                    self._invalidate_entry(normalized, reason="file_deleted")
                    self._stats.misses += 1
                    return None

                current_mtime = file_path.stat().st_mtime
                if current_mtime != entry.mtime:
                    self._invalidate_entry(normalized, reason="mtime_changed")
                    self._stats.misses += 1
                    return None

            except OSError:
                self._invalidate_entry(normalized, reason="os_error")
                self._stats.misses += 1
                return None

            # Cache hit!
            entry.hits += 1
            self._stats.hits += 1
            self._stats.total_bytes_saved += entry.size

            # Update access order for LRU
            if normalized in self._access_order:
                self._access_order.remove(normalized)
            self._access_order.append(normalized)

            logger.debug(
                "File cache HIT: %s (hits=%d, saved=%d bytes)",
                normalized,
                entry.hits,
                entry.size,
            )
            return entry.content

    def set(self, path: str, content: str, mtime: float, size: int) -> bool:
        """Cache file content.

        May trigger LRU eviction if cache is full.

        Args:
            path: File path (will be normalized)
            content: File content to cache
            mtime: File modification time
            size: File size in bytes

        Returns:
            True if cached successfully
        """
        normalized = self._normalize_path(path)

        with self._lock:
            # Check if we need to evict entries
            self._maybe_evict(size)

            # Store the entry
            self._cache[normalized] = CachedFileContent(
                content=content,
                mtime=mtime,
                cached_at=time.time(),
                size=size,
                hits=0,
            )
            self._stats.total_bytes_cached += size

            # Update access order
            if normalized in self._access_order:
                self._access_order.remove(normalized)
            self._access_order.append(normalized)

            logger.debug("File cache SET: %s (%d bytes)", normalized, size)
            return True

    def invalidate(self, path: str) -> bool:
        """Invalidate a cache entry.

        Call this when a file is written to ensure fresh reads.

        Args:
            path: File path to invalidate

        Returns:
            True if entry was found and invalidated
        """
        normalized = self._normalize_path(path)
        with self._lock:
            return self._invalidate_entry(normalized, reason="explicit_invalidate")

    def _invalidate_entry(self, normalized: str, reason: str = "") -> bool:
        """Internal invalidation (already holds lock)."""
        entry = self._cache.pop(normalized, None)
        if entry:
            self._stats.invalidations += 1
            self._stats.total_bytes_cached -= entry.size
            if normalized in self._access_order:
                self._access_order.remove(normalized)
            logger.debug("File cache INVALIDATE: %s (reason=%s)", normalized, reason)
            return True
        return False

    def _maybe_evict(self, incoming_bytes: int) -> None:
        """Evict entries if necessary (already holds lock)."""
        # Check entry count
        while len(self._cache) >= self._max_entries and self._access_order:
            oldest = self._access_order.pop(0)
            entry = self._cache.pop(oldest, None)
            if entry:
                self._stats.evictions += 1
                self._stats.total_bytes_cached -= entry.size
                logger.debug("File cache EVICT (count): %s", oldest)

        # Check total bytes (soft limit - only evict if significantly over)
        while (
            self._stats.total_bytes_cached + incoming_bytes > self._max_total_bytes * 1.2
            and self._access_order
        ):
            oldest = self._access_order.pop(0)
            entry = self._cache.pop(oldest, None)
            if entry:
                self._stats.evictions += 1
                self._stats.total_bytes_cached -= entry.size
                logger.debug("File cache EVICT (bytes): %s", oldest)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            self._stats.total_bytes_cached = 0
            logger.info("File cache CLEARED: %d entries removed", count)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                self._stats.hits / (self._stats.hits + self._stats.misses) * 100
                if (self._stats.hits + self._stats.misses) > 0
                else 0.0
            )
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate_percent": round(hit_rate, 1),
                "invalidations": self._stats.invalidations,
                "evictions": self._stats.evictions,
                "entries": len(self._cache),
                "total_bytes_cached": self._stats.total_bytes_cached,
                "total_bytes_saved": self._stats.total_bytes_saved,
                "max_entries": self._max_entries,
                "max_total_bytes": self._max_total_bytes,
            }

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


# Global file content cache instance (session-level)
# This is shared across all filesystem tool invocations within a session
_file_content_cache: Optional[FileContentCache] = None
_cache_enabled: bool = True  # Can be disabled via settings


# ============================================================================
# SANDBOX PATH ENFORCEMENT (P4 - Explore/Plan Mode Restrictions)
# ============================================================================
# In EXPLORE and PLAN modes, file writes are restricted to the sandbox directory
# (.victor/sandbox/) to prevent accidental modifications to the main codebase.


def get_sandbox_path() -> Optional[Path]:
    """Get the sandbox directory path for the current mode.

    Returns:
        Path to sandbox directory if in restricted mode, None if unrestricted.
    """
    try:
        from victor.agent.mode_controller import get_mode_controller

        controller = get_mode_controller()
        config = controller.config

        if config.sandbox_dir and config.allow_sandbox_edits:
            # Sandbox is enabled for this mode
            return Path.cwd() / config.sandbox_dir
        elif not config.allow_all_tools and (
            "write_file" in config.disallowed_tools or "edit_files" in config.disallowed_tools
        ):
            # Writing is disallowed in this mode (no sandbox exception)
            return None

        # Unrestricted mode (BUILD) - no sandbox enforcement
        return None
    except Exception:
        # If mode controller not available, default to unrestricted
        return None


def enforce_sandbox_path(file_path: Path) -> None:
    """Enforce sandbox restrictions for file writes in restricted modes.

    Args:
        file_path: The resolved file path being written to

    Raises:
        PermissionError: If path is outside sandbox in restricted mode
    """
    sandbox = get_sandbox_path()

    if sandbox is None:
        # No sandbox restriction in effect
        return

    # Ensure sandbox exists
    sandbox.mkdir(parents=True, exist_ok=True)

    # Check if the path is within the sandbox
    try:
        file_path.resolve().relative_to(sandbox.resolve())
    except ValueError:
        # Path is not within sandbox
        from victor.agent.mode_controller import get_mode_controller

        mode = get_mode_controller().current_mode.value.upper()
        raise PermissionError(
            f"[{mode} MODE] Cannot write to '{file_path}'.\n"
            f"In {mode} mode, edits are restricted to the sandbox directory: {sandbox}\n"
            f"Use /mode build to switch to build mode for unrestricted file access.\n"
            f"Or prefix your path with .victor/sandbox/ to write within the sandbox."
        )


def get_file_content_cache() -> FileContentCache:
    """Get or create the global file content cache."""
    global _file_content_cache
    if _file_content_cache is None:
        _file_content_cache = FileContentCache()
    return _file_content_cache


def clear_file_content_cache(reset_stats: bool = True) -> None:
    """Clear the global file content cache (call on session end).

    Args:
        reset_stats: If True (default), also reset statistics.
                     Set to False to preserve stats across clear.
    """
    global _file_content_cache
    if _file_content_cache is not None:
        _file_content_cache.clear()
        if reset_stats:
            # Reset the stats for a fresh session
            _file_content_cache._stats = FileContentCacheStats()


def is_file_cache_enabled() -> bool:
    """Check if file content cache is enabled."""
    return _cache_enabled


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
    extensions: tuple[str, ...]  # Expected extensions for this type
    magic_bytes: Optional[bytes] = None  # Magic signature if detected
    magic_offset: int = 0  # Offset where magic bytes are found
    suggestion: str = ""  # Help message for unsupported types


# Magic bytes signatures for common file types
# Format: (magic_bytes, offset, FileTypeInfo)
# Order matters: more specific signatures should come first
MAGIC_SIGNATURES: list[tuple[bytes, int, FileTypeInfo]] = [
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
_BINARY_HANDLERS: dict[FileCategory, BinaryFileHandler] = {}


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
    execution_category="read_only",  # Safe for parallel execution
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
        "explain",
        "describe",
        "what does",
    ],  # Force inclusion
    priority_hints=[
        "TRUNCATION: 1500 lines/64KB. Always ends on complete lines.",
        "PAGINATION: When truncated, output includes 'Use offset=N to continue' - use that exact offset value.",
        "Use for TEXT and CODE files only (.py, .js, .json, .yaml, .md, etc.)",
        "NOT for binary files (.pdf, .docx, .db, .pyc, images, archives)",
        "Use search parameter for efficient grep-like targeted lookups",
        "Use ls first to check file sizes before reading",
    ],
)
async def read(
    path: str = "",
    offset: int = 0,
    limit: int = 0,
    search: str = "",
    ctx: int = 2,
    regex: bool = False,
    # Parameter aliases for models that use different names (e.g., gpt-oss)
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    # Capture malformed calls with incorrect parameter names
    **kwargs,
) -> str:
    """Read text/code file. Binary files rejected.

    TRUNCATION LIMITS:
    - Maximum 1500 lines OR 64KB (whichever is reached first)
    - Always truncates at complete line boundaries (never mid-line)
    - When truncated, includes: "[... N more lines. Use offset=X to continue ...]"

    PAGINATION (for large files):
    - Use offset/limit: read(path, offset=0, limit=200), then offset=200, etc.
    - Or let auto-truncation guide you with the offset value in output
    - Use search param to find specific content without reading entire file

    Args:
        path: File path
        offset: Start line (0=beginning). Use for pagination of large files.
        limit: Max lines to read (0=auto, applies configured limits).
               Set explicit limit to override auto-truncation.
        search: Grep pattern - efficient for finding specific content
        ctx: Context lines around matches
        regex: Pattern is regex
        line_start: Alias for offset (some models use this name)
        line_end: Alias for limit (some models use this name)
        **kwargs: Captures malformed calls with incorrect parameter names
                  (e.g., pattern instead of search, file instead of path)

    Returns:
        File content with line numbers. If truncated, includes continuation hint
        with exact offset to use for next read.
    """
    # ==========================================================================
    # MALFORMED CALL RECOVERY
    # ==========================================================================
    # Some models (especially Ollama/local models) use incorrect parameter names.
    # We detect and recover from common mistakes:
    #   - pattern -> search (grep pattern)
    #   - file/file_path/filepath -> path
    #   - query/grep -> search
    #   - start/begin -> offset
    #   - end/max/count -> limit
    # ==========================================================================
    if kwargs:
        # Path aliases
        for key in ["file", "file_path", "filepath", "filename"]:
            if key in kwargs and not path:
                path = kwargs.pop(key)
                break

        # Search/pattern aliases
        for key in ["pattern", "query", "grep", "find", "match"]:
            if key in kwargs and not search:
                search = kwargs.pop(key)
                break

        # Offset aliases
        for key in ["start", "begin", "from_line", "start_line"]:
            if key in kwargs and offset == 0:
                offset = kwargs.pop(key)
                break

        # Limit aliases
        for key in ["end", "max", "count", "max_lines", "num_lines"]:
            if key in kwargs and limit == 0:
                limit = kwargs.pop(key)
                break

        # Context aliases
        for key in ["context", "context_lines"]:
            if key in kwargs:
                ctx = kwargs.pop(key)
                break

        # Log remaining unexpected kwargs for debugging
        if kwargs:
            logger.debug(f"read() received unexpected kwargs (ignored): {list(kwargs.keys())}")

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
        # Try PathResolver for intelligent path normalization and suggestions
        resolver = get_path_resolver()
        try:
            result = resolver.resolve_file(path)
            # PathResolver found a valid path after normalization
            file_path = result.resolved_path
            if result.was_normalized:
                logger.debug(
                    f"Path normalized by PathResolver: '{path}' -> '{result.resolved_path}' "
                    f"(applied: {result.normalization_applied})"
                )
                try:
                    path = str(result.resolved_path.relative_to(Path.cwd()))
                except ValueError:
                    path = str(result.resolved_path)
        except FileNotFoundError:
            # PathResolver couldn't find the file - provide helpful suggestions
            suggestions = resolver.suggest_similar(path, limit=5)
            if suggestions:
                suggestion_list = "\n  - ".join(suggestions[:5])
                raise FileNotFoundError(
                    f"File not found: {path}\n" f"Did you mean one of these?\n  - {suggestion_list}"
                )
            else:
                raise FileNotFoundError(f"File not found: {path}")
        except IsADirectoryError:
            raise IsADirectoryError(
                f"Cannot read directory as file: {path}\n"
                f"Suggestion: Use list_directory(path='{path}') to explore its contents, "
                f"or specify a file path within this directory."
            )
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
    ALL_BINARY_EXTENSIONS: set[str] = set()
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
    # TEXT FILE READING (with session-level caching)
    # =========================================================================
    # Check session cache first (prevents redundant reads for providers like DeepSeek)
    content: Optional[str] = None
    cache_hit = False

    if is_file_cache_enabled():
        cache = get_file_content_cache()
        cached_content = cache.get(str(file_path))
        if cached_content is not None:
            content = cached_content
            cache_hit = True
            logger.debug("File read from cache: %s", path)

    # Try to read the file if not in cache
    if content is None:
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

        # Cache the content for future reads
        if is_file_cache_enabled() and not cache_hit:
            try:
                file_stat = file_path.stat()
                cache = get_file_content_cache()
                cache.set(
                    str(file_path),
                    content,
                    mtime=file_stat.st_mtime,
                    size=file_stat.st_size,
                )
            except OSError:
                pass  # Don't fail if we can't cache

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

        grep_result = grep_lines(
            content=content,
            pattern=search,
            context_before=ctx,
            context_after=ctx,
            case_sensitive=True,
            is_regex=regex,
            max_matches=50,
            file_path=path,
        )
        return grep_result.to_string(show_line_numbers=True, max_matches=50)

    # Import truncation utilities
    from victor.tools.output_utils import (
        truncate_by_lines,
        format_with_line_numbers,
    )

    # Determine truncation limits based on model context
    # Cloud models (Anthropic, OpenAI, etc.): ~2500 lines / 100KB (~25K tokens)
    # Local models (Ollama, LMStudio, vLLM): ~2000 lines / 32KB (~8K tokens)
    def _get_truncation_limits() -> tuple:
        """Get appropriate truncation limits based on current provider/model."""
        try:
            from victor.config.settings import get_settings
            from victor.config.config_loaders import get_provider_limits

            settings = get_settings()
            provider = getattr(settings, "provider", "")
            model = getattr(settings, "model", "")

            # Check for airgapped mode
            if settings.airgapped_mode:
                return 1500, 65536  # ~2000 lines, 64KB for airgapped mode

            # Get context window from provider_context_limits.yaml
            # This supports provider defaults + model-specific overrides
            limits = get_provider_limits(provider, model)
            context_window = limits.context_window

            if context_window > 0:
                # Use ~25% of context window for file reads
                # This leaves room for the rest of the conversation
                max_tokens = context_window // 4
                # Estimate ~3 bytes per token (average), ~40 chars per line
                max_lines = min(1500, max(100, max_tokens // 3))  # Ensure at least 100 lines
                max_bytes = min(65536, max_tokens * 3)  # ~3 bytes per token average
                return max_lines, max_bytes

            # Fallback to standard limits
            return 1500, 65536
        except Exception:
            # Default limits if config loading fails
            return 1500, 65536  # ~1500 lines, 64KB

    MAX_LINES, MAX_BYTES = _get_truncation_limits()

    lines = content.split("\n")
    total_lines = len(lines)

    # Clamp offset to valid range
    offset = max(0, min(offset, total_lines))

    # Apply offset first
    if offset > 0:
        remaining_content = "\n".join(lines[offset:])
    else:
        remaining_content = content

    # Determine effective limit
    if limit > 0:
        # User specified explicit limit - honor it (but still apply byte limit for safety)
        effective_max_lines = limit
    else:
        # Auto-truncation mode - apply default limits
        effective_max_lines = MAX_LINES

    # Apply truncation (always ends on complete lines)
    truncated_content, trunc_info = truncate_by_lines(
        remaining_content,
        max_lines=effective_max_lines,
        max_bytes=MAX_BYTES,
        start_line=0,  # Already applied offset above
    )
    # trunc_info is already a TruncationInfo from the function return

    # Format with line numbers (1-indexed, adjusted for offset)
    numbered_content = format_with_line_numbers(truncated_content, start_line=offset + 1)

    # Get total file size for complete metadata
    file_size_bytes = file_path.stat().st_size
    file_size_kb = file_size_bytes / 1024

    # Build informative header with explicit units to prevent LLM misinterpretation
    actual_end_line = offset + trunc_info.lines_returned
    header_parts = [
        f"File: {path}",
        f"Showing lines {offset + 1}-{actual_end_line} of {total_lines} total lines",
        f"This file is {file_size_bytes:,} bytes ({file_size_kb:.1f} KB)",
        f"Showing {trunc_info.bytes_returned:,} bytes of content",
    ]

    # Add truncation warning with specific details
    if trunc_info.was_truncated:
        remaining = total_lines - actual_end_line
        if trunc_info.truncation_reason == "line_limit":
            header_parts.append(
                f"[TRUNCATED: Hit {effective_max_lines} line limit. "
                f"{remaining} lines remaining. Use offset={actual_end_line} to continue]"
            )
        elif trunc_info.truncation_reason == "byte_limit":
            header_parts.append(
                f"[TRUNCATED: Hit {MAX_BYTES // 1024}KB byte limit at line {actual_end_line}. "
                f"{remaining} lines remaining. Use offset={actual_end_line} to continue]"
            )

    header = "\n".join(header_parts) + "\n\n"

    # Add summary footer with explicit metrics to reinforce correct interpretation
    # This defense-in-depth approach prevents LLM from confusing bytes/lines
    summary_footer = (
        f"\n[End of excerpt from {path}]\n"
        f"File has {total_lines} lines total, {file_size_bytes:,} bytes ({file_size_kb:.1f} KB)\n"
    )

    return header + numbered_content + summary_footer


@tool(
    category="filesystem",
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.WRITE,  # Creates/overwrites files
    danger_level=DangerLevel.LOW,  # Minor risk, easily undoable
    # Registry-driven metadata for tool selection and cache invalidation
    progress_params=["path"],  # Same file = loop, regardless of content
    stages=["execution"],  # Conversation stages where relevant
    task_types=["edit", "generation", "action"],  # Task types for classification-aware selection
    execution_category="write",  # Cannot run in parallel with conflicting ops
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

    Note:
        In EXPLORE/PLAN modes, writes are restricted to .victor/sandbox/.
        Use /mode build to enable unrestricted file writes.
    """
    from victor.agent.change_tracker import ChangeType, get_change_tracker

    file_path = Path(path).expanduser().resolve()

    # Enforce sandbox restrictions in EXPLORE/PLAN modes
    enforce_sandbox_path(file_path)

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

    # Invalidate file content cache (ensures fresh reads after write)
    if is_file_cache_enabled():
        cache = get_file_content_cache()
        cache.invalidate(str(file_path))

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
    execution_category="read_only",  # Safe for parallel execution
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
) -> list[dict[str, Any]]:
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

    # Normalize limit (handle non-int input from model)
    if not isinstance(limit, int):
        limit = int(limit) if isinstance(limit, str) and limit.isdigit() else 1000  # type: ignore[unreachable]

    items = []
    count = 0

    # Normalize depth (handle non-int input from model)
    if not isinstance(depth, int):
        depth = int(depth) if isinstance(depth, str) and depth.isdigit() else 1  # type: ignore[unreachable]

    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            # Try PathResolver for intelligent path normalization and suggestions
            resolver = get_path_resolver()
            try:
                result = resolver.resolve_directory(path)
                # PathResolver found a valid directory after normalization
                dir_path = result.resolved_path
                if result.was_normalized:
                    logger.debug(
                        f"Directory path normalized by PathResolver: '{path}' -> '{result.resolved_path}' "
                        f"(applied: {result.normalization_applied})"
                    )
                    try:
                        path = str(result.resolved_path.relative_to(Path.cwd()))
                    except ValueError:
                        path = str(result.resolved_path)
            except FileNotFoundError:
                # PathResolver couldn't find the directory - provide helpful suggestions
                suggestions = resolver.suggest_similar(path, limit=5)
                if suggestions:
                    suggestion_list = "\n  - ".join(suggestions[:5])
                    raise FileNotFoundError(
                        f"Directory not found: {path}\n"
                        f"Did you mean one of these?\n  - {suggestion_list}"
                    )
                else:
                    raise FileNotFoundError(f"Directory not found: {path}")
            except NotADirectoryError:
                raise NotADirectoryError(
                    f"Path is not a directory: {path}\n"
                    f"Suggestion: Use read_file(path='{path}') to read this file instead."
                )

        # Explicit check: path must be a directory
        if dir_path.exists() and not dir_path.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {path}\n"
                f"Suggestion: Use read_file(path='{path}') to read this file instead."
            )

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

        # Always include cwd context and relative target path in response
        for item in items:
            if pattern:
                item["filter"] = pattern
            if count >= limit:
                item["truncated"] = True

        # Always wrap in metadata dict within a list for consistency
        result_dict = {
            "items": items,
            "count": len(items),
            "truncated": count >= limit,
            "cwd": str(dir_path),  # Current working directory for context
            "target": str(dir_path),  # Target path that was listed
        }

        # Add filter key to result dict if pattern was used
        if pattern:
            result_dict["filter"] = pattern

        # Return as list for consistency with tests
        return [result_dict]

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
) -> list[dict[str, Any]]:
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
    import os

    try:
        base_path = Path(path).expanduser().resolve()

        if not base_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        results = []
        count = 0

        # Walk the directory tree
        for root, dirs, files in os.walk(base_path):
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
                        dir_path = Path(root) / d
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
                        file_path = Path(root) / f
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
    execution_category="read_only",  # Safe for parallel execution
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
) -> dict[str, Any]:
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
                raise NotADirectoryError(
                    f"Path is not a directory: {path}. Use the parent directory or a directory path."
                )

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
        all_files: list[dict[str, Any]] = []
        directories: list[dict[str, Any]] = []
        important_docs: list[dict[str, Any]] = []

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
