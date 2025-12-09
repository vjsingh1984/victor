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

"""Project context loader for init.md files.

This module provides functionality similar to Claude Code's CLAUDE.md,
allowing projects to define context, instructions, and configuration
that Victor uses when working in that codebase.

Configuration is driven by settings.py:
- VICTOR_DIR_NAME: Directory name (default: .victor)
- VICTOR_CONTEXT_FILE: Context file name (default: init.md)

Primary location: {project_root}/.victor/init.md
Legacy locations: .victor.md, VICTOR.md (for backwards compatibility)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from victor.config.settings import (
    VICTOR_DIR_NAME,
    VICTOR_CONTEXT_FILE,
    get_project_paths,
)

# Cache for ProjectContext instances and their content
# Key: (root_path, mtime) -> (content, parsed_sections)
_context_cache: Dict[Tuple[str, float], Tuple[str, Dict[str, str]]] = {}
_cache_lock = threading.Lock()
_cache_ttl = 60.0  # Seconds to cache before checking mtime again
_last_cache_check: Dict[str, float] = {}

logger = logging.getLogger(__name__)

# Context file location: .victor/init.md (configurable via settings.py)
# No legacy locations - clean integration for future code
CONTEXT_FILE_PATH = f"{VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE}"


class ProjectContext:
    """Loads and manages project-specific context from init.md files.

    Location: .victor/init.md (configurable via settings.py)
    """

    def __init__(self, root_path: Optional[str] = None):
        """Initialize project context loader.

        Args:
            root_path: Root directory to search for context files.
                      Defaults to current working directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._context_file: Optional[Path] = None
        self._content: Optional[str] = None
        self._parsed_sections: Dict[str, str] = {}

    def find_context_file(self) -> Optional[Path]:
        """Find the project context file.

        Looks for .victor/init.md (configurable via settings.py).
        Searches current directory and parent directories up to git root.

        Returns:
            Path to context file if found, None otherwise.
        """
        search_path = self.root_path

        # Search up the directory tree
        while search_path != search_path.parent:
            # Use settings-driven path
            context_file = search_path / CONTEXT_FILE_PATH
            if context_file.exists() and context_file.is_file():
                logger.info(f"Found project context file: {context_file}")
                return context_file

            # Stop at git root if found
            if (search_path / ".git").exists():
                break

            search_path = search_path.parent

        return None

    def load(self, force_reload: bool = False) -> bool:
        """Load project context from file with caching.

        Uses mtime-based caching to avoid re-reading unchanged files.
        Cache is automatically invalidated when the file is modified.

        Args:
            force_reload: Force re-reading the file even if cached.

        Returns:
            True if context was loaded successfully, False otherwise.
        """
        self._context_file = self.find_context_file()

        if not self._context_file:
            logger.debug("No project context file found")
            return False

        try:
            root_key = str(self.root_path)
            now = time.time()

            # Check if we should skip cache check (within TTL)
            if not force_reload and root_key in _last_cache_check:
                if now - _last_cache_check[root_key] < _cache_ttl:
                    # Use cached content if available
                    for (cached_root, _), (content, sections) in _context_cache.items():
                        if cached_root == root_key:
                            self._content = content
                            self._parsed_sections = sections
                            logger.debug(f"Using cached project context for {root_key}")
                            return True

            # Get file mtime for cache key
            mtime = self._context_file.stat().st_mtime
            cache_key = (root_key, mtime)

            with _cache_lock:
                # Check cache with mtime
                if not force_reload and cache_key in _context_cache:
                    self._content, self._parsed_sections = _context_cache[cache_key]
                    _last_cache_check[root_key] = now
                    logger.debug(f"Using cached project context for {self._context_file}")
                    return True

                # Load from file
                self._content = self._context_file.read_text(encoding="utf-8")
                self._parse_sections()

                # Update cache (clear old entries for same root)
                keys_to_remove = [k for k in _context_cache if k[0] == root_key]
                for k in keys_to_remove:
                    del _context_cache[k]
                _context_cache[cache_key] = (self._content, self._parsed_sections)
                _last_cache_check[root_key] = now

            logger.info(f"Loaded project context from {self._context_file}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load project context: {e}")
            return False

    @staticmethod
    def clear_cache() -> None:
        """Clear the project context cache (for testing or hot-reload)."""
        with _cache_lock:
            _context_cache.clear()
            _last_cache_check.clear()
        logger.debug("Cleared project context cache")

    def _parse_sections(self) -> None:
        """Parse markdown content into sections by headers."""
        if not self._content:
            return

        self._parsed_sections = {}
        current_section = "overview"
        current_content: List[str] = []

        for line in self._content.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_content:
                    self._parsed_sections[current_section] = "\n".join(current_content).strip()
                # Start new section
                current_section = line[3:].strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            self._parsed_sections[current_section] = "\n".join(current_content).strip()

    @property
    def content(self) -> str:
        """Get the full context content."""
        return self._content or ""

    @property
    def context_file(self) -> Optional[Path]:
        """Get the path to the loaded context file."""
        return self._context_file

    def get_section(self, section_name: str) -> str:
        """Get a specific section from the context.

        Args:
            section_name: Name of the section (case-insensitive, spaces to underscores)

        Returns:
            Section content, or empty string if not found.
        """
        key = section_name.lower().replace(" ", "_")
        return self._parsed_sections.get(key, "")

    def get_system_prompt_addition(self) -> str:
        """Get context formatted for inclusion in system prompt.

        Returns:
            Formatted context string for system prompt injection.
        """
        if not self._content:
            return ""

        # Use actual file name or default to configured name
        file_name = self._context_file.name if self._context_file else VICTOR_CONTEXT_FILE

        return f"""
<project-context>
The following is project-specific context from {file_name}:

{self._content}
</project-context>
"""

    def get_package_layout_hint(self) -> str:
        """Extract package layout hints from context.

        Returns:
            Package layout guidance if specified in context.
        """
        # Look for package layout or directory structure sections
        for section_name in [
            "package_layout",
            "directory_structure",
            "project_structure",
            "architecture",
        ]:
            content = self.get_section(section_name)
            if content:
                return content
        return ""


def generate_victor_md(root_path: Optional[str] = None) -> str:
    """Generate project context file by analyzing the codebase.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.

    Returns:
        Generated markdown content for init.md
    """
    root = Path(root_path) if root_path else Path.cwd()

    sections = []

    # Header - use configurable file name
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    project_name = root.name
    sections.append("## Project Overview\n")

    # Try to detect project type and description
    readme_files = ["README.md", "README.rst", "README.txt", "readme.md"]
    readme_content = ""
    for readme in readme_files:
        readme_path = root / readme
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8")
                # Extract first meaningful paragraph (skip headers, images, HTML, badges)
                lines = content.split("\n\n")
                for para in lines:
                    stripped = para.strip()
                    # Skip empty, headers, images, HTML tags, badges, links-only lines
                    if not stripped:
                        continue
                    if stripped.startswith(("#", "![", "<", "[!", "---", "```")):
                        continue
                    if stripped.startswith("[") and stripped.endswith(")"):
                        continue  # Skip badge/link lines like [License](url)
                    # Found a text paragraph
                    readme_content = stripped[:500]
                    break
            except Exception:
                pass
            break

    if readme_content:
        sections.append(f"**{project_name}**: {readme_content}\n")
    else:
        sections.append(f"**{project_name}**: [Add project description here]\n")

    # Detect package layout
    sections.append("## Package Layout\n")
    layout_hints = []

    # Find Python package directories at root level (directories with __init__.py)
    def is_python_package(path: Path) -> bool:
        return path.is_dir() and (path / "__init__.py").exists()

    root_packages = [
        d.name
        for d in root.iterdir()
        if is_python_package(d) and not d.name.startswith((".", "_", "venv", "test"))
    ]

    # Check for common Python project structures
    has_src = (root / "src").is_dir()
    expected_pkg = project_name.replace("-", "_")

    if root_packages:
        # Found Python packages at root level - these are likely the active code
        main_pkg = root_packages[0]  # Take first package found
        layout_hints.append(f"- **Active code**: `{main_pkg}/` (main package)")
        if has_src:
            layout_hints.append("- **Legacy/Deprecated**: `src/` (DO NOT USE)")
    elif has_src:
        # Only src/ exists, check if it has nested packages
        src_packages = [d.name for d in (root / "src").iterdir() if is_python_package(d)]
        if src_packages:
            layout_hints.append(f"- **Source code**: `src/{src_packages[0]}/` (src layout)")
        else:
            layout_hints.append("- **Source code**: `src/` (src layout)")
    elif (root / expected_pkg).is_dir():
        layout_hints.append(f"- **Main package**: `{expected_pkg}/`")

    # Check for tests
    if (root / "tests").is_dir():
        layout_hints.append("- **Tests**: `tests/`")
    elif (root / "test").is_dir():
        layout_hints.append("- **Tests**: `test/`")

    # Check for docs
    if (root / "docs").is_dir():
        layout_hints.append("- **Documentation**: `docs/`")

    if layout_hints:
        sections.append("\n".join(layout_hints) + "\n")
    else:
        sections.append("[Add package layout description here]\n")

    # Common Commands
    sections.append("## Common Commands\n")
    commands = []

    # Detect build system
    if (root / "pyproject.toml").exists():
        commands.append("```bash")
        commands.append("# Install with dev dependencies")
        commands.append('pip install -e ".[dev]"')
        commands.append("")
        commands.append("# Run tests")
        commands.append("pytest")
        commands.append("")
        commands.append("# Format code")
        commands.append("black .")
        commands.append("```")
    elif (root / "setup.py").exists():
        commands.append("```bash")
        commands.append("pip install -e .")
        commands.append("pytest")
        commands.append("```")
    elif (root / "package.json").exists():
        commands.append("```bash")
        commands.append("npm install")
        commands.append("npm test")
        commands.append("npm run build")
        commands.append("```")
    elif (root / "Cargo.toml").exists():
        commands.append("```bash")
        commands.append("cargo build")
        commands.append("cargo test")
        commands.append("```")
    elif (root / "go.mod").exists():
        commands.append("```bash")
        commands.append("go build ./...")
        commands.append("go test ./...")
        commands.append("```")
    else:
        commands.append("[Add common commands here]")

    sections.append("\n".join(commands) + "\n")

    # Architecture (placeholder)
    sections.append("## Architecture\n")
    sections.append("[Add architecture overview here]\n")

    # Important Notes
    sections.append("## Important Notes\n")
    sections.append("- [Add project-specific notes and conventions here]\n")

    return "\n".join(sections)


def init_victor_md(root_path: Optional[str] = None, force: bool = False) -> Optional[Path]:
    """Initialize project context file in .victor/init.md.

    Creates the file at the configured location (default: .victor/init.md).
    Location is configurable via settings.py (VICTOR_DIR_NAME, VICTOR_CONTEXT_FILE).

    Args:
        root_path: Root directory to create file in. Defaults to current directory.
        force: If True, overwrite existing file.

    Returns:
        Path to created file, or None if file exists and force=False.
    """
    root = Path(root_path) if root_path else Path.cwd()

    # Use settings-driven path
    paths = get_project_paths(root)
    target_file = paths.project_context_file

    # Ensure .victor directory exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if target_file.exists() and not force:
        logger.warning(f"{VICTOR_CONTEXT_FILE} already exists at {target_file}")
        return None

    content = generate_victor_md(root_path)

    try:
        target_file.write_text(content, encoding="utf-8")
        logger.info(f"Created {VICTOR_CONTEXT_FILE} at {target_file}")
        return target_file
    except Exception as e:
        logger.error(f"Failed to create {VICTOR_CONTEXT_FILE}: {e}")
        return None
