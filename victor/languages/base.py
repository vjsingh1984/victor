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

"""Base types for language plugins.

Defines the interfaces and data structures used by language plugins
to provide language-specific functionality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable


class CommentStyle(Enum):
    """Comment styles for different languages."""

    C_STYLE = "c_style"  # /* */ and //
    HASH = "hash"  # #
    DOUBLE_DASH = "double_dash"  # --
    SEMICOLON = "semicolon"  # ;
    HTML = "html"  # <!-- -->
    NONE = "none"


@dataclass
class LanguageConfig:
    """Configuration for a programming language.

    Contains all the metadata needed to work with a language,
    from syntax details to tooling commands.
    """

    # Identity
    name: str  # Canonical name (e.g., "python")
    display_name: str  # Human-readable name (e.g., "Python")
    aliases: List[str] = field(default_factory=list)  # Alternative names

    # File identification
    extensions: List[str] = field(default_factory=list)  # .py, .pyw
    filenames: List[str] = field(default_factory=list)  # Makefile, Dockerfile
    shebangs: List[str] = field(default_factory=list)  # #!/usr/bin/python

    # Syntax
    comment_style: CommentStyle = CommentStyle.HASH
    line_comment: Optional[str] = "#"
    block_comment_start: Optional[str] = None
    block_comment_end: Optional[str] = None
    string_delimiters: List[str] = field(default_factory=lambda: ['"', "'"])

    # Indentation
    indent_size: int = 4
    use_tabs: bool = False

    # Tooling
    package_managers: List[str] = field(default_factory=list)  # pip, npm
    build_systems: List[str] = field(default_factory=list)  # make, cargo
    test_frameworks: List[str] = field(default_factory=list)  # pytest, jest

    # LSP
    language_server: Optional[str] = None  # Command to start LSP
    language_server_name: Optional[str] = None  # Name for identification

    # Tree-sitter
    tree_sitter_language: Optional[str] = None  # tree-sitter grammar name


@dataclass
class TestRunner:
    """Configuration for a test runner."""

    name: str
    command: List[str]  # Base command (e.g., ["pytest"])
    file_pattern: str = "test_*.py"  # Pattern for test files
    discover_args: List[str] = field(default_factory=list)  # Args for discovery
    run_args: List[str] = field(default_factory=list)  # Args for running
    coverage_args: List[str] = field(default_factory=list)  # Args for coverage
    parallel_args: List[str] = field(default_factory=list)  # Args for parallel run

    # Output parsing
    output_format: str = "text"  # text, json, junit
    json_args: List[str] = field(default_factory=list)  # Args for JSON output


@dataclass
class BuildSystem:
    """Configuration for a build system."""

    name: str
    build_command: List[str]  # e.g., ["cargo", "build"]
    run_command: List[str]  # e.g., ["cargo", "run"]
    clean_command: List[str] = field(default_factory=list)
    install_command: List[str] = field(default_factory=list)

    # Build modes
    debug_args: List[str] = field(default_factory=list)
    release_args: List[str] = field(default_factory=list)

    # Manifest file
    manifest_file: Optional[str] = None  # Cargo.toml, package.json


@dataclass
class Formatter:
    """Configuration for a code formatter."""

    name: str
    command: List[str]  # e.g., ["black", "."]
    check_args: List[str] = field(default_factory=list)  # Args for check mode
    config_file: Optional[str] = None  # pyproject.toml, .prettierrc


@dataclass
class Linter:
    """Configuration for a linter."""

    name: str
    command: List[str]  # e.g., ["ruff", "check"]
    fix_args: List[str] = field(default_factory=list)  # Args for auto-fix
    config_file: Optional[str] = None
    output_format: str = "text"  # text, json, sarif


@dataclass
class LanguageCapabilities:
    """Capabilities of a language plugin."""

    # Analysis
    supports_syntax_analysis: bool = False
    supports_semantic_analysis: bool = False
    supports_type_checking: bool = False

    # Refactoring
    supports_rename: bool = False
    supports_extract_function: bool = False
    supports_inline: bool = False
    supports_organize_imports: bool = False

    # Testing
    supports_test_discovery: bool = False
    supports_test_execution: bool = False
    supports_coverage: bool = False

    # Debugging
    supports_debugging: bool = False
    supports_breakpoints: bool = False
    supports_step_debugging: bool = False

    # Other
    supports_formatting: bool = False
    supports_linting: bool = False
    supports_completion: bool = False


@runtime_checkable
class LanguagePlugin(Protocol):
    """Protocol for language plugins.

    Each language implements this protocol to provide
    language-specific functionality to Victor.
    """

    @property
    def config(self) -> LanguageConfig:
        """Get language configuration."""
        ...

    @property
    def capabilities(self) -> LanguageCapabilities:
        """Get language capabilities."""
        ...

    def detect_from_file(self, path: Path) -> bool:
        """Check if this language handles the given file.

        Args:
            path: File path to check

        Returns:
            True if this language handles the file
        """
        ...

    def detect_from_content(self, content: str, filename: Optional[str] = None) -> float:
        """Estimate confidence that content is in this language.

        Args:
            content: File content
            filename: Optional filename hint

        Returns:
            Confidence score 0.0 to 1.0
        """
        ...

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get appropriate test runner for a project.

        Args:
            project_root: Project root directory

        Returns:
            TestRunner config or None if not available
        """
        ...

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get appropriate build system for a project.

        Args:
            project_root: Project root directory

        Returns:
            BuildSystem config or None if not available
        """
        ...

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get appropriate formatter for a project.

        Args:
            project_root: Project root directory

        Returns:
            Formatter config or None if not available
        """
        ...

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get appropriate linter for a project.

        Args:
            project_root: Project root directory

        Returns:
            Linter config or None if not available
        """
        ...


class BaseLanguagePlugin(ABC):
    """Base class for language plugins with common functionality."""

    def __init__(self):
        """Initialize plugin."""
        self._config: Optional[LanguageConfig] = None
        self._capabilities: Optional[LanguageCapabilities] = None

    @property
    def config(self) -> LanguageConfig:
        """Get language configuration."""
        if self._config is None:
            self._config = self._create_config()
        return self._config

    @property
    def capabilities(self) -> LanguageCapabilities:
        """Get language capabilities."""
        if self._capabilities is None:
            self._capabilities = self._create_capabilities()
        return self._capabilities

    @abstractmethod
    def _create_config(self) -> LanguageConfig:
        """Create language configuration."""
        ...

    @abstractmethod
    def _create_capabilities(self) -> LanguageCapabilities:
        """Create capabilities description."""
        ...

    def detect_from_file(self, path: Path) -> bool:
        """Check if this language handles the file."""
        # Check extension
        if path.suffix.lower() in [e.lower() for e in self.config.extensions]:
            return True

        # Check filename
        if path.name in self.config.filenames:
            return True

        # Check shebang
        if self.config.shebangs and path.exists():
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline().strip()
                    for shebang in self.config.shebangs:
                        if first_line.startswith("#!") and shebang in first_line:
                            return True
            except Exception:
                pass

        return False

    def detect_from_content(self, content: str, filename: Optional[str] = None) -> float:
        """Estimate confidence from content."""
        score = 0.0

        # Check filename hint
        if filename:
            path = Path(filename)
            if path.suffix.lower() in [e.lower() for e in self.config.extensions]:
                score += 0.5
            if path.name in self.config.filenames:
                score += 0.5

        # Check shebang in content
        if content.startswith("#!"):
            first_line = content.split("\n")[0]
            for shebang in self.config.shebangs:
                if shebang in first_line:
                    score += 0.3
                    break

        return min(score, 1.0)

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Default: no test runner."""
        return None

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Default: no build system."""
        return None

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Default: no formatter."""
        return None

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Default: no linter."""
        return None
