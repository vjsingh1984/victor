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

"""
Vertical integration protocols for framework-vertical dependency-free integration.

This module provides protocol definitions that enable framework tools to work
with vertical-specific implementations (victor-coding, victor-rag, etc.) without
creating direct dependencies on external packages.

SOLID Principles:
- DIP (Dependency Inversion): Framework depends on protocols, not concrete implementations
- ISP (Interface Segregation): Focused protocols for each capability
- OCP (Open/Closed): New verticals can implement protocols without framework changes
- LSP (Liskov Substitution): All protocol implementations are interchangeable
- SRP (Single Responsibility): Each protocol handles one capability

Usage:
    from victor.framework.vertical_protocols import EditorProtocol

    # Use any editor implementation
    editor: EditorProtocol = get_editor()
    result = await editor.apply()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# =============================================================================
# File Editing Protocols
# =============================================================================


@runtime_checkable
class EditorProtocol(Protocol):
    """
    Protocol for transaction-based file editing operations.

    This protocol defines the interface for file editors that support
    transaction-based editing with undo/redo capabilities.

    The protocol supports queuing multiple operations and applying them
    atomically with automatic backup and rollback support.

    Example:
        editor.start_transaction("Fix bug in auth")
        editor.add_create("/path/to/new.py", content)
        editor.add_modify("/path/to/existing.py", new_content)
        result = await editor.apply()  # Applies all or rolls back on error

    Methods:
        start_transaction: Begin a new transaction
        add_create: Queue a file creation
        add_modify: Queue a file modification
        add_delete: Queue a file deletion
        add_rename: Queue a file rename/move
        add_replace: Queue a string replacement
        apply: Execute all queued operations
    """

    def start_transaction(self, desc: str = "") -> str:
        """Start a new transaction and return transaction ID."""
        ...

    def add_create(self, path: str, content: str) -> None:
        """Add a create operation to the transaction."""
        ...

    def add_modify(self, path: str, content: str) -> None:
        """Add a modify operation to the transaction."""
        ...

    def add_delete(self, path: str) -> None:
        """Add a delete operation to the transaction."""
        ...

    def add_rename(self, path: str, new_path: str) -> None:
        """Add a rename operation to the transaction."""
        ...

    def add_replace(self, path: str, old_str: str, new_str: str) -> None:
        """Add a replace operation to the transaction."""
        ...

    async def apply(self) -> Dict[str, Any]:
        """Apply all queued operations in the transaction."""
        ...

    def get_editor_info(self) -> Dict[str, Any]:
        """Get editor metadata."""
        ...


@dataclass
class EditOperation:
    """Single edit operation.

    Attributes:
        old_str: String to find in file (must match exactly)
        new_str: Replacement string
        start_line: Optional start line for context (0-indexed)
        end_line: Optional end line for context (0-indexed)
        allow_multiple: If False, error if old_str appears multiple times
    """

    old_str: str
    new_str: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    allow_multiple: bool = False

    def __post_init__(self) -> None:
        """Validate edit operation parameters."""
        if not self.old_str:
            raise ValueError("old_str cannot be empty")

        if self.start_line is not None and self.start_line < 0:
            raise ValueError("start_line must be >= 0")

        if self.end_line is not None and self.start_line is not None:
            if self.end_line < self.start_line:
                raise ValueError("start_line must be <= end_line")


@dataclass
class EditResult:
    """Result of file edit operation.

    Attributes:
        success: Whether the edit was successful
        file_path: Path to the edited file
        edits_applied: Number of edits successfully applied
        edits_failed: Number of edits that failed
        preview: Optional preview of changes
        error: Optional error message if edit failed
        metadata: Optional implementation-specific metadata
    """

    success: bool
    file_path: str
    edits_applied: int
    edits_failed: int = 0
    preview: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_edits(self) -> int:
        """Total number of edits processed."""
        return self.edits_applied + self.edits_failed


@dataclass
class EditValidationResult:
    """Result of edit validation.

    Attributes:
        valid: Whether the edit is valid
        file_path: Path to the file validated against
        old_str_found: Whether old_str was found in file
        match_count: Number of times old_str appears in file
        error: Optional error message if validation failed
        warnings: Optional list of warnings about the edit
    """

    valid: bool
    file_path: str
    old_str_found: bool
    match_count: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_safe_to_apply(self) -> bool:
        """Whether the edit is safe to apply (valid and single match)."""
        return self.valid and (self.match_count == 1 or self.match_count == 0)


# =============================================================================
# Codebase Analysis Protocols
# =============================================================================


@runtime_checkable
class CodebaseAnalyzerProtocol(Protocol):
    """
    Protocol for codebase analysis operations.

    Codebase analysis provides:
    - File discovery and filtering
    - Syntax parsing
    - Structure extraction (classes, functions, imports)
    - Dependency analysis

    Implementations can be provided by:
    - victor-coding (advanced coding-specific analysis)
    - victor.contrib.codebase (basic tree-sitter parser)
    - Custom analyzers

    Example:
        analyzer: CodebaseAnalyzerProtocol = get_analyzer()
        analysis = await analyzer.analyze_codebase(
            root_path="/path/to/code",
            include_patterns=["**/*.py"],
        )
    """

    async def analyze_codebase(
        self,
        root_path: Path,
        include_patterns: List[str],
        exclude_patterns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "CodebaseAnalysis":
        """Analyze a codebase."""
        ...

    async def parse_file(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> "ParsedFile":
        """Parse a single source file."""
        ...

    async def get_dependencies(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> "FileDependencies":
        """Get dependencies for a file."""
        ...

    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get analyzer metadata."""
        ...


@dataclass
class CodebaseAnalysis:
    """Results of codebase analysis.

    Attributes:
        root_path: Root directory of the analyzed codebase
        files: List of analyzed files
        total_files: Total number of files found
        total_lines: Total lines of code
        languages: Languages found and their file counts
        dependencies: Aggregated dependencies
        structure: Codebase structure (modules, packages, etc.)
        metadata: Additional implementation-specific metadata
    """

    root_path: Path
    files: List[str]
    total_files: int
    total_lines: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFile:
    """Results of parsing a single file.

    Attributes:
        file_path: Path to the parsed file
        language: Programming language
        lines: Number of lines
        classes: List of class definitions
        functions: List of function definitions
        imports: List of import statements
        syntax_tree: Optional syntax tree representation
        errors: List of parsing errors
    """

    file_path: Path
    language: str
    lines: int
    classes: List["ClassInfo"] = field(default_factory=list)
    functions: List["FunctionInfo"] = field(default_factory=list)
    imports: List["ImportInfo"] = field(default_factory=list)
    syntax_tree: Optional[Any] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class definition.

    Attributes:
        name: Class name
        line_number: Line number where class is defined
        end_line_number: Optional end line number
        bases: Base classes
        methods: List of methods
        decorators: List of decorators
        docstring: Optional docstring
    """

    name: str
    line_number: int
    end_line_number: Optional[int] = None
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class FunctionInfo:
    """Information about a function definition.

    Attributes:
        name: Function name
        line_number: Line number where function is defined
        end_line_number: Optional end line number
        parameters: Function parameters
        return_type: Optional return type annotation
        decorators: List of decorators
        docstring: Optional docstring
        is_async: Whether function is async
        is_method: Whether function is a method
    """

    name: str
    line_number: int
    end_line_number: Optional[int] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_async: bool = False
    is_method: bool = False


@dataclass
class ImportInfo:
    """Information about an import statement.

    Attributes:
        module: Module being imported
        names: Specific names being imported (for "from x import y")
        alias: Import alias (if any)
        line_number: Line number of import
        is_from_import: Whether this is a "from" import
    """

    module: str
    names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    line_number: int = 0
    is_from_import: bool = False


@dataclass
class FileDependencies:
    """Dependencies for a single file.

    Attributes:
        file_path: Path to the file
        imports: List of imports
        external_packages: External package dependencies
        internal_modules: Internal module dependencies
        requirements: Requirements.txt style dependencies
    """

    file_path: Path
    imports: List[ImportInfo] = field(default_factory=list)
    external_packages: List[str] = field(default_factory=list)
    internal_modules: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)


# =============================================================================
# LSP (Language Server Protocol) Protocols
# =============================================================================


class CompletionItemKind(Enum):
    """LSP completion item kinds."""

    TEXT = 1
    METHOD = 2
    FUNCTION = 3
    CONSTRUCTOR = 4
    FIELD = 5
    VARIABLE = 6
    CLASS = 7
    INTERFACE = 8
    MODULE = 9
    PROPERTY = 10
    UNIT = 11
    VALUE = 12
    ENUM = 13
    KEYWORD = 14
    SNIPPET = 15
    COLOR = 16
    FILE = 17
    REFERENCE = 18
    FOLDER = 19
    ENUM_MEMBER = 20
    CONSTANT = 21
    STRUCT = 22
    EVENT = 23
    OPERATOR = 24
    TYPE_PARAMETER = 25


@runtime_checkable
class LanguageServerProtocol(Protocol):
    """
    Protocol for Language Server Protocol (LSP) integration.

    LSP provides IDE-like features:
    - Code completion
    - Go-to-definition
    - Hover information
    - Diagnostics
    - Code actions
    - References

    Implementations can be provided by:
    - victor-coding (advanced coding-specific LSP integration)
    - victor.contrib.lsp (basic LSP client)
    - Custom LSP implementations

    Example:
        lsp: LanguageServerProtocol = get_lsp_client()
        completions = await lsp.get_completions(
            file_path="/path/to/file.py",
            line=10,
            character=5,
        )
    """

    async def start_server(
        self,
        language: str,
        file_path: Path,
        **kwargs: Any,
    ) -> bool:
        """Start language server for a file."""
        ...

    async def stop_server(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> bool:
        """Stop language server for a file."""
        ...

    async def get_completions(
        self,
        file_path: Path,
        line: int,
        character: int,
        **kwargs: Any,
    ) -> List["CompletionItem"]:
        """Get code completions at position."""
        ...

    async def get_definition(
        self,
        file_path: Path,
        line: int,
        character: int,
        **kwargs: Any,
    ) -> Optional["Location"]:
        """Go to definition at position."""
        ...

    async def get_hover(
        self,
        file_path: Path,
        line: int,
        character: int,
        **kwargs: Any,
    ) -> Optional["HoverInfo"]:
        """Get hover information at position."""
        ...

    async def get_diagnostics(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> List["Diagnostic"]:
        """Get diagnostics for a file."""
        ...

    def get_server_info(self) -> Dict[str, Any]:
        """Get server metadata."""
        ...


@dataclass
class CompletionItem:
    """Code completion item.

    Attributes:
        label: Completion text shown to user
        kind: Type of completion (class, function, etc.)
        detail: Additional detail (e.g., type signature)
        documentation: Documentation text
        insert_text: Text to insert (may differ from label)
        sort_text: Text for sorting
        filter_text: Text for filtering
    """

    label: str
    kind: CompletionItemKind
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None
    sort_text: Optional[str] = None
    filter_text: Optional[str] = None


@dataclass
class Location:
    """Source code location.

    Attributes:
        file_path: Path to the file
        line: Line number (0-indexed)
        character: Character position (0-indexed)
    """

    file_path: Path
    line: int
    character: int


@dataclass
class HoverInfo:
    """Hover information.

    Attributes:
        contents: Hover text (may contain markdown)
        range: Optional range of the hover
    """

    contents: str
    range: Optional[tuple[int, int, int, int]] = None


@dataclass
class Diagnostic:
    """Diagnostic information.

    Attributes:
        message: Diagnostic message
        severity: Severity level (error, warning, etc.)
        range: Range of the diagnostic
        source: Source of the diagnostic (e.g., "pylint")
        code: Diagnostic code
    """

    message: str
    severity: int  # 1=error, 2=warning, 3=info, 4=hint
    range: tuple[int, int, int, int]
    source: Optional[str] = None
    code: Optional[str] = None


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Editing protocols
    "EditorProtocol",
    "EditOperation",
    "EditResult",
    "EditValidationResult",
    # Codebase protocols
    "CodebaseAnalyzerProtocol",
    "CodebaseAnalysis",
    "ParsedFile",
    "ClassInfo",
    "FunctionInfo",
    "ImportInfo",
    "FileDependencies",
    # LSP protocols
    "LanguageServerProtocol",
    "CompletionItemKind",
    "CompletionItem",
    "Location",
    "HoverInfo",
    "Diagnostic",
]
