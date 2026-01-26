"""
Core data structures for the Unified Language Capability System.

This module defines the data types used by both indexing (code understanding)
and validation (code grounding) systems. Single source of truth for what
a language supports.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple


class LanguageTier(Enum):
    """
    Language support tiers.

    Tier determines which parsing strategies are available:
    - TIER_1: Full support - Native AST + Tree-sitter + LSP
    - TIER_2: Good support - Native AST or Tree-sitter + LSP
    - TIER_3: Basic support - Tree-sitter only
    - UNSUPPORTED: No parsing support
    """
    TIER_1 = 1  # Full support: Native AST + Tree-sitter + LSP
    TIER_2 = 2  # Good support: Native AST/Tree-sitter + LSP
    TIER_3 = 3  # Basic support: Tree-sitter only
    UNSUPPORTED = 0


class ASTAccessMethod(Enum):
    """
    Method for accessing AST parsing capabilities.

    Each method has different trade-offs:
    - NATIVE: Built-in Python module (fastest, Python only)
    - PYTHON_LIB: Pure Python library (portable, may be incomplete)
    - FFI: Foreign function interface (fast, requires native libs)
    - SUBPROCESS: External process (full features, slower startup)
    - LSP: Language Server Protocol (full features, async)
    - TREE_SITTER: Tree-sitter parser (fast, error-tolerant)
    """
    NATIVE = "native"          # Built-in module (Python ast)
    PYTHON_LIB = "python_lib"  # Pure Python library (javalang, gopygo)
    FFI = "ffi"                # Foreign function interface (libclang)
    SUBPROCESS = "subprocess"  # External process (Node.js, Ruby)
    LSP = "lsp"                # Via Language Server Protocol
    TREE_SITTER = "tree_sitter"  # Tree-sitter parser


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"        # Syntax/semantic error - blocks write
    WARNING = "warning"    # Potential issue - logged but allowed
    INFO = "info"          # Informational - always allowed
    HINT = "hint"          # Suggestion - always allowed


@dataclass(frozen=True)
class ASTCapability:
    """
    Describes AST parsing capability for a language.

    Attributes:
        library: Name of the parsing library (e.g., "ast", "javalang", "libclang")
        access_method: How to access the library
        python_package: pip package name for installation
        requires_runtime: External runtime requirement (e.g., "node", "java")
        has_type_info: Whether the parser provides type information
        has_error_recovery: Whether the parser can produce partial AST on errors
        has_semantic_analysis: Whether semantic analysis is available
        subprocess_command: Command template for subprocess method
        output_format: Output format for subprocess method (json, sexp, xml)
    """
    library: str
    access_method: ASTAccessMethod
    python_package: Optional[str] = None
    requires_runtime: Optional[str] = None

    # Capability flags
    has_type_info: bool = False
    has_error_recovery: bool = False
    has_semantic_analysis: bool = False

    # For subprocess method
    subprocess_command: Optional[List[str]] = None
    output_format: str = "json"


@dataclass(frozen=True)
class LSPCapability:
    """
    Describes Language Server Protocol support for a language.

    Attributes:
        server_name: Name of the language server (e.g., "pylsp", "tsserver")
        language_id: LSP language identifier
        install_command: Command to install the language server
        has_diagnostics: Whether diagnostics (errors/warnings) are available
        has_completion: Whether code completion is available
        has_hover: Whether hover information is available
        has_type_info: Whether type information is available via hover
    """
    server_name: str
    language_id: str
    install_command: Optional[str] = None
    has_diagnostics: bool = True
    has_completion: bool = True
    has_hover: bool = True
    has_type_info: bool = False


@dataclass(frozen=True)
class TreeSitterCapability:
    """
    Describes Tree-sitter support for a language.

    Attributes:
        grammar_package: Package containing the grammar (e.g., "tree_sitter_python")
        language_function: Function name to get Language object (usually "language")
        has_highlights: Whether syntax highlighting queries are available
        has_injections: Whether language injection is supported (for embedded languages)
    """
    grammar_package: str
    language_function: str = "language"
    has_highlights: bool = True
    has_injections: bool = False


@dataclass
class ValidationIssue:
    """
    A single validation issue found in code.

    Attributes:
        line: 1-indexed line number
        column: 0-indexed column number
        message: Human-readable description of the issue
        severity: Severity level (error, warning, info, hint)
        source: Validator that detected this issue
        end_line: Optional end line for multi-line issues
        end_column: Optional end column
        code: Optional error code (e.g., "E0001")
        suggestion: Optional fix suggestion
    """
    line: int
    column: int
    message: str
    severity: ValidationSeverity
    source: str
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    code: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc = f"{self.line}:{self.column}"
        if self.end_line:
            loc += f"-{self.end_line}:{self.end_column or 0}"
        return f"[{self.severity.value}] {loc}: {self.message} ({self.source})"


@dataclass
class CodeValidationResult:
    """
    Result of code validation.

    Attributes:
        is_valid: Whether the code passed validation
        language: Detected or specified language
        tier: Language tier (if detected)
        validators_used: List of validators that ran
        issues: List of validation issues found
        warnings: Deprecated, use issues with WARNING severity
        metadata: Additional metadata from validators
    """
    is_valid: bool
    language: str = "unknown"
    tier: Optional[LanguageTier] = None
    validators_used: List[str] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False

    def merge(self, other: "CodeValidationResult") -> None:
        """Merge another result into this one."""
        self.issues.extend(other.issues)
        self.warnings.extend(other.warnings)
        self.validators_used.extend(other.validators_used)
        if not other.is_valid:
            self.is_valid = False
        self.metadata.update(other.metadata)


@dataclass
class ValidationConfig:
    """
    Configuration for code validation.

    Attributes:
        strict: Block on any validation failure
        check_syntax: Perform syntax validation
        check_semantics: Perform semantic validation (if available)
        check_types: Perform type checking (if available)
        max_errors: Maximum errors before stopping (0 = unlimited)
        timeout_seconds: Timeout for validation (0 = no timeout)
        custom_rules: Additional validation rules
    """
    strict: bool = False
    check_syntax: bool = True
    check_semantics: bool = False
    check_types: bool = False
    max_errors: int = 0
    timeout_seconds: float = 0.0
    custom_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedLanguageCapability:
    """
    Unified capability definition for a language.

    Used by BOTH indexing and enforcement systems.
    Single source of truth for what a language supports.

    Attributes:
        name: Language identifier (e.g., "python", "typescript")
        tier: Support tier (TIER_1, TIER_2, TIER_3)
        extensions: File extensions (e.g., [".py", ".pyw"])
        filenames: Special filenames (e.g., ["Makefile", "Dockerfile"])
        native_ast: Native AST capability (if available)
        tree_sitter: Tree-sitter capability (if available)
        lsp: LSP capability (if available)
        indexing_enabled: Whether indexing is enabled
        validation_enabled: Whether validation is enabled
        indexing_strategy: Ordered list of methods to try for indexing
        validation_strategy: Ordered list of methods to try for validation
        fallback_on_unavailable: Behavior when no method available
        fallback_on_error: Behavior when validation fails
    """
    name: str
    tier: LanguageTier
    extensions: List[str]
    filenames: List[str] = field(default_factory=list)

    # Capability modules (all optional)
    native_ast: Optional[ASTCapability] = None
    tree_sitter: Optional[TreeSitterCapability] = None
    lsp: Optional[LSPCapability] = None

    # Feature flags (runtime toggleable)
    indexing_enabled: bool = True
    validation_enabled: bool = True

    # Strategy preferences (ordered by priority)
    indexing_strategy: List[ASTAccessMethod] = field(
        default_factory=lambda: [
            ASTAccessMethod.NATIVE,
            ASTAccessMethod.TREE_SITTER,
            ASTAccessMethod.LSP
        ]
    )
    validation_strategy: List[ASTAccessMethod] = field(
        default_factory=lambda: [
            ASTAccessMethod.NATIVE,
            ASTAccessMethod.TREE_SITTER,
            ASTAccessMethod.LSP
        ]
    )

    # Fallback behavior
    fallback_on_unavailable: str = "allow"  # allow, warn, block
    fallback_on_error: str = "warn"  # allow, warn, block

    def get_best_indexing_method(self) -> Optional[ASTAccessMethod]:
        """Get the best available method for indexing."""
        for method in self.indexing_strategy:
            if self._method_available(method):
                return method
        return None

    def get_best_validation_method(self) -> Optional[ASTAccessMethod]:
        """Get the best available method for validation."""
        for method in self.validation_strategy:
            if self._method_available(method):
                return method
        return None

    def _method_available(self, method: ASTAccessMethod) -> bool:
        """Check if a method is available for this language."""
        if method == ASTAccessMethod.NATIVE:
            return (
                self.native_ast is not None
                and self.native_ast.access_method == ASTAccessMethod.NATIVE
            )
        elif method == ASTAccessMethod.PYTHON_LIB:
            return (
                self.native_ast is not None
                and self.native_ast.access_method == ASTAccessMethod.PYTHON_LIB
            )
        elif method == ASTAccessMethod.FFI:
            return (
                self.native_ast is not None
                and self.native_ast.access_method == ASTAccessMethod.FFI
            )
        elif method == ASTAccessMethod.SUBPROCESS:
            return (
                self.native_ast is not None
                and self.native_ast.access_method == ASTAccessMethod.SUBPROCESS
            )
        elif method == ASTAccessMethod.TREE_SITTER:
            return self.tree_sitter is not None
        elif method == ASTAccessMethod.LSP:
            return self.lsp is not None
        return False

    def get_available_methods(self) -> List[ASTAccessMethod]:
        """Get all available methods for this language."""
        available = []
        for method in ASTAccessMethod:
            if self._method_available(method):
                available.append(method)
        return available

    def supports_syntax_validation(self) -> bool:
        """Check if syntax validation is supported."""
        return bool(self.native_ast or self.tree_sitter)

    def supports_semantic_validation(self) -> bool:
        """Check if semantic validation is supported."""
        if self.native_ast and self.native_ast.has_semantic_analysis:
            return True
        if self.lsp and self.lsp.has_diagnostics:
            return True
        return False

    def supports_type_checking(self) -> bool:
        """Check if type checking is supported."""
        if self.native_ast and self.native_ast.has_type_info:
            return True
        if self.lsp and self.lsp.has_type_info:
            return True
        return False


@dataclass
class ExtractedSymbol:
    """
    A symbol extracted from source code.

    Attributes:
        name: Symbol name
        symbol_type: Type of symbol (class, function, method, etc.)
        file_path: Path to the source file
        line_number: 1-indexed start line
        end_line: Optional end line
        signature: Optional signature (for functions/methods)
        docstring: Optional documentation string
        parent_symbol: Optional parent symbol name (for nested symbols)
        return_type: Optional return type
        parameters: List of parameter names/types
        visibility: Visibility modifier (public, private, protected)
        is_async: Whether this is an async function
        decorators: List of decorator names
        metadata: Additional metadata
    """
    name: str
    symbol_type: str
    file_path: str
    line_number: int
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None
    return_type: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    visibility: Optional[str] = None
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name (parent.name)."""
        if self.parent_symbol:
            return f"{self.parent_symbol}.{self.name}"
        return self.name
