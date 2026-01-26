"""
Unified Language Analysis & Code Grounding System.

This module provides a single source of truth for language capabilities,
used by both indexing (code understanding) and validation (code enforcement).

Architecture Overview:
----------------------

    +--------------------------------------------------+
    |          LanguageCapabilityRegistry              |
    |          (Single Source of Truth)                |
    |  - 40+ languages with tier-based capabilities    |
    |  - YAML-configurable (capabilities.yaml)         |
    +--------------------------------------------------+
                         |
         +---------------+---------------+
         |                               |
    +----v----+                    +-----v-----+
    | INDEXING |                   | VALIDATION |
    | Pipeline |                   | Pipeline   |
    +----------+                    +-----------+
    | Extractors:                   | Validators:
    | - PythonASTExtractor          | - PythonASTValidator
    | - GoExtractor                 | - GoValidator
    | - JavaExtractor               | - JavaValidator
    | - CppExtractor                | - CppValidator
    | - TreeSitterExtractor         | - TreeSitterValidator
    +----------+                    +-----------+
         |                               |
         v                               v
    Symbols, Types              CodeValidationResult
    (for search/embeddings)     (for write blocking)

Key Components:
--------------
- LanguageCapabilityRegistry: Central registry for 40+ languages
- UnifiedLanguageExtractor: Symbol extraction using best available method
- UnifiedLanguageValidator: Code validation using best available method
- CodeGroundingHook: Pre-write validation integration

Language Tiers:
--------------
- Tier 1: Full support (Native AST + Tree-sitter + LSP)
  - Python, TypeScript, JavaScript, JSX, TSX
- Tier 2: Good support (Tree-sitter + LSP)
  - Go, Rust, Java, C, C++, C#, Kotlin, Swift, Scala
- Tier 3: Basic support (Tree-sitter only)
  - Ruby, PHP, Lua, and 30+ more languages

Optional Native Libraries:
-------------------------
- Python: Built-in ast module (always available)
- Go: gopygo (pip install gopygo)
- Java: javalang (pip install javalang)
- C/C++: libclang (pip install libclang)

All languages fall back to tree-sitter when native libraries aren't available.

Usage Examples:
--------------
    from victor.core.language_capabilities import (
        LanguageCapabilityRegistry,
        UnifiedLanguageValidator,
        CodeGroundingHook,
    )

    # Get registry instance
    registry = LanguageCapabilityRegistry.instance()

    # Detect language from file
    cap = registry.get_for_file(Path("example.py"))
    print(f"Language: {cap.name}, Tier: {cap.tier}")

    # Validate code before write
    hook = CodeGroundingHook.instance()
    should_proceed, result = hook.validate_before_write_sync(
        content="def foo(:",
        file_path=Path("example.py"),
    )

    # Quick validation check
    is_valid = hook.quick_validate("def foo(): pass", Path("test.py"))

Feature Flags:
-------------
Environment variables for runtime control:
- VICTOR_INDEXING_ENABLED: Enable/disable indexing (default: true)
- VICTOR_VALIDATION_ENABLED: Enable/disable validation (default: true)
- VICTOR_STRICT_VALIDATION: Enable strict validation mode (default: false)
"""

from .types import (
    ASTAccessMethod,
    ASTCapability,
    CodeValidationResult,
    ExtractedSymbol,
    LanguageTier,
    LSPCapability,
    TreeSitterCapability,
    UnifiedLanguageCapability,
    ValidationConfig,
    ValidationIssue,
    ValidationSeverity,
)
from .registry import LanguageCapabilityRegistry
from .feature_flags import (
    FeatureFlagManager,
    GlobalFeatureFlags,
    LanguageFeatureFlags,
)
from .hooks import (
    CodeGroundingHook,
    validate_code_before_write,
    validate_code_before_write_async,
)
from .extractors import (
    UnifiedLanguageExtractor,
    PythonASTExtractor,
    TreeSitterExtractor,
    GoExtractor,
    JavaExtractor,
    CppExtractor,
)
from .validators import (
    UnifiedLanguageValidator,
    PythonASTValidator,
    TreeSitterValidator,
    GoValidator,
    JavaValidator,
    CppValidator,
)
from .config_loader import (
    load_capabilities_from_yaml,
    load_feature_flags_from_yaml,
)

__all__ = [
    # Types
    "ASTAccessMethod",
    "ASTCapability",
    "CodeValidationResult",
    "ExtractedSymbol",
    "LanguageTier",
    "LSPCapability",
    "TreeSitterCapability",
    "UnifiedLanguageCapability",
    "ValidationConfig",
    "ValidationIssue",
    "ValidationSeverity",
    # Registry
    "LanguageCapabilityRegistry",
    # Feature Flags
    "FeatureFlagManager",
    "GlobalFeatureFlags",
    "LanguageFeatureFlags",
    # Hooks
    "CodeGroundingHook",
    "validate_code_before_write",
    "validate_code_before_write_async",
    # Extractors
    "UnifiedLanguageExtractor",
    "PythonASTExtractor",
    "TreeSitterExtractor",
    "GoExtractor",
    "JavaExtractor",
    "CppExtractor",
    # Validators
    "UnifiedLanguageValidator",
    "PythonASTValidator",
    "TreeSitterValidator",
    "GoValidator",
    "JavaValidator",
    "CppValidator",
    # Config loaders
    "load_capabilities_from_yaml",
    "load_feature_flags_from_yaml",
]
