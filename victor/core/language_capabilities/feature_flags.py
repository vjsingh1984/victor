"""
Feature flag management for language capabilities.

Provides runtime control over indexing and validation features,
with support for:
- Environment variable configuration
- Per-language overrides
- Runtime API for dynamic control
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LanguageFeatureFlags:
    """Per-language feature flag overrides."""

    indexing_enabled: bool = True
    validation_enabled: bool = True
    native_ast_enabled: bool = True
    tree_sitter_enabled: bool = True
    lsp_enabled: bool = True


@dataclass
class GlobalFeatureFlags:
    """
    Global feature flags for the language capability system.

    Attributes:
        indexing_enabled: Master switch for indexing
        validation_enabled: Master switch for validation
        native_ast_enabled: Enable native AST parsing
        tree_sitter_enabled: Enable tree-sitter parsing
        lsp_enabled: Enable LSP integration
        strict_mode: Block writes on any validation failure
        cache_enabled: Enable caching of parse results
        cache_ttl_seconds: Cache time-to-live
        parallel_processing: Enable parallel processing
        language_overrides: Per-language flag overrides
    """

    # Master switches
    indexing_enabled: bool = True
    validation_enabled: bool = True

    # Mechanism toggles
    native_ast_enabled: bool = True
    tree_sitter_enabled: bool = True
    lsp_enabled: bool = True

    # Behavior
    strict_mode: bool = False
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    parallel_processing: bool = True

    # Per-language overrides
    language_overrides: Dict[str, LanguageFeatureFlags] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Manages feature flags for language capabilities.

    Sources (in priority order):
    1. Environment variables (VICTOR_LANG_*)
    2. Runtime API calls
    3. YAML configuration

    Thread-safe for concurrent access.
    """

    _instance: Optional["FeatureFlagManager"] = None

    def __init__(self) -> None:
        self._global = GlobalFeatureFlags()
        self._lock = threading.RLock()
        self._load_from_env()

    @classmethod
    def instance(cls) -> "FeatureFlagManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _load_from_env(self) -> None:
        """Load flags from environment variables."""
        self._global.indexing_enabled = self._env_bool("VICTOR_INDEXING_ENABLED", True)
        self._global.validation_enabled = self._env_bool("VICTOR_VALIDATION_ENABLED", True)
        self._global.strict_mode = self._env_bool("VICTOR_STRICT_VALIDATION", False)
        self._global.native_ast_enabled = self._env_bool("VICTOR_NATIVE_AST_ENABLED", True)
        self._global.tree_sitter_enabled = self._env_bool("VICTOR_TREE_SITTER_ENABLED", True)
        self._global.lsp_enabled = self._env_bool("VICTOR_LSP_ENABLED", True)
        self._global.cache_enabled = self._env_bool("VICTOR_LANG_CACHE_ENABLED", True)
        self._global.parallel_processing = self._env_bool("VICTOR_PARALLEL_PROCESSING", True)

        # Load cache TTL
        ttl_str = os.getenv("VICTOR_LANG_CACHE_TTL")
        if ttl_str:
            try:
                self._global.cache_ttl_seconds = int(ttl_str)
            except ValueError:
                pass

    def _env_bool(self, key: str, default: bool) -> bool:
        """Get boolean from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    @property
    def global_flags(self) -> GlobalFeatureFlags:
        """Get the global flags (read-only access)."""
        return self._global

    def is_indexing_enabled(self, language: str) -> bool:
        """Check if indexing is enabled for a language."""
        if not self._global.indexing_enabled:
            return False
        override = self._global.language_overrides.get(language)
        if override:
            return override.indexing_enabled
        return True

    def is_validation_enabled(self, language: str) -> bool:
        """Check if validation is enabled for a language."""
        if not self._global.validation_enabled:
            return False
        override = self._global.language_overrides.get(language)
        if override:
            return override.validation_enabled
        return True

    def is_native_ast_enabled(self, language: str) -> bool:
        """Check if native AST is enabled for a language."""
        if not self._global.native_ast_enabled:
            return False
        override = self._global.language_overrides.get(language)
        if override:
            return override.native_ast_enabled
        return True

    def is_tree_sitter_enabled(self, language: str) -> bool:
        """Check if tree-sitter is enabled for a language."""
        if not self._global.tree_sitter_enabled:
            return False
        override = self._global.language_overrides.get(language)
        if override:
            return override.tree_sitter_enabled
        return True

    def is_lsp_enabled(self, language: str) -> bool:
        """Check if LSP is enabled for a language."""
        if not self._global.lsp_enabled:
            return False
        override = self._global.language_overrides.get(language)
        if override:
            return override.lsp_enabled
        return True

    def is_strict_mode(self) -> bool:
        """Check if strict validation mode is enabled."""
        return self._global.strict_mode

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._global.cache_enabled

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return self._global.cache_ttl_seconds

    def is_parallel_enabled(self) -> bool:
        """Check if parallel processing is enabled."""
        return self._global.parallel_processing

    def set_global_flag(self, name: str, value: bool) -> None:
        """Set a global flag at runtime."""
        with self._lock:
            if hasattr(self._global, name):
                setattr(self._global, name, value)

    def set_language_flags(self, language: str, flags: LanguageFeatureFlags) -> None:
        """Set flags for a specific language at runtime."""
        with self._lock:
            self._global.language_overrides[language] = flags

    def clear_language_flags(self, language: str) -> None:
        """Clear flags for a specific language."""
        with self._lock:
            self._global.language_overrides.pop(language, None)

    def get_language_flags(self, language: str) -> Optional[LanguageFeatureFlags]:
        """Get flags for a specific language."""
        return self._global.language_overrides.get(language)

    def disable_language(self, language: str) -> None:
        """Disable all features for a language."""
        self.set_language_flags(
            language,
            LanguageFeatureFlags(
                indexing_enabled=False,
                validation_enabled=False,
                native_ast_enabled=False,
                tree_sitter_enabled=False,
                lsp_enabled=False,
            ),
        )

    def enable_language(self, language: str) -> None:
        """Enable all features for a language (remove override)."""
        self.clear_language_flags(language)

    def to_dict(self) -> Dict[str, Any]:
        """Export current flags as a dictionary."""
        return {
            "global": {
                "indexing_enabled": self._global.indexing_enabled,
                "validation_enabled": self._global.validation_enabled,
                "native_ast_enabled": self._global.native_ast_enabled,
                "tree_sitter_enabled": self._global.tree_sitter_enabled,
                "lsp_enabled": self._global.lsp_enabled,
                "strict_mode": self._global.strict_mode,
                "cache_enabled": self._global.cache_enabled,
                "cache_ttl_seconds": self._global.cache_ttl_seconds,
                "parallel_processing": self._global.parallel_processing,
            },
            "language_overrides": {
                lang: {
                    "indexing_enabled": flags.indexing_enabled,
                    "validation_enabled": flags.validation_enabled,
                    "native_ast_enabled": flags.native_ast_enabled,
                    "tree_sitter_enabled": flags.tree_sitter_enabled,
                    "lsp_enabled": flags.lsp_enabled,
                }
                for lang, flags in self._global.language_overrides.items()
            },
        }
