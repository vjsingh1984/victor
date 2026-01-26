"""Tests for LanguageCapabilityRegistry."""

from pathlib import Path

import pytest

from victor.core.language_capabilities import (
    ASTAccessMethod,
    LanguageCapabilityRegistry,
    LanguageTier,
    UnifiedLanguageCapability,
    TreeSitterCapability,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton between tests."""
    LanguageCapabilityRegistry.reset_instance()
    yield
    LanguageCapabilityRegistry.reset_instance()


class TestLanguageCapabilityRegistry:
    """Tests for the language capability registry."""

    def test_singleton_instance(self):
        """Registry should be a singleton."""
        registry1 = LanguageCapabilityRegistry.instance()
        registry2 = LanguageCapabilityRegistry.instance()
        assert registry1 is registry2

    def test_default_languages_loaded(self):
        """Default languages should be loaded."""
        registry = LanguageCapabilityRegistry.instance()
        languages = registry.list_supported_languages()

        # Should have tier 1 languages
        assert "python" in languages
        assert "typescript" in languages
        assert "javascript" in languages

        # Should have tier 2 languages
        assert "go" in languages
        assert "rust" in languages
        assert "java" in languages

        # Should have tier 3 languages
        assert "ruby" in languages
        assert "php" in languages

    def test_get_python_capability(self):
        """Should get Python capability correctly."""
        registry = LanguageCapabilityRegistry.instance()
        cap = registry.get("python")

        assert cap is not None
        assert cap.name == "python"
        assert cap.tier == LanguageTier.TIER_1
        assert ".py" in cap.extensions
        assert cap.native_ast is not None
        assert cap.native_ast.library == "ast"
        assert cap.tree_sitter is not None
        assert cap.lsp is not None

    def test_get_for_file_python(self):
        """Should detect Python from file path."""
        registry = LanguageCapabilityRegistry.instance()

        cap = registry.get_for_file(Path("test.py"))
        assert cap is not None
        assert cap.name == "python"

        cap = registry.get_for_file(Path("src/module.pyi"))
        assert cap is not None
        assert cap.name == "python"

    def test_get_for_file_typescript(self):
        """Should detect TypeScript from file path."""
        registry = LanguageCapabilityRegistry.instance()

        cap = registry.get_for_file(Path("app.ts"))
        assert cap is not None
        assert cap.name == "typescript"

        cap = registry.get_for_file(Path("Component.tsx"))
        assert cap is not None
        assert cap.name == "tsx"

    def test_get_for_file_by_filename(self):
        """Should detect language by special filenames."""
        registry = LanguageCapabilityRegistry.instance()

        cap = registry.get_for_file(Path("Dockerfile"))
        assert cap is not None
        assert cap.name == "dockerfile"

        cap = registry.get_for_file(Path("Makefile"))
        assert cap is not None
        assert cap.name == "make"

    def test_get_for_unknown_file(self):
        """Should return None for unknown file types."""
        registry = LanguageCapabilityRegistry.instance()

        cap = registry.get_for_file(Path("unknown.xyz"))
        assert cap is None

    def test_list_by_tier(self):
        """Should list languages by tier."""
        registry = LanguageCapabilityRegistry.instance()

        tier1 = registry.list_supported_languages(LanguageTier.TIER_1)
        assert "python" in tier1
        assert "typescript" in tier1

        tier2 = registry.list_supported_languages(LanguageTier.TIER_2)
        assert "go" in tier2
        assert "rust" in tier2

        tier3 = registry.list_supported_languages(LanguageTier.TIER_3)
        assert "ruby" in tier3
        assert "php" in tier3

    def test_get_languages_by_tier(self):
        """Should group languages by tier."""
        registry = LanguageCapabilityRegistry.instance()
        by_tier = registry.get_languages_by_tier()

        assert LanguageTier.TIER_1 in by_tier
        assert LanguageTier.TIER_2 in by_tier
        assert LanguageTier.TIER_3 in by_tier

        assert "python" in by_tier[LanguageTier.TIER_1]
        assert "go" in by_tier[LanguageTier.TIER_2]
        assert "ruby" in by_tier[LanguageTier.TIER_3]

    def test_register_custom_language(self):
        """Should allow registering custom languages."""
        registry = LanguageCapabilityRegistry.instance()

        custom = UnifiedLanguageCapability(
            name="custom_lang",
            tier=LanguageTier.TIER_3,
            extensions=[".custom"],
            tree_sitter=TreeSitterCapability(grammar_package="tree_sitter_custom"),
        )

        registry.register(custom)

        cap = registry.get("custom_lang")
        assert cap is not None
        assert cap.name == "custom_lang"

        cap = registry.get_for_file(Path("file.custom"))
        assert cap is not None
        assert cap.name == "custom_lang"

    def test_unregister_language(self):
        """Should allow unregistering languages."""
        registry = LanguageCapabilityRegistry.instance()

        # Register then unregister
        custom = UnifiedLanguageCapability(
            name="to_remove",
            tier=LanguageTier.TIER_3,
            extensions=[".remove"],
        )
        registry.register(custom)

        removed = registry.unregister("to_remove")
        assert removed is not None
        assert removed.name == "to_remove"

        assert registry.get("to_remove") is None
        assert registry.get_for_file(Path("test.remove")) is None

    def test_detect_language(self):
        """Should detect language from file path."""
        registry = LanguageCapabilityRegistry.instance()

        assert registry.detect_language(Path("test.py")) == "python"
        assert registry.detect_language(Path("test.ts")) == "typescript"
        assert registry.detect_language(Path("test.rs")) == "rust"
        assert registry.detect_language(Path("unknown.xyz")) is None

    def test_get_indexing_method(self):
        """Should get best indexing method for language."""
        registry = LanguageCapabilityRegistry.instance()

        # Python should use native AST
        method = registry.get_indexing_method("python")
        assert method == ASTAccessMethod.NATIVE

        # Rust should use tree-sitter (no native AST)
        method = registry.get_indexing_method("rust")
        assert method == ASTAccessMethod.TREE_SITTER

    def test_get_validation_method(self):
        """Should get best validation method for language."""
        registry = LanguageCapabilityRegistry.instance()

        # Python should use native AST
        method = registry.get_validation_method("python")
        assert method == ASTAccessMethod.NATIVE

        # Rust should use tree-sitter
        method = registry.get_validation_method("rust")
        assert method == ASTAccessMethod.TREE_SITTER

    def test_list_extensions(self):
        """Should list all supported extensions."""
        registry = LanguageCapabilityRegistry.instance()
        extensions = registry.list_extensions()

        assert ".py" in extensions
        assert ".ts" in extensions
        assert ".go" in extensions
        assert ".rs" in extensions

    def test_list_filenames(self):
        """Should list all supported filenames."""
        registry = LanguageCapabilityRegistry.instance()
        filenames = registry.list_filenames()

        assert "dockerfile" in filenames
        assert "makefile" in filenames


class TestUnifiedLanguageCapability:
    """Tests for UnifiedLanguageCapability dataclass."""

    def test_get_best_indexing_method_native(self):
        """Should prefer native AST when available."""
        from victor.core.language_capabilities.types import ASTCapability

        cap = UnifiedLanguageCapability(
            name="test",
            tier=LanguageTier.TIER_1,
            extensions=[".test"],
            native_ast=ASTCapability(
                library="test",
                access_method=ASTAccessMethod.NATIVE,
            ),
            tree_sitter=TreeSitterCapability(grammar_package="tree_sitter_test"),
            indexing_strategy=[
                ASTAccessMethod.NATIVE,
                ASTAccessMethod.TREE_SITTER,
            ],
        )

        method = cap.get_best_indexing_method()
        assert method == ASTAccessMethod.NATIVE

    def test_get_best_indexing_method_fallback(self):
        """Should fall back to tree-sitter when native not available."""
        cap = UnifiedLanguageCapability(
            name="test",
            tier=LanguageTier.TIER_2,
            extensions=[".test"],
            native_ast=None,  # No native AST
            tree_sitter=TreeSitterCapability(grammar_package="tree_sitter_test"),
            indexing_strategy=[
                ASTAccessMethod.NATIVE,
                ASTAccessMethod.TREE_SITTER,
            ],
        )

        method = cap.get_best_indexing_method()
        assert method == ASTAccessMethod.TREE_SITTER

    def test_supports_syntax_validation(self):
        """Should check syntax validation support."""
        from victor.core.language_capabilities.types import ASTCapability

        cap_with_native = UnifiedLanguageCapability(
            name="test",
            tier=LanguageTier.TIER_1,
            extensions=[".test"],
            native_ast=ASTCapability(
                library="test",
                access_method=ASTAccessMethod.NATIVE,
            ),
        )
        assert cap_with_native.supports_syntax_validation()

        cap_with_ts = UnifiedLanguageCapability(
            name="test",
            tier=LanguageTier.TIER_3,
            extensions=[".test"],
            tree_sitter=TreeSitterCapability(grammar_package="tree_sitter_test"),
        )
        assert cap_with_ts.supports_syntax_validation()

        cap_without = UnifiedLanguageCapability(
            name="test",
            tier=LanguageTier.UNSUPPORTED,
            extensions=[".test"],
        )
        assert not cap_without.supports_syntax_validation()

    def test_get_available_methods(self):
        """Should list all available methods."""
        from victor.core.language_capabilities.types import ASTCapability, LSPCapability

        cap = UnifiedLanguageCapability(
            name="test",
            tier=LanguageTier.TIER_1,
            extensions=[".test"],
            native_ast=ASTCapability(
                library="test",
                access_method=ASTAccessMethod.NATIVE,
            ),
            tree_sitter=TreeSitterCapability(grammar_package="tree_sitter_test"),
            lsp=LSPCapability(
                server_name="test-lsp",
                language_id="test",
            ),
        )

        methods = cap.get_available_methods()
        assert ASTAccessMethod.NATIVE in methods
        assert ASTAccessMethod.TREE_SITTER in methods
        assert ASTAccessMethod.LSP in methods
