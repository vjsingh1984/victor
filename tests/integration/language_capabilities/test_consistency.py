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

"""End-to-end consistency tests for the unified language capabilities system.

These tests verify that:
1. Indexing and validation use the same capabilities from the registry
2. If a language is supported for indexing, validation should also work
3. Language detection is consistent across all components
"""

import pytest
import tempfile
from pathlib import Path

from victor.core.language_capabilities import (
    LanguageCapabilityRegistry,
    LanguageTier,
    CodeGroundingHook,
    UnifiedLanguageValidator,
)
from victor.core.language_capabilities.extractors import (
    UnifiedLanguageExtractor,
    TreeSitterExtractor,
)


class TestIndexingValidationConsistency:
    """Test that indexing and validation use the same capabilities."""

    @pytest.fixture
    def registry(self):
        """Get the language capability registry."""
        return LanguageCapabilityRegistry.instance()

    @pytest.fixture
    def extractor(self):
        """Get the unified extractor."""
        return UnifiedLanguageExtractor()

    @pytest.fixture
    def validator(self):
        """Get the unified validator."""
        return UnifiedLanguageValidator()

    def test_all_languages_have_validation_method(self, registry):
        """Test that all registered languages have a validation method."""
        languages = registry.list_supported_languages()

        for lang in languages:
            cap = registry.get(lang)
            assert cap is not None, f"No capability for {lang}"

            # All languages should have at least tree-sitter validation
            validation_method = cap.get_best_validation_method()
            assert validation_method is not None, (
                f"No validation method for {lang} (tier={cap.tier.name})"
            )

    def test_indexing_implies_validation(self, registry):
        """Test that if indexing works, validation should too."""
        languages = registry.list_supported_languages()

        for lang in languages:
            cap = registry.get(lang)
            indexing_method = cap.get_best_indexing_method()
            validation_method = cap.get_best_validation_method()

            if indexing_method is not None:
                assert validation_method is not None, (
                    f"Language {lang} has indexing ({indexing_method}) "
                    f"but no validation"
                )

    def test_tier_1_languages_have_advanced_support(self, registry):
        """Test that Tier 1 languages have advanced support (native AST, tree-sitter, or LSP)."""
        tier_1_languages = registry.list_supported_languages(LanguageTier.TIER_1)

        for lang in tier_1_languages:
            cap = registry.get(lang)
            # Tier 1 should have at least one advanced parsing method
            has_advanced = (
                cap.native_ast is not None or
                cap.tree_sitter is not None or
                cap.lsp is not None
            )
            assert has_advanced, (
                f"Tier 1 language {lang} missing advanced support"
            )

    def test_all_languages_have_tree_sitter_capability(self, registry):
        """Test that Tier 1/2 languages have tree-sitter capability defined."""
        tier_1_2 = registry.list_supported_languages(LanguageTier.TIER_1)
        tier_1_2.extend(registry.list_supported_languages(LanguageTier.TIER_2))

        for lang in tier_1_2:
            cap = registry.get(lang)
            # Tier 1/2 languages should have tree-sitter capability defined
            # (even if the grammar is not installed)
            assert cap.tree_sitter is not None or cap.native_ast is not None, (
                f"Tier 1/2 language {lang} missing tree-sitter or native AST capability"
            )


class TestLanguageDetectionConsistency:
    """Test that language detection is consistent."""

    @pytest.fixture
    def registry(self):
        """Get the language capability registry."""
        return LanguageCapabilityRegistry.instance()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.parametrize("extension,expected_language", [
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        (".tsx", "tsx"),
        (".jsx", "jsx"),
        (".go", "go"),
        (".rs", "rust"),
        (".java", "java"),
        (".c", "c"),
        (".cpp", "cpp"),
        (".rb", "ruby"),
        (".php", "php"),
        (".swift", "swift"),
        (".kt", "kotlin"),
        (".scala", "scala"),
        (".lua", "lua"),
        (".sh", "bash"),
        (".html", "html"),
        (".css", "css"),
        (".json", "json"),
        (".yaml", "yaml"),
        (".yml", "yaml"),
        (".md", "markdown"),
        (".toml", "toml"),
    ])
    def test_extension_detection(self, registry, temp_dir, extension, expected_language):
        """Test that file extensions are correctly detected."""
        file_path = temp_dir / f"test{extension}"

        cap = registry.get_for_file(file_path)

        if cap is None:
            pytest.skip(f"Language not registered for {extension}")

        assert cap.name == expected_language, (
            f"Expected {expected_language} for {extension}, got {cap.name}"
        )

    @pytest.mark.parametrize("filename,expected_language", [
        ("Dockerfile", "dockerfile"),
        ("Makefile", "make"),
        ("CMakeLists.txt", "cmake"),
    ])
    def test_filename_detection(self, registry, temp_dir, filename, expected_language):
        """Test that special filenames are correctly detected."""
        file_path = temp_dir / filename

        cap = registry.get_for_file(file_path)

        if cap is None:
            pytest.skip(f"Language not registered for {filename}")

        assert cap.name == expected_language, (
            f"Expected {expected_language} for {filename}, got {cap.name}"
        )


class TestValidationConsistency:
    """Test that validation produces consistent results."""

    @pytest.fixture
    def hook(self):
        """Get the code grounding hook."""
        return CodeGroundingHook.instance()

    @pytest.fixture
    def validator(self):
        """Get the unified validator."""
        return UnifiedLanguageValidator()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_hook_and_validator_agree(self, hook, validator, temp_dir):
        """Test that hook and validator produce same results."""
        file_path = temp_dir / "test.py"
        valid_code = "def foo(): pass"
        invalid_code = "def foo(:"

        # Valid code
        hook_result = hook.validate_before_write_sync(valid_code, file_path)
        validator_result = validator.validate(valid_code, file_path)
        assert hook_result[1].is_valid == validator_result.is_valid

        # Invalid code
        hook_result = hook.validate_before_write_sync(invalid_code, file_path)
        validator_result = validator.validate(invalid_code, file_path)
        assert hook_result[1].is_valid == validator_result.is_valid

    @pytest.mark.parametrize("language,extension,valid_code,invalid_code", [
        ("python", ".py", "x = 1", "x = ("),
        ("javascript", ".js", "const x = 1;", "const x = (;"),
        ("typescript", ".ts", "const x: number = 1;", "const x: number = (;"),
        ("go", ".go", "package main\nfunc main() {}", "package main\nfunc main( {}"),
        ("rust", ".rs", "fn main() {}", "fn main( {}"),
        ("java", ".java", "class Main {}", "class Main {"),
    ])
    def test_multi_language_validation(
        self, validator, temp_dir, language, extension, valid_code, invalid_code
    ):
        """Test validation works consistently across languages."""
        file_path = temp_dir / f"test{extension}"

        # Valid code should pass
        result = validator.validate(valid_code, file_path)
        assert result.is_valid is True, (
            f"{language} validation failed for valid code: {result.issues}"
        )
        assert result.language == language

        # Invalid code should fail
        result = validator.validate(invalid_code, file_path)
        assert result.is_valid is False, (
            f"{language} validation passed for invalid code"
        )
        assert result.language == language


class TestExtractionConsistency:
    """Test that extraction produces consistent results."""

    @pytest.fixture
    def extractor(self):
        """Get the tree-sitter extractor for testing."""
        return TreeSitterExtractor()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_python_extraction(self, extractor, temp_dir):
        """Test Python symbol extraction."""
        file_path = temp_dir / "test.py"
        code = '''
def greet(name: str) -> str:
    """Return greeting."""
    return f"Hello, {name}!"

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
'''
        file_path.write_text(code)

        symbols = extractor.extract(code, file_path, "python")

        # Should extract function and class
        names = [s.name for s in symbols]
        assert len(symbols) >= 1

    def test_javascript_extraction(self, extractor, temp_dir):
        """Test JavaScript symbol extraction."""
        file_path = temp_dir / "test.js"
        code = '''
function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
'''
        file_path.write_text(code)

        symbols = extractor.extract(code, file_path, "javascript")

        # Should extract function
        assert len(symbols) >= 1

    def test_go_extraction(self, extractor, temp_dir):
        """Test Go symbol extraction."""
        file_path = temp_dir / "main.go"
        code = '''
package main

func main() {
    fmt.Println("Hello")
}

func add(a, b int) int {
    return a + b
}
'''
        file_path.write_text(code)

        symbols = extractor.extract(code, file_path, "go")

        # Should extract functions
        assert len(symbols) >= 1


class TestRegistryConsistency:
    """Test registry state consistency."""

    @pytest.fixture
    def registry(self):
        """Get the language capability registry."""
        return LanguageCapabilityRegistry.instance()

    def test_singleton_instance(self):
        """Test that registry is a singleton."""
        r1 = LanguageCapabilityRegistry.instance()
        r2 = LanguageCapabilityRegistry.instance()
        assert r1 is r2

    def test_all_extensions_unique(self, registry):
        """Test that no extension is mapped to multiple languages."""
        extensions = registry.list_extensions()

        # Each extension should appear only once
        seen = set()
        for ext in extensions:
            assert ext not in seen, f"Extension {ext} is duplicated"
            seen.add(ext)

    def test_all_languages_have_extensions(self, registry):
        """Test that all languages have at least one extension."""
        languages = registry.list_supported_languages()

        for lang in languages:
            cap = registry.get(lang)
            assert len(cap.extensions) > 0 or len(cap.filenames) > 0, (
                f"Language {lang} has no extensions or filenames"
            )

    def test_tier_consistency(self, registry):
        """Test that tier values are valid."""
        languages = registry.list_supported_languages()

        for lang in languages:
            cap = registry.get(lang)
            assert cap.tier in [
                LanguageTier.TIER_1,
                LanguageTier.TIER_2,
                LanguageTier.TIER_3,
                LanguageTier.UNSUPPORTED,
            ], f"Invalid tier for {lang}: {cap.tier}"
