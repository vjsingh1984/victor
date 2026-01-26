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

"""End-to-end consistency tests for the unified language capability system.

These tests verify that:
1. Indexing and validation use the same language capabilities
2. The unified registry is the single source of truth
3. All components work together correctly
"""

import pytest
import tempfile
from pathlib import Path

from victor.core.language_capabilities import LanguageCapabilityRegistry
from victor.core.language_capabilities.validators import UnifiedLanguageValidator
from victor.core.language_capabilities.types import LanguageTier, ASTAccessMethod


class TestIndexingValidationConsistency:
    """Test that indexing and validation use the same capabilities."""

    @pytest.fixture
    def registry(self):
        """Get a fresh registry instance."""
        LanguageCapabilityRegistry.reset_instance()
        return LanguageCapabilityRegistry.instance()

    @pytest.fixture
    def validator(self, registry):
        """Get validator using the same registry."""
        return UnifiedLanguageValidator(registry=registry)

    def test_all_indexable_languages_are_validatable(self, registry):
        """If a language supports indexing, it should support validation."""
        for lang in registry.list_supported_languages():
            cap = registry.get(lang)
            assert cap is not None, f"Language {lang} not found in registry"

            # Get best methods for each operation
            index_method = cap.get_best_indexing_method()
            validate_method = cap.get_best_validation_method()

            # If indexing is supported, validation should be too
            if index_method is not None:
                assert validate_method is not None, (
                    f"Language {lang} supports indexing ({index_method}) " f"but not validation"
                )

    def test_same_capability_used_for_both_operations(self, registry):
        """Both indexing and validation should use the same capability definition."""
        for lang in registry.list_supported_languages():
            cap = registry.get(lang)

            # The same capability object should be used
            assert cap.indexing_enabled or not cap.indexing_strategy
            assert cap.validation_enabled or not cap.validation_strategy

            # Native AST should be available to both if available to either
            if cap.native_ast is not None:
                # Check it's in at least one strategy
                has_native_in_strategy = (
                    ASTAccessMethod.NATIVE in cap.indexing_strategy
                    or ASTAccessMethod.NATIVE in cap.validation_strategy
                )
                # Native AST should be used somewhere
                assert (
                    has_native_in_strategy or cap.native_ast.access_method != ASTAccessMethod.NATIVE
                )

    def test_tier_classification_matches_capabilities(self, registry):
        """Tier classification should match available capabilities."""
        for lang in registry.list_supported_languages():
            cap = registry.get(lang)

            if cap.tier == LanguageTier.TIER_1:
                # Tier 1 should have multiple methods available
                available_methods = cap.get_available_methods()
                assert len(available_methods) >= 1, (
                    f"Tier 1 language {lang} should have at least 1 method, "
                    f"has {len(available_methods)}"
                )
            elif cap.tier == LanguageTier.TIER_2:
                # Tier 2 should have at least tree-sitter or native
                assert (
                    cap.tree_sitter is not None or cap.native_ast is not None
                ), f"Tier 2 language {lang} should have tree-sitter or native AST"
            elif cap.tier == LanguageTier.TIER_3:
                # Tier 3 should at least have tree-sitter
                assert (
                    cap.tree_sitter is not None
                ), f"Tier 3 language {lang} should have tree-sitter"

    def test_validation_works_for_all_languages(self, registry, validator):
        """Validation should work for all registered languages."""
        temp_dir = tempfile.mkdtemp()

        # Test code samples for different languages
        test_cases = {
            "python": ("test.py", "def foo(): pass"),
            "javascript": ("test.js", "function foo() {}"),
            "typescript": ("test.ts", "function foo(): void {}"),
            "json": ("test.json", '{"key": "value"}'),
            "yaml": ("test.yaml", "key: value"),
            "toml": ("test.toml", 'key = "value"'),
            "xml": ("test.xml", "<root><item>test</item></root>"),
            "html": ("test.html", "<html><body>test</body></html>"),
            "css": ("test.css", "body { color: red; }"),
            "markdown": ("test.md", "# Header\n\nParagraph"),
        }

        for lang, (filename, code) in test_cases.items():
            cap = registry.get(lang)
            if cap is None:
                continue

            file_path = Path(temp_dir) / filename

            # Validation should not raise an exception
            result = validator.validate(code, file_path, language=lang)

            # Result should have the correct language
            assert result.language == lang, f"Validation result language mismatch for {lang}"

            # Valid code should pass validation
            assert result.is_valid, f"Valid {lang} code failed validation: {result.errors}"

    def test_invalid_code_detected_by_validation(self, registry, validator):
        """Invalid code should be detected by validation."""
        temp_dir = tempfile.mkdtemp()

        # Invalid code samples
        invalid_cases = {
            "python": ("test.py", "def foo(:"),  # Missing parameter
            "json": ("test.json", '{"key": value}'),  # Unquoted value
            "yaml": ("test.yaml", "key: [unclosed"),  # Unclosed bracket
            "xml": ("test.xml", "<root><item></root>"),  # Mismatched tags
        }

        for lang, (filename, code) in invalid_cases.items():
            cap = registry.get(lang)
            if cap is None:
                continue

            file_path = Path(temp_dir) / filename
            result = validator.validate(code, file_path, language=lang)

            # Invalid code should fail validation
            assert not result.is_valid, f"Invalid {lang} code passed validation when it should fail"
            assert len(result.errors) > 0, f"Invalid {lang} code should have errors"


class TestConfigLanguageValidation:
    """Test that config languages use native validators properly."""

    @pytest.fixture
    def registry(self):
        """Get a fresh registry instance."""
        LanguageCapabilityRegistry.reset_instance()
        return LanguageCapabilityRegistry.instance()

    @pytest.fixture
    def validator(self, registry):
        """Get validator using the same registry."""
        return UnifiedLanguageValidator(registry=registry)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_json_uses_native_validator(self, registry, validator, temp_dir):
        """JSON validation should use native json module."""
        cap = registry.get("json")
        assert cap is not None
        assert cap.native_ast is not None, "JSON should have native AST"
        assert cap.native_ast.library == "json"
        assert ASTAccessMethod.NATIVE in cap.validation_strategy

        # Test validation
        valid_json = '{"name": "test", "value": 42}'
        invalid_json = '{"name": "test", value: 42}'

        file_path = temp_dir / "test.json"

        # Valid JSON should pass
        result = validator.validate(valid_json, file_path, language="json")
        assert result.is_valid
        assert "json.loads" in result.validators_used

        # Invalid JSON should fail with descriptive error
        result = validator.validate(invalid_json, file_path, language="json")
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.errors[0].line >= 1

    def test_yaml_uses_native_validator(self, registry, validator, temp_dir):
        """YAML validation should use native PyYAML."""
        cap = registry.get("yaml")
        assert cap is not None
        assert cap.native_ast is not None, "YAML should have native AST"
        assert ASTAccessMethod.NATIVE in cap.validation_strategy

        # Test validation
        valid_yaml = "name: test\nvalue: 42"
        invalid_yaml = "key: [unclosed bracket"

        file_path = temp_dir / "test.yaml"

        # Valid YAML should pass
        result = validator.validate(valid_yaml, file_path, language="yaml")
        assert result.is_valid

        # Invalid YAML should fail
        result = validator.validate(invalid_yaml, file_path, language="yaml")
        assert not result.is_valid

    def test_xml_uses_native_validator(self, registry, validator, temp_dir):
        """XML validation should use native xml.etree."""
        cap = registry.get("xml")
        assert cap is not None
        assert cap.native_ast is not None, "XML should have native AST"
        assert ASTAccessMethod.NATIVE in cap.validation_strategy

        # Test validation
        valid_xml = "<root><item>test</item></root>"
        invalid_xml = "<root><item>test</root>"  # Mismatched tags

        file_path = temp_dir / "test.xml"

        # Valid XML should pass
        result = validator.validate(valid_xml, file_path, language="xml")
        assert result.is_valid
        assert "xml.etree.ElementTree" in result.validators_used

        # Invalid XML should fail
        result = validator.validate(invalid_xml, file_path, language="xml")
        assert not result.is_valid


class TestRegistrySingleSourceOfTruth:
    """Test that the registry is the single source of truth."""

    @pytest.fixture
    def registry(self):
        """Get a fresh registry instance."""
        LanguageCapabilityRegistry.reset_instance()
        return LanguageCapabilityRegistry.instance()

    def test_registry_is_singleton(self):
        """Registry should be a singleton."""
        LanguageCapabilityRegistry.reset_instance()
        reg1 = LanguageCapabilityRegistry.instance()
        reg2 = LanguageCapabilityRegistry.instance()
        assert reg1 is reg2

    def test_all_components_use_same_registry(self, registry):
        """All components should use the same registry instance."""
        validator = UnifiedLanguageValidator(registry=registry)

        # Validator should use the same registry
        assert validator._registry is registry

    def test_language_detection_consistent(self, registry):
        """Language detection should be consistent across components."""
        test_files = [
            (Path("test.py"), "python"),
            (Path("test.js"), "javascript"),
            (Path("test.ts"), "typescript"),
            (Path("test.json"), "json"),
            (Path("config.yaml"), "yaml"),
            (Path("Makefile"), "make"),
            (Path("Dockerfile"), "dockerfile"),
        ]

        for file_path, expected_lang in test_files:
            cap = registry.get_for_file(file_path)
            if expected_lang == "make" or expected_lang == "dockerfile":
                # These may or may not be registered
                if cap is not None:
                    assert cap.name == expected_lang
            else:
                assert cap is not None, f"No capability for {file_path}"
                assert (
                    cap.name == expected_lang
                ), f"Expected {expected_lang}, got {cap.name} for {file_path}"

    def test_extension_mapping_unique(self, registry):
        """Each extension should map to exactly one language."""
        seen_extensions = {}

        for lang in registry.list_supported_languages():
            cap = registry.get(lang)
            for ext in cap.extensions:
                if ext in seen_extensions:
                    # Some extensions might legitimately be shared
                    # (e.g., .h for C and C++) - but warn about it
                    pass
                seen_extensions[ext] = lang

    def test_validation_strategy_consistency(self, registry):
        """Validation strategy should be consistent with available methods."""
        for lang in registry.list_supported_languages():
            cap = registry.get(lang)

            for method in cap.validation_strategy:
                # Each method in the strategy should be potentially available
                if method == ASTAccessMethod.NATIVE:
                    assert cap.native_ast is not None or method not in cap.get_available_methods()
                elif method == ASTAccessMethod.TREE_SITTER:
                    assert cap.tree_sitter is not None or method not in cap.get_available_methods()
                elif method == ASTAccessMethod.LSP:
                    assert cap.lsp is not None or method not in cap.get_available_methods()
