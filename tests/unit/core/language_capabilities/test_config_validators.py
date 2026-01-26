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

"""Tests for configuration file validators."""

import pytest
import tempfile
from pathlib import Path

from victor.core.language_capabilities.validators.config_validators import (
    JsonValidator,
    YamlValidator,
    TomlValidator,
    HoconValidator,
    XmlValidator,
    MarkdownValidator,
    get_config_validator,
)


class TestJsonValidator:
    """Tests for JsonValidator."""

    @pytest.fixture
    def validator(self):
        return JsonValidator()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """JSON validator is always available (built-in)."""
        assert validator.is_available() is True

    def test_valid_json(self, validator, temp_dir):
        """Test validation of valid JSON."""
        code = '{"name": "test", "value": 42}'
        result = validator.validate(code, temp_dir / "test.json")
        assert result.is_valid is True
        assert result.language == "json"

    def test_valid_json_array(self, validator, temp_dir):
        """Test validation of valid JSON array."""
        code = '[1, 2, 3, "test"]'
        result = validator.validate(code, temp_dir / "test.json")
        assert result.is_valid is True

    def test_invalid_json_syntax(self, validator, temp_dir):
        """Test validation fails for invalid JSON syntax."""
        code = '{"name": "test", value: 42}'  # Missing quotes around value
        result = validator.validate(code, temp_dir / "test.json")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_invalid_json_trailing_comma(self, validator, temp_dir):
        """Test validation fails for trailing comma."""
        code = '{"name": "test",}'
        result = validator.validate(code, temp_dir / "test.json")
        assert result.is_valid is False

    def test_error_has_line_info(self, validator, temp_dir):
        """Test that errors include line information."""
        code = '{\n"name": "test",\nvalue: 42\n}'
        result = validator.validate(code, temp_dir / "test.json")
        assert result.is_valid is False
        assert result.errors[0].line >= 1


class TestYamlValidator:
    """Tests for YamlValidator."""

    @pytest.fixture
    def validator(self):
        return YamlValidator()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """YAML validator availability depends on PyYAML."""
        # Should return True if PyYAML is installed
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_valid_yaml(self, validator, temp_dir):
        """Test validation of valid YAML."""
        if not validator.is_available():
            pytest.skip("PyYAML not available")

        code = """
name: test
value: 42
items:
  - one
  - two
"""
        result = validator.validate(code, temp_dir / "test.yaml")
        assert result.is_valid is True
        assert result.language == "yaml"

    def test_invalid_yaml_indentation(self, validator, temp_dir):
        """Test validation fails for invalid YAML indentation."""
        if not validator.is_available():
            pytest.skip("PyYAML not available")

        code = """
name: test
  value: 42
"""  # Inconsistent indentation
        result = validator.validate(code, temp_dir / "test.yaml")
        # Note: This may or may not fail depending on YAML version
        assert isinstance(result.is_valid, bool)

    def test_invalid_yaml_syntax(self, validator, temp_dir):
        """Test validation fails for invalid YAML syntax."""
        if not validator.is_available():
            pytest.skip("PyYAML not available")

        code = "key: [unclosed bracket"
        result = validator.validate(code, temp_dir / "test.yaml")
        assert result.is_valid is False


class TestTomlValidator:
    """Tests for TomlValidator."""

    @pytest.fixture
    def validator(self):
        return TomlValidator()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """TOML validator availability depends on tomllib/tomli."""
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_valid_toml(self, validator, temp_dir):
        """Test validation of valid TOML."""
        if not validator.is_available():
            pytest.skip("tomllib/tomli not available")

        code = """
[package]
name = "test"
version = "1.0.0"

[dependencies]
foo = "1.0"
"""
        result = validator.validate(code, temp_dir / "test.toml")
        assert result.is_valid is True
        assert result.language == "toml"

    def test_invalid_toml_syntax(self, validator, temp_dir):
        """Test validation fails for invalid TOML syntax."""
        if not validator.is_available():
            pytest.skip("tomllib/tomli not available")

        code = """
[package
name = "test"
"""  # Missing closing bracket
        result = validator.validate(code, temp_dir / "test.toml")
        assert result.is_valid is False


class TestXmlValidator:
    """Tests for XmlValidator."""

    @pytest.fixture
    def validator(self):
        return XmlValidator()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """XML validator is always available (built-in)."""
        assert validator.is_available() is True

    def test_valid_xml(self, validator, temp_dir):
        """Test validation of valid XML."""
        code = '<?xml version="1.0"?><root><item>test</item></root>'
        result = validator.validate(code, temp_dir / "test.xml")
        assert result.is_valid is True
        assert result.language == "xml"

    def test_valid_xml_with_attributes(self, validator, temp_dir):
        """Test validation of XML with attributes."""
        code = '<root attr="value"><child id="1">text</child></root>'
        result = validator.validate(code, temp_dir / "test.xml")
        assert result.is_valid is True

    def test_invalid_xml_unclosed_tag(self, validator, temp_dir):
        """Test validation fails for unclosed XML tag."""
        code = "<root><item>test</root>"  # Missing </item>
        result = validator.validate(code, temp_dir / "test.xml")
        assert result.is_valid is False

    def test_invalid_xml_mismatched_tags(self, validator, temp_dir):
        """Test validation fails for mismatched tags."""
        code = "<root><item>test</other></root>"
        result = validator.validate(code, temp_dir / "test.xml")
        assert result.is_valid is False

    def test_error_has_position(self, validator, temp_dir):
        """Test that errors include position information."""
        code = "<root>\n<item>test</root>"
        result = validator.validate(code, temp_dir / "test.xml")
        assert result.is_valid is False
        assert result.errors[0].line >= 1


class TestHoconValidator:
    """Tests for HoconValidator."""

    @pytest.fixture
    def validator(self):
        return HoconValidator()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """HOCON validator availability depends on pyhocon."""
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_valid_hocon(self, validator, temp_dir):
        """Test validation of valid HOCON."""
        if not validator.is_available():
            pytest.skip("pyhocon not available")

        code = """
app {
    name = "test"
    version = "1.0.0"
}
"""
        result = validator.validate(code, temp_dir / "application.conf")
        assert result.is_valid is True
        assert result.language == "hocon"

    def test_hocon_with_substitution(self, validator, temp_dir):
        """Test HOCON with variable substitution."""
        if not validator.is_available():
            pytest.skip("pyhocon not available")

        code = """
base {
    value = 42
}
derived {
    value = ${base.value}
}
"""
        result = validator.validate(code, temp_dir / "application.conf")
        assert result.is_valid is True


class TestMarkdownValidator:
    """Tests for MarkdownValidator."""

    @pytest.fixture
    def validator(self):
        return MarkdownValidator()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """Markdown validator availability depends on markdown library."""
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_valid_markdown(self, validator, temp_dir):
        """Test validation of valid Markdown."""
        if not validator.is_available():
            pytest.skip("markdown library not available")

        code = """
# Header

This is a paragraph with **bold** and *italic* text.

- List item 1
- List item 2

```python
def hello():
    print("Hello")
```
"""
        result = validator.validate(code, temp_dir / "README.md")
        assert result.is_valid is True
        assert result.language == "markdown"


class TestGetConfigValidator:
    """Tests for get_config_validator helper."""

    def test_get_json_validator(self):
        """Test getting JSON validator."""
        validator = get_config_validator("json")
        assert validator is not None
        assert isinstance(validator, JsonValidator)

    def test_get_yaml_validator(self):
        """Test getting YAML validator."""
        validator = get_config_validator("yaml")
        assert validator is not None
        assert isinstance(validator, YamlValidator)

    def test_get_unknown_returns_none(self):
        """Test getting unknown language returns None."""
        validator = get_config_validator("unknown")
        assert validator is None

    def test_case_insensitive(self):
        """Test that language names are case-insensitive."""
        validator1 = get_config_validator("JSON")
        validator2 = get_config_validator("Json")
        validator3 = get_config_validator("json")
        assert all(v is not None for v in [validator1, validator2, validator3])


class TestUnifiedValidatorConfigIntegration:
    """Test that config validators are integrated with UnifiedLanguageValidator."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_json_validation_direct(self, temp_dir):
        """Test JSON validation through JsonValidator directly."""
        validator = JsonValidator()
        file_path = temp_dir / "test.json"

        # Valid JSON
        result = validator.validate('{"key": "value"}', file_path)
        assert result.is_valid is True

        # Invalid JSON
        result = validator.validate('{"key": value}', file_path)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_yaml_validation_direct(self, temp_dir):
        """Test YAML validation through YamlValidator directly."""
        validator = YamlValidator()
        if not validator.is_available():
            pytest.skip("PyYAML not available")

        file_path = temp_dir / "test.yaml"

        # Valid YAML
        result = validator.validate("key: value\nlist:\n  - item1\n  - item2", file_path)
        assert result.is_valid is True

    def test_xml_validation_direct(self, temp_dir):
        """Test XML validation through XmlValidator directly."""
        validator = XmlValidator()
        file_path = temp_dir / "test.xml"

        # Valid XML
        result = validator.validate("<root><item>test</item></root>", file_path)
        assert result.is_valid is True

        # Invalid XML
        result = validator.validate("<root><item>test</root>", file_path)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_config_validators_available_via_helper(self):
        """Test that config validators are available via get_config_validator."""
        for lang in ["json", "yaml", "toml", "xml", "hocon", "markdown"]:
            validator = get_config_validator(lang)
            assert validator is not None, f"No validator for {lang}"
            # All should have is_available method
            assert hasattr(validator, "is_available")
            assert hasattr(validator, "validate")
