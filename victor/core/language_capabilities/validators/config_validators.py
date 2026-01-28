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

"""Native validators for configuration file formats.

Uses native Python libraries for better error messages and semantic validation:
- JSON: Built-in json module
- YAML: PyYAML with yaml.safe_load()
- TOML: tomli/tomllib (Python 3.11+)
- HOCON: pyhocon (optional)

All validators fall back to tree-sitter if native libraries aren't available.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..types import (
    CodeValidationResult,
    ValidationConfig,
    ValidationIssue,
    ValidationSeverity,
)
from .tree_sitter_validator import TreeSitterValidator

logger = logging.getLogger(__name__)


class JsonValidator:
    """
    JSON validator using Python's built-in json module.

    Provides detailed error messages with line and column information.
    """

    def __init__(self, tree_sitter_validator: Optional[TreeSitterValidator] = None):
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()

    def is_available(self) -> bool:
        """JSON module is always available (built-in)."""
        return True

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate JSON code using the built-in json module."""
        try:
            json.loads(code)
            return CodeValidationResult(
                is_valid=True,
                language="json",
                validators_used=["json.loads"],
            )
        except json.JSONDecodeError as e:
            result = CodeValidationResult(
                is_valid=False,
                language="json",
                validators_used=["json.loads"],
            )
            result.add_issue(
                ValidationIssue(
                    line=e.lineno,
                    column=e.colno,
                    message=e.msg,
                    severity=ValidationSeverity.ERROR,
                    source="json.loads",
                )
            )
            return result


class YamlValidator:
    """
    YAML validator using PyYAML with safe_load.

    Provides detailed error messages for syntax and structure errors.
    Falls back to tree-sitter if PyYAML is not installed.
    """

    def __init__(self, tree_sitter_validator: Optional[TreeSitterValidator] = None):
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()
        self._yaml_available = False
        try:
            import yaml

            self._yaml = yaml
            self._yaml_available = True
        except ImportError:
            logger.debug("PyYAML not available, falling back to tree-sitter")

    def is_available(self) -> bool:
        """Check if PyYAML is available."""
        return self._yaml_available

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate YAML code using PyYAML's safe_load."""
        if not self._yaml_available:
            return self._ts_validator.validate(code, file_path, "yaml", config)

        try:
            # Use safe_load to prevent code execution
            self._yaml.safe_load(code)
            return CodeValidationResult(
                is_valid=True,
                language="yaml",
                validators_used=["yaml.safe_load"],
            )
        except self._yaml.YAMLError as e:
            line = 1
            column = 0
            message = str(e)

            # Extract position information if available
            if hasattr(e, "problem_mark") and e.problem_mark:
                line = e.problem_mark.line + 1
                column = e.problem_mark.column
                message = e.problem if hasattr(e, "problem") else str(e)

            result = CodeValidationResult(
                is_valid=False,
                language="yaml",
                validators_used=["yaml.safe_load"],
            )
            result.add_issue(
                ValidationIssue(
                    line=line,
                    column=column,
                    message=message,
                    severity=ValidationSeverity.ERROR,
                    source="yaml.safe_load",
                )
            )
            return result


class TomlValidator:
    """
    TOML validator using tomli/tomllib.

    Uses tomllib (Python 3.11+) or tomli as fallback.
    Falls back to tree-sitter if neither is available.
    """

    def __init__(self, tree_sitter_validator: Optional[TreeSitterValidator] = None):
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()
        self._toml_available = False
        self._toml_module = None

        # Try tomllib (Python 3.11+) first
        try:
            import tomllib  # type: ignore[import-not-found]

            self._toml_module = tomllib
            self._toml_available = True
        except ImportError:
            # Fall back to tomli
            try:
                import tomli  # type: ignore[import-not-found]

                self._toml_module = tomli
                self._toml_available = True
            except ImportError:
                logger.debug("Neither tomllib nor tomli available, falling back to tree-sitter")

    def is_available(self) -> bool:
        """Check if tomllib/tomli is available."""
        return self._toml_available

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate TOML code using tomllib/tomli."""
        if not self._toml_available or self._toml_module is None:
            return self._ts_validator.validate(code, file_path, "toml", config)

        try:
            self._toml_module.loads(code)
            return CodeValidationResult(
                is_valid=True,
                language="toml",
                validators_used=["tomllib.loads"],
            )
        except self._toml_module.TOMLDecodeError as e:
            line = 1
            column = 0
            message = str(e)

            # Extract line number from error message if available
            if hasattr(e, "lineno"):
                line = e.lineno
            if hasattr(e, "colno"):
                column = e.colno

            result = CodeValidationResult(
                is_valid=False,
                language="toml",
                validators_used=["tomllib.loads"],
            )
            result.add_issue(
                ValidationIssue(
                    line=line,
                    column=column,
                    message=message,
                    severity=ValidationSeverity.ERROR,
                    source="tomllib.loads",
                )
            )
            return result


class HoconValidator:
    """
    HOCON validator using pyhocon.

    HOCON (Human-Optimized Config Object Notation) is used by Akka, Play Framework, etc.
    Falls back to tree-sitter if pyhocon is not installed.
    """

    def __init__(self, tree_sitter_validator: Optional[TreeSitterValidator] = None):
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()
        self._hocon_available = False

        try:
            from pyhocon import ConfigFactory

            self._config_factory = ConfigFactory
            self._hocon_available = True
        except ImportError:
            logger.debug("pyhocon not available, falling back to tree-sitter")

    def is_available(self) -> bool:
        """Check if pyhocon is available."""
        return self._hocon_available

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate HOCON code using pyhocon."""
        if not self._hocon_available:
            return self._ts_validator.validate(code, file_path, "hocon", config)

        try:
            self._config_factory.parse_string(code)
            return CodeValidationResult(
                is_valid=True,
                language="hocon",
                validators_used=["pyhocon.ConfigFactory"],
            )
        except Exception as e:
            # pyhocon raises various exceptions
            message = str(e)
            line = 1

            # Try to extract line number from error message
            import re

            line_match = re.search(r"line\s+(\d+)", message, re.IGNORECASE)
            if line_match:
                line = int(line_match.group(1))

            result = CodeValidationResult(
                is_valid=False,
                language="hocon",
                validators_used=["pyhocon.ConfigFactory"],
            )
            result.add_issue(
                ValidationIssue(
                    line=line,
                    column=0,
                    message=message,
                    severity=ValidationSeverity.ERROR,
                    source="pyhocon.ConfigFactory",
                )
            )
            return result


class XmlValidator:
    """
    XML validator using Python's built-in xml.etree.ElementTree.

    Validates well-formedness of XML documents.
    """

    def __init__(self, tree_sitter_validator: Optional[TreeSitterValidator] = None):
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()

    def is_available(self) -> bool:
        """XML module is always available (built-in)."""
        return True

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate XML code using ElementTree."""
        import xml.etree.ElementTree as ET

        try:
            ET.fromstring(code)
            return CodeValidationResult(
                is_valid=True,
                language="xml",
                validators_used=["xml.etree.ElementTree"],
            )
        except ET.ParseError as e:
            # ParseError includes position info
            line = 1
            column = 0
            if e.position:
                line, column = e.position

            result = CodeValidationResult(
                is_valid=False,
                language="xml",
                validators_used=["xml.etree.ElementTree"],
            )
            result.add_issue(
                ValidationIssue(
                    line=line,
                    column=column,
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    source="xml.etree.ElementTree",
                )
            )
            return result


class MarkdownValidator:
    """
    Markdown validator using markdown-it-py.

    Note: Markdown is very permissive, so this mainly checks for structural issues.
    Falls back to tree-sitter if markdown-it-py is not installed.
    """

    def __init__(self, tree_sitter_validator: Optional[TreeSitterValidator] = None):
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()
        self._md_available = False

        try:
            from markdown_it import MarkdownIt

            self._md = MarkdownIt()
            self._md_available = True
        except ImportError:
            # Try mistune as fallback
            try:
                import mistune

                self._mistune = mistune
                self._md_available = True
            except ImportError:
                logger.debug("No markdown library available, falling back to tree-sitter")

    def is_available(self) -> bool:
        """Check if a markdown library is available."""
        return self._md_available

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate Markdown code."""
        # Markdown is extremely permissive, so we mostly just check it parses
        if not self._md_available:
            return self._ts_validator.validate(code, file_path, "markdown", config)

        try:
            if hasattr(self, "_md"):
                # markdown-it-py
                self._md.parse(code)
            elif hasattr(self, "_mistune"):
                # mistune
                self._mistune.html(code)

            return CodeValidationResult(
                is_valid=True,
                language="markdown",
                validators_used=["markdown-it-py"],
            )
        except Exception as e:
            result = CodeValidationResult(
                is_valid=False,
                language="markdown",
                validators_used=["markdown-it-py"],
            )
            result.add_issue(
                ValidationIssue(
                    line=1,
                    column=0,
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    source="markdown-it-py",
                )
            )
            return result


# Mapping of language names to their validators
CONFIG_VALIDATORS = {
    "json": JsonValidator,
    "yaml": YamlValidator,
    "toml": TomlValidator,
    "hocon": HoconValidator,
    "xml": XmlValidator,
    "markdown": MarkdownValidator,
}


def get_config_validator(
    language: str, ts_validator: Optional[TreeSitterValidator] = None
) -> Optional[Union[JsonValidator, YamlValidator, TomlValidator, XmlValidator, MarkdownValidator]]:
    """
    Get the appropriate validator for a configuration file format.

    Args:
        language: The language/format name (e.g., "json", "yaml")
        ts_validator: Optional tree-sitter validator for fallback

    Returns:
        Validator instance or None if not supported
    """
    validator_class = CONFIG_VALIDATORS.get(language.lower())
    if validator_class:
        result = validator_class(ts_validator)
        return result  # type: ignore[return-value]
    return None
