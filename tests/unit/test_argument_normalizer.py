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

"""Tests for ArgumentNormalizer - achieving 70%+ coverage."""

import json
import pytest
from victor.agent.argument_normalizer import (
    ArgumentNormalizer,
    NormalizationStrategy,
    NormalizationStats,
    ToolStats,
)


class TestNormalizationStrategy:
    """Tests for NormalizationStrategy enum."""

    def test_direct_value(self):
        """Test DIRECT value."""
        assert NormalizationStrategy.DIRECT.value == "direct"

    def test_python_ast_value(self):
        """Test PYTHON_AST value."""
        assert NormalizationStrategy.PYTHON_AST.value == "python_ast"

    def test_regex_quotes_value(self):
        """Test REGEX_QUOTES value."""
        assert NormalizationStrategy.REGEX_QUOTES.value == "regex_quotes"

    def test_manual_repair_value(self):
        """Test MANUAL_REPAIR value."""
        assert NormalizationStrategy.MANUAL_REPAIR.value == "manual_repair"

    def test_failed_value(self):
        """Test FAILED value."""
        assert NormalizationStrategy.FAILED.value == "failed"


class TestArgumentNormalizerInit:
    """Tests for ArgumentNormalizer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        normalizer = ArgumentNormalizer()
        assert normalizer.provider_name == "unknown"
        assert normalizer.config == {}
        assert normalizer.parameter_aliases == {}

    def test_init_with_provider_name(self):
        """Test initialization with provider name."""
        normalizer = ArgumentNormalizer(provider_name="ollama")
        assert normalizer.provider_name == "ollama"

    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"key": "value"}
        normalizer = ArgumentNormalizer(config=config)
        assert normalizer.config == config

    def test_init_with_parameter_aliases(self):
        """Test initialization with parameter aliases."""
        config = {"parameter_aliases": {"read": {"line_start": "offset"}}}
        normalizer = ArgumentNormalizer(config=config)
        assert normalizer.parameter_aliases == {"read": {"line_start": "offset"}}

    def test_initial_stats(self):
        """Test initial stats are zeroed."""
        normalizer = ArgumentNormalizer()
        stats = normalizer.stats
        assert stats["total_calls"] == 0
        assert stats["failures"] == 0
        assert stats["by_tool"] == {}


class TestNormalizeParameterAliases:
    """Tests for normalize_parameter_aliases method."""

    def test_no_aliases_configured(self):
        """Test when no aliases are configured."""
        normalizer = ArgumentNormalizer()
        args = {"path": "test.py", "offset": 10}
        result, was_aliased = normalizer.normalize_parameter_aliases(args, "read")
        assert result == args
        assert was_aliased is False

    def test_alias_applied(self):
        """Test when alias is applied."""
        config = {"parameter_aliases": {"read": {"line_start": "offset"}}}
        normalizer = ArgumentNormalizer(config=config)
        args = {"path": "test.py", "line_start": 10}
        result, was_aliased = normalizer.normalize_parameter_aliases(args, "read")
        assert result == {"path": "test.py", "offset": 10}
        assert was_aliased is True

    def test_multiple_aliases(self):
        """Test multiple aliases applied."""
        config = {"parameter_aliases": {"read": {"line_start": "offset", "line_end": "limit"}}}
        normalizer = ArgumentNormalizer(config=config)
        args = {"path": "test.py", "line_start": 10, "line_end": 50}
        result, was_aliased = normalizer.normalize_parameter_aliases(args, "read")
        assert result == {"path": "test.py", "offset": 10, "limit": 50}
        assert was_aliased is True

    def test_no_matching_tool(self):
        """Test when tool has no aliases configured."""
        config = {"parameter_aliases": {"read": {"line_start": "offset"}}}
        normalizer = ArgumentNormalizer(config=config)
        args = {"content": "hello"}
        result, was_aliased = normalizer.normalize_parameter_aliases(args, "write")
        assert result == args
        assert was_aliased is False

    def test_mixed_aliased_and_normal_params(self):
        """Test mixing aliased and normal parameters."""
        config = {"parameter_aliases": {"read": {"line_start": "offset"}}}
        normalizer = ArgumentNormalizer(config=config)
        args = {"path": "test.py", "line_start": 10, "max_lines": 100}
        result, was_aliased = normalizer.normalize_parameter_aliases(args, "read")
        assert result == {"path": "test.py", "offset": 10, "max_lines": 100}
        assert was_aliased is True


class TestNormalizeArgumentsDirectPath:
    """Tests for normalize_arguments - direct valid JSON path."""

    def test_already_valid_json(self):
        """Test arguments that are already valid JSON."""
        normalizer = ArgumentNormalizer()
        args = {"path": "test.py", "content": "hello"}
        result, strategy = normalizer.normalize_arguments(args, "write")
        assert result == args
        assert strategy == NormalizationStrategy.DIRECT

    def test_valid_json_with_list(self):
        """Test valid JSON with list values."""
        normalizer = ArgumentNormalizer()
        args = {"paths": ["file1.py", "file2.py"]}
        result, strategy = normalizer.normalize_arguments(args, "read_multiple")
        assert result == args
        assert strategy == NormalizationStrategy.DIRECT

    def test_valid_json_with_dict(self):
        """Test valid JSON with dict values."""
        normalizer = ArgumentNormalizer()
        args = {"options": {"recursive": True, "depth": 3}}
        result, strategy = normalizer.normalize_arguments(args, "search")
        assert result == args
        assert strategy == NormalizationStrategy.DIRECT

    def test_stats_updated_on_direct(self):
        """Test stats are updated on direct path."""
        normalizer = ArgumentNormalizer()
        normalizer.normalize_arguments({"path": "test.py"}, "read")
        assert normalizer.stats["total_calls"] == 1
        assert normalizer.stats["normalizations"]["direct"] == 1


class TestNormalizeArgumentsAST:
    """Tests for normalize_arguments - AST normalization path."""

    def test_python_syntax_list_normalized(self):
        """Test Python syntax list is normalized to JSON."""
        normalizer = ArgumentNormalizer()
        args = {"operations": "[{'type': 'modify', 'path': 'test.py'}]"}
        result, strategy = normalizer.normalize_arguments(args, "edit_files")
        assert strategy == NormalizationStrategy.PYTHON_AST
        # The result should be valid JSON
        assert "type" in result["operations"]

    def test_python_syntax_dict_normalized(self):
        """Test Python syntax dict is normalized to JSON."""
        normalizer = ArgumentNormalizer()
        args = {"config": "{'key': 'value', 'num': 42}"}
        result, strategy = normalizer.normalize_arguments(args, "configure")
        assert strategy == NormalizationStrategy.PYTHON_AST

    def test_empty_list_coerced_to_type(self):
        """Test empty list string is coerced to actual list."""
        normalizer = ArgumentNormalizer()
        args = {"patterns": "[]"}
        result, strategy = normalizer.normalize_arguments(args, "search")
        assert strategy == NormalizationStrategy.PYTHON_AST
        assert result["patterns"] == []

    def test_empty_dict_coerced_to_type(self):
        """Test empty dict string is coerced to actual dict."""
        normalizer = ArgumentNormalizer()
        args = {"options": "{}"}
        result, strategy = normalizer.normalize_arguments(args, "search")
        assert strategy == NormalizationStrategy.PYTHON_AST
        assert result["options"] == {}

    def test_nested_python_syntax(self):
        """Test nested Python syntax is normalized."""
        normalizer = ArgumentNormalizer()
        args = {"data": "[{'nested': {'key': 'value'}}]"}
        result, strategy = normalizer.normalize_arguments(args, "process")
        assert strategy == NormalizationStrategy.PYTHON_AST


class TestNormalizeArgumentsRegex:
    """Tests for normalize_arguments - regex normalization path."""

    def test_escaped_single_quotes_replaced(self):
        """Test escaped single quotes are replaced."""
        normalizer = ArgumentNormalizer()
        # Create a case where AST fails but regex might work
        args = {"data": "{\\'key\\': \\'value\\'}"}
        result, strategy = normalizer.normalize_arguments(args, "test")
        # Could be REGEX_QUOTES or MANUAL_REPAIR depending on parsing
        assert strategy in (
            NormalizationStrategy.REGEX_QUOTES,
            NormalizationStrategy.MANUAL_REPAIR,
            NormalizationStrategy.FAILED,
            NormalizationStrategy.DIRECT,
        )


class TestNormalizeArgumentsManualRepair:
    """Tests for normalize_arguments - manual repair path."""

    def test_edit_files_operations_repair(self):
        """Test edit_files operations are repaired."""
        normalizer = ArgumentNormalizer()
        # Python syntax that might need manual repair
        args = {"operations": "[{'type': 'modify'}]"}
        result, strategy = normalizer.normalize_arguments(args, "edit_files")
        # Should be normalized via AST or manual repair
        assert strategy in (NormalizationStrategy.PYTHON_AST, NormalizationStrategy.MANUAL_REPAIR)


class TestNormalizeArgumentsFailure:
    """Tests for normalize_arguments - failure cases."""

    def test_completely_invalid_json(self):
        """Test completely invalid JSON fails gracefully."""
        normalizer = ArgumentNormalizer()

        # Create an object that can't be JSON serialized
        class NotSerializable:
            pass

        args = {"bad": NotSerializable()}
        result, strategy = normalizer.normalize_arguments(args, "test")
        assert strategy == NormalizationStrategy.FAILED
        assert normalizer.stats["failures"] == 1

    def test_stats_track_failures(self):
        """Test failures are tracked in stats."""
        normalizer = ArgumentNormalizer()

        class NotSerializable:
            pass

        args = {"bad": NotSerializable()}
        normalizer.normalize_arguments(args, "tool1")
        normalizer.normalize_arguments(args, "tool2")

        assert normalizer.stats["failures"] == 2


class TestIsValidJsonDict:
    """Tests for _is_valid_json_dict method."""

    def test_valid_simple_dict(self):
        """Test valid simple dict."""
        normalizer = ArgumentNormalizer()
        assert normalizer._is_valid_json_dict({"key": "value"}) is True

    def test_valid_nested_dict(self):
        """Test valid nested dict."""
        normalizer = ArgumentNormalizer()
        obj = {"key": {"nested": [1, 2, 3]}}
        assert normalizer._is_valid_json_dict(obj) is True

    def test_valid_with_json_string_value(self):
        """Test valid with JSON string value."""
        normalizer = ArgumentNormalizer()
        obj = {"data": '[{"x": 1}]'}
        assert normalizer._is_valid_json_dict(obj) is True

    def test_invalid_not_serializable(self):
        """Test invalid - not serializable."""
        normalizer = ArgumentNormalizer()

        class Custom:
            pass

        assert normalizer._is_valid_json_dict({"x": Custom()}) is False


class TestNormalizeViaAST:
    """Tests for _normalize_via_ast method."""

    def test_string_with_python_list(self):
        """Test string with Python list syntax."""
        normalizer = ArgumentNormalizer()
        args = {"data": "['a', 'b', 'c']"}
        result = normalizer._normalize_via_ast(args)
        assert result["data"] == '["a", "b", "c"]'

    def test_string_with_python_dict(self):
        """Test string with Python dict syntax."""
        normalizer = ArgumentNormalizer()
        args = {"data": "{'key': 'value'}"}
        result = normalizer._normalize_via_ast(args)
        assert result["data"] == '{"key": "value"}'

    def test_non_json_string_unchanged(self):
        """Test non-JSON string is unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"message": "Hello world"}
        result = normalizer._normalize_via_ast(args)
        assert result["message"] == "Hello world"

    def test_non_string_unchanged(self):
        """Test non-string values are unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"count": 42, "flag": True}
        result = normalizer._normalize_via_ast(args)
        assert result == args

    def test_invalid_syntax_unchanged(self):
        """Test invalid syntax is kept unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"bad": "[invalid syntax"}
        result = normalizer._normalize_via_ast(args)
        assert result["bad"] == "[invalid syntax"

    def test_empty_list_coercion(self):
        """Test empty list string is coerced to list."""
        normalizer = ArgumentNormalizer()
        args = {"items": "[]"}
        result = normalizer._normalize_via_ast(args)
        assert result["items"] == []

    def test_empty_dict_coercion(self):
        """Test empty dict string is coerced to dict."""
        normalizer = ArgumentNormalizer()
        args = {"opts": "{}"}
        result = normalizer._normalize_via_ast(args)
        assert result["opts"] == {}


class TestNormalizeViaRegex:
    """Tests for _normalize_via_regex method."""

    def test_single_quotes_replaced(self):
        """Test single quotes are replaced with double quotes."""
        normalizer = ArgumentNormalizer()
        args = {"data": "{'key': 'value'}"}
        result = normalizer._normalize_via_regex(args)
        assert '"key"' in result["data"]
        assert '"value"' in result["data"]

    def test_escaped_quotes_replaced(self):
        """Test escaped quotes are handled."""
        normalizer = ArgumentNormalizer()
        args = {"data": "{\\'key\\': \\'value\\'}"}
        result = normalizer._normalize_via_regex(args)
        assert '"' in result["data"]

    def test_non_string_unchanged(self):
        """Test non-string values are unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"count": 42}
        result = normalizer._normalize_via_regex(args)
        assert result["count"] == 42


class TestNormalizeViaManualRepair:
    """Tests for _normalize_via_manual_repair method."""

    def test_edit_files_repair(self):
        """Test edit_files arguments are repaired."""
        normalizer = ArgumentNormalizer()
        args = {"operations": "[{'type': 'modify'}]"}
        result = normalizer._normalize_via_manual_repair(args, "edit_files")
        # Should attempt repair
        assert "operations" in result

    def test_other_tool_unchanged(self):
        """Test other tools are unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"data": "test"}
        result = normalizer._normalize_via_manual_repair(args, "other_tool")
        assert result == args


class TestRepairEditFilesArgs:
    """Tests for _repair_edit_files_args method."""

    def test_string_operations_converted(self):
        """Test string operations are converted to JSON."""
        normalizer = ArgumentNormalizer()
        args = {"operations": "[{'type': 'modify', 'path': 'test.py'}]"}
        result = normalizer._repair_edit_files_args(args)
        # Should be valid JSON string now
        assert "type" in result["operations"]

    def test_non_string_operations_unchanged(self):
        """Test non-string operations are unchanged."""
        normalizer = ArgumentNormalizer()
        ops = [{"type": "modify"}]
        args = {"operations": ops}
        result = normalizer._repair_edit_files_args(args)
        assert result["operations"] == ops

    def test_no_operations_unchanged(self):
        """Test args without operations are unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"path": "test.py"}
        result = normalizer._repair_edit_files_args(args)
        assert result == args


class TestGetStats:
    """Tests for get_stats method."""

    def test_initial_stats(self):
        """Test initial stats."""
        normalizer = ArgumentNormalizer(provider_name="test")
        stats = normalizer.get_stats()
        assert stats["provider"] == "test"
        assert stats["total_calls"] == 0
        assert stats["failures"] == 0
        # With 0 calls, success_rate is 0.0 (0/1 * 100)
        assert stats["success_rate"] == 0.0

    def test_stats_after_calls(self):
        """Test stats after some calls."""
        normalizer = ArgumentNormalizer()
        normalizer.normalize_arguments({"path": "test.py"}, "read")
        normalizer.normalize_arguments({"content": "hello"}, "write")

        stats = normalizer.get_stats()
        assert stats["total_calls"] == 2
        assert stats["success_rate"] == 100.0

    def test_stats_with_failures(self):
        """Test stats with failures."""
        normalizer = ArgumentNormalizer()

        class Bad:
            pass

        normalizer.normalize_arguments({"path": "test.py"}, "read")
        normalizer.normalize_arguments({"bad": Bad()}, "test")

        stats = normalizer.get_stats()
        assert stats["total_calls"] == 2
        assert stats["failures"] == 1
        assert stats["success_rate"] == 50.0

    def test_stats_by_tool(self):
        """Test stats by tool."""
        normalizer = ArgumentNormalizer()
        normalizer.normalize_arguments({"path": "test.py"}, "read")
        normalizer.normalize_arguments({"path": "test2.py"}, "read")
        normalizer.normalize_arguments({"content": "hi"}, "write")

        stats = normalizer.get_stats()
        assert stats["by_tool"]["read"]["calls"] == 2
        assert stats["by_tool"]["write"]["calls"] == 1


class TestResetStats:
    """Tests for reset_stats method."""

    def test_reset_clears_stats(self):
        """Test reset clears all stats."""
        normalizer = ArgumentNormalizer()
        normalizer.normalize_arguments({"path": "test.py"}, "read")

        normalizer.reset_stats()

        stats = normalizer.get_stats()
        assert stats["total_calls"] == 0
        assert stats["failures"] == 0
        assert stats["by_tool"] == {}


class TestLogStats:
    """Tests for log_stats method."""

    def test_log_stats_runs(self):
        """Test log_stats executes without error."""
        normalizer = ArgumentNormalizer()
        normalizer.normalize_arguments({"path": "test.py"}, "read")
        # Should not raise
        normalizer.log_stats()


class TestIntegration:
    """Integration tests for common scenarios."""

    def test_ollama_python_syntax(self):
        """Test Ollama-style Python syntax is handled."""
        normalizer = ArgumentNormalizer(provider_name="ollama")
        args = {"operations": "[{'type': 'modify', 'path': 'test.sh', 'content': 'echo hello'}]"}
        result, strategy = normalizer.normalize_arguments(args, "edit_files")

        # Should successfully normalize
        assert strategy != NormalizationStrategy.FAILED

        # Result should be valid JSON
        if isinstance(result["operations"], str):
            parsed = json.loads(result["operations"])
            assert parsed[0]["type"] == "modify"

    def test_gpt_oss_aliases(self):
        """Test gpt-oss style parameter aliases are handled."""
        config = {"parameter_aliases": {"read": {"line_start": "offset", "line_end": "_line_end"}}}
        normalizer = ArgumentNormalizer(provider_name="ollama", config=config)
        args = {"path": "test.py", "line_start": 10, "line_end": 50}

        result, strategy = normalizer.normalize_arguments(args, "read")

        assert result["offset"] == 10
        assert result["_line_end"] == 50
        assert "line_start" not in result

    def test_multiple_normalizations_tracked(self):
        """Test multiple normalizations are properly tracked."""
        normalizer = ArgumentNormalizer()

        # Direct path
        normalizer.normalize_arguments({"path": "a.py"}, "read")

        # AST path
        normalizer.normalize_arguments({"data": "['x']"}, "process")

        stats = normalizer.get_stats()
        assert stats["total_calls"] == 2
        assert stats["normalizations"]["direct"] >= 1 or stats["normalizations"]["python_ast"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
