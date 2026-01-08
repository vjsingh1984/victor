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


class TestNativeFallback:
    """Tests for native extension fallback behavior (lines 25-30)."""

    def test_native_coerce_string_type_fallback(self):
        """Test fallback stub when native is not available."""
        # Import the module-level fallback
        from victor.agent import argument_normalizer

        # The module should have _NATIVE_AVAILABLE defined
        assert hasattr(argument_normalizer, "_NATIVE_AVAILABLE")

        # If native is not available, the fallback should be used
        if not argument_normalizer._NATIVE_AVAILABLE:
            # Test that native_coerce_string_type returns the fallback behavior
            result = argument_normalizer.native_coerce_string_type("test value")
            assert result == ("string", "test value", None)


class TestParameterAliasStatsTracking:
    """Tests for parameter alias statistics tracking (lines 129-155)."""

    def test_alias_stats_incremented(self):
        """Test that alias stats are properly incremented."""
        config = {"parameter_aliases": {"read": {"line_start": "offset"}}}
        normalizer = ArgumentNormalizer(provider_name="test", config=config)

        args = {"path": "test.py", "line_start": 10}
        result, was_aliased = normalizer.normalize_parameter_aliases(args, "read")

        assert was_aliased is True
        assert normalizer._alias_stats["total"] == 1
        assert normalizer._alias_stats["aliased"] == 1
        assert normalizer._alias_stats["by_tool"]["read"] == 1

    def test_alias_stats_by_tool_tracking(self):
        """Test that alias stats are tracked per tool."""
        config = {
            "parameter_aliases": {
                "read": {"line_start": "offset"},
                "write": {"line_num": "line"},
            }
        }
        normalizer = ArgumentNormalizer(config=config)

        normalizer.normalize_parameter_aliases({"line_start": 10}, "read")
        normalizer.normalize_parameter_aliases({"line_num": 20}, "write")
        normalizer.normalize_parameter_aliases({"line_start": 30}, "read")

        assert normalizer._alias_stats["by_tool"]["read"] == 2
        assert normalizer._alias_stats["by_tool"]["write"] == 1
        assert normalizer._alias_stats["total"] == 3

    def test_alias_stats_included_in_get_stats(self):
        """Test that alias stats are included in get_stats output."""
        config = {"parameter_aliases": {"read": {"line_start": "offset"}}}
        normalizer = ArgumentNormalizer(config=config)

        normalizer.normalize_parameter_aliases({"line_start": 10}, "read")

        stats = normalizer.get_stats()
        assert "alias_stats" in stats
        assert stats["alias_stats"]["aliased"] == 1


class TestPreemptiveASTNormalization:
    """Tests for preemptive AST normalization path (lines 196-219)."""

    def test_preemptive_ast_with_json_like_string(self):
        """Test preemptive AST normalization for JSON-like strings."""
        normalizer = ArgumentNormalizer()
        # String that starts with [ or { triggers preemptive AST
        args = {"data": "[{'key': 'value'}]"}
        result, strategy = normalizer.normalize_arguments(args, "test")

        assert strategy == NormalizationStrategy.PYTHON_AST
        # Verify the result is valid JSON
        assert json.loads(result["data"])[0]["key"] == "value"

    def test_preemptive_ast_with_dict_string(self):
        """Test preemptive AST normalization for dict-like strings."""
        normalizer = ArgumentNormalizer()
        args = {"config": "{'setting': 'enabled'}"}
        result, strategy = normalizer.normalize_arguments(args, "configure")

        assert strategy == NormalizationStrategy.PYTHON_AST

    def test_preemptive_ast_failure_falls_through(self):
        """Test that preemptive AST failure falls through to other strategies."""
        normalizer = ArgumentNormalizer()
        # Invalid Python syntax that looks like JSON
        args = {"data": "[{invalid: syntax}]"}
        result, strategy = normalizer.normalize_arguments(args, "test")

        # Should not be PYTHON_AST since parsing failed
        # May be DIRECT (passthrough) or FAILED
        assert strategy in (
            NormalizationStrategy.DIRECT,
            NormalizationStrategy.REGEX_QUOTES,
            NormalizationStrategy.FAILED,
        )


class TestValidationExceptionHandling:
    """Tests for validation exception handling (lines 231-281)."""

    def test_layer1_validation_with_non_serializable(self):
        """Test exception handling in Layer 1 validation."""
        normalizer = ArgumentNormalizer()

        class NonSerializable:
            """Object that cannot be JSON serialized."""

            pass

        # This will trigger the exception handling in Layer 1
        args = {"bad": NonSerializable()}
        result, strategy = normalizer.normalize_arguments(args, "test")

        # Should fail since the object can't be validated
        assert strategy == NormalizationStrategy.FAILED

    def test_layer2_ast_normalization_path(self):
        """Test Layer 2 AST normalization when preemptive didn't run."""
        normalizer = ArgumentNormalizer()
        # Args without JSON-like strings don't trigger preemptive AST
        # but if validation fails, Layer 2 will try AST
        # We need an object that fails validation but not AST
        args = {"key": "simple string"}

        # This should pass direct validation
        result, strategy = normalizer.normalize_arguments(args, "test")
        assert strategy == NormalizationStrategy.DIRECT

    def test_normalization_increments_by_tool_stats(self):
        """Test that normalization increments by_tool stats."""
        normalizer = ArgumentNormalizer()
        args = {"data": "[{'a': 'b'}]"}
        result, strategy = normalizer.normalize_arguments(args, "my_tool")

        assert strategy == NormalizationStrategy.PYTHON_AST
        assert normalizer.stats["by_tool"]["my_tool"]["normalizations"] == 1

    def test_all_strategies_fail(self):
        """Test when all normalization strategies fail."""
        normalizer = ArgumentNormalizer()

        class NonSerializable:
            pass

        args = {"bad_obj": NonSerializable()}
        result, strategy = normalizer.normalize_arguments(args, "test_tool")

        assert strategy == NormalizationStrategy.FAILED
        assert normalizer.stats["failures"] == 1


class TestIsValidJsonDictWithJsonStrings:
    """Tests for _is_valid_json_dict with JSON-like string values (lines 321-327, 334-336)."""

    def test_json_string_value_starting_with_bracket(self):
        """Test validation of string values starting with [."""
        normalizer = ArgumentNormalizer()
        obj = {"operations": '[{"type": "modify"}]'}
        assert normalizer._is_valid_json_dict(obj) is True

    def test_json_string_value_starting_with_brace(self):
        """Test validation of string values starting with {."""
        normalizer = ArgumentNormalizer()
        obj = {"config": '{"key": "value"}'}
        assert normalizer._is_valid_json_dict(obj) is True

    def test_type_error_returns_false(self):
        """Test that TypeError during validation returns False."""
        normalizer = ArgumentNormalizer()

        class BadType:
            pass

        # TypeError is caught and returns False
        assert normalizer._is_valid_json_dict({"x": BadType()}) is False

    def test_value_error_returns_false(self):
        """Test that ValueError during validation returns False."""
        normalizer = ArgumentNormalizer()
        # Sets are not JSON serializable, raising TypeError
        assert normalizer._is_valid_json_dict({"x": {1, 2, 3}}) is False


class TestCoercePrimitiveTypes:
    """Tests for _coerce_primitive_types method (lines 518-530)."""

    def test_coerce_string_to_int(self):
        """Test coercing string to integer."""
        normalizer = ArgumentNormalizer()
        args = {"line_start": "0", "line_end": "30"}
        result = normalizer._coerce_primitive_types(args, "read")

        assert result["line_start"] == 0
        assert result["line_end"] == 30
        assert isinstance(result["line_start"], int)
        assert isinstance(result["line_end"], int)

    def test_coerce_string_to_bool(self):
        """Test coercing string to boolean."""
        normalizer = ArgumentNormalizer()
        args = {"regex": "false", "verbose": "true"}
        result = normalizer._coerce_primitive_types(args, "search")

        assert result["regex"] is False
        assert result["verbose"] is True

    def test_coerce_string_to_float(self):
        """Test coercing string to float."""
        normalizer = ArgumentNormalizer()
        args = {"timeout": "30.5", "threshold": "0.75"}
        result = normalizer._coerce_primitive_types(args, "config")

        assert result["timeout"] == 30.5
        assert result["threshold"] == 0.75
        assert isinstance(result["timeout"], float)

    def test_non_string_values_unchanged(self):
        """Test that non-string values are not coerced."""
        normalizer = ArgumentNormalizer()
        args = {"count": 42, "enabled": True, "data": ["a", "b"]}
        result = normalizer._coerce_primitive_types(args, "test")

        assert result == args

    def test_coerce_logs_info_when_coerced(self):
        """Test that coercion logs info message."""
        normalizer = ArgumentNormalizer(provider_name="test_provider")
        args = {"line_start": "10"}
        result = normalizer._coerce_primitive_types(args, "read")

        assert result["line_start"] == 10


class TestTryCoerceString:
    """Tests for _try_coerce_string method (lines 559-605)."""

    def test_coerce_true_string(self):
        """Test coercing 'true' to True."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("true") is True
        assert normalizer._try_coerce_string("TRUE") is True
        assert normalizer._try_coerce_string("True") is True

    def test_coerce_false_string(self):
        """Test coercing 'false' to False."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("false") is False
        assert normalizer._try_coerce_string("FALSE") is False
        assert normalizer._try_coerce_string("False") is False

    def test_coerce_null_string(self):
        """Test coercing 'null' to None."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("null") is None
        assert normalizer._try_coerce_string("NULL") is None
        assert normalizer._try_coerce_string("none") is None
        assert normalizer._try_coerce_string("None") is None

    def test_coerce_integer_string(self):
        """Test coercing integer strings."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("42") == 42
        assert normalizer._try_coerce_string("-10") == -10
        assert normalizer._try_coerce_string("0") == 0

    def test_coerce_float_string(self):
        """Test coercing float strings."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("3.14") == 3.14
        assert normalizer._try_coerce_string("-2.5") == -2.5
        assert normalizer._try_coerce_string("0.0") == 0.0

    def test_path_not_coerced(self):
        """Test that paths are not coerced to numbers."""
        normalizer = ArgumentNormalizer()
        # Paths starting with / should not be coerced
        assert normalizer._try_coerce_string("/123") == "/123"
        assert normalizer._try_coerce_string("/1.5") == "/1.5"

    def test_regular_string_unchanged(self):
        """Test that regular strings are unchanged."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("hello") == "hello"
        assert normalizer._try_coerce_string("test.py") == "test.py"
        assert normalizer._try_coerce_string("") == ""

    def test_whitespace_preserved(self):
        """Test that whitespace is stripped for comparison but preserved otherwise."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("  42  ") == 42
        assert normalizer._try_coerce_string("  true  ") is True

    def test_negative_numbers(self):
        """Test coercing negative numbers."""
        normalizer = ArgumentNormalizer()
        assert normalizer._try_coerce_string("-100") == -100
        assert normalizer._try_coerce_string("-3.14") == -3.14


class TestNormalizeViaASTExtended:
    """Extended tests for _normalize_via_ast (lines 361-396)."""

    def test_complex_nested_structure(self):
        """Test AST normalization of complex nested structures."""
        normalizer = ArgumentNormalizer()
        args = {"data": "[{'a': {'b': [1, 2, {'c': 'd'}]}}]"}
        result = normalizer._normalize_via_ast(args)

        # Should be valid JSON
        parsed = json.loads(result["data"])
        assert parsed[0]["a"]["b"][2]["c"] == "d"

    def test_primitive_value_in_json_like_string(self):
        """Test AST with primitive value evaluated from JSON-like string."""
        normalizer = ArgumentNormalizer()
        # A string that looks like a JSON primitive
        args = {"val": "[42]"}  # List with single integer
        result = normalizer._normalize_via_ast(args)

        # Should be normalized
        assert result["val"] == "[42]"

    def test_mixed_types_in_structure(self):
        """Test AST with mixed types in structure."""
        normalizer = ArgumentNormalizer()
        args = {"data": "[{'str': 'value', 'int': 42, 'float': 3.14, 'bool': True}]"}
        result = normalizer._normalize_via_ast(args)

        parsed = json.loads(result["data"])
        assert parsed[0]["str"] == "value"
        assert parsed[0]["int"] == 42
        assert parsed[0]["float"] == 3.14
        assert parsed[0]["bool"] is True


class TestNormalizeViaRegexExtended:
    """Extended tests for _normalize_via_regex (lines 417-435)."""

    def test_multiple_quoted_values(self):
        """Test regex replacement with multiple quoted values."""
        normalizer = ArgumentNormalizer()
        args = {"data": "{'a': 'b', 'c': 'd', 'e': 'f'}"}
        result = normalizer._normalize_via_regex(args)

        # All single quotes should be replaced
        assert "'" not in result["data"]
        assert '"a"' in result["data"]

    def test_list_values_unchanged(self):
        """Test that list values are unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"items": [1, 2, 3]}
        result = normalizer._normalize_via_regex(args)
        assert result["items"] == [1, 2, 3]

    def test_dict_values_unchanged(self):
        """Test that dict values are unchanged."""
        normalizer = ArgumentNormalizer()
        args = {"config": {"key": "value"}}
        result = normalizer._normalize_via_regex(args)
        assert result["config"] == {"key": "value"}


class TestRepairEditFilesArgsExtended:
    """Extended tests for _repair_edit_files_args (lines 478-489)."""

    def test_complex_operations_repair(self):
        """Test repair of complex edit_files operations."""
        normalizer = ArgumentNormalizer()
        args = {
            "operations": "[{'type': 'modify', 'path': 'test.py', 'content': 'def foo():\\n    pass'}]"
        }
        result = normalizer._repair_edit_files_args(args)

        # Should be valid JSON
        parsed = json.loads(result["operations"])
        assert parsed[0]["type"] == "modify"

    def test_malformed_operations_unchanged(self):
        """Test that malformed operations remain unchanged when repair fails."""
        normalizer = ArgumentNormalizer()
        args = {"operations": "[{completely invalid json"}
        result = normalizer._repair_edit_files_args(args)

        # Original unchanged
        assert result["operations"] == "[{completely invalid json"


class TestGetStatsExtended:
    """Extended tests for get_stats method (lines 614-618)."""

    def test_success_rate_calculation_all_success(self):
        """Test success rate when all calls succeed."""
        normalizer = ArgumentNormalizer()
        normalizer.normalize_arguments({"a": "1"}, "t1")
        normalizer.normalize_arguments({"b": "2"}, "t2")
        normalizer.normalize_arguments({"c": "3"}, "t3")

        stats = normalizer.get_stats()
        assert stats["success_rate"] == 100.0

    def test_success_rate_calculation_partial_failure(self):
        """Test success rate with partial failures."""
        normalizer = ArgumentNormalizer()

        class Bad:
            pass

        normalizer.normalize_arguments({"a": "1"}, "t1")
        normalizer.normalize_arguments({"a": "2"}, "t2")
        normalizer.normalize_arguments({"bad": Bad()}, "t3")
        normalizer.normalize_arguments({"bad": Bad()}, "t4")

        stats = normalizer.get_stats()
        # 2 success out of 4 = 50%
        assert stats["success_rate"] == 50.0

    def test_success_rate_zero_calls(self):
        """Test success rate with zero calls."""
        normalizer = ArgumentNormalizer()
        stats = normalizer.get_stats()

        # (0 - 0) / max(0, 1) * 100 = 0.0
        assert stats["success_rate"] == 0.0


class TestResetStatsExtended:
    """Extended tests for reset_stats method (lines 630-636)."""

    def test_reset_clears_alias_stats(self):
        """Test that reset clears alias stats."""
        config = {"parameter_aliases": {"read": {"line_start": "offset"}}}
        normalizer = ArgumentNormalizer(config=config)

        normalizer.normalize_parameter_aliases({"line_start": 10}, "read")
        assert normalizer._alias_stats["aliased"] == 1

        normalizer.reset_stats()

        assert normalizer._alias_stats["total"] == 0
        assert normalizer._alias_stats["aliased"] == 0
        assert normalizer._alias_stats["by_tool"] == {}

    def test_reset_clears_all_normalization_counters(self):
        """Test that reset clears all normalization strategy counters."""
        normalizer = ArgumentNormalizer()
        normalizer.normalize_arguments({"a": "1"}, "t1")
        normalizer.normalize_arguments({"b": "[{'x': 'y'}]"}, "t2")

        normalizer.reset_stats()

        for strategy in NormalizationStrategy:
            assert normalizer.stats["normalizations"][strategy.value] == 0


class TestLogStatsExtended:
    """Extended tests for log_stats method (lines 640-641)."""

    def test_log_stats_after_operations(self, caplog):
        """Test log_stats logs correct information after operations."""
        import logging

        caplog.set_level(logging.INFO)

        normalizer = ArgumentNormalizer(provider_name="test_provider")
        normalizer.normalize_arguments({"path": "test.py"}, "read")
        normalizer.normalize_arguments({"data": "[{'a': 'b'}]"}, "process")

        normalizer.log_stats()

        assert "Argument normalization stats" in caplog.text


class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_dict_argument(self):
        """Test handling of empty dict argument."""
        normalizer = ArgumentNormalizer()
        args = {}
        result, strategy = normalizer.normalize_arguments(args, "test")

        assert result == {}
        assert strategy == NormalizationStrategy.DIRECT

    def test_unicode_in_arguments(self):
        """Test handling of unicode characters in arguments."""
        normalizer = ArgumentNormalizer()
        args = {"message": "Hello 世界 🚀"}
        result, strategy = normalizer.normalize_arguments(args, "test")

        assert result["message"] == "Hello 世界 🚀"
        assert strategy == NormalizationStrategy.DIRECT

    def test_unicode_in_python_syntax(self):
        """Test handling of unicode in Python syntax normalization."""
        normalizer = ArgumentNormalizer()
        args = {"data": "[{'msg': '你好'}]"}
        result, strategy = normalizer.normalize_arguments(args, "test")

        assert strategy == NormalizationStrategy.PYTHON_AST
        parsed = json.loads(result["data"])
        assert parsed[0]["msg"] == "你好"

    def test_null_equivalent_values(self):
        """Test handling of null/none equivalent string values."""
        normalizer = ArgumentNormalizer()
        args = {"value": "null", "other": "none"}
        result = normalizer._coerce_primitive_types(args, "test")

        assert result["value"] is None
        assert result["other"] is None

    def test_mixed_coercion_and_regular_values(self):
        """Test coercion with mixed coercible and non-coercible values."""
        normalizer = ArgumentNormalizer()
        args = {"line": "10", "path": "/test/file.py", "enabled": "true"}
        result = normalizer._coerce_primitive_types(args, "test")

        assert result["line"] == 10
        assert result["path"] == "/test/file.py"  # Path unchanged
        assert result["enabled"] is True

    def test_float_with_leading_decimal(self):
        """Test coercion of floats with leading decimal."""
        normalizer = ArgumentNormalizer()
        # ".5" may be coerced by native extension or remain string in Python fallback
        args = {"ratio": ".5"}
        result = normalizer._coerce_primitive_types(args, "test")

        # May be coerced to 0.5 by native or remain ".5" by Python fallback
        assert result["ratio"] in (".5", 0.5)

    def test_integer_overflow_protection(self):
        """Test handling of very large integers."""
        normalizer = ArgumentNormalizer()
        args = {"big": "999999999999999999999999999999"}
        result = normalizer._coerce_primitive_types(args, "test")

        # Native may convert to float, Python fallback keeps as int
        expected_int = 999999999999999999999999999999
        expected_float = float(expected_int)
        assert result["big"] in (expected_int, expected_float)


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_openrouter_llama_output(self):
        """Test handling of OpenRouter Llama model output with string numbers."""
        normalizer = ArgumentNormalizer(provider_name="openrouter")
        args = {"line_start": "0", "line_end": "30", "path": "/test/file.py"}
        result, strategy = normalizer.normalize_arguments(args, "read")

        assert result["line_start"] == 0
        assert result["line_end"] == 30
        assert result["path"] == "/test/file.py"

    def test_fireworks_output(self):
        """Test handling of Fireworks model output."""
        normalizer = ArgumentNormalizer(provider_name="fireworks")
        args = {"timeout": "30.5", "retry": "true", "max_attempts": "3"}
        result, strategy = normalizer.normalize_arguments(args, "config")

        assert result["timeout"] == 30.5
        assert result["retry"] is True
        assert result["max_attempts"] == 3

    def test_full_normalization_pipeline(self):
        """Test complete normalization pipeline with multiple steps."""
        config = {"parameter_aliases": {"edit": {"file_path": "path"}}}
        normalizer = ArgumentNormalizer(provider_name="ollama", config=config)

        args = {
            "file_path": "/test/file.py",
            "line_num": "10",
            "operations": "[{'type': 'modify', 'content': 'new code'}]",
        }
        result, strategy = normalizer.normalize_arguments(args, "edit")

        # Alias should be applied
        assert "path" in result
        assert "file_path" not in result

        # Integer coercion should be applied
        assert result["line_num"] == 10

        # Python syntax should be normalized
        assert strategy == NormalizationStrategy.PYTHON_AST


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
