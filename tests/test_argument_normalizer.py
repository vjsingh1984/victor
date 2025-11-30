"""
Unit tests for argument normalization system.

Tests the multi-layer normalization pipeline that handles malformed
tool arguments (e.g., Python syntax vs JSON syntax).
"""

import pytest
import json
from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy


class TestArgumentNormalizer:
    """Test suite for ArgumentNormalizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.normalizer = ArgumentNormalizer(provider_name="test_provider")

    def test_valid_json_fast_path(self):
        """Test that valid JSON passes through unchanged (fast path)."""
        # Valid JSON dict
        valid_args = {
            "operations": '[{"type": "modify", "path": "file.sh", "content": "echo hello"}]',
            "path": "/test/file.txt",
        }

        normalized, strategy = self.normalizer.normalize_arguments(valid_args, "edit_files")

        assert normalized == valid_args
        assert strategy == NormalizationStrategy.DIRECT
        assert self.normalizer.stats["normalizations"]["direct"] == 1

    def test_python_syntax_ast_normalization(self):
        """Test normalization of Python dict syntax via AST."""
        # Python dict syntax (single quotes)
        python_args = {
            "operations": "[{'type': 'modify', 'path': 'fibonacci.sh', 'content': '#!/bin/bash\\necho test'}]"
        }

        normalized, strategy = self.normalizer.normalize_arguments(python_args, "edit_files")

        # Should be normalized to valid JSON
        assert strategy == NormalizationStrategy.PYTHON_AST

        # Verify the normalized value is valid JSON
        operations_json = json.loads(normalized["operations"])
        assert isinstance(operations_json, list)
        assert operations_json[0]["type"] == "modify"
        assert operations_json[0]["path"] == "fibonacci.sh"

    def test_escaped_quotes_ast_normalization(self):
        """Test normalization of escaped single quotes."""
        # Escaped single quotes (from model output)
        escaped_args = {"operations": "[{\\'type\\': \\'modify\\', \\'path\\': \\'test.sh\\'}]"}

        normalized, strategy = self.normalizer.normalize_arguments(escaped_args, "edit_files")

        # Should be normalized (either AST or regex)
        assert strategy in [NormalizationStrategy.PYTHON_AST, NormalizationStrategy.REGEX_QUOTES]

        # Verify the result
        assert "operations" in normalized

    def test_mixed_valid_and_invalid_args(self):
        """Test that valid args pass through while invalid ones are normalized."""
        mixed_args = {
            "valid_field": "already valid",
            "invalid_field": "[{'key': 'value'}]",  # Python syntax
        }

        normalized, strategy = self.normalizer.normalize_arguments(mixed_args, "test_tool")

        # Invalid field should be normalized
        assert strategy == NormalizationStrategy.PYTHON_AST
        assert normalized["valid_field"] == "already valid"

        # Invalid field should now be valid JSON
        parsed = json.loads(normalized["invalid_field"])
        assert parsed[0]["key"] == "value"

    def test_nested_structures(self):
        """Test normalization of nested dicts and lists."""
        nested_args = {
            "operations": "[{'type': 'modify', 'metadata': {'author': 'test', 'version': 1}}]"
        }

        normalized, strategy = self.normalizer.normalize_arguments(nested_args, "edit_files")

        assert strategy == NormalizationStrategy.PYTHON_AST

        # Verify nested structure is preserved
        operations = json.loads(normalized["operations"])
        assert operations[0]["metadata"]["author"] == "test"
        assert operations[0]["metadata"]["version"] == 1

    def test_non_string_args_unchanged(self):
        """Test that non-string arguments pass through unchanged."""
        numeric_args = {
            "count": 42,
            "enabled": True,
            "ratio": 3.14,
            "items": ["a", "b", "c"],  # Already a list, not a string
        }

        normalized, strategy = self.normalizer.normalize_arguments(numeric_args, "test_tool")

        assert normalized == numeric_args
        assert strategy == NormalizationStrategy.DIRECT

    def test_empty_arguments(self):
        """Test handling of empty arguments."""
        empty_args = {}

        normalized, strategy = self.normalizer.normalize_arguments(empty_args, "test_tool")

        assert normalized == {}
        assert strategy == NormalizationStrategy.DIRECT

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        self.normalizer.reset_stats()

        # Valid JSON
        self.normalizer.normalize_arguments({"valid": "value"}, "tool1")

        # Python syntax
        self.normalizer.normalize_arguments({"invalid": "{'key': 'value'}"}, "tool2")

        # Another valid
        self.normalizer.normalize_arguments({"valid": "value2"}, "tool3")

        stats = self.normalizer.get_stats()

        assert stats["total_calls"] == 3
        assert stats["normalizations"]["direct"] == 2
        assert stats["normalizations"]["python_ast"] >= 1
        assert stats["failures"] == 0
        assert stats["success_rate"] == 100.0

    def test_per_tool_statistics(self):
        """Test that per-tool statistics are tracked."""
        self.normalizer.reset_stats()

        # Call same tool multiple times
        self.normalizer.normalize_arguments({"valid": "1"}, "edit_files")
        self.normalizer.normalize_arguments({"ops": "{'a': 'b'}"}, "edit_files")
        self.normalizer.normalize_arguments({"valid": "2"}, "read_file")

        stats = self.normalizer.get_stats()

        assert "edit_files" in stats["by_tool"]
        assert stats["by_tool"]["edit_files"]["calls"] == 2
        assert stats["by_tool"]["edit_files"]["normalizations"] == 1

        assert "read_file" in stats["by_tool"]
        assert stats["by_tool"]["read_file"]["calls"] == 1
        assert stats["by_tool"]["read_file"]["normalizations"] == 0

    def test_malformed_json_all_strategies_fail(self):
        """Test handling when all normalization strategies fail."""
        # Completely malformed input that can't be parsed
        # Note: Must START with [ or { to trigger normalization attempts
        malformed_args = {"operations": "[this is not JSON or Python {{{["}

        normalized, strategy = self.normalizer.normalize_arguments(malformed_args, "edit_files")

        # Should return original (failed to normalize)
        assert strategy == NormalizationStrategy.FAILED
        assert normalized == malformed_args
        assert self.normalizer.stats["failures"] == 1

    def test_special_characters_preserved(self):
        """Test that special characters are preserved during normalization."""
        special_args = {
            "operations": "[{'content': '#!/bin/bash\\necho \"Hello, World!\"\\ncd /tmp'}]"
        }

        normalized, strategy = self.normalizer.normalize_arguments(special_args, "edit_files")

        assert strategy == NormalizationStrategy.PYTHON_AST

        # Verify special characters are properly converted (Python escape → actual newline)
        operations = json.loads(normalized["operations"])
        content = operations[0]["content"]
        # After normalization, \n should be actual newlines, not escaped
        assert "\n" in content  # Actual newline character
        assert "#!/bin/bash" in content
        assert "Hello, World!" in content

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        unicode_args = {"operations": "[{'message': '你好世界', 'emoji': '🚀'}]"}

        normalized, strategy = self.normalizer.normalize_arguments(unicode_args, "test_tool")

        assert strategy == NormalizationStrategy.PYTHON_AST

        # Verify unicode preserved
        operations = json.loads(normalized["operations"])
        assert operations[0]["message"] == "你好世界"
        assert operations[0]["emoji"] == "🚀"

    def test_performance_fast_path(self):
        """Test that fast path (valid JSON) has minimal overhead."""
        import time

        valid_args = {"path": "/test/file.txt", "content": "test content"}

        # Warm up
        for _ in range(100):
            self.normalizer.normalize_arguments(valid_args, "test_tool")

        # Benchmark
        start = time.perf_counter()
        for _ in range(10000):
            self.normalizer.normalize_arguments(valid_args, "test_tool")
        elapsed = time.perf_counter() - start

        # Should be fast (< 1ms average)
        avg_time_ms = (elapsed / 10000) * 1000
        assert avg_time_ms < 1.0, f"Fast path too slow: {avg_time_ms:.3f}ms"

    def test_edit_files_specific_repair(self):
        """Test edit_files tool-specific repair logic."""
        # Real example from Qwen model output
        qwen_output = {
            "operations": "[{'type': 'modify', 'path': 'fibonacci.sh', 'content': '#!/bin/bash\\n\\nfib=0\\nb=1\\n\\nfor ((i=0; i<5; i++)); do\\n  echo -n \"$a \"\\n  next=$((a + b))\\n  a=$b\\n  b=$next\\ndone\\necho'}]"
        }

        normalized, strategy = self.normalizer.normalize_arguments(qwen_output, "edit_files")

        # Should be normalized
        assert strategy != NormalizationStrategy.FAILED

        # Verify it's valid JSON
        operations = json.loads(normalized["operations"])
        assert operations[0]["type"] == "modify"
        assert operations[0]["path"] == "fibonacci.sh"
        assert "#!/bin/bash" in operations[0]["content"]


class TestNormalizationStrategy:
    """Test the NormalizationStrategy enum."""

    def test_strategy_values(self):
        """Test that strategy enum has expected values."""
        assert NormalizationStrategy.DIRECT.value == "direct"
        assert NormalizationStrategy.PYTHON_AST.value == "python_ast"
        assert NormalizationStrategy.REGEX_QUOTES.value == "regex_quotes"
        assert NormalizationStrategy.MANUAL_REPAIR.value == "manual_repair"
        assert NormalizationStrategy.FAILED.value == "failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
