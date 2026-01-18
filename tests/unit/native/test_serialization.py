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

"""Unit tests for the serialization module.

Tests both native and fallback implementations to ensure consistent behavior.
"""

import json
import tempfile
from pathlib import Path

import pytest

try:
    from victor.native.python.serialization import (
        apply_json_patches,
        compute_json_diff,
        convert_json_to_yaml,
        convert_yaml_to_json,
        create_incremental_json_parser,
        deep_get_json,
        deep_set_json,
        dump_json,
        dump_json_batch,
        dump_yaml,
        extract_json_fields,
        get_performance_info,
        is_valid_json,
        is_valid_json_batch,
        load_config,
        load_json,
        load_json_batch,
        load_yaml,
        load_yaml_multi_doc,
        merge_json,
        query_json,
        SerializationError,
    )

    SERIALIZATION_AVAILABLE = True
except ImportError:
    SERIALIZATION_AVAILABLE = False


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestJSONOperations:
    """Test JSON parsing and serialization operations."""

    def test_load_json_simple(self):
        """Test basic JSON parsing."""
        json_str = '{"key": "value", "number": 42}'
        result = load_json(json_str)
        assert result == {"key": "value", "number": 42}

    def test_load_json_nested(self):
        """Test nested JSON parsing."""
        json_str = '{"users": [{"name": "Alice"}, {"name": "Bob"}]}'
        result = load_json(json_str)
        assert len(result["users"]) == 2
        assert result["users"][0]["name"] == "Alice"

    def test_load_json_array(self):
        """Test JSON array parsing."""
        json_str = "[1, 2, 3, 4, 5]"
        result = load_json(json_str)
        assert result == [1, 2, 3, 4, 5]

    def test_load_json_invalid(self):
        """Test invalid JSON raises error."""
        json_str = '{"key": invalid}'
        with pytest.raises(SerializationError):
            load_json(json_str)

    def test_dump_json_simple(self):
        """Test basic JSON serialization."""
        obj = {"key": "value", "number": 42}
        result = dump_json(obj)
        assert json.loads(result) == obj

    def test_dump_json_pretty(self):
        """Test JSON serialization with pretty printing."""
        obj = {"key": "value"}
        result = dump_json(obj, pretty=True)
        assert "\n" in result
        assert json.loads(result) == obj

    def test_load_json_batch(self):
        """Test batch JSON parsing."""
        json_strings = ['{"name": "Alice"}', '{"name": "Bob"}', '{"name": "Charlie"}']
        results = load_json_batch(json_strings)
        assert len(results) == 3
        assert results[0]["name"] == "Alice"
        assert results[1]["name"] == "Bob"
        assert results[2]["name"] == "Charlie"

    def test_dump_json_batch(self):
        """Test batch JSON serialization."""
        objects = [{"name": "Alice"}, {"name": "Bob"}]
        results = dump_json_batch(objects)
        assert len(results) == 2
        assert json.loads(results[0]) == {"name": "Alice"}
        assert json.loads(results[1]) == {"name": "Bob"}

    def test_is_valid_json_true(self):
        """Test JSON validation with valid JSON."""
        assert is_valid_json('{"key": "value"}')
        assert is_valid_json("[1, 2, 3]")
        assert is_valid_json('"string"')

    def test_is_valid_json_false(self):
        """Test JSON validation with invalid JSON."""
        assert not is_valid_json('{"key": invalid}')
        assert not is_valid_json("{unclosed brace")
        assert not is_valid_json("")

    def test_is_valid_json_batch(self):
        """Test batch JSON validation."""
        json_strings = ['{"key": "value"}', '{"invalid": }', '{"valid": "true"}']
        results = is_valid_json_batch(json_strings)
        assert results == [True, False, True]

    def test_query_json_dot_notation(self):
        """Test JSON querying with dot notation."""
        json_str = '{"users": [{"name": "Alice"}, {"name": "Bob"}]}'
        result = query_json(json_str, "users.0.name")
        assert result == "Alice"

    def test_query_json_nested(self):
        """Test JSON querying with nested path."""
        json_str = '{"data": {"nested": {"value": 42}}}'
        result = query_json(json_str, "data.nested.value")
        assert result == 42

    def test_extract_json_fields(self):
        """Test extracting specific fields from JSON."""
        json_str = '{"name": "Alice", "age": 30, "city": "NYC"}'
        result = extract_json_fields(json_str, ["name", "age"])
        assert result == {"name": "Alice", "age": 30}

    def test_extract_json_fields_missing(self):
        """Test extracting fields with missing keys."""
        json_str = '{"name": "Alice", "age": 30}'
        result = extract_json_fields(json_str, ["name", "city"])
        assert result == {"name": "Alice"}


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestYAMLOperations:
    """Test YAML parsing and serialization operations."""

    def test_load_yaml_simple(self):
        """Test basic YAML parsing."""
        yaml_str = "key: value\nnumber: 42"
        result = load_yaml(yaml_str)
        assert result == {"key": "value", "number": 42}

    def test_load_yaml_list(self):
        """Test YAML list parsing."""
        yaml_str = "- item1\n- item2\n- item3"
        result = load_yaml(yaml_str)
        assert result == ["item1", "item2", "item3"]

    def test_load_yaml_nested(self):
        """Test nested YAML parsing."""
        yaml_str = "users:\n  - name: Alice\n  - name: Bob"
        result = load_yaml(yaml_str)
        assert len(result["users"]) == 2
        assert result["users"][0]["name"] == "Alice"

    def test_dump_yaml_simple(self):
        """Test YAML serialization."""
        obj = {"key": "value", "number": 42}
        result = dump_yaml(obj)
        assert "key: value" in result

    def test_load_yaml_multi_doc(self):
        """Test multi-document YAML parsing."""
        yaml_str = "---\nname: doc1\n---\nname: doc2"
        results = load_yaml_multi_doc(yaml_str)
        assert len(results) == 2
        assert results[0]["name"] == "doc1"
        assert results[1]["name"] == "doc2"

    def test_convert_yaml_to_json(self):
        """Test YAML to JSON conversion."""
        yaml_str = "key: value\nlist:\n  - item1"
        result = convert_yaml_to_json(yaml_str)
        result_obj = json.loads(result)
        assert result_obj["key"] == "value"
        assert result_obj["list"] == ["item1"]

    def test_convert_json_to_yaml(self):
        """Test JSON to YAML conversion."""
        json_str = '{"key": "value"}'
        result = convert_json_to_yaml(json_str)
        assert "key: value" in result


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_config_json(self):
        """Test loading JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            path = Path(f.name)

        try:
            result = load_config(path)
            assert result == {"key": "value"}
        finally:
            path.unlink()

    def test_load_config_yaml(self):
        """Test loading YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\nnumber: 42\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_config(path)
            assert result == {"key": "value", "number": 42}
        finally:
            path.unlink()

    def test_load_config_auto_detect(self):
        """Test automatic format detection."""
        # Test JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            path = Path(f.name)

        try:
            result = load_config(path)
            assert result == {"key": "value"}
        finally:
            path.unlink()

        # Test YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_config(path)
            assert result == {"key": "value"}
        finally:
            path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestIncrementalParsing:
    """Test incremental JSON parsing for streaming data."""

    def test_incremental_parser_simple(self):
        """Test incremental parser with simple JSON."""
        parser = create_incremental_json_parser()

        # Feed incomplete data
        result = parser.feed('{"users": [')
        assert result is None

        # Feed rest of data
        result = parser.feed('{"name": "Alice"}]}')
        assert result is not None
        assert result["users"][0]["name"] == "Alice"

    def test_incremental_parser_reset(self):
        """Test resetting incremental parser."""
        parser = create_incremental_json_parser()

        # Feed incomplete JSON to ensure buffer has content
        result = parser.feed('{"key": "incomplete')

        # Buffer should have content since JSON is incomplete
        assert parser.buffer_length > 0

        parser.reset()
        assert parser.buffer_length == 0

    def test_incremental_parser_multiple_feeds(self):
        """Test incremental parser with multiple small chunks."""
        parser = create_incremental_json_parser()

        chunks = ['{"use', 'rs": [{"na', 'me": "Alice"}]}']
        result = None

        for chunk in chunks:
            result = parser.feed(chunk)

        assert result is not None
        assert result["users"][0]["name"] == "Alice"


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestJSONDiffAndPatch:
    """Test JSON diffing and patching operations."""

    def test_compute_json_diff_add(self):
        """Test computing JSON diff for added fields."""
        original = '{"users": [{"name": "Alice"}]}'
        modified = '{"users": [{"name": "Alice", "age": 30}]}'

        patches = compute_json_diff(original, modified)
        assert len(patches) > 0

    def test_compute_json_diff_replace(self):
        """Test computing JSON diff for replaced values."""
        original = '{"value": 10}'
        modified = '{"value": 20}'

        patches = compute_json_diff(original, modified)
        assert len(patches) > 0

    def test_apply_json_patches_add(self):
        """Test applying add patch."""
        json_str = '{"users": [{"name": "Alice"}]}'
        patches = [{"op": "add", "path": "/users/0/age", "value": 30}]

        result = apply_json_patches(json_str, patches)
        result_obj = json.loads(result)
        assert result_obj["users"][0]["age"] == 30

    def test_apply_json_patches_replace(self):
        """Test applying replace patch."""
        json_str = '{"value": 10}'
        patches = [{"op": "replace", "path": "/value", "value": 20}]

        result = apply_json_patches(json_str, patches)
        result_obj = json.loads(result)
        assert result_obj["value"] == 20


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestJSONMergeAndDeepOperations:
    """Test JSON merging and deep operations."""

    def test_merge_json_simple(self):
        """Test simple JSON merge."""
        base = '{"users": {"name": "Alice"}}'
        merge_data = '{"users": {"age": 30}, "city": "NYC"}'

        result = merge_json(base, merge_data)
        result_obj = json.loads(result)

        assert result_obj["users"]["name"] == "Alice"
        assert result_obj["users"]["age"] == 30
        assert result_obj["city"] == "NYC"

    def test_deep_get_json_simple(self):
        """Test deep get with simple path."""
        json_str = '{"users": [{"name": "Alice"}]}'
        result = deep_get_json(json_str, ["users", "0", "name"])
        assert result == "Alice"

    def test_deep_get_json_nested(self):
        """Test deep get with nested objects."""
        json_str = '{"data": {"nested": {"value": 42}}}'
        result = deep_get_json(json_str, ["data", "nested", "value"])
        assert result == 42

    def test_deep_set_json_simple(self):
        """Test deep set with simple path."""
        json_str = '{"users": [{}]}'
        result = deep_set_json(json_str, ["users", "0", "name"], "Alice")
        result_obj = json.loads(result)
        assert result_obj["users"][0]["name"] == "Alice"

    def test_deep_set_json_nested(self):
        """Test deep set creating intermediate objects."""
        json_str = '{"data": {}}'
        result = deep_set_json(json_str, ["data", "nested", "value"], 42)
        result_obj = json.loads(result)
        assert result_obj["data"]["nested"]["value"] == 42


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestPerformanceInfo:
    """Test performance information utilities."""

    def test_get_performance_info(self):
        """Test getting performance information."""
        info = get_performance_info()

        assert "native_available" in info
        assert "json_speedup" in info
        assert "batch_json_speedup" in info
        assert "yaml_speedup" in info
        assert isinstance(info["native_available"], bool)


@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestErrorHandling:
    """Test error handling in serialization operations."""

    def test_load_json_invalid(self):
        """Test error handling for invalid JSON."""
        with pytest.raises(SerializationError):
            load_json('{"invalid": }')

    def test_load_yaml_invalid(self):
        """Test error handling for invalid YAML."""
        with pytest.raises(SerializationError):
            load_yaml("invalid: yaml: [unclosed")

    def test_query_json_invalid_path(self):
        """Test error handling for invalid query path."""
        with pytest.raises(SerializationError):
            query_json('{"key": "value"}', "nonexistent.key")

    def test_deep_get_json_invalid_path(self):
        """Test error handling for invalid deep path."""
        with pytest.raises(SerializationError):
            deep_get_json('{"key": "value"}', ["nonexistent", "key"])

    def test_load_config_unsupported_format(self):
        """Test error handling for unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(SerializationError):
                load_config(path, format="unsupported")
        finally:
            path.unlink()


@pytest.mark.integration
@pytest.mark.skipif(not SERIALIZATION_AVAILABLE, reason="Serialization module not available")
class TestIntegration:
    """Integration tests for serialization module."""

    def test_full_workflow_json(self):
        """Test complete workflow with JSON operations."""
        # Create config
        config = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}

        # Serialize
        json_str = dump_json(config)
        assert is_valid_json(json_str)

        # Parse
        parsed = load_json(json_str)
        assert parsed == config

        # Query
        name = query_json(json_str, "users.0.name")
        assert name == "Alice"

        # Extract fields
        fields = extract_json_fields(json_str, ["users"])
        assert "users" in fields

    def test_full_workflow_yaml(self):
        """Test complete workflow with YAML operations."""
        # Create config
        config = {"key": "value", "nested": {"item": "value2"}}

        # Serialize to YAML
        yaml_str = dump_yaml(config)

        # Parse back
        parsed = load_yaml(yaml_str)
        assert parsed == config

        # Convert to JSON
        json_str = convert_yaml_to_json(yaml_str)
        assert is_valid_json(json_str)

    def test_config_file_workflow(self):
        """Test config file operations."""
        # Create temporary config
        config = {"app_name": "test", "version": "1.0", "settings": {"debug": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            path = Path(f.name)

        try:
            # Load config
            loaded = load_config(path)
            assert loaded == config

            # Modify and save
            loaded["version"] = "2.0"
            modified_json = dump_json(loaded)

            # Parse modified
            modified = load_json(modified_json)
            assert modified["version"] == "2.0"

        finally:
            path.unlink()

    def test_batch_operations_workflow(self):
        """Test batch operations workflow."""
        # Create batch data
        data = [{"id": i, "value": f"item{i}"} for i in range(100)]

        # Serialize batch
        json_strings = dump_json_batch(data)
        assert len(json_strings) == 100

        # Parse batch
        parsed = load_json_batch(json_strings)
        assert len(parsed) == 100
        assert parsed[0] == data[0]

        # Validate batch
        valid = is_valid_json_batch(json_strings)
        assert all(valid)
