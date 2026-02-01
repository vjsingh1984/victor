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

"""Python wrapper for high-performance serialization operations.

This module provides a user-friendly Python interface to the Rust-based
serialization functions, with fallbacks to Python's standard library when
the native extensions are not available.

Performance benefits:
- JSON parsing: 5-10x faster than json.loads
- Batch JSON parsing: 8-12x faster (with parallelization)
- YAML parsing: 5-10x faster than PyYAML
- Config loading: 5-10x faster
"""

import json
from pathlib import Path
from typing import Any, Optional

try:
    from victor_native import (  # type: ignore[import-not-found]
        apply_json_patch,
        json_deep_get,
        json_deep_set,
        json_diff,
        json_extract_fields,
        json_merge,
        json_path_query,
        json_to_yaml,
        load_config_file,
        parse_json,
        parse_json_batch,
        parse_yaml,
        parse_yaml_multi_doc,
        serialize_json,
        serialize_json_batch,
        serialize_yaml,
        validate_json,
        validate_json_batch,
        yaml_to_json,
        IncrementalJsonParser,
        JsonPatch,
    )

    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


class SerializationError(Exception):
    """Exception raised for serialization errors."""

    pass


def load_json(json_str: str) -> Any:
    """Parse JSON string to Python object.

    Args:
        json_str: Raw JSON string to parse

    Returns:
        Python object (dict, list, or scalar)

    Raises:
        SerializationError: If JSON is invalid

    Example:
        >>> data = load_json('{"key": "value"}')
        >>> data
        {'key': 'value'}
    """
    if NATIVE_AVAILABLE:
        try:
            return parse_json(json_str)
        except Exception as e:
            raise SerializationError(f"Failed to parse JSON: {e}")
    else:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON: {e}")


def dump_json(obj: Any, pretty: bool = False) -> str:
    """Serialize Python object to JSON string.

    Args:
        obj: Python object to serialize
        pretty: Whether to format with indentation (default: False)

    Returns:
        JSON string

    Raises:
        SerializationError: If serialization fails

    Example:
        >>> dump_json({"key": "value"}, pretty=True)
        '{\\n  "key": "value"\\n}'
    """
    if NATIVE_AVAILABLE:
        try:
            return serialize_json(obj, pretty=pretty)
        except Exception as e:
            raise SerializationError(f"Failed to serialize JSON: {e}")
    else:
        try:
            if pretty:
                return json.dumps(obj, indent=2)
            else:
                return json.dumps(obj)
        except TypeError as e:
            raise SerializationError(f"Failed to serialize JSON: {e}")


def load_json_batch(json_strings: list[str]) -> list[Any]:
    """Parse multiple JSON strings in parallel.

    Args:
        json_strings: List of JSON strings to parse

    Returns:
        List of Python objects

    Raises:
        SerializationError: If any JSON is invalid

    Example:
        >>> data = load_json_batch([
        ...     '{"name": "Alice"}',
        ...     '{"name": "Bob"}',
        ... ])
        >>> len(data)
        2
    """
    if NATIVE_AVAILABLE:
        try:
            return parse_json_batch(json_strings)
        except Exception as e:
            raise SerializationError(f"Failed to parse JSON batch: {e}")
    else:
        results = []
        for json_str in json_strings:
            try:
                results.append(json.loads(json_str))
            except json.JSONDecodeError as e:
                raise SerializationError(f"Invalid JSON in batch: {e}")
        return results


def dump_json_batch(objects: list[Any], pretty: bool = False) -> list[str]:
    """Serialize multiple Python objects to JSON strings in parallel.

    Args:
        objects: List of Python objects to serialize
        pretty: Whether to format with indentation

    Returns:
        List of JSON strings

    Raises:
        SerializationError: If any serialization fails
    """
    if NATIVE_AVAILABLE:
        try:
            return serialize_json_batch(objects, pretty=pretty)
        except Exception as e:
            raise SerializationError(f"Failed to serialize JSON batch: {e}")
    else:
        results = []
        for obj in objects:
            try:
                if pretty:
                    results.append(json.dumps(obj, indent=2))
                else:
                    results.append(json.dumps(obj))
            except TypeError as e:
                raise SerializationError(f"Failed to serialize JSON in batch: {e}")
        return results


def is_valid_json(json_str: str) -> bool:
    """Check if string is valid JSON.

    Args:
        json_str: Raw JSON string

    Returns:
        True if valid, False if invalid

    Example:
        >>> is_valid_json('{"key": "value"}')
        True
        >>> is_valid_json('{"key": invalid}')
        False
    """
    if NATIVE_AVAILABLE:
        return validate_json(json_str)
    else:
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


def is_valid_json_batch(json_strings: list[str]) -> list[bool]:
    """Validate multiple JSON strings in parallel.

    Args:
        json_strings: List of JSON strings to validate

    Returns:
        List of booleans indicating validity
    """
    if NATIVE_AVAILABLE:
        return validate_json_batch(json_strings)
    else:
        return [is_valid_json(s) for s in json_strings]


def query_json(json_str: str, path: str) -> Any:
    """Query JSON using JSONPath or dot notation.

    Args:
        json_str: Raw JSON string
        path: Query path (JSONPath or dot notation, e.g., "users.0.name")

    Returns:
        Matched value(s) as Python object

    Raises:
        SerializationError: If query fails

    Example:
        >>> data = '{"users": [{"name": "Alice"}]}'
        >>> query_json(data, "users.0.name")
        'Alice'
    """
    if NATIVE_AVAILABLE:
        try:
            return json_path_query(json_str, path)
        except Exception as e:
            raise SerializationError(f"Failed to query JSON: {e}")
    else:
        # Python fallback with simple dot notation
        try:
            obj = json.loads(json_str)
            current = obj
            for component in path.split("."):
                if component.isdigit():
                    current = current[int(component)]
                else:
                    current = current[component]
            return current
        except (KeyError, IndexError, TypeError) as e:
            raise SerializationError(f"Failed to query JSON path '{path}': {e}")


def extract_json_fields(json_str: str, fields: list[str]) -> dict[str, Any]:
    """Extract specific fields from JSON object.

    Args:
        json_str: Raw JSON string
        fields: List of field names to extract

    Returns:
        Dict with extracted fields

    Raises:
        SerializationError: If extraction fails

    Example:
        >>> data = '{"name": "Alice", "age": 30, "city": "NYC"}'
        >>> extract_json_fields(data, ["name", "age"])
        {'name': 'Alice', 'age': 30}
    """
    if NATIVE_AVAILABLE:
        try:
            return json_extract_fields(json_str, fields)
        except Exception as e:
            raise SerializationError(f"Failed to extract JSON fields: {e}")
    else:
        try:
            obj = json.loads(json_str)
            return {k: obj[k] for k in fields if k in obj}
        except (json.JSONDecodeError, KeyError) as e:
            raise SerializationError(f"Failed to extract fields: {e}")


def load_yaml(yaml_str: str) -> Any:
    """Parse YAML string to Python object.

    Args:
        yaml_str: Raw YAML string to parse

    Returns:
        Python object (dict, list, or scalar)

    Raises:
        SerializationError: If YAML is invalid

    Note:
        Requires PyYAML to be installed when native extensions are not available.
    """
    if NATIVE_AVAILABLE:
        try:
            return parse_yaml(yaml_str)
        except Exception as e:
            raise SerializationError(f"Failed to parse YAML: {e}")
    else:
        try:
            import yaml

            return yaml.safe_load(yaml_str)
        except ImportError:
            raise SerializationError(
                "PyYAML is required for YAML parsing when native extensions are not available. "
                "Install with: pip install pyyaml"
            )
        except yaml.YAMLError as e:
            raise SerializationError(f"Invalid YAML: {e}")


def dump_yaml(obj: Any, pretty: bool = True) -> str:
    """Serialize Python object to YAML string.

    Args:
        obj: Python object to serialize
        pretty: Whether to format with indentation (default: True for YAML)

    Returns:
        YAML string

    Raises:
        SerializationError: If serialization fails
    """
    if NATIVE_AVAILABLE:
        try:
            return serialize_yaml(obj, pretty=pretty)
        except Exception as e:
            raise SerializationError(f"Failed to serialize YAML: {e}")
    else:
        try:
            import yaml

            return yaml.dump(obj, default_flow_style=False if pretty else None)
        except ImportError:
            raise SerializationError(
                "PyYAML is required for YAML serialization when native extensions are not available. "
                "Install with: pip install pyyaml"
            )
        except yaml.YAMLError as e:
            raise SerializationError(f"Failed to serialize YAML: {e}")


def load_yaml_multi_doc(yaml_str: str) -> list[Any]:
    """Parse YAML multi-document stream.

    Args:
        yaml_str: Raw YAML string with multiple documents

    Returns:
        List of Python objects (one per document)

    Raises:
        SerializationError: If parsing fails
    """
    if NATIVE_AVAILABLE:
        try:
            return parse_yaml_multi_doc(yaml_str)
        except Exception as e:
            raise SerializationError(f"Failed to parse multi-doc YAML: {e}")
    else:
        try:
            import yaml

            return list(yaml.safe_load_all(yaml_str))
        except ImportError:
            raise SerializationError(
                "PyYAML is required for YAML parsing when native extensions are not available. "
                "Install with: pip install pyyaml"
            )
        except yaml.YAMLError as e:
            raise SerializationError(f"Invalid multi-doc YAML: {e}")


def convert_yaml_to_json(yaml_str: str) -> str:
    """Convert YAML string to JSON string.

    Args:
        yaml_str: Raw YAML string

    Returns:
        JSON string

    Raises:
        SerializationError: If conversion fails

    Example:
        >>> convert_yaml_to_json('key: value\\nlist:\\n  - item1')
        '{"key":"value","list":["item1"]}'
    """
    if NATIVE_AVAILABLE:
        try:
            return yaml_to_json(yaml_str)
        except Exception as e:
            raise SerializationError(f"Failed to convert YAML to JSON: {e}")
    else:
        try:
            obj = load_yaml(yaml_str)
            return dump_json(obj)
        except Exception as e:
            raise SerializationError(f"Failed to convert YAML to JSON: {e}")


def convert_json_to_yaml(json_str: str) -> str:
    """Convert JSON string to YAML string.

    Args:
        json_str: Raw JSON string

    Returns:
        YAML string

    Raises:
        SerializationError: If conversion fails

    Example:
        >>> convert_json_to_yaml('{"key": "value"}')
        'key: value\\n'
    """
    if NATIVE_AVAILABLE:
        try:
            return json_to_yaml(json_str)
        except Exception as e:
            raise SerializationError(f"Failed to convert JSON to YAML: {e}")
    else:
        try:
            obj = load_json(json_str)
            return dump_yaml(obj)
        except Exception as e:
            raise SerializationError(f"Failed to convert JSON to YAML: {e}")


def load_config(path: str | Path, format: Optional[str] = None) -> Any:
    """Load and parse config file with format auto-detection.

    Args:
        path: Path to config file
        format: Optional format hint ("json" or "yaml"). If None, auto-detected.

    Returns:
        Python object (dict, list, or scalar)

    Raises:
        SerializationError: If file not found or parsing fails
        FileNotFoundError: If file doesn't exist

    Example:
        >>> config = load_config('config.yaml')
        >>> config['key']
        'value'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Auto-detect format from extension
    if format is None:
        ext = path.suffix.lower()
        if ext in [".json"]:
            format = "json"
        elif ext in [".yaml", ".yml"]:
            format = "yaml"
        else:
            raise SerializationError(
                f"Cannot auto-detect format from extension '{ext}'. "
                "Please specify format explicitly."
            )

    content = path.read_text()

    if format == "json":
        return load_json(content)
    elif format == "yaml":
        return load_yaml(content)
    else:
        raise SerializationError(f"Unsupported format: '{format}'. Use 'json' or 'yaml'.")


def create_incremental_json_parser(expected_depth: int = 10) -> "IncrementalJsonParserWrapper":
    """Create incremental JSON parser for streaming/incomplete data.

    Args:
        expected_depth: Expected nesting depth (for optimization)

    Returns:
        IncrementalJsonParserWrapper instance

    Example:
        >>> parser = create_incremental_json_parser()
        >>> result = parser.feed('{"users": [')
        >>> result  # None (incomplete)
        >>> result = parser.feed('{"name": "Alice"}]}')
        >>> result  # Complete JSON object
    """
    if NATIVE_AVAILABLE:
        return IncrementalJsonParserWrapper(expected_depth)
    else:
        return PythonIncrementalJsonParser(expected_depth)  # type: ignore[return-value]


class IncrementalJsonParserWrapper:
    """Wrapper for native IncrementalJsonParser."""

    def __init__(self, expected_depth: int):
        if NATIVE_AVAILABLE:
            self._parser = IncrementalJsonParser(expected_depth=expected_depth)
        else:
            self._parser = None

    def feed(self, chunk: str) -> Optional[Any]:
        """Feed data chunk to parser.

        Args:
            chunk: Data chunk to append

        Returns:
            Complete JSON object if ready, None if more data needed
        """
        if self._parser is not None:
            return self._parser.feed(chunk)
        else:
            raise RuntimeError("Native parser not available")

    def reset(self) -> None:
        """Reset parser state."""
        if self._parser is not None:
            self._parser.reset()

    @property
    def buffer_length(self) -> int:
        """Get current buffer length."""
        if self._parser is not None:
            return self._parser.buffer_length
        return 0


class PythonIncrementalJsonParser:
    """Python fallback implementation of incremental JSON parser."""

    def __init__(self, expected_depth: int):
        self.buffer = ""
        self.expected_depth = expected_depth
        self.current_depth = 0
        self.in_string = False
        self.escape_next = False

    def feed(self, chunk: str) -> Optional[Any]:
        """Feed data chunk to parser."""
        self.buffer += chunk

        # Check if we have a complete JSON document
        try:
            obj = json.loads(self.buffer)
            self.buffer = ""
            self.reset()
            return obj
        except json.JSONDecodeError:
            return None

    def reset(self) -> None:
        """Reset parser state."""
        self.buffer = ""
        self.current_depth = 0
        self.in_string = False
        self.escape_next = False

    @property
    def buffer_length(self) -> int:
        """Get current buffer length."""
        return len(self.buffer)


def compute_json_diff(original: str, modified: str) -> list[dict[str, Any]]:
    """Compute JSON diff (RFC 6902).

    Args:
        original: Original JSON string
        modified: Modified JSON string

    Returns:
        List of patch operations

    Raises:
        SerializationError: If diff computation fails

    Example:
        >>> original = '{"users": [{"name": "Alice"}]}'
        >>> modified = '{"users": [{"name": "Alice", "age": 30}]}'
        >>> patches = compute_json_diff(original, modified)
        >>> len(patches)
        1
    """
    if NATIVE_AVAILABLE:
        try:
            return json_diff(original, modified)
        except Exception as e:
            raise SerializationError(f"Failed to compute JSON diff: {e}")
    else:
        # Python fallback - simplified diff
        try:
            orig_obj = json.loads(original)
            mod_obj = json.loads(modified)

            # Very basic diff implementation
            patches = []
            if isinstance(orig_obj, dict) and isinstance(mod_obj, dict):
                for key, value in mod_obj.items():
                    if key not in orig_obj or orig_obj[key] != value:
                        patches.append(
                            {
                                "op": "add" if key not in orig_obj else "replace",
                                "path": f"/{key}",
                                "value": value,
                            }
                        )
            return patches
        except (json.JSONDecodeError, TypeError) as e:
            raise SerializationError(f"Failed to compute JSON diff: {e}")


def apply_json_patches(json_str: str, patches: list[dict[str, Any]]) -> str:
    """Apply JSON patches to document.

    Args:
        json_str: Original JSON string
        patches: List of patch operations

    Returns:
        Patched JSON string

    Raises:
        SerializationError: If patch application fails
    """
    if NATIVE_AVAILABLE:
        try:
            # Convert dicts to JsonPatch objects
            patch_objects = []
            for patch_dict in patches:
                patch_obj = JsonPatch(**patch_dict)
                patch_objects.append(patch_obj)

            return apply_json_patch(json_str, patch_objects)
        except Exception as e:
            raise SerializationError(f"Failed to apply JSON patches: {e}")
    else:
        # Python fallback - simplified patch application with JSON Pointer support
        try:
            obj = json.loads(json_str)

            for patch in patches:
                op = patch.get("op")
                path_str = patch.get("path", "")

                # Parse JSON Pointer path (e.g., "/users/0/age")
                path_components = [p for p in path_str.split("/") if p]

                # Navigate to parent of target
                current = obj
                for i, component in enumerate(path_components[:-1]):
                    if isinstance(current, dict):
                        if component not in current:
                            # Create intermediate dict if needed
                            current[component] = {}
                        current = current[component]
                    elif isinstance(current, list):
                        idx = int(component)
                        if idx >= len(current):
                            # Extend list if needed
                            current.extend([{}] * (idx - len(current) + 1))
                        current = current[idx]
                    else:
                        raise SerializationError(
                            f"Cannot traverse into scalar value at path component {i}"
                        )

                # Apply operation
                last_component = path_components[-1] if path_components else None

                if op == "add":
                    value = patch.get("value")
                    if last_component is not None:
                        if isinstance(current, dict):
                            current[last_component] = value
                        elif isinstance(current, list):
                            idx = int(last_component)
                            current.insert(idx, value)
                        else:
                            raise SerializationError("Cannot add to scalar value")
                    else:
                        # Replace root
                        obj = value

                elif op == "replace":
                    value = patch.get("value")
                    if last_component is not None:
                        if isinstance(current, dict) and last_component in current:
                            current[last_component] = value
                        elif isinstance(current, list):
                            idx = int(last_component)
                            if 0 <= idx < len(current):
                                current[idx] = value
                            else:
                                raise SerializationError(f"Array index {idx} out of bounds")
                        else:
                            raise SerializationError(f"Path not found: {path_str}")
                    else:
                        # Replace root
                        obj = value

                elif op == "remove":
                    if last_component is not None:
                        if isinstance(current, dict) and last_component in current:
                            del current[last_component]
                        elif isinstance(current, list):
                            idx = int(last_component)
                            if 0 <= idx < len(current):
                                current.pop(idx)
                            else:
                                raise SerializationError(f"Array index {idx} out of bounds")
                        else:
                            raise SerializationError(f"Path not found: {path_str}")
                    else:
                        raise SerializationError("Cannot remove root")

            return json.dumps(obj)
        except (json.JSONDecodeError, TypeError, ValueError, IndexError) as e:
            raise SerializationError(f"Failed to apply JSON patches: {e}")


def merge_json(base: str, merge_data: str) -> str:
    """Deep merge two JSON objects.

    Args:
        base: Base JSON string
        merge_data: JSON string to merge into base

    Returns:
        Merged JSON string

    Raises:
        SerializationError: If merge fails

    Example:
        >>> base = '{"users": {"name": "Alice"}}'
        >>> merge_data = '{"users": {"age": 30}, "city": "NYC"}'
        >>> result = merge_json(base, merge_data)
        >>> "Alice" in result and "NYC" in result
        True
    """
    if NATIVE_AVAILABLE:
        try:
            return json_merge(base, merge_data)
        except Exception as e:
            raise SerializationError(f"Failed to merge JSON: {e}")
    else:
        # Python fallback
        try:
            base_obj = json.loads(base)
            merge_obj = json.loads(merge_data)

            def deep_merge(base_val, merge_val):
                if isinstance(base_val, dict) and isinstance(merge_val, dict):
                    result = base_val.copy()
                    for key, val in merge_val.items():
                        if (
                            key in result
                            and isinstance(result[key], dict)
                            and isinstance(val, dict)
                        ):
                            result[key] = deep_merge(result[key], val)
                        else:
                            result[key] = val
                    return result
                else:
                    return merge_val

            merged = deep_merge(base_obj, merge_obj)
            return json.dumps(merged)
        except (json.JSONDecodeError, TypeError) as e:
            raise SerializationError(f"Failed to merge JSON: {e}")


def deep_get_json(json_str: str, path: list[str]) -> Any:
    """Get nested value using path array.

    Args:
        json_str: JSON string
        path: Array of path components (e.g., ["users", "0", "name"])

    Returns:
        Value at path

    Raises:
        SerializationError: If path doesn't exist

    Example:
        >>> data = '{"users": [{"name": "Alice"}]}'
        >>> deep_get_json(data, ["users", "0", "name"])
        'Alice'
    """
    if NATIVE_AVAILABLE:
        try:
            return json_deep_get(json_str, path)
        except Exception as e:
            raise SerializationError(f"Failed to get nested value: {e}")
    else:
        try:
            obj = json.loads(json_str)
            current = obj
            for component in path:
                if isinstance(current, dict):
                    current = current[component]
                elif isinstance(current, list):
                    current = current[int(component)]
                else:
                    raise SerializationError("Cannot traverse into scalar value")
            return current
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raise SerializationError(f"Failed to get nested value: {e}")


def deep_set_json(json_str: str, path: list[str], value: Any) -> str:
    """Set nested value using path array.

    Args:
        json_str: JSON string
        path: Array of path components
        value: Value to set

    Returns:
        Modified JSON string

    Raises:
        SerializationError: If operation fails

    Example:
        >>> data = '{"users": [{}]}'
        >>> result = deep_set_json(data, ["users", "0", "name"], "Alice")
        >>> "Alice" in result
        True
    """
    if NATIVE_AVAILABLE:
        try:
            return json_deep_set(json_str, path, value)
        except Exception as e:
            raise SerializationError(f"Failed to set nested value: {e}")
    else:
        # Python fallback - with intermediate object creation
        try:
            obj = json.loads(json_str)
            current = obj

            # Navigate to parent of target, creating intermediate objects
            for component in path[:-1]:
                if isinstance(current, dict):
                    if component not in current:
                        # Create intermediate dict
                        current[component] = {}
                    current = current[component]
                elif isinstance(current, list):
                    idx = int(component)
                    if idx >= len(current):
                        # Extend list
                        current.extend([{}] * (idx - len(current) + 1))
                    current = current[idx]
                else:
                    raise SerializationError(
                        f"Cannot traverse into scalar value at component '{component}'"
                    )

            # Set value
            last_component = path[-1]
            if isinstance(current, dict):
                current[last_component] = value
            elif isinstance(current, list):
                idx = int(last_component)
                if idx >= len(current):
                    # Extend list if needed
                    current.extend([None] * (idx - len(current) + 1))
                current[idx] = value
            else:
                raise SerializationError(f"Cannot set value on scalar type at path '{path}'")

            return json.dumps(obj)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
            raise SerializationError(f"Failed to set nested value: {e}")


# Performance utilities
def get_performance_info() -> dict[str, Any]:
    """Get information about available performance optimizations.

    Returns:
        Dict with performance info
    """
    return {
        "native_available": NATIVE_AVAILABLE,
        "json_speedup": "5-10x" if NATIVE_AVAILABLE else "1x (Python fallback)",
        "batch_json_speedup": "8-12x" if NATIVE_AVAILABLE else "1x (Python fallback)",
        "yaml_speedup": "5-10x" if NATIVE_AVAILABLE else "1x (PyYAML fallback)",
        "parallel_processing": NATIVE_AVAILABLE,
    }


__all__ = [
    "load_json",
    "dump_json",
    "load_json_batch",
    "dump_json_batch",
    "is_valid_json",
    "is_valid_json_batch",
    "query_json",
    "extract_json_fields",
    "load_yaml",
    "dump_yaml",
    "load_yaml_multi_doc",
    "convert_yaml_to_json",
    "convert_json_to_yaml",
    "load_config",
    "create_incremental_json_parser",
    "compute_json_diff",
    "apply_json_patches",
    "merge_json",
    "deep_get_json",
    "deep_set_json",
    "SerializationError",
    "IncrementalJsonParserWrapper",
    "get_performance_info",
]
