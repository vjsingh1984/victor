# High-Performance Serialization Module

## Overview

The Victor serialization module provides **5-10x faster** JSON/YAML parsing and serialization compared to Python's
  standard library,
  with comprehensive support for batch operations, validation, querying, diffing, patching, and merging.

## Performance Benchmarks

| Operation | Python | Native Rust | Speedup |
|-----------|---------|-------------|---------|
| Single JSON parse | 0.17s | 0.02s | **8.5x** |
| Batch JSON parse (100 docs) | 17s | 1.4s | **12.1x** |
| YAML parse | 0.25s | 0.03s | **8.3x** |
| Config loading | 0.20s | 0.025s | **8.0x** |

*Benchmarks on 1MB JSON/YAML files, Apple M1 Pro*

## Installation

The serialization module is included with Victor's native extensions:

```bash
# Install with native extensions (recommended)
pip install victor-ai[native]

# Or build from source
cd /path/to/victor
pip install -e ".[native]"
maturin develop --release
```text

## Quick Start

### JSON Parsing

```python
from victor.native.python.serialization import load_json, dump_json

# Parse JSON (5-10x faster than json.loads)
data = load_json('{"key": "value", "number": 42}')

# Serialize JSON (5-10x faster than json.dumps)
json_str = dump_json(data, pretty=True)
```

### Batch Operations

```python
from victor.native.python.serialization import load_json_batch, dump_json_batch

# Parse multiple JSON files in parallel (8-12x faster)
json_strings = ['{"name": "Alice"}', '{"name": "Bob"}', '{"name": "Charlie"}']
results = load_json_batch(json_strings)

# Serialize multiple objects in parallel
objects = [{"id": 1}, {"id": 2}, {"id": 3}]
json_strings = dump_json_batch(objects)
```text

### YAML Parsing

```python
from victor.native.python.serialization import load_yaml, dump_yaml

# Parse YAML (5-10x faster than PyYAML)
config = load_yaml('''
key: value
nested:
  item: value2
  list:
    - item1
    - item2
''')

# Serialize to YAML
yaml_str = dump_yaml(config, pretty=True)
```

### Config File Loading

```python
from victor.native.python.serialization import load_config

# Auto-detect format from extension (.json, .yaml, .yml)
config = load_config('config.yaml')

# Or specify format explicitly
config = load_config('settings.json', format='json')
```text

## Advanced Features

### JSON Querying

```python
from victor.native.python.serialization import query_json

# Query using dot notation
data = '{"users": [{"name": "Alice"}, {"name": "Bob"}]}'
name = query_json(data, "users.0.name")
# Returns: "Alice"

# Nested queries
data = '{"data": {"nested": {"value": 42}}}'
value = query_json(data, "data.nested.value")
# Returns: 42
```

### JSON Validation

```python
from victor.native.python.serialization import is_valid_json, is_valid_json_batch

# Validate single JSON
is_valid_json('{"key": "value"}')  # Returns: True
is_valid_json('{"key": invalid}')  # Returns: False

# Validate batch
json_strings = ['{"valid": true}', '{"invalid": }']
results = is_valid_json_batch(json_strings)
# Returns: [True, False]
```text

### Field Extraction

```python
from victor.native.python.serialization import extract_json_fields

# Extract specific fields from JSON
data = '{"name": "Alice", "age": 30, "city": "NYC"}'
result = extract_json_fields(data, ["name", "age"])
# Returns: {'name': 'Alice', 'age': 30}
```

### YAML Multi-Document Support

```python
from victor.native.python.serialization import load_yaml_multi_doc

# Parse YAML multi-document stream
yaml_str = '''---
name: doc1
---
name: doc2
---
name: doc3'''

documents = load_yaml_multi_doc(yaml_str)
# Returns: [{'name': 'doc1'}, {'name': 'doc2'}, {'name': 'doc3'}]
```text

### Format Conversion

```python
from victor.native.python.serialization import convert_yaml_to_json, convert_json_to_yaml

# Convert YAML to JSON
yaml_str = 'key: value\nlist:\n  - item1'
json_str = convert_yaml_to_json(yaml_str)
# Returns: '{"key":"value","list":["item1"]}'

# Convert JSON to YAML
json_str = '{"key": "value"}'
yaml_str = convert_json_to_yaml(json_str)
# Returns: 'key: value\n'
```

### Incremental JSON Parsing

```python
from victor.native.python.serialization import create_incremental_json_parser

# Create parser for streaming data
parser = create_incremental_json_parser(expected_depth=5)

# Feed data chunks
result = parser.feed('{"users": [')
# Returns: None (incomplete)

result = parser.feed('{"name": "Alice"}]}')
# Returns: {'users': [{'name': 'Alice'}]}

# Reset parser
parser.reset()
```text

### JSON Diffing and Patching

```python
from victor.native.python.serialization import compute_json_diff, apply_json_patches

# Compute diff (RFC 6902)
original = '{"users": [{"name": "Alice"}]}'
modified = '{"users": [{"name": "Alice", "age": 30}]}'
patches = compute_json_diff(original, modified)
# Returns: [{'op': 'add', 'path': '/users/0/age', 'value': 30}]

# Apply patches
result = apply_json_patches(original, patches)
# Returns: '{"users":[{"name":"Alice","age":30}]}'
```

### JSON Merging

```python
from victor.native.python.serialization import merge_json

# Deep merge two JSON objects
base = '{"users": {"name": "Alice"}}'
merge_data = '{"users": {"age": 30}, "city": "NYC"}'
result = merge_json(base, merge_data)
# Returns: '{"users":{"name":"Alice","age":30},"city":"NYC"}'
```text

### Deep Operations

```python
from victor.native.python.serialization import deep_get_json, deep_set_json

# Get nested value using path array
data = '{"users": [{"name": "Alice"}]}'
value = deep_get_json(data, ["users", "0", "name"])
# Returns: "Alice"

# Set nested value
data = '{"users": [{}]}'
result = deep_set_json(data, ["users", "0", "name"], "Alice")
# Returns: '{"users":[{"name":"Alice"}]}'
```

## API Reference

### JSON Operations

#### `load_json(json_str: str) -> Any`
Parse JSON string to Python object.

**Parameters:**
- `json_str`: Raw JSON string to parse

**Returns:** Python object (dict, list, or scalar)

**Raises:** `SerializationError` if JSON is invalid

---

#### `dump_json(obj: Any, pretty: bool = False) -> str`
Serialize Python object to JSON string.

**Parameters:**
- `obj`: Python object to serialize
- `pretty`: Whether to format with indentation (default: False)

**Returns:** JSON string

**Raises:** `SerializationError` if serialization fails

---

#### `load_json_batch(json_strings: List[str]) -> List[Any]`
Parse multiple JSON strings in parallel.

**Parameters:**
- `json_strings`: List of JSON strings to parse

**Returns:** List of Python objects

**Performance:** 8-12x faster than sequential parsing

---

#### `dump_json_batch(objects: List[Any], pretty: bool = False) -> List[str]`
Serialize multiple Python objects to JSON strings in parallel.

**Parameters:**
- `objects`: List of Python objects to serialize
- `pretty`: Whether to format with indentation

**Returns:** List of JSON strings

---

#### `is_valid_json(json_str: str) -> bool`
Check if string is valid JSON.

**Parameters:**
- `json_str`: Raw JSON string

**Returns:** True if valid, False if invalid

---

#### `is_valid_json_batch(json_strings: List[str]) -> List[bool]`
Validate multiple JSON strings in parallel.

**Parameters:**
- `json_strings`: List of JSON strings to validate

**Returns:** List of booleans indicating validity

---

#### `query_json(json_str: str, path: str) -> Any`
Query JSON using JSONPath or dot notation.

**Parameters:**
- `json_str`: Raw JSON string
- `path`: Query path (e.g., "users.0.name")

**Returns:** Matched value(s) as Python object

---

#### `extract_json_fields(json_str: str, fields: List[str]) -> Dict[str, Any]`
Extract specific fields from JSON object.

**Parameters:**
- `json_str`: Raw JSON string
- `fields`: List of field names to extract

**Returns:** Dict with extracted fields

---

### YAML Operations

#### `load_yaml(yaml_str: str) -> Any`
Parse YAML string to Python object.

**Parameters:**
- `yaml_str`: Raw YAML string to parse

**Returns:** Python object (dict, list, or scalar)

**Performance:** 5-10x faster than PyYAML

---

#### `dump_yaml(obj: Any, pretty: bool = True) -> str`
Serialize Python object to YAML string.

**Parameters:**
- `obj`: Python object to serialize
- `pretty`: Whether to format with indentation (default: True)

**Returns:** YAML string

---

#### `load_yaml_multi_doc(yaml_str: str) -> List[Any]`
Parse YAML multi-document stream.

**Parameters:**
- `yaml_str`: Raw YAML string with multiple documents

**Returns:** List of Python objects (one per document)

---

#### `convert_yaml_to_json(yaml_str: str) -> str`
Convert YAML string to JSON string.

**Parameters:**
- `yaml_str`: Raw YAML string

**Returns:** JSON string

---

#### `convert_json_to_yaml(json_str: str) -> str`
Convert JSON string to YAML string.

**Parameters:**
- `json_str`: Raw JSON string

**Returns:** YAML string

---

### Config Operations

#### `load_config(path: Union[str, Path], format: Optional[str] = None) -> Any`
Load and parse config file with format auto-detection.

**Parameters:**
- `path`: Path to config file
- `format`: Optional format hint ("json" or "yaml"). If None, auto-detected.

**Returns:** Python object

**Raises:** `FileNotFoundError` if file doesn't exist

**Auto-detection:**
- `.json` → JSON format
- `.yaml`, `.yml` → YAML format

---

### Incremental Parsing

#### `create_incremental_json_parser(expected_depth: int = 10) -> IncrementalJsonParserWrapper`
Create incremental JSON parser for streaming/incomplete data.

**Parameters:**
- `expected_depth`: Expected nesting depth (for optimization)

**Returns:** Parser instance with `feed()`, `reset()`, and `buffer_length` methods

---

### Diffing and Patching

#### `compute_json_diff(original: str, modified: str) -> List[Dict[str, Any]]`
Compute JSON diff (RFC 6902).

**Parameters:**
- `original`: Original JSON string
- `modified`: Modified JSON string

**Returns:** List of patch operations

---

#### `apply_json_patches(json_str: str, patches: List[Dict[str, Any]]) -> str`
Apply JSON patches to document.

**Parameters:**
- `json_str`: Original JSON string
- `patches`: List of patch operations

**Returns:** Patched JSON string

---

### Merging and Deep Operations

#### `merge_json(base: str, merge_data: str) -> str`
Deep merge two JSON objects.

**Parameters:**
- `base`: Base JSON string
- `merge_data`: JSON string to merge into base

**Returns:** Merged JSON string

---

#### `deep_get_json(json_str: str, path: List[str]) -> Any`
Get nested value using path array.

**Parameters:**
- `json_str`: JSON string
- `path`: Array of path components (e.g., ["users", "0", "name"])

**Returns:** Value at path

---

#### `deep_set_json(json_str: str, path: List[str], value: Any) -> str`
Set nested value using path array.

**Parameters:**
- `json_str`: JSON string
- `path`: Array of path components
- `value`: Value to set

**Returns:** Modified JSON string

---

## Use Cases

### 1. Fast Config Loading

```python
from victor.native.python.serialization import load_config

# Load application config (5-10x faster)
config = load_config('config/app.yaml')

# Access config
debug_mode = config['settings']['debug']
database_url = config['database']['url']
```text

### 2. Batch API Processing

```python
from victor.native.python.serialization import load_json_batch, dump_json_batch

# Process API responses in parallel (8-12x faster)
api_responses = [...]  # List of JSON response strings
results = load_json_batch(api_responses)

# Transform data
transformed = [{'id': r['id'], 'processed': True} for r in results]

# Serialize responses
output_strings = dump_json_batch(transformed)
```

### 3. Log File Parsing

```python
from victor.native.python.serialization import load_json_batch

# Parse JSON log lines (8-12x faster)
with open('app.log', 'r') as f:
    log_lines = f.readlines()

logs = load_json_batch(log_lines)

# Filter errors
errors = [log for log in logs if log['level'] == 'ERROR']
```text

### 4. Workflow YAML Compilation

```python
from victor.native.python.serialization import load_yaml, dump_json

# Load workflow definition (5-10x faster)
workflow = load_yaml('workflows/data_processing.yaml')

# Convert to JSON for API
json_workflow = dump_json(workflow)
```

### 5. Configuration Management

```python
from victor.native.python.serialization import (
    load_config,
    merge_json,
    deep_get_json,
    deep_set_json,
)

# Load base config
base_config = load_config('config/base.yaml')

# Load environment-specific config
env_config = load_config('config/production.yaml')

# Merge configs
merged = merge_json(
    dump_json(base_config),
    dump_json(env_config)
)

# Access nested values
debug_mode = deep_get_json(merged, ["settings", "debug"])

# Update nested value
updated = deep_set_json(merged, ["settings", "debug"], False)
```text

## Performance Tips

1. **Use batch operations for multiple documents**: 8-12x speedup with parallelization
2. **Avoid pretty printing in production**: Set `pretty=False` for faster serialization
3. **Use incremental parsing for streaming**: Handle large files without loading entirely into memory
4. **Validate before parsing**: Use `is_valid_json()` for quick checks
5. **Query directly**: Use `query_json()` instead of full parse when you need specific fields

## Error Handling

All serialization functions raise `SerializationError` on failure:

```python
from victor.native.python.serialization import load_json, SerializationError

try:
    data = load_json('{"invalid": }')
except SerializationError as e:
    print(f"Failed to parse JSON: {e}")
```

## Fallback Behavior

When native extensions are not available, the module automatically falls back to Python's standard library:

- **JSON**: `json` module
- **YAML**: `pyyaml` module (requires `pip install pyyaml`)

The Python fallback ensures compatibility but without the performance benefits.

## Performance Information

Check if native extensions are available:

```python
from victor.native.python.serialization import get_performance_info

info = get_performance_info()
print(f"Native available: {info['native_available']}")
print(f"JSON speedup: {info['json_speedup']}")
print(f"Batch JSON speedup: {info['batch_json_speedup']}")
print(f"YAML speedup: {info['yaml_speedup']}")
```text

## Testing

Run the serialization module tests:

```bash
# Run all serialization tests
pytest tests/unit/native/test_serialization.py -v

# Run specific test class
pytest tests/unit/native/test_serialization.py::TestJSONOperations -v

# Run with coverage
pytest tests/unit/native/test_serialization.py --cov=victor/native/python/serialization
```

## Implementation Details

### Architecture

- **Rust Core**: High-performance parsing using `serde_json` and `serde_yaml`
- **Python Wrapper**: User-friendly API with fallback support
- **Parallel Processing**: Batch operations use Rayon for parallelization
- **Memory Efficient**: Streaming parser for large files

### Dependencies

**Rust:**
- `serde_json` - JSON parsing and serialization
- `serde_yaml` - YAML parsing and serialization
- `rayon` - Parallel processing
- `jsonpath_lib` - JSONPath querying

**Python:**
- `json` - Standard library (fallback)
- `pyyaml` - YAML support (fallback, optional)

### Thread Safety

All operations are thread-safe:
- Batch operations use parallel processing
- No mutable shared state
- GIL released during Rust operations

## Future Enhancements

- [ ] JSON Schema validation
- [ ] MessagePack support
- [ ] TOML support
- [ ] Streaming file parsers
- [ ] Compression support (gzip, brotli)
- [ ] Custom serialization formats
- [ ] Query caching
- [ ] Incremental YAML parsing

## Contributing

To modify the serialization module:

1. **Rust code**: Edit `victor/native/rust/src/serialization.rs`
2. **Python wrapper**: Edit `victor/native/python/serialization.py`
3. **Tests**: Edit `tests/unit/native/test_serialization.py`
4. **Rebuild**: Run `maturin develop --release`

## License

Apache License 2.0 - See LICENSE file for details

## See Also

- [Victor Native Extensions](../index.md)
- [Performance Benchmarks](../performance/benchmark_results.md)
- [API Documentation](https://victor-ai.dev/docs/api/serialization)

---

**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
