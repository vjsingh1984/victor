# File Operations Module - High-Performance Parallel File System Operations

## Overview

The File Operations module provides Rust-accelerated file system operations with 2-3x faster directory traversal and 3-5x faster metadata collection compared to pure Python implementations.

## Performance Characteristics

| Operation | Performance | Description |
|-----------|-------------|-------------|
| `walk_directory_parallel` | 2-3x faster than os.walk | Parallel directory traversal with Rayon |
| `collect_metadata` | 3-5x faster than individual stat calls | Batch metadata collection |
| `filter_by_extension` | Near-instant | Set-based O(1) lookup |
| `filter_by_size` | Parallel | Rayon-based parallel filtering |
| `get_directory_stats` | Batch | Single-pass statistics collection |

## Installation

```bash
# Install with native extensions
pip install victor-ai[native]

# Or build from source
cd rust
cargo build --release
pip install -e ..
```

## Quick Start

```python
from victor.native.rust import file_ops

# Walk directory with glob patterns
files = file_ops.walk_directory(
    "src",
    patterns=["*.py", "**/*.rs"],
    max_depth=10,
    ignore_patterns=["*.pyc", "__pycache__"]
)

# Filter by extension
py_files = file_ops.filter_files_by_extension(files, ["py"])

# Get metadata
metadata = file_ops.get_file_metadata([f.path for f in py_files])

# Get directory statistics
stats = file_ops.get_directory_statistics("src")
print(f"Total size: {stats['total_size']} bytes")
```

## API Reference

### Classes

#### FileInfo

Represents a file or directory with metadata.

**Attributes:**
- `path` (str): Full path to the file/directory
- `file_type` (str): Type of entry: "file", "directory", or "symlink"
- `size` (int): Size in bytes (0 for directories)
- `modified` (Optional[int]): Last modification time as Unix timestamp
- `depth` (int): Depth in directory tree (0 = root)

**Methods:**
- `matches(pattern: str) -> bool`: Check if file matches glob pattern
- `to_dict() -> Dict`: Convert to dictionary

**Example:**
```python
info = FileInfo(
    path="/test/file.txt",
    file_type="file",
    size=1024,
    modified=1234567890,
    depth=1
)

print(info.path)      # "/test/file.txt"
print(info.matches("*.txt"))  # True
```

#### FileMetadata

Detailed file metadata from stat calls.

**Attributes:**
- `path` (str): Full path to the file
- `size` (int): Size in bytes
- `modified` (int): Last modification time as Unix timestamp
- `is_file` (bool): True if this is a regular file
- `is_dir` (bool): True if this is a directory
- `is_symlink` (bool): True if this is a symbolic link
- `is_readonly` (bool): True if file is read-only

**Methods:**
- `to_dict() -> Dict`: Convert to dictionary

**Example:**
```python
metadata = file_ops.get_file_metadata(["src/main.py"])[0]
print(f"{metadata.path}: {metadata.size} bytes")
print(f"Read-only: {metadata.is_readonly}")
```

### Functions

#### walk_directory

Walk directory tree in parallel with pattern matching.

**Signature:**
```python
def walk_directory(
    root: Union[str, Path],
    patterns: Optional[List[str]] = None,
    max_depth: int = 100,
    follow_symlinks: bool = False,
    ignore_patterns: Optional[List[str]] = None,
) -> List[FileInfo]
```

**Parameters:**
- `root`: Root directory path to traverse
- `patterns`: List of glob patterns (e.g., ["*.py", "**/*.rs"])
- `max_depth`: Maximum traversal depth (0 = root only, default: 100)
- `follow_symlinks`: Whether to follow symbolic links
- `ignore_patterns`: Patterns to ignore (e.g., ["*.pyc", "__pycache__"])

**Returns:**
- List of FileInfo objects with matched files/directories

**Examples:**
```python
# Find all Python files in src directory
files = file_ops.walk_directory("src", patterns=["*.py"])

# Find all code files recursively
files = file_ops.walk_directory(
    "src",
    patterns=["*.py", "*.rs", "*.java"],
    max_depth=20
)

# Exclude cache directories
files = file_ops.walk_directory(
    "src",
    patterns=["*"],
    ignore_patterns=["__pycache__", "*.pyc", ".git"]
)
```

#### get_file_metadata

Collect metadata for multiple files in parallel.

**Signature:**
```python
def get_file_metadata(paths: List[Union[str, Path]]) -> List[FileMetadata]
```

**Parameters:**
- `paths`: List of file paths to get metadata for

**Returns:**
- List of FileMetadata objects. Skips paths that don't exist.

**Example:**
```python
metadata = file_ops.get_file_metadata(["src/main.py", "README.md"])
for m in metadata:
    print(f"{m.path}: {m.size} bytes")
```

#### filter_files_by_extension

Filter files by extension using efficient set-based lookup.

**Signature:**
```python
def filter_files_by_extension(
    files: List[FileInfo],
    extensions: List[str]
) -> List[FileInfo]
```

**Parameters:**
- `files`: List of FileInfo objects to filter
- `extensions`: List of extensions (e.g., ["py", "rs", "java"])

**Returns:**
- Filtered list of FileInfo objects matching extensions

**Example:**
```python
files = file_ops.walk_directory("src", patterns=["*"])
code_files = file_ops.filter_files_by_extension(files, ["py", "rs"])
```

#### filter_files_by_size

Filter files by size range.

**Signature:**
```python
def filter_files_by_size(
    files: List[FileInfo],
    min_size: int = 0,
    max_size: int = 0
) -> List[FileInfo]
```

**Parameters:**
- `files`: List of FileInfo objects to filter
- `min_size`: Minimum file size in bytes (0 = no minimum)
- `max_size`: Maximum file size in bytes (0 = no maximum)

**Returns:**
- Filtered list of FileInfo objects within size range

**Example:**
```python
# Find medium-sized files (1KB to 1MB)
medium_files = file_ops.filter_files_by_size(
    files,
    min_size=1024,
    max_size=1024*1024
)
```

#### get_directory_statistics

Get directory statistics including total size and largest files.

**Signature:**
```python
def get_directory_statistics(
    root: Union[str, Path],
    max_depth: int = 100
) -> Dict[str, Union[int, List[tuple]]]
```

**Parameters:**
- `root`: Root directory path to analyze
- `max_depth`: Maximum traversal depth

**Returns:**
- Dictionary with:
  - `total_size`: Total size in bytes
  - `file_count`: Number of files
  - `dir_count`: Number of directories
  - `largest_files`: List of (path, size) tuples for top 10 files

**Example:**
```python
stats = file_ops.get_directory_statistics("src")
print(f"Total size: {stats['total_size']} bytes")
print(f"Files: {stats['file_count']}, Dirs: {stats['dir_count']}")
```

#### group_files_by_directory

Group files by their parent directory.

**Signature:**
```python
def group_files_by_directory(
    files: List[FileInfo]
) -> Dict[str, List[FileInfo]]
```

**Parameters:**
- `files`: List of FileInfo objects to group

**Returns:**
- Dictionary mapping directory paths to lists of FileInfo objects

**Example:**
```python
files = file_ops.walk_directory("src", patterns=["*.py"])
grouped = file_ops.group_files_by_directory(files)
for dir_path, dir_files in grouped.items():
    print(f"{dir_path}: {len(dir_files)} files")
```

#### filter_files_by_modified_time

Filter files by modification time.

**Signature:**
```python
def filter_files_by_modified_time(
    files: List[FileInfo],
    since: int,
    until: int = 0
) -> List[FileInfo]
```

**Parameters:**
- `files`: List of FileInfo objects to filter
- `since`: Unix timestamp for earliest modification time
- `until`: Unix timestamp for latest modification time (0 = now)

**Returns:**
- Filtered list of FileInfo objects modified in the time range

**Example:**
```python
import time
# Find files modified in the last 24 hours
one_day_ago = int(time.time()) - 86400
recent = file_ops.filter_files_by_modified_time(files, since=one_day_ago)
```

#### find_code_files

Convenience function to find code files in a directory.

**Signature:**
```python
def find_code_files(
    root: Union[str, Path],
    extensions: Optional[List[str]] = None,
    ignore_dirs: Optional[List[str]] = None,
    max_depth: int = 100,
) -> List[FileInfo]
```

**Parameters:**
- `root`: Root directory to search
- `extensions`: List of file extensions (default: common code extensions)
- `ignore_dirs`: Directories to ignore (default: common ignore patterns)
- `max_depth`: Maximum traversal depth

**Returns:**
- List of FileInfo objects for code files

**Example:**
```python
# Find all code files with default extensions
code_files = file_ops.find_code_files("src")

# Find only Python and Rust files
code_files = file_ops.find_code_files("src", extensions=["py", "rs"])
```

## Glob Pattern Support

The module supports various glob patterns:

### Basic Patterns

- `*.py` - Match all Python files in current directory
- `test.*` - Match all files starting with "test"
- `*` - Match all files

### Recursive Patterns

- `**/*.py` - Match all Python files recursively
- `src/**/*.rs` - Match all Rust files in src directory tree

### Negation

- `!*.pyc` - Exclude all .pyc files
- `!__pycache__` - Exclude pycache directories

### Brace Expansion

- `{*.py,*.rs}` - Match Python and Rust files (planned)

### Examples

```python
# Find all test files
files = file_ops.walk_directory("tests", patterns=["test_*.py"])

# Find all documentation
files = file_ops.walk_directory(".", patterns=["*.md", "*.rst", "**/*.md"])

# Exclude compiled files
files = file_ops.walk_directory(
    "src",
    patterns=["*.py"],
    ignore_patterns=["*.pyc", "!*.py"]  # Exclude .pyc, keep .py
)
```

## Performance Benchmarks

### Directory Traversal

```
Test: Walk victor/ directory (3,000+ files, depth 5)
- Rust (parallel): 0.15s
- Python (os.walk): 0.35s
- Speedup: 2.3x
```

### Metadata Collection

```
Test: Collect metadata for 1,000 files
- Rust (batch): 0.08s
- Python (individual stat): 0.32s
- Speedup: 4.0x
```

### Extension Filtering

```
Test: Filter 1,000 files by extension
- Rust (set-based): 0.002s
- Python (list comprehension): 0.008s
- Speedup: 4.0x
```

## Error Handling

The module uses custom `FileOpsError` exceptions for clear error reporting:

```python
from victor.native.rust.file_ops import FileOpsError

try:
    files = file_ops.walk_directory("/nonexistent/path")
except FileOpsError as e:
    print(f"Error: {e}")
```

Common errors:
- `FileOpsError: Root directory does not exist`
- `FileOpsError: Root path is not a directory`
- `FileOpsError: Rust extension not available`

## Best Practices

### 1. Use Appropriate Max Depth

Limit traversal depth for large directory trees:

```python
# Good: Limited depth
files = file_ops.walk_directory("src", max_depth=5)

# Avoid: Unlimited depth in huge trees
files = file_ops.walk_directory("/", max_depth=1000)  # Slow!
```

### 2. Leverage Ignore Patterns

Exclude unnecessary directories:

```python
files = file_ops.walk_directory(
    "src",
    ignore_patterns=["__pycache__", ".git", "node_modules", "*.pyc"]
)
```

### 3. Batch Operations

Use batch functions for multiple files:

```python
# Good: Batch metadata collection
metadata = file_ops.get_file_metadata(paths)

# Avoid: Individual calls
metadata = [file_ops.get_file_metadata([p])[0] for p in paths]
```

### 4. Combine Filters

Chain filters for efficient querying:

```python
# Get all Python files modified recently
files = file_ops.walk_directory("src", patterns=["*.py"])
py_files = file_ops.filter_files_by_extension(files, ["py"])
recent = file_ops.filter_files_by_modified_time(py_files, since=one_day_ago)
```

## Integration with Victor

The file operations module integrates seamlessly with Victor's codebase analysis:

```python
from victor.native.rust import file_ops

# Find all Python files in codebase
code_files = file_ops.find_code_files("victor")

# Group by directory for parallel analysis
grouped = file_ops.group_files_by_directory(code_files)

# Process each directory
for dir_path, files in grouped.items():
    print(f"Analyzing {dir_path} ({len(files)} files)")
    # Integrate with AST processor, etc.
```

## Implementation Details

### Architecture

- **Rust Core**: High-performance file operations using walkdir, Rayon
- **PyO3 Bindings**: Zero-copy Python/Rust interop
- **Python Wrapper**: User-friendly API with error handling

### Key Dependencies

- `walkdir`: Efficient directory traversal
- `rayon`: Parallel processing
- `ignore`: .gitignore-style pattern matching
- `glob`: Glob pattern matching
- `pyo3`: Python bindings

### Thread Safety

All operations are thread-safe and can be used from multiple threads:

```python
from concurrent.futures import ThreadPoolExecutor

def process_directory(dir_path):
    files = file_ops.walk_directory(dir_path)
    # Process files...
    return len(files)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_directory, ["src1", "src2", "src3", "src4"])
```

## See Also

- [AST Processor](ast_processor_integration.md) - Code analysis with tree-sitter
- [Embedding Operations](embedding_ops_implementation.md) - SIMD-optimized embeddings
- [Tool Selector](tool_selector_accelerator.md) - High-performance tool selection

## License

Apache License 2.0 - See LICENSE file for details.
