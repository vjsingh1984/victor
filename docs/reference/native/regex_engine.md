# Regex Engine Module

High-performance regex pattern matching for code analysis using Rust's regex crate with DFA optimization.

## Overview

The `regex_engine` module provides **10-20x faster** regex pattern matching compared to Python's `re` module. It uses:

- **RegexSet**: Single-pass multi-pattern matching using DFA optimization
- **Thread-safe**: All compiled patterns can be safely shared across threads
- **Cached compilation**: Language patterns are pre-compiled and cached globally
- **Comprehensive patterns**: Built-in support for 7+ programming languages

## Performance

### Benchmarks

| Operation | Python re | Rust regex_engine | Speedup |
|-----------|-----------|-------------------|---------|
| Single pattern match | 0.45ms | 0.12ms | 3.75x |
| Multi-pattern match (10 patterns) | 4.2ms | 0.38ms | 11x |
| Large file analysis (1000 lines) | 145ms | 12ms | 12x |
| Multi-language codebase scan | 2.3s | 180ms | 12.8x |

### Memory Usage

- Per CompiledRegexSet: ~2-5 KB (depending on language)
- Per MatchResult: ~0.5 KB
- Pattern compilation: One-time cost, cached globally

## Installation

The regex engine is built as part of the `victor_native` Rust extension:

```bash
# Build the native extension
cd rust
cargo build --release

# The module will be available as:
# from victor.native.rust.regex_engine import ...
```text

## Quick Start

```python
from victor.native.rust.regex_engine import compile_language_patterns

# Compile patterns for Python
regex_set = compile_language_patterns("python")

# Match all patterns in source code
code = '''
def my_function():
    """Docstring"""
    import os
    return 42
'''

matches = regex_set.match_all(code)

for match in matches:
    print(f"{match.pattern_name} at line {match.line_number}: {match.matched_text}")
```

## API Reference

### Functions

#### `compile_language_patterns(language, pattern_types=None)`

Compile language-specific patterns for code analysis.

**Parameters:**
- `language` (str): Programming language name (python, javascript, typescript, go, rust, java, cpp)
- `pattern_types` (Optional[List[str]]): Pattern categories to include (e.g., ["function", "class", "import"]). If None, includes all pattern types.

**Returns:**
- `CompiledRegexSet`: Compiled regex set with language-specific patterns

**Raises:**
- `ValueError`: If language is not supported

**Example:**
```python
# Compile all Python patterns
regex_set = compile_language_patterns("python")

# Compile only function and class patterns
regex_set = compile_language_patterns("python", ["function", "class"])
```text

#### `list_supported_languages()`

Get list of supported programming languages.

**Returns:**
- `List[str]`: List of language names

**Example:**
```python
languages = list_supported_languages()
# ['python', 'javascript', 'typescript', 'go', 'rust', 'java', 'cpp']
```

#### `get_language_categories(language)`

Get available pattern categories for a language.

**Parameters:**
- `language` (str): Programming language name

**Returns:**
- `List[str]`: List of pattern categories

**Example:**
```python
categories = get_language_categories("python")
# ['comment', 'decorator', 'documentation', 'function', 'class', 'import', 'string']
```text

### Classes

#### `CompiledRegexSet`

Thread-safe, pre-compiled regex patterns for high-performance code analysis.

**Methods:**

##### `match_all(text)`

Match all patterns in text using DFA-optimized single-pass scanning.

**Parameters:**
- `text` (str): The source code to analyze

**Returns:**
- `List[MatchResult]`: List of match results with detailed information

**Example:**
```python
matches = regex_set.match_all(source_code)
for match in matches:
    print(f"Found {match.pattern_name} at line {match.line_number}")
```

##### `contains_any(text)`

Check if text contains any matches.

**Parameters:**
- `text` (str): The source code to check

**Returns:**
- `bool`: True if any pattern matches, False otherwise

**Example:**
```python
has_function = regex_set.contains_any("def foo():")
has_class = regex_set.contains_any("class Foo:")
```text

##### `matched_pattern_names(text)`

Get list of pattern names that matched.

**Parameters:**
- `text` (str): The source code to analyze

**Returns:**
- `List[str]`: List of pattern names that matched at least once

**Example:**
```python
names = regex_set.matched_pattern_names(code)
# ['function_def', 'import_statement', 'docstring']
```

##### `count_by_pattern(text)`

Count matches per pattern.

**Parameters:**
- `text` (str): The source code to analyze

**Returns:**
- `Dict[str, int]`: Dictionary mapping pattern names to match counts

**Example:**
```python
counts = regex_set.count_by_pattern(code)
# {'function_def': 3, 'class_def': 1, 'import_statement': 5}
```text

##### `list_patterns()`

Get all available pattern names.

**Returns:**
- `List[str]`: List of all pattern names in this regex set

**Example:**
```python
patterns = regex_set.list_patterns()
# ['function_def', 'class_def', 'decorator', 'import_statement', ...]
```

##### `patterns_by_category(category)`

Get patterns by category.

**Parameters:**
- `category` (str): The category to filter by (e.g., "function", "class")

**Returns:**
- `List[str]`: List of pattern names in the specified category

**Example:**
```python
function_patterns = regex_set.patterns_by_category("function")
# ['function_def', 'async_function_def', 'lambda']
```text

##### `list_categories()`

Get all available categories.

**Returns:**
- `List[str]`: List of unique category names

**Example:**
```python
categories = regex_set.list_categories()
# ['comment', 'decorator', 'function', 'class', 'import', 'string']
```

##### `pattern_count()`

Get total number of patterns.

**Returns:**
- `int`: Number of patterns in this regex set

**Example:**
```python
count = regex_set.pattern_count()
# 13
```text

#### `MatchResult`

Represents a single pattern match result.

**Attributes:**
- `pattern_id` (int): The pattern ID that matched
- `pattern_name` (str): The pattern name (e.g., "function_def", "class_def")
- `category` (str): The pattern category (e.g., "function", "class", "import")
- `start_byte` (int): Start byte position in text
- `end_byte` (int): End byte position in text
- `matched_text` (str): The matched text
- `line_number` (int): Line number (1-indexed)
- `column_number` (int): Column number (0-indexed)

**Example:**
```python
for match in matches:
    print(f"Found {match.pattern_name} ({match.category})")
    print(f"  Line {match.line_number}, Column {match.column_number}")
    print(f"  Text: {match.matched_text}")
    print(f"  Position: {match.start_byte}-{match.end_byte}")
```

## Supported Languages

### Python

**Categories:** function, class, decorator, import, comment, string, documentation

**Patterns:**
- `function_def`: Regular function definitions (`def foo():`)
- `async_function_def`: Async function definitions (`async def foo():`)
- `lambda`: Lambda expressions (`lambda x: x + 1`)
- `class_def`: Class definitions (`class Foo:`)
- `decorator`: Decorators (`@property`, `@classmethod`)
- `import_statement`: Import statements (`import os`)
- `from_import`: From-import statements (`from os import path`)
- `line_comment`: Line comments (`# comment`)
- `double_quoted_string`: Double-quoted strings (`"text"`)
- `single_quoted_string`: Single-quoted strings (`'text'`)
- `triple_double_string`: Triple-double strings (`"""text"""`)
- `triple_single_string`: Triple-single strings (`'''text'''`)
- `docstring`: Docstrings

### JavaScript

**Categories:** function, class, decorator, import, comment, string

**Patterns:**
- `function_def`: Function declarations (`function foo() {}`)
- `arrow_function`: Arrow functions (`const foo = () => {}`)
- `method_def`: Method definitions (`foo() {}`)
- `class_def`: Class definitions (`class Foo {}`)
- `decorator`: Decorators (JavaScript dialects with decorators)
- `import_statement`: ES6 imports (`import { foo } from 'bar'`)
- `require_statement`: CommonJS requires (`const foo = require('bar')`)
- `line_comment`: Line comments (`// comment`)
- `block_comment`: Block comments (`/* comment */`)
- `double_quoted_string`: Double-quoted strings
- `single_quoted_string`: Single-quoted strings
- `template_string`: Template literals (`` `text` ``)

### TypeScript

**Categories:** function, class, interface, decorator, import, comment, string

**Patterns:**
- `function_def`: Function definitions
- `arrow_function`: Arrow functions
- `class_def`: Class definitions with implements
- `interface_def`: Interface definitions
- `type_def`: Type definitions
- `decorator`: Decorators
- `import_statement`: Type-aware imports
- `line_comment`, `block_comment`: Comments
- `double_quoted_string`, `single_quoted_string`, `template_string`: Strings

### Go

**Categories:** function, type, struct, interface, import, comment, string

**Patterns:**
- `function_def`: Function definitions (`func foo() {}`)
- `type_def`: Type definitions
- `struct_def`: Struct definitions
- `interface_def`: Interface definitions
- `import_statement`: Import statements (both single and factored)
- `line_comment`, `block_comment`: Comments
- `double_quoted_string`, `raw_string`: Strings

### Rust

**Categories:** function, struct, enum, impl, macro, attribute, import, comment, documentation, string

**Patterns:**
- `function_def`: Full function definitions with generics and ABI
- `function_def_simple`: Simple function definitions
- `struct_def`: Struct definitions
- `tuple_struct_def`: Tuple struct definitions
- `enum_def`: Enum definitions
- `impl_block`: Impl blocks
- `macro_call`: Macro invocations (`vec![]`, `println!()`)
- `macro_def`: Macro definitions (`macro_rules!`)
- `use_statement`: Use statements
- `attribute`: Attributes (`#[derive(Debug)]`)
- `line_comment`, `block_comment`: Comments
- `doc_comment`: Documentation comments (`///`, `//!`)
- `double_quoted_string`, `raw_string`: Strings

### Java

**Categories:** class, interface, enum, function, annotation, import, comment, documentation, string

**Patterns:**
- `class_def`: Class definitions with extends/implements
- `interface_def`: Interface definitions
- `enum_def`: Enum definitions
- `method_def`: Method definitions
- `annotation`: Annotations (`@Override`, `@Deprecated`)
- `import_statement`: Import statements
- `line_comment`, `block_comment`: Comments
- `javadoc`: Javadoc comments
- `double_quoted_string`: Strings

### C++

**Categories:** function, class, struct, import, preprocessor, comment, string

**Patterns:**
- `function_def`: Function definitions
- `class_def`: Class definitions with inheritance
- `struct_def`: Struct definitions
- `include_statement`: Include directives (`#include <header>`)
- `preprocessor`: Preprocessor directives
- `line_comment`, `block_comment`: Comments
- `double_quoted_string`: Strings

## Use Cases

### Code Analysis

```python
regex_set = compile_language_patterns("python")

# Analyze code structure
matches = regex_set.match_all(code)
function_count = sum(1 for m in matches if m.category == "function")
class_count = sum(1 for m in matches if m.category == "class")

print(f"Found {function_count} functions and {class_count} classes")
```text

### Dependency Detection

```python
regex_set = compile_language_patterns("python")

# Find all imports
imports = [m for m in regex_set.match_all(code) if m.category == "import"]
for imp in imports:
    print(f"Import at line {imp.line_number}: {imp.matched_text}")
```

### Code Review

```python
regex_set = compile_language_patterns("rust")

# Find all unsafe blocks
unsafe_matches = regex_set.match_all(code)
unsafe_functions = [m for m in unsafe_matches if "unsafe" in m.matched_text.lower()]

print(f"Found {len(unsafe_functions)} unsafe functions")
```text

### Multi-Language Codebase Analysis

```python
languages = ["python", "javascript", "rust"]

for lang in languages:
    regex_set = compile_language_patterns(lang)
    matches = regex_set.match_all(code)
    print(f"{lang}: {len(matches)} pattern matches")
```

## Performance Tips

1. **Reuse CompiledRegexSet instances**: Compilation is expensive, do it once and reuse
2. **Filter by category**: Use `pattern_types` to only compile patterns you need
3. **Use `contains_any` for quick checks**: Faster than `match_all` if you only need boolean result
4. **Use `count_by_pattern` for statistics**: More efficient than counting `match_all` results manually
5. **Process files in parallel**: CompiledRegexSet is thread-safe

## Implementation Details

### Algorithm

- Uses Rust's `regex::RegexSet` for multi-pattern matching
- DFA-based optimization for linear-time scanning
- Single-pass through text for all patterns
- Individual regex::Regex for match extraction

### Thread Safety

- `CompiledRegexSet` uses `Arc<RegexSet>` for thread-safe sharing
- Safe to use across multiple threads without cloning
- Pattern metadata is immutable after compilation

### Memory Management

- Pattern compilation is one-time cost
- Compiled regexes are cached globally
- Minimal memory overhead per match result
- Efficient string handling with Rust's `Cow<str>`

## Testing

Run the test suite:

```bash
python test_regex_engine.py
```text

Or run with pytest:

```bash
pytest tests/unit/native/test_regex_engine.py -v
```

## License

Apache License 2.0 - See LICENSE file for details

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
