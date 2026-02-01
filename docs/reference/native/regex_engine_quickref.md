# Regex Engine - Quick Reference

## Installation

```bash
cd /Users/vijaysingh/code/codingagent/rust
cargo build --release
```

## Basic Usage

```python
from victor.native.rust.regex_engine import compile_language_patterns

# Compile patterns
regex_set = compile_language_patterns("python")

# Match patterns
matches = regex_set.match_all(source_code)
```

## Supported Languages

- `python` - Python
- `javascript` / `js` - JavaScript
- `typescript` / `ts` - TypeScript
- `go` / `golang` - Go
- `rust` - Rust
- `java` - Java
- `cpp` / `c++` - C++

## API Methods

### `compile_language_patterns(language, pattern_types=None)`
Compile language-specific patterns.

```python
# All patterns
regex_set = compile_language_patterns("python")

# Specific categories
regex_set = compile_language_patterns("python", ["function", "class"])
```

### `CompiledRegexSet.match_all(text)`
Match all patterns, return detailed results.

```python
matches = regex_set.match_all(code)
for match in matches:
    print(f"{match.pattern_name} at line {match.line_number}")
```

### `CompiledRegexSet.contains_any(text)`
Check if any patterns match.

```python
has_function = regex_set.contains_any("def foo():")
```

### `CompiledRegexSet.matched_pattern_names(text)`
Get names of matched patterns.

```python
names = regex_set.matched_pattern_names(code)
# ['function_def', 'import_statement', 'docstring']
```

### `CompiledRegexSet.count_by_pattern(text)`
Count matches per pattern.

```python
counts = regex_set.count_by_pattern(code)
# {'function_def': 3, 'class_def': 1, 'import_statement': 5}
```

### `CompiledRegexSet.list_patterns()`
List all pattern names.

```python
patterns = regex_set.list_patterns()
# ['function_def', 'async_function_def', 'class_def', ...]
```

### `CompiledRegexSet.patterns_by_category(category)`
Get patterns in a category.

```python
function_patterns = regex_set.patterns_by_category("function")
# ['function_def', 'async_function_def', 'lambda']
```

### `CompiledRegexSet.list_categories()`
List all categories.

```python
categories = regex_set.list_categories()
# ['comment', 'decorator', 'function', 'class', 'import', 'string']
```

### `CompiledRegexSet.pattern_count()`
Get total number of patterns.

```python
count = regex_set.pattern_count()
# 13
```

## MatchResult Attributes

```python
match.pattern_id      # int: Pattern index
match.pattern_name    # str: Pattern name
match.category        # str: Pattern category
match.start_byte      # int: Start position
match.end_byte        # int: End position
match.matched_text    # str: Matched text
match.line_number     # int: Line number (1-indexed)
match.column_number   # int: Column number (0-indexed)
```

## Pattern Categories

### Python
- `function` - Function definitions
- `class` - Class definitions
- `decorator` - Decorators
- `import` - Import statements
- `comment` - Comments
- `string` - String literals
- `documentation` - Docstrings

### JavaScript
- `function` - Function definitions
- `class` - Class definitions
- `decorator` - Decorators
- `import` - Import/require statements
- `comment` - Comments
- `string` - String literals

### TypeScript
- `function` - Function definitions
- `class` - Class definitions
- `interface` - Interface definitions
- `decorator` - Decorators
- `import` - Import statements
- `comment` - Comments
- `string` - String literals

### Go
- `function` - Function definitions
- `type` - Type definitions
- `struct` - Struct definitions
- `interface` - Interface definitions
- `import` - Import statements
- `comment` - Comments
- `string` - String literals

### Rust
- `function` - Function definitions
- `struct` - Struct definitions
- `enum` - Enum definitions
- `impl` - Impl blocks
- `macro` - Macros
- `attribute` - Attributes
- `import` - Use statements
- `comment` - Comments
- `documentation` - Doc comments
- `string` - String literals

### Java
- `class` - Class definitions
- `interface` - Interface definitions
- `enum` - Enum definitions
- `function` - Method definitions
- `annotation` - Annotations
- `import` - Import statements
- `comment` - Comments
- `documentation` - Javadoc
- `string` - String literals

### C++
- `function` - Function definitions
- `class` - Class definitions
- `struct` - Struct definitions
- `import` - Include directives
- `preprocessor` - Preprocessor directives
- `comment` - Comments
- `string` - String literals

## Common Patterns

### Count Functions and Classes

```python
regex_set = compile_language_patterns("python")
matches = regex_set.match_all(code)

function_count = sum(1 for m in matches if m.category == "function")
class_count = sum(1 for m in matches if m.category == "class")
```

### Find All Imports

```python
regex_set = compile_language_patterns("python")
matches = regex_set.match_all(code)
imports = [m for m in matches if m.category == "import"]
```

### Filter by Line Range

```python
matches = regex_set.match_all(code)
matches_in_range = [m for m in matches if 10 <= m.line_number <= 50]
```

### Group by Category

```python
matches = regex_set.match_all(code)
by_category = {}
for match in matches:
    if match.category not in by_category:
        by_category[match.category] = []
    by_category[match.category].append(match)
```

## Performance Tips

1. **Reuse CompiledRegexSet** - Compilation is expensive
2. **Filter by category** - Use `pattern_types` to limit patterns
3. **Use `contains_any`** - Faster than `match_all` for boolean checks
4. **Use `count_by_pattern`** - More efficient than counting manually
5. **Process in parallel** - Thread-safe, can use multiprocessing

## Testing

```bash
# Unit tests
pytest tests/unit/native/test_regex_engine.py -v

# Integration test
python test_regex_engine.py
```

## Files

- **Rust**: `/Users/vijaysingh/code/codingagent/rust/src/regex_engine.rs`
- **Python**: `/Users/vijaysingh/code/codingagent/victor/native/rust/regex_engine.py`
- **Docs**: `/Users/vijaysingh/code/codingagent/docs/native/regex_engine.md`
- **Tests**: `/Users/vijaysingh/code/codingagent/tests/unit/native/test_regex_engine.py`

## Performance

- **Single pattern**: 3.75x faster than Python re
- **Multi-pattern (10)**: 11x faster
- **Large file (1000 lines)**: 12x faster
- **Multi-language scan**: 12.8x faster

## Support

For issues or questions:
- Documentation: `/Users/vijaysingh/code/codingagent/docs/native/regex_engine.md`
- Summary: `/Users/vijaysingh/code/codingagent/docs/native/REGEX_ENGINE_SUMMARY.md`

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
