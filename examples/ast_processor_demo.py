#!/usr/bin/env python3
"""Example demonstrating the AST Processor Accelerator.

This example shows how to use the Rust-backed AST processor for
high-performance code analysis operations.
"""

from victor.native.accelerators import get_ast_processor


def main():
    """Demonstrate AST processor capabilities."""
    print("=" * 60)
    print("AST Processor Accelerator Demo")
    print("=" * 60)

    # Get the singleton instance
    processor = get_ast_processor(max_cache_size=1000)

    # Check backend availability
    print(f"\nBackend Information:")
    print(f"  Rust Available: {processor.is_rust_available()}")
    print(f"  Version: {processor.get_version()}")

    # Sample Python code
    python_code = '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
'''

    print("\n" + "-" * 60)
    print("1. Parsing Python Code")
    print("-" * 60)

    # Parse the code
    ast = processor.parse_to_ast(python_code, "python", "example.py")
    print(f"✓ Parsed successfully")

    # Show parse statistics
    stats = processor.parse_stats
    print(f"  Total parses: {stats['total_parses']}")
    print(f"  Avg duration: {stats['avg_duration_ms']:.2f}ms")

    print("\n" + "-" * 60)
    print("2. Extracting Functions")
    print("-" * 60)

    # Extract function definitions
    functions = processor.extract_symbols(ast, ["function_definition"])
    print(f"✓ Found {len(functions)} functions")
    for func in functions:
        name = func.get("name", "unknown")
        line = func.get("start_line", 0)
        print(f"  - {name} (line {line})")

    print("\n" + "-" * 60)
    print("3. Extracting Classes")
    print("-" * 60)

    # Extract class definitions
    classes = processor.extract_symbols(ast, ["class_definition"])
    print(f"✓ Found {len(classes)} classes")
    for cls in classes:
        name = cls.get("name", "unknown")
        line = cls.get("start_line", 0)
        print(f"  - {name} (line {line})")

    print("\n" + "-" * 60)
    print("4. Custom Query: Functions with Type Hints")
    print("-" * 60)

    # Execute custom query for functions with return type hints
    query = """
    (function_definition
        name: (identifier) @name
        return_type: (type) @return_type)
    """
    results = processor.execute_query(ast, query)
    print(f"✓ Found {results.matches} functions with type hints")
    print(f"  Query duration: {results.duration_ms:.2f}ms")

    print("\n" + "-" * 60)
    print("5. Parallel Processing: Multiple Files")
    print("-" * 60)

    # Process multiple files in parallel
    files = [
        ("python", "def foo(): pass\n"),
        ("python", "def bar(): pass\nclass Baz: pass\n"),
        ("javascript", "function qux() { return 42; }\n"),
    ]

    results = processor.extract_symbols_parallel(
        files, symbol_types=["function_definition", "class_definition"]
    )

    for idx, symbols in results.items():
        lang = files[idx][0]
        print(f"  File {idx} ({lang}): {len(symbols)} symbols")

    print("\n" + "-" * 60)
    print("6. Cache Statistics")
    print("-" * 60)

    cache_stats = processor.cache_stats
    print(f"  Cache size: {cache_stats.get('size', 'N/A')}")
    print(f"  Max size: {cache_stats.get('max_size', 'N/A')}")

    parse_stats = processor.parse_stats
    print(f"  Total parses: {parse_stats['total_parses']}")
    print(f"  Cache hits: {parse_stats['cache_hits']}")
    print(f"  Cache misses: {parse_stats['cache_misses']}")
    print(f"  Hit rate: {parse_stats['cache_hit_rate']:.1f}%")

    print("\n" + "-" * 60)
    print("7. Supported Languages")
    print("-" * 60)

    languages = processor.get_supported_languages()
    print(f"✓ Supports {len(languages)} languages")
    print(f"  Sample: {', '.join(languages[:10])}...")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
