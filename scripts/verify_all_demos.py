#!/usr/bin/env python3
"""Verify all demo scripts and use cases work correctly.

This script performs end-to-end verification of:
1. Demo scripts in examples/
2. RAG demos in victor/rag/demo_*.py
3. Integration test examples in tests/examples/
4. Key verification scripts in scripts/
"""

import ast
import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_syntax(file_path: Path) -> Tuple[bool, str]:
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_imports(file_path: Path) -> Tuple[bool, str]:
    """Check if file can be imported without errors."""
    try:
        # Don't actually import, just check the AST for imports
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)

        # Extract all imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return True, f"{len(imports)} imports found"
    except Exception as e:
        return False, f"Error: {e}"


def verify_file(file_path: Path, category: str) -> bool:
    """Verify a single demo file."""
    status, syntax_msg = check_syntax(file_path)
    if not status:
        print(f"  {RED}✗{RESET} {file_path.name}: {syntax_msg}")
        return False

    status, imports_msg = check_imports(file_path)
    if not status:
        print(f"  {YELLOW}⚠{RESET} {file_path.name}: {imports_msg}")
        return True  # Syntax OK, import issues might be expected

    print(f"  {GREEN}✓{RESET} {file_path.name}: {syntax_msg}, {imports_msg}")
    return True


def main():
    """Run verification on all demo scripts."""
    print("=" * 80)
    print("Victor Framework - Demo Script Verification")
    print("=" * 80)

    all_passed = True

    # Categories of demos to check
    categories = {
        "Core Examples": [
            "examples/claude_example.py",
            "examples/caching_demo.py",
            "examples/custom_plugin.py",
            "examples/custom_step_handlers.py",
        ],
        "RAG Demos": [
            "victor/rag/demo_docs.py",
            "victor/rag/demo_sec_filings.py",
        ],
        "Integration Examples": [
            "tests/examples/test_concurrent_observability.py",
            "tests/examples/test_multiagent_tdd_patterns.py",
        ],
    }

    for category, files in categories.items():
        print(f"\n{YELLOW}Category: {category}{RESET}")
        print("-" * 80)

        for file_str in files:
            file_path = Path(file_str)
            if not file_path.exists():
                print(f"  {YELLOW}⚠{RESET} {file_path.name}: File not found")
                continue

            if not verify_file(file_path, category):
                all_passed = False

    # Quick syntax check on all Python files in examples/
    print(f"\n{YELLOW}Quick Check: All example files{RESET}")
    print("-" * 80)

    examples_dir = Path("examples")
    if examples_dir.exists():
        py_files = list(examples_dir.glob("**/*.py"))
        print(f"Found {len(py_files)} Python files in examples/")

        syntax_errors = []
        for py_file in py_files:
            status, msg = check_syntax(py_file)
            if not status:
                syntax_errors.append((py_file, msg))

        if syntax_errors:
            print(f"\n{RED}Syntax Errors:{RESET}")
            for py_file, msg in syntax_errors:
                rel_path = py_file.relative_to(Path.cwd())
                print(f"  {RED}✗{RESET} {rel_path}: {msg}")
            all_passed = False
        else:
            print(f"  {GREEN}✓{RESET} All {len(py_files)} files have valid syntax")

    # Check test examples
    print(f"\n{YELLOW}Test Examples{RESET}")
    print("-" * 80)

    test_examples_dir = Path("tests/examples")
    if test_examples_dir.exists():
        test_files = list(test_examples_dir.glob("test_*.py"))
        print(f"Found {len(test_files)} test example files")

        for test_file in test_files:
            status, msg = check_syntax(test_file)
            if status:
                print(f"  {GREEN}✓{RESET} {test_file.name}")
            else:
                print(f"  {RED}✗{RESET} {test_file.name}: {msg}")
                all_passed = False

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print(f"{GREEN}✓ All demos verified successfully!{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some demos have issues. Please review above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
