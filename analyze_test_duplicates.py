#!/usr/bin/env python3
"""Analyze test files across load/performance/benchmark modules for true duplicates.

This script examines test content (not just names) to identify:
1. Tests with identical purpose and assertions (true duplicates)
2. Tests with similar names but different purposes (different systems)
3. Tests that can be consolidated (same system, different coverage)
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TestInfo:
    """Information about a test function."""
    file_path: str
    test_name: str
    docstring: str
    assertions: List[str]
    system_under_test: str
    purpose: str


class TestAnalyzer(ast.NodeVisitor):
    """Extract test information from AST."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tests: List[TestInfo] = []

    def visit_FunctionDef(self, node):
        if node.name.startswith('test_'):
            # Get docstring
            docstring = ast.get_docstring(node) or ""

            # Find assertions in the function body
            assertions = self._extract_assertions(node)

            # Extract system under test from imports or docstring
            system_under_test = self._extract_system_under_test(node, docstring)

            # Extract purpose from docstring first line
            purpose = docstring.split('\n')[0] if docstring else ""

            self.tests.append(TestInfo(
                file_path=self.file_path,
                test_name=node.name,
                docstring=docstring,
                assertions=assertions,
                system_under_test=system_under_test,
                purpose=purpose
            ))

        self.generic_visit(node)

    def _extract_assertions(self, node) -> List[str]:
        """Extract assertion statements from function body."""
        assertions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                # Convert AST back to source-like string
                assertion = ast.unparse(child) if hasattr(ast, 'unparse') else str(child.lineno)
                assertions.append(assertion)
        return assertions

    def _extract_system_under_test(self, node, docstring: str) -> str:
        """Extract what system/class is being tested."""
        # Try to extract from docstring
        if "for " in docstring.lower():
            match = re.search(r'(?:for|testing|benchmark)\s+([A-Z][a-zA-Z0-9_]+)', docstring, re.IGNORECASE)
            if match:
                return match.group(1)

        # Check for common patterns
        if "cache" in docstring.lower() or "cache" in node.name.lower():
            return "Cache"
        if "memory" in docstring.lower() or "memory" in node.name.lower():
            return "Memory"
        if "throughput" in docstring.lower() or "latency" in docstring.lower():
            return "Performance"
        if "team" in node.name.lower():
            return "TeamNode"
        if "tool" in node.name.lower() and "selection" in node.name.lower():
            return "ToolSelection"

        return "Unknown"


def analyze_file(file_path: str) -> List[TestInfo]:
    """Analyze a single test file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        analyzer = TestAnalyzer(file_path)
        analyzer.visit(tree)
        return analyzer.tests
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []


def find_duplicates_by_name(tests: List[TestInfo]) -> Dict[str, List[TestInfo]]:
    """Group tests by name to find potential duplicates."""
    grouped = defaultdict(list)
    for test in tests:
        grouped[test.test_name].append(test)
    return grouped


def find_similar_purpose(tests: List[TestInfo]) -> List[Tuple[str, List[TestInfo]]]:
    """Find tests with similar purposes."""
    # Group by keywords in purpose
    keyword_groups = defaultdict(list)

    for test in tests:
        # Extract key terms from purpose
        keywords = re.findall(r'(cache|memory|throughput|latency|performance|leak|concurrent|parallel)', test.purpose.lower())
        for keyword in keywords:
            keyword_groups[keyword].append(test)

    # Return groups with multiple tests
    return [(kw, tests) for kw, tests in keyword_groups.items() if len(tests) > 1]


def compare_test_content(test1: TestInfo, test2: TestInfo) -> float:
    """Compare two tests to estimate similarity (0-1)."""
    similarity = 0.0

    # Same system under test
    if test1.system_under_test == test2.system_under_test:
        similarity += 0.3

    # Similar purpose (words in common)
    words1 = set(test1.purpose.lower().split())
    words2 = set(test2.purpose.lower().split())
    if words1 and words2:
        intersection = words1 & words2
        union = words1 | words2
        similarity += 0.4 * (len(intersection) / len(union))

    # Similar assertion count
    if test1.assertions and test2.assertions:
        assert_ratio = min(len(test1.assertions), len(test2.assertions)) / max(len(test1.assertions), len(test2.assertions))
        similarity += 0.3 * assert_ratio

    return similarity


def main():
    # Find all test files in the three directories
    test_dirs = [
        'tests/benchmark',
        'tests/performance',
        'tests/load'
    ]

    all_tests: List[TestInfo] = []

    print("Scanning test files...")
    for test_dir in test_dirs:
        path = Path(test_dir)
        if path.exists():
            for py_file in path.glob('test_*.py'):
                print(f"  Analyzing {py_file}...")
                tests = analyze_file(str(py_file))
                all_tests.extend(tests)

    print(f"\nFound {len(all_tests)} total tests")

    # Group by test name
    print("\n" + "="*80)
    print("DUPLICATES BY TEST NAME")
    print("="*80)

    grouped = find_duplicates_by_name(all_tests)

    duplicates_found = 0
    for test_name, test_list in sorted(grouped.items()):
        if len(test_list) > 1:
            duplicates_found += 1
            print(f"\n{test_name} ({len(test_list)} occurrences):")
            for test in test_list:
                rel_path = test.file_path.replace('/Users/vijaysingh/code/codingagent/', '')
                print(f"  - {rel_path}")
                print(f"    System: {test.system_under_test}")
                print(f"    Purpose: {test.purpose[:80]}...")

    print(f"\nTotal tests with duplicate names: {duplicates_found}")

    # Find similar purposes
    print("\n" + "="*80)
    print("SIMILAR TEST PURPOSES")
    print("="*80)

    similar = find_similar_purpose(all_tests)
    for keyword, test_list in similar:
        print(f"\nKeyword: {keyword} ({len(test_list)} tests)")
        # Group by system
        by_system = defaultdict(list)
        for test in test_list:
            by_system[test.system_under_test].append(test)
        for system, tests in by_system.items():
            print(f"  {system}: {len(tests)} tests")

    # Find high-similarity test pairs
    print("\n" + "="*80)
    print("HIGH SIMILARITY TEST PAIRS (>70% similarity)")
    print("="*80)

    high_similarity_pairs = []
    for i, test1 in enumerate(all_tests):
        for test2 in all_tests[i+1:]:
            if test1.test_name != test2.test_name:
                continue
            similarity = compare_test_content(test1, test2)
            if similarity > 0.7:
                high_similarity_pairs.append((test1, test2, similarity))

    for test1, test2, similarity in high_similarity_pairs:
        print(f"\n{test1.test_name}: {similarity:.1%} similar")
        print(f"  File 1: {test1.file_path.split('/')[-1]}")
        print(f"  System 1: {test1.system_under_test}")
        print(f"  File 2: {test2.file_path.split('/')[-1]}")
        print(f"  System 2: {test2.system_under_test}")

    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Check specific duplicates found earlier
    specific_duplicates = [
        "test_memory_leak_detection",
        "test_team_node_performance_summary",
        "test_cache_hit_rate",
        "test_formation_performance",
    ]

    for test_name in specific_duplicates:
        if test_name in grouped and len(grouped[test_name]) > 1:
            test_list = grouped[test_name]
            print(f"\n{test_name}:")
            print(f"  Found in {len(test_list)} files")

            # Check if testing same system
            systems = set(t.system_under_test for t in test_list)
            if len(systems) == 1:
                print(f"  ✓ All test same system: {systems.pop()}")
                print(f"  Recommendation: Consolidate into most comprehensive test")
            else:
                print(f"  ✗ Test different systems: {', '.join(systems)}")
                print(f"  Recommendation: Rename tests to be more specific")


if __name__ == "__main__":
    main()
