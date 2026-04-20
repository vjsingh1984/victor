#!/usr/bin/env python3
"""Analyze all tools for parameter count and complexity."""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

def extract_tool_functions(filepath: Path) -> List[Dict]:
    """Extract tool functions and their parameters from a file."""
    tools = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        # Find @tool decorators
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if has @tool decorator
                has_tool_decorator = any(
                    (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == 'tool') or
                    (isinstance(dec, ast.Name) and dec.id == 'tool')
                    for dec in node.decorator_list
                )

                if has_tool_decorator:
                    # Count parameters
                    params = []
                    for arg in node.args.args:
                        if arg.arg not in ['self', 'cls']:
                            params.append(arg.arg)

                    # Get defaults
                    defaults_count = len(node.args.defaults)
                    required_params = len(params) - defaults_count

                    # Get docstring
                    docstring = ast.get_docstring(node) or ""

                    tools.append({
                        'name': node.name,
                        'file': str(filepath),
                        'total_params': len(params),
                        'required_params': required_params,
                        'optional_params': defaults_count,
                        'params': params,
                        'docstring_length': len(docstring),
                    })
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return tools

def main():
    """Analyze all tools in victor/tools/ and victor-* plugins."""
    base_dir = Path('/Users/vijaysingh/code/codingagent')

    # Core victor tools
    search_paths = [
        base_dir / 'victor' / 'tools',
    ]

    # victor-* plugins
    for plugin_dir in base_dir.parent.glob('victor-*'):
        if (plugin_dir / 'victor' / 'tools').exists():
            search_paths.append(plugin_dir / 'victor' / 'tools')
        elif (plugin_dir / 'tools').exists():
            search_paths.append(plugin_dir / 'tools')

    all_tools = []
    tool_sources = []

    for tools_dir in search_paths:
        if not tools_dir.exists():
            continue

        source_name = str(tools_dir).replace('/Users/vijaysingh/code/', '').replace('codingagent/', '')

        for py_file in tools_dir.glob('*.py'):
            if py_file.name.startswith('__'):
                continue
            if py_file.name in ['base.py', 'decorators.py', 'common.py', 'enums.py']:
                continue

            tools = extract_tool_functions(py_file)
            for tool in tools:
                tool['source'] = source_name
            all_tools.extend(tools)

    # Sort by parameter count
    all_tools.sort(key=lambda x: x['total_params'], reverse=True)

    # Print report
    print("=" * 120)
    print("TOOL PARAMETER ANALYSIS")
    print("=" * 120)
    print(f"\nTotal tools found: {len(all_tools)}\n")

    # Summary statistics
    param_counts = [t['total_params'] for t in all_tools]
    avg_params = sum(param_counts) / len(param_counts) if param_counts else 0
    max_params = max(param_counts) if param_counts else 0

    print(f"Average parameters per tool: {avg_params:.1f}")
    print(f"Maximum parameters in a tool: {max_params}")
    print()

    # High-parameter tools (> 8 params)
    high_param_tools = [t for t in all_tools if t['total_params'] > 8]
    print("=" * 120)
    print(f"HIGH-PARAMETER TOOLS (> 8 parameters): {len(high_param_tools)} tools")
    print("=" * 120)

    for tool in high_param_tools:
        source = tool.get('source', tool['file'])
        print(f"\n{tool['name']}")
        print(f"  Source: {source}")
        print(f"  Total: {tool['total_params']} params | Required: {tool['required_params']} | Optional: {tool['optional_params']}")
        print(f"  Parameters: {', '.join(tool['params'][:10])}")
        if len(tool['params']) > 10:
            print(f"    ... and {len(tool['params']) - 10} more")

    # Medium-parameter tools (5-8 params)
    medium_param_tools = [t for t in all_tools if 5 <= t['total_params'] <= 8]
    print("\n" + "=" * 120)
    print(f"MEDIUM-PARAMETER TOOLS (5-8 parameters): {len(medium_param_tools)} tools")
    print("=" * 120)

    for tool in medium_param_tools[:20]:  # Show first 20
        source = tool.get('source', tool['file'])
        print(f"\n{tool['name']}")
        print(f"  Source: {source}")
        print(f"  Total: {tool['total_params']} params | Required: {tool['required_params']} | Optional: {tool['optional_params']}")

    if len(medium_param_tools) > 20:
        print(f"\n... and {len(medium_param_tools) - 20} more medium-parameter tools")

    # Low-parameter tools (< 5 params)
    low_param_tools = [t for t in all_tools if t['total_params'] < 5]
    print("\n" + "=" * 120)
    print(f"LOW-PARAMETER TOOLS (< 5 parameters): {len(low_param_tools)} tools")
    print("=" * 120)
    print("(These are generally well-designed)")

    # Find potential issues
    print("\n" + "=" * 120)
    print("POTENTIAL ISSUES")
    print("=" * 120)

    # Tools with excessive optional params
    excessive_optional = [t for t in all_tools if t['optional_params'] > 6]
    if excessive_optional:
        print(f"\n⚠️  Tools with excessive optional parameters (> 6): {len(excessive_optional)}")
        for tool in excessive_optional[:10]:
            print(f"  - {tool['name']}: {tool['optional_params']} optional params")

    # Tools with long docstrings (metadata explosion risk)
    long_docs = [t for t in all_tools if t['docstring_length'] > 2000]
    if long_docs:
        print(f"\n⚠️  Tools with very long docstrings (> 2000 chars): {len(long_docs)}")
        for tool in long_docs[:10]:
            print(f"  - {tool['name']}: {tool['docstring_length']} chars")

    # Tools with high param-to-docstring ratio (inefficient)
    inefficient = [t for t in all_tools if t['total_params'] > 5 and t['docstring_length'] < t['total_params'] * 20]
    if inefficient:
        print(f"\n⚠️  Tools with poor param-to-docstring ratio (> 5 params, < 20 chars per param): {len(inefficient)}")
        for tool in inefficient[:10]:
            print(f"  - {tool['name']}: {tool['total_params']} params, {tool['docstring_length']} chars docstring")

    print("\n" + "=" * 120)

    # Create detailed table
    print("\nDETAILED TOOL TABLE")
    print("=" * 140)
    print(f"{'Tool Name':<30} {'Source':<40} {'Total':<6} {'Required':<8} {'Optional':<8} {'Doc Length':<10}")
    print("-" * 140)

    for tool in all_tools[:50]:  # Show first 50
        source = tool.get('source', 'unknown')
        # Truncate source if too long
        if len(source) > 38:
            source = '...' + source[-35:]
        print(f"{tool['name']:<30} {source:<40} {tool['total_params']:<6} {tool['required_params']:<8} {tool['optional_params']:<8} {tool['docstring_length']:<10}")

    if len(all_tools) > 50:
        print(f"\n... and {len(all_tools) - 50} more tools")

if __name__ == '__main__':
    main()
