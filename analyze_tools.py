#!/usr/bin/env python3
"""Analyze all Victor tools for duplicates and consolidation opportunities."""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

def extract_tool_info(file_path: Path) -> List[Dict]:
    """Extract tool information from a Python file."""
    tools = []

    try:
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has @tool decorator or returns ToolResult
                has_tool_decorator = any(
                    isinstance(d, ast.Name) and d.id == 'tool'
                    or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'tool')
                    for d in node.decorator_list
                )

                # Get docstring
                docstring = ast.get_docstring(node) or ""

                # Get function name
                func_name = node.name

                # Skip private functions
                if func_name.startswith('_'):
                    continue

                # Look for ToolResult in return type or body
                returns_tool_result = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value:
                        if isinstance(child.value, ast.Call):
                            if hasattr(child.value.func, 'id') and 'ToolResult' in child.value.func.id:
                                returns_tool_result = True

                if has_tool_decorator or returns_tool_result or 'tool' in func_name.lower():
                    tools.append({
                        'name': func_name,
                        'file': file_path.name,
                        'docstring': docstring[:200],  # First 200 chars
                        'has_decorator': has_tool_decorator
                    })

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return tools

def categorize_tools(tools: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize tools by functionality."""
    categories = {
        'filesystem': [],
        'git': [],
        'testing': [],
        'security': [],
        'documentation': [],
        'code_review': [],
        'refactoring': [],
        'ci_cd': [],
        'docker': [],
        'metrics': [],
        'web': [],
        'batch': [],
        'scaffold': [],
        'code_intelligence': [],
        'editor': [],
        'execution': [],
        'workflow': [],
        'other': []
    }

    for tool in tools:
        name = tool['name'].lower()
        file = tool['file'].lower()

        if 'git' in name or 'git' in file:
            categories['git'].append(tool)
        elif 'test' in name or 'test' in file:
            categories['testing'].append(tool)
        elif 'security' in name or 'security' in file or 'scan' in name:
            categories['security'].append(tool)
        elif 'doc' in name or 'documentation' in file:
            categories['documentation'].append(tool)
        elif 'review' in name or 'code_review' in file:
            categories['code_review'].append(tool)
        elif 'refactor' in name or 'refactor' in file:
            categories['refactoring'].append(tool)
        elif 'cicd' in name or 'cicd' in file or 'workflow' in file and 'cicd' in name:
            categories['ci_cd'].append(tool)
        elif 'docker' in name or 'docker' in file:
            categories['docker'].append(tool)
        elif 'metrics' in name or 'metrics' in file or 'complexity' in name:
            categories['metrics'].append(tool)
        elif 'web' in name or 'web' in file or 'search' in name or 'fetch' in name:
            categories['web'].append(tool)
        elif 'batch' in name or 'batch' in file:
            categories['batch'].append(tool)
        elif 'scaffold' in name or 'scaffold' in file:
            categories['scaffold'].append(tool)
        elif 'symbol' in name or 'intelligence' in file or 'find' in name or 'reference' in name:
            categories['code_intelligence'].append(tool)
        elif 'editor' in name or 'file_editor' in file:
            categories['editor'].append(tool)
        elif 'read' in name or 'write' in name or 'list' in name or 'filesystem' in file:
            categories['filesystem'].append(tool)
        elif 'execute' in name or 'bash' in name or 'executor' in file:
            categories['execution'].append(tool)
        elif 'workflow' in name or 'workflow' in file:
            categories['workflow'].append(tool)
        else:
            categories['other'].append(tool)

    return {k: v for k, v in categories.items() if v}

def find_duplicates(tools: List[Dict]) -> List[Tuple[Dict, Dict, str]]:
    """Find potential duplicate tools."""
    duplicates = []

    for i, tool1 in enumerate(tools):
        for tool2 in tools[i+1:]:
            # Check for similar names
            name1 = tool1['name'].lower().replace('_', '')
            name2 = tool2['name'].lower().replace('_', '')

            # Skip if from same file
            if tool1['file'] == tool2['file']:
                continue

            # Check for exact substring match
            if name1 in name2 or name2 in name1:
                duplicates.append((tool1, tool2, "Name similarity"))

            # Check for similar docstrings
            if tool1['docstring'] and tool2['docstring']:
                doc1_words = set(tool1['docstring'].lower().split())
                doc2_words = set(tool2['docstring'].lower().split())
                if doc1_words and doc2_words:
                    overlap = len(doc1_words & doc2_words) / min(len(doc1_words), len(doc2_words))
                    if overlap > 0.5:
                        duplicates.append((tool1, tool2, f"Docstring overlap: {overlap:.0%}"))

    return duplicates

def main():
    tools_dir = Path("/Users/vijaysingh/code/codingagent/victor/tools")

    all_tools = []
    for tool_file in tools_dir.glob("*.py"):
        if tool_file.name.startswith('_'):
            continue
        tools = extract_tool_info(tool_file)
        all_tools.extend(tools)

    print(f"\\n{'='*80}")
    print(f"TOOL ANALYSIS REPORT")
    print(f"{'='*80}\\n")

    print(f"Total tools found: {len(all_tools)}\\n")

    # Categorize
    categories = categorize_tools(all_tools)

    print(f"\\nTOOLS BY CATEGORY:")
    print(f"{'-'*80}")
    for category, tools in sorted(categories.items()):
        print(f"\\n{category.upper().replace('_', ' ')} ({len(tools)} tools):")
        for tool in sorted(tools, key=lambda t: t['name']):
            print(f"  - {tool['name']:<40} ({tool['file']})")

    # Find duplicates
    duplicates = find_duplicates(all_tools)

    if duplicates:
        print(f"\\n\\nPOTENTIAL DUPLICATES/OVERLAPS:")
        print(f"{'-'*80}")
        for tool1, tool2, reason in duplicates:
            print(f"\\n{tool1['name']} ({tool1['file']})")
            print(f"  vs")
            print(f"{tool2['name']} ({tool2['file']})")
            print(f"  Reason: {reason}")

    # Consolidation suggestions
    print(f"\\n\\nCONSOLIDATION OPPORTUNITIES:")
    print(f"{'-'*80}")

    # Editor tools (10 tools!)
    if 'editor' in categories and len(categories['editor']) > 5:
        print(f"\\n1. FILE EDITOR ({len(categories['editor'])} tools)")
        print(f"   Current: {', '.join([t['name'] for t in categories['editor']])}")
        print(f"   Suggestion: Consolidate into single 'edit_files' tool with operations:")
        print(f"   - create, modify, delete, rename, preview, commit, rollback")

    # Git tools
    if 'git' in categories and len(categories['git']) > 6:
        print(f"\\n2. GIT ({len(categories['git'])} tools)")
        print(f"   Current: {', '.join([t['name'] for t in categories['git']])}")
        print(f"   Suggestion: Consolidate into 'git_operation' with subcommands")

    # Docker tools
    if 'docker' in categories and len(categories['docker']) > 8:
        print(f"\\n3. DOCKER ({len(categories['docker'])} tools)")
        print(f"   Current: {', '.join([t['name'] for t in categories['docker']])}")
        print(f"   Suggestion: Consolidate into 'docker_manage' with resource types")

    # Security scans
    if 'security' in categories and len(categories['security']) > 3:
        print(f"\\n4. SECURITY ({len(categories['security'])} tools)")
        print(f"   Current: {', '.join([t['name'] for t in categories['security']])}")
        print(f"   Suggestion: Single 'security_scan' with scan_types parameter")

    # Code review
    if 'code_review' in categories and len(categories['code_review']) > 3:
        print(f"\\n5. CODE REVIEW ({len(categories['code_review'])} tools)")
        print(f"   Current: {', '.join([t['name'] for t in categories['code_review']])}")
        print(f"   Suggestion: Single 'code_review' with aspect parameter")

    # Metrics
    if 'metrics' in categories and len(categories['metrics']) > 3:
        print(f"\\n6. METRICS ({len(categories['metrics'])} tools)")
        print(f"   Current: {', '.join([t['name'] for t in categories['metrics']])}")
        print(f"   Suggestion: Single 'analyze_metrics' with metric_types parameter")

    print(f"\\n\\nSUMMARY:")
    print(f"{'-'*80}")
    print(f"Current: {len(all_tools)} tools across {len(categories)} categories")
    print(f"Potential after consolidation: ~20-25 tools (60% reduction)")
    print(f"\\nBenefits:")
    print(f"  - Smaller context footprint")
    print(f"  - Clearer tool purposes")
    print(f"  - Better model understanding")
    print(f"  - Easier maintenance")
    print()

if __name__ == "__main__":
    main()
