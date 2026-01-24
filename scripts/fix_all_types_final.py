#!/usr/bin/env python3
"""
Final comprehensive fix for ALL missing type parameters.
Handles nested cases like Optional[Dict], Union[List, None], etc.
"""
import re
import sys
from pathlib import Path

# Type replacements
TYPE_REPLACEMENTS = {
    'Callable': 'Callable[..., Any]',
    'Dict': 'Dict[str, Any]',
    'dict': 'dict[str, Any]',
    'List': 'List[Any]',
    'list': 'list[Any]',
    'Tuple': 'Tuple[Any, ...]',
    'tuple': 'tuple[Any, ...]',
    'Set': 'Set[Any]',
    'set': 'set[Any]',
    'frozenset': 'frozenset[Any]',
    'Type': 'Type[Any]',
    'Pattern': 'Pattern[str]',
    'Counter': 'Counter[str]',
    'deque': 'deque[Any]',
    'OrderedDict': 'OrderedDict[str, Any]',
    'ItemsView': 'ItemsView[Any, Any]',
    'ObjectPool': 'ObjectPool[Any]',
    'LRUCache': 'LRUCache[Any, Any]',
    'TimedCache': 'TimedCache[Any, Any]',
    'weakref.ref': 'weakref.ref[Any]',
    'CompletedProcess': 'CompletedProcess[Any]',
    'Queue': 'Queue[Any]',
    'Future': 'Future[Any]',
    'Awaitable': 'Awaitable[Any]',
    'Popen': 'Popen[Any]',
    'Task': 'Task[Any, Any]',
    'TaskInput': 'TaskInput[Any]',
    'CompiledGraph': 'CompiledGraph[Any]',
    'StateGraph': 'StateGraph[Any]',
    'WorkflowGraph': 'WorkflowGraph[Any]',
    'WorkflowGraphCompiler': 'WorkflowGraphCompiler[Any]',
    'GraphExecutionResult': 'GraphExecutionResult[Any]',
    'Runnable': 'Runnable[Any]',
    'MiddlewareContext': 'MiddlewareContext[Any]',
    'Middleware': 'Middleware[Any]',
    'MiddlewarePipeline': 'MiddlewarePipeline[Any]',
    'Query': 'Query[Any, Any]',
    'QueryHandler': 'QueryHandler[Any, Any]',
    'QueryHandlerFunc': 'QueryHandlerFunc[Any, Any]',
    'QueryResult': 'QueryResult[Any]',
    'CommandResult': 'CommandResult[Any]',
    'CommandHandler': 'CommandHandler[Any]',
    'Repository': 'Repository[Any, Any]',
    'SingletonRegistry': 'SingletonRegistry[Any]',
    'UniversalRegistry': 'UniversalRegistry[Any]',
    'ServiceDescriptor': 'ServiceDescriptor[Any]',
    'DeprecatedConstantDescriptor': 'DeprecatedConstantDescriptor[Any]',
}

def fix_file(file_path: Path) -> tuple[int, list[str]]:
    """Fix type parameters in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        # For each type that needs fixing
        for type_name, replacement in TYPE_REPLACEMENTS.items():
            # Multiple patterns to catch different contexts
            patterns = [
                # Optional[Type], Union[Type, ...], etc.
                rf'(\w+\[{re.escape(type_name)}\])',
                # Direct bare usage: variable: Type,
                rf':\s*{re.escape(type_name)}\s*([,\)\]\}:\n])',
                # In assignments: = Type()
                rf'=\s*{re.escape(type_name)}\s*\(',
            ]

            for pattern in patterns:
                matches = list(re.finditer(pattern, content))
                if not matches:
                    continue

                # Filter and apply fixes
                valid_matches = []
                for m in matches:
                    text = m.group(0)

                    # Skip if already has type parameters
                    if f'{type_name}[' in text:
                        # Check if it's the bare version
                        # e.g., "Dict]" but not "Dict[str, Any]]"
                        if f'{type_name}[' in text and ']' in text:
                            # Need to check if it already has params
                            idx = text.find(f'{type_name}[')
                            after = text[idx + len(type_name) + 1:]
                            if after and not after.startswith(']'):
                                # Already has type params
                                continue

                    valid_matches.append(m)

                if valid_matches:
                    # Apply replacements in reverse order
                    for m in reversed(valid_matches):
                        old_text = m.group(0)

                        # Handle different patterns
                        if pattern.startswith(r'(\w+['):
                            # Wrapper types: Optional[Dict] -> Optional[Dict[str, Any]]
                            wrapper = m.group(1)
                            wrapper_name = wrapper.split('[')[0]
                            new_text = f'{wrapper_name}[{replacement}]'
                        elif '=' in old_text:
                            # Assignment: = Dict( -> = Dict[str, Any](
                            new_text = f'= {replacement}('
                        else:
                            # Type annotation: : Dict, -> : Dict[str, Any],
                            delimiter = m.group(1) if m.lastindex and len(m.groups()) > 0 else ''
                            new_text = f': {replacement}{delimiter}'

                        content = content[:m.start()] + new_text + content[m.end():]

                    changes_made.append(f"  - {type_name} -> {replacement}: {len(valid_matches)} occurrence(s)")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(changes_made), changes_made

        return 0, []
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return 0, []

def main():
    """Main function."""
    victor_dir = Path('/Users/vijaysingh/code/codingagent/victor')

    # Find all Python files
    python_files = list(victor_dir.rglob('*.py'))

    print(f"Processing {len(python_files)} Python files...")
    print()

    total_files_modified = 0
    total_changes = 0

    for file_path in sorted(python_files):
        changes_count, changes = fix_file(file_path)
        if changes_count > 0:
            total_files_modified += 1
            total_changes += changes_count
            print(f"Modified: {file_path.relative_to(victor_dir)}")
            for change in changes:
                print(change)
            print()

    print(f"\nSummary:")
    print(f"  Files modified: {total_files_modified}")
    print(f"  Total change types applied: {total_changes}")
    print()

    return 0

if __name__ == '__main__':
    sys.exit(main())
