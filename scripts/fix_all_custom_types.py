#!/usr/bin/env python3
"""
Comprehensive fix for ALL missing type parameters including framework types.
"""
import re
import sys
from pathlib import Path

# Complete mapping of all types needing parameters
ALL_TYPE_FIXES = {
    # Standard library types
    "Callable": ("Callable[..., Any]", None),
    "Dict": ("Dict[str, Any]", None),
    "dict": ("dict[str, Any]", None),
    "List": ("List[Any]", None),
    "list": ("list[Any]", None),
    "Tuple": ("Tuple[Any, ...]", None),
    "tuple": ("tuple[Any, ...]", None),
    "Set": ("Set[Any]", None),
    "set": ("set[Any]", None),
    "frozenset": ("frozenset[Any]", None),
    "Type": ("Type[Any]", None),
    "Pattern": ("Pattern[str]", None),
    "Counter": ("Counter[str]", None),
    "deque": ("deque[Any]", None),
    "OrderedDict": ("OrderedDict[str, Any]", None),
    "ItemsView": ("ItemsView[Any, Any]", None),
    "ObjectPool": ("ObjectPool[Any]", None),
    "LRUCache": ("LRUCache[Any, Any]", None),
    "TimedCache": ("TimedCache[Any, Any]", None),
    "weakref.ref": ("weakref.ref[Any]", None),
    "CompletedProcess": ("CompletedProcess[Any]", None),
    "Queue": ("Queue[Any]", None),
    "Future": ("Future[Any]", None),
    "Awaitable": ("Awaitable[Any]", None),
    "Popen": ("Popen[Any]", None),
    # Framework types
    "Task": ("Task[Any, Any]", None),
    "TaskInput": ("TaskInput[Any]", None),
    "CompiledGraph": ("CompiledGraph[Any]", None),
    "StateGraph": ("StateGraph[Any]", None),
    "WorkflowGraph": ("WorkflowGraph[Any]", None),
    "WorkflowGraphCompiler": ("WorkflowGraphCompiler[Any]", None),
    "GraphExecutionResult": ("GraphExecutionResult[Any]", None),
    "Runnable": ("Runnable[Any]", None),
    "MiddlewareContext": ("MiddlewareContext[Any]", None),
    "Middleware": ("Middleware[Any]", None),
    "MiddlewarePipeline": ("MiddlewarePipeline[Any]", None),
    "Query": ("Query[Any, Any]", None),
    "QueryHandler": ("QueryHandler[Any, Any]", None),
    "QueryHandlerFunc": ("QueryHandlerFunc[Any, Any]", None),
    "QueryResult": ("QueryResult[Any]", None),
    "CommandResult": ("CommandResult[Any]", None),
    "CommandHandler": ("CommandHandler[Any]", None),
    "Repository": ("Repository[Any, Any]", None),
    "SingletonRegistry": ("SingletonRegistry[Any]", None),
    "UniversalRegistry": ("UniversalRegistry[Any]", None),
    "ServiceDescriptor": ("ServiceDescriptor[Any]", None),
    "DeprecatedConstantDescriptor": ("DeprecatedConstantDescriptor[Any]", None),
}


def fix_file(file_path: Path) -> tuple[int, list[str]]:
    """Fix type parameters in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        changes_made = []

        # Apply each fix
        for type_name, (replacement, _) in ALL_TYPE_FIXES.items():
            # Pattern to match bare type annotations
            # Match: : Type followed by delimiter, but NOT: Type[...]
            delimiter_class = r"[,\)\]\}:\n]"
            patterns = [
                rf":\s*{re.escape(type_name)}\s*({delimiter_class})",
            ]

            for pattern in patterns:
                matches = list(re.finditer(pattern, content))
                if matches:
                    # Filter out matches that already have [
                    valid_matches = []
                    for m in matches:
                        # Check if next non-space char is [
                        pos = m.end()
                        while (
                            pos < len(content) and content[pos].isspace() and content[pos] != "\n"
                        ):
                            pos += 1
                        if pos >= len(content) or (
                            content[pos] != "[" and not content[pos].isalpha()
                        ):
                            valid_matches.append(m)

                    if valid_matches:
                        # Apply replacements in reverse order
                        for m in reversed(valid_matches):
                            delimiter = m.group(1)
                            old_text = m.group(0)
                            new_text = f": {replacement}{delimiter}"
                            content = content[: m.start()] + new_text + content[m.end() :]

                        changes_made.append(
                            f"  - {type_name} -> {replacement}: {len(valid_matches)} occurrence(s)"
                        )

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return len(changes_made), changes_made

        return 0, []
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return 0, []


def main():
    """Main function."""
    victor_dir = Path("/Users/vijaysingh/code/codingagent/victor")

    # Find all Python files
    python_files = list(victor_dir.rglob("*.py"))

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


if __name__ == "__main__":
    sys.exit(main())
