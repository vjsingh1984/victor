#!/usr/bin/env python3
"""
Final comprehensive fix for ALL missing type parameters.
"""
import re
import sys
from pathlib import Path

# Type replacements
TYPE_REPLACEMENTS = {
    "Callable": "Callable[..., Any]",
    "Dict": "Dict[str, Any]",
    "dict": "dict[str, Any]",
    "List": "List[Any]",
    "list": "list[Any]",
    "Tuple": "Tuple[Any, ...]",
    "tuple": "tuple[Any, ...]",
    "Set": "Set[Any]",
    "set": "set[Any]",
    "frozenset": "frozenset[Any]",
    "Type": "Type[Any]",
    "Pattern": "Pattern[str]",
    "Counter": "Counter[str]",
    "deque": "deque[Any]",
    "OrderedDict": "OrderedDict[str, Any]",
    "Task": "Task[Any, Any]",
    "TaskInput": "TaskInput[Any]",
    "CompiledGraph": "CompiledGraph[Any]",
    "StateGraph": "StateGraph[Any]",
    "Middleware": "Middleware[Any]",
    "MiddlewareContext": "MiddlewareContext[Any]",
    "Runnable": "Runnable[Any]",
    "Popen": "Popen[Any]",
    "Future": "Future[Any]",
    "Awaitable": "Awaitable[Any]",
}


def fix_file(file_path: Path) -> tuple[int, int]:
    """Fix type parameters in a file. Return (fixes_applied, errors_fixed)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        total_fixes = 0

        # For each type
        for type_name, replacement in TYPE_REPLACEMENTS.items():
            # Fix wrapper types: Optional[Dict] -> Optional[Dict[str, Any]]
            wrapper_pattern = r"((?:Optional|Union)\[" + re.escape(type_name) + r"\])"
            matches = list(re.finditer(wrapper_pattern, content))
            for m in reversed(matches):
                wrapper_text = m.group(1)
                wrapper_name = wrapper_text.split("[")[0]
                new_text = f"{wrapper_name}[{replacement}]"
                content = content[: m.start()] + new_text + content[m.end() :]
                total_fixes += 1

            # Fix bare annotations: : Dict, -> : Dict[str, Any],
            annotation_pattern = r":\s*" + re.escape(type_name) + r"\s*([,\)\]\}:\n])"
            matches = list(re.finditer(annotation_pattern, content))
            for m in reversed(matches):
                delimiter = m.group(1)
                new_text = f": {replacement}{delimiter}"
                content = content[: m.start()] + new_text + content[m.end() :]
                total_fixes += 1

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return 1, total_fixes

        return 0, 0
    except Exception as e:
        print(f"  Error in {file_path}: {e}", file=sys.stderr)
        return 0, 0


def main():
    """Main function."""
    victor_dir = Path("/Users/vijaysingh/code/codingagent/victor")
    python_files = list(victor_dir.rglob("*.py"))

    print(f"Processing {len(python_files)} Python files...")

    total_files = 0
    total_fixes = 0

    for file_path in sorted(python_files):
        files_fixed, fixes = fix_file(file_path)
        if files_fixed:
            total_files += files_fixed
            total_fixes += fixes
            print(f"Fixed: {file_path.relative_to(victor_dir)} ({fixes} fixes)")

    print(f"\nTotal files modified: {total_files}")
    print(f"Total fixes applied: {total_fixes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
