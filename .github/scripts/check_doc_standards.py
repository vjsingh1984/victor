#!/usr/bin/env python3
"""Check documentation compliance with standards."""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_file_standards(file_path: Path) -> List[str]:
    """Check a single file against documentation standards."""
    violations = []

    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
    except Exception as e:
        return [f"Error reading file: {e}"]

    # Check 1: Has H1 heading
    if not lines[0].startswith("# "):
        violations.append("Missing H1 heading (document title)")

    # Check 2: Has metadata section (Last Updated, Reading Time)
    has_last_updated = "Last Updated" in content
    has_reading_time = "Reading Time" in content or "Time Commitment" in content

    if not has_last_updated:
        violations.append("Missing 'Last Updated' date in footer")

    # Check 3: Lines are not excessively long (max 100 chars for code, 120 for text)
    for i, line in enumerate(lines, 1):
        if len(line) > 120 and not line.startswith('```'):
            if not line.startswith('|'):  # Allow tables to be longer
                violations.append(f"Line {i}: exceeds 120 characters ({len(line)} chars)")

    # Check 4: Has code examples with syntax highlighting
    code_blocks = re.findall(r'```(\w+)', content)
    plain_code_blocks = content.count('```\n') - len(code_blocks) * 2

    if plain_code_blocks > 0:
        violations.append(f"Found {plain_code_blocks} code blocks without syntax highlighting")

    # Check 5: Has links to related content
    if "See Also" not in content and "Related" not in content and "Next Steps" not in content:
        # Only check for non-index files
        if file_path.name != "index.md":
            violations.append("Missing 'See Also', 'Related', or 'Next Steps' section")

    # Check 6: Has diagrams for long content
    if len(lines) > 300:
        has_mermaid = "```mermaid" in content
        has_diagram_ref = ".mmd" in content or ".svg" in content

        if not has_mermaid and not has_diagram_ref:
            violations.append("Long content (>300 lines) should include diagrams")

    return violations


def main():
    """Main entry point."""
    docs_dir = Path("docs")

    if not docs_dir.exists():
        print(f"Error: {docs_dir} directory not found", file=sys.stderr)
        sys.exit(1)

    all_violations = []
    file_count = 0

    # Check all markdown files except archive
    for md_file in docs_dir.rglob("*.md"):
        if "docs/archive/" in str(md_file):
            continue

        file_count += 1
        violations = check_file_standards(md_file)

        if violations:
            all_violations.append((md_file, violations))

    if all_violations:
        print("❌ Documentation standards violations:", file=sys.stderr)
        print()

        for file_path, violations in all_violations:
            rel_path = file_path.relative_to(docs_dir)
            print(f"  {rel_path}:", file=sys.stderr)
            for violation in violations:
                print(f"    - {violation}", file=sys.stderr)
            print()

        print(f"Files with violations: {len(all_violations)}/{file_count}", file=sys.stderr)
        print()
        print("To fix:", file=sys.stderr)
        print("  1. Review the Documentation Standards (docs/STANDARDS.md)", file=sys.stderr)
        print("  2. Add missing metadata to file footers", file=sys.stderr)
        print("  3. Add syntax highlighting to code blocks", file=sys.stderr)
        print("  4. Keep lines under 120 characters", file=sys.stderr)
        print("  5. Add diagrams for long content", file=sys.stderr)
        print("  6. Add 'See Also' or 'Next Steps' sections", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"✅ All {file_count} files meet documentation standards")
        sys.exit(0)


if __name__ == "__main__":
    main()
