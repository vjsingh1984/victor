#!/usr/bin/env python3
"""
Comprehensive doc linting fixes for remaining violations.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

def add_see_also_section(content: str, file_path: Path) -> Tuple[str, int]:
    """Add See Also section if missing and appropriate."""
    # Don't add to index files or very short files
    if file_path.name == "index.md" or len(content.split('\n')) < 50:
        return content, 0

    if "See Also" in content or "Related" in content or "Next Steps" in content:
        return content, 0

    # Add See Also section before ---
    lines = content.split('\n')

    # Find the last ---
    last_dash = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '---':
            last_dash = i
            break

    if last_dash == -1:
        return content, 0

    # Add See Also section
    see_also = "\n## See Also\n\n- [Documentation Home](../../README.md)\n"

    if last_dash < len(lines) - 1:
        lines.insert(last_dash + 1, see_also)
    else:
        lines.append(see_also)

    return '\n'.join(lines), 1

def fix_long_lines_aggressive(content: str) -> Tuple[str, int]:
    """Fix long lines more aggressively."""
    fixes = 0
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        if len(line) <= 120:
            fixed_lines.append(line)
            continue

        # Skip tables, code blocks, references
        if (line.startswith('|') or
            line.strip().startswith('```') or
            line.strip().startswith('>') or
            line.strip().startswith('- ') or
            line.strip().startswith('* ')):
            fixed_lines.append(line)
            continue

        # Break long inline code
        if '`' in line and line.count('`') >= 2:
            # Find code spans and break line
            parts = re.split(r'(`[^`]+`)', line)
            new_line = parts[0]
            for i, part in enumerate(parts[1:], 1):
                if i < len(parts) - 1:
                    # Not the last part
                    if len(new_line) + len(part) > 100:
                        new_line += '\n' + part
                        fixes += 1
                    else:
                        new_line += part
                else:
                    new_line += part
            fixed_lines.append(new_line)
            continue

        # Break long lines at logical break points
        if ', ' in line and len(line) > 150:
            # Comma-separated list
            parts = line.split(', ')
            if len(parts) > 1:
                new_lines = []
                current = parts[0]
                for part in parts[1:]:
                    if len(current) + len(part) + 2 > 120:
                        new_lines.append(current + ',')
                        current = '  ' + part
                        fixes += 1
                    else:
                        current += ', ' + part
                new_lines.append(current)
                fixed_lines.extend(new_lines)
                continue

        # Last resort: break at space
        if ' ' in line:
            words = line.split(' ')
            new_line = words[0]
            for word in words[1:]:
                if len(new_line) + len(word) + 1 > 120:
                    fixed_lines.append(new_line)
                    new_line = '  ' + word
                    fixes += 1
                else:
                    new_line += ' ' + word
            fixed_lines.append(new_line)
            continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes

def fix_h1_heading(content: str, file_path: Path) -> Tuple[str, int]:
    """Add H1 heading if missing."""
    lines = content.split('\n')

    # Check if first line is H1
    if lines[0].startswith("# "):
        return content, 0

    # Check for title in first few lines
    title = None
    for i, line in enumerate(lines[:10]):
        if line.startswith("# "):
            # Found H1, but not at start
            if i > 0:
                # Move to top
                title = lines[i]
                del lines[i]
                break
            return content, 0
        elif line.startswith("#") and not line.startswith("##"):
            # This is H1, move to top
            title = line
            del lines[i]
            break
        elif line.strip() and not line.startswith("#"):
            # Found text, use as title
            title = f"# {line}"
            del lines[i]
            break

    if title:
        lines.insert(0, title)
        lines.insert(1, "")
        return '\n'.join(lines), 1

    return content, 0

def fix_missing_metadata(content: str, file_path: Path) -> Tuple[str, int]:
    """Ensure all required metadata is present."""
    fixes = 0

    # Check for Last Updated
    if "Last Updated:" not in content:
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")

        # Add before the last ---
        lines = content.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == '---' and i > 0:
                lines.insert(i, f'\n**Last Updated:** {today}**')
                fixes += 1
                break
        content = '\n'.join(lines)

    # Check for Reading Time
    if 'Reading Time:' not in content and 'Time Commitment:' not in content:
        word_count = len(content.split())
        minutes = max(1, word_count // 200)

        if '**Last Updated:**' in content:
            content = content.replace(
                '**Last Updated:**',
                f'**Reading Time:** {minutes} min\n**Last Updated:**'
            )
            fixes += 1

    return content, fixes

def fix_file_comprehensive(file_path: Path) -> Tuple[int, List[str]]:
    """Apply comprehensive fixes to a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        fixes = 0
        details = []

        # Apply all fixes
        content, updated = fix_missing_metadata(content, file_path)
        fixes += updated

        content, updated = fix_h1_heading(content, file_path)
        fixes += updated
        if updated:
            details.append("Added H1 heading")

        content, updated = add_see_also_section(content, file_path)
        fixes += updated
        if updated:
            details.append("Added 'See Also' section")

        content, updated = fix_long_lines_aggressive(content)
        fixes += updated
        if updated:
            details.append(f"Fixed {updated} long lines")

        if fixes > 0:
            file_path.write_text(content, encoding='utf-8')

        return fixes, details

    except Exception as e:
        return 0, [f"Error: {e}"]

def main():
    """Main entry point."""
    docs_dir = Path("docs")

    if not docs_dir.exists():
        print(f"Error: {docs_dir} not found")
        sys.exit(1)

    print("🔧 Applying comprehensive fixes...\n")

    total_fixes = 0
    fixed_files = 0

    # Process all markdown files
    md_files = list(docs_dir.rglob("*.md"))

    for md_file in md_files:
        if "node_modules" in str(md_file):
            continue

        fixes, details = fix_file_comprehensive(md_file)

        if fixes > 0:
            fixed_files += 1
            total_fixes += fixes
            rel_path = md_file.relative_to(docs_dir)
            print(f"✅ {rel_path}: {fixes} fixes")
            for detail in details:
                print(f"   - {detail}")

    print(f"\n📊 Summary:")
    print(f"   Fixed files: {fixed_files}")
    print(f"   Total fixes: {total_fixes}")

if __name__ == "__main__":
    main()
