#!/usr/bin/env python3
"""
Auto-fix common documentation linting issues.

This script fixes:
1. Missing 'Last Updated' dates
2. Code blocks without syntax highlighting
3. Some long lines (URLs, table rows)
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Language mappings for common code patterns
LANGUAGE_MAP = {
    # Python
    r'```.*python.*?\n': '```python\n',
    r'```.*py.*?\n': '```python\n',
    r'```.*pip.*?\n': '```bash\n',

    # Shell/Bash
    r'```.*bash.*?\n': '```bash\n',
    r'```.*shell.*?\n': '```bash\n',
    r'```.*sh.*?\n': '```bash\n',

    # JavaScript/TypeScript
    r'```.*javascript.*?\n': '```javascript\n',
    r'```.*js.*?\n': '```javascript\n',
    r'```.*typescript.*?\n': '```typescript\n',
    r'```.*ts.*?\n': '```typescript\n',

    # YAML
    r'```.*yaml.*?\n': '```yaml\n',
    r'```.*yml.*?\n': '```yaml\n',

    # JSON
    r'```.*json.*?\n': '```json\n',

    # Other common languages
    r'```.*markdown.*?\n': '```markdown\n',
    r'```.*md.*?\n': '```markdown\n',
    r'```.*dockerfile.*?\n': '```dockerfile\n',
    r'```.*makefile.*?\n': '```makefile\n',
    r'```.*sql.*?\n': '```sql\n',
    r'```.*html.*?\n': '```html\n',
    r'```.*css.*?\n': '```css\n',
    r'```.*xml.*?\n': '```xml\n',
    r'```.*mermaid.*?\n': '```mermaid\n',

    # Commands
    r'```.*command.*?\n': '```bash\n',
    r'```.*terminal.*?\n': '```bash\n',
    r'```.*console.*?\n': '```bash\n',
}

def fix_last_updated(content: str, file_path: Path) -> Tuple[str, int]:
    """Add or update Last Updated date."""
    today = datetime.now().strftime("%B %d, %Y")

    # Check if file already has Last Updated
    if "Last Updated:" in content:
        # Update existing date
        content = re.sub(
            r'\*\*Last Updated:\*\* [^\n]+\*\*',
            f'**Last Updated:** {today}**',
            content
        )
        return content, 1

    # Add Last Updated before the end
    if content.endswith('---\n') or '---' in content.split('\n')[-5:]:
        # Insert before the last ---
        lines = content.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == '---':
                lines.insert(i, f'\n**Last Updated:** {today}**\n')
                break
        return '\n'.join(lines), 1
    else:
        # Add at end
        content = content.rstrip() + f'\n\n---\n\n**Last Updated:** {today}**\n'
        return content, 1

def fix_code_blocks(content: str) -> Tuple[str, int]:
    """Add syntax highlighting to code blocks."""
    fixes = 0

    # Find all code blocks without language specification
    def replace_code_block(match):
        nonlocal fixes
        code_content = match.group(1)

        # Try to detect language from content
        for pattern, lang in LANGUAGE_MAP.items():
            if re.search(pattern, code_content, re.IGNORECASE):
                fixes += 1
                return f'```\n{code_content}'

        # Default to bash for commands
        if any(cmd in code_content for cmd in ['curl ', 'git ', 'npm ', 'pip ', 'apt-get']):
            fixes += 1
            return f'```bash\n{code_content}'

        return match.group(0)

    # Replace ``` with ```language
    content = re.sub(
        r'```\n(.*?)(?=```\n|$)',
        replace_code_block,
        content,
        flags=re.DOTALL
    )

    return content, fixes

def fix_long_lines(content: str, file_path: Path) -> Tuple[str, int]:
    """Fix overly long lines where possible."""
    fixes = 0
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Skip if line is already short enough
        if len(line) <= 120:
            fixed_lines.append(line)
            continue

        # Skip tables and code blocks
        if line.startswith('|') or line.strip().startswith('```'):
            fixed_lines.append(line)
            continue

        # Fix long URLs by breaking them
        if 'http://' in line or 'https://' in line:
            # Break long URLs
            url_pattern = r'(https?://[^\s]+)'
            matches = list(re.finditer(url_pattern, line))

            if matches and len(matches) == 1:
                # Single long URL - keep as is
                fixed_lines.append(line)
            else:
                # Multiple URLs or no URLs - try to break
                for match in matches:
                    url = match.group(0)
                    if len(url) > 80:
                        # Break the line at URL
                        url_start = line[:match.start()]
                        url_end = line[match.end():]
                        fixed_lines.append(url_start)
                        fixed_lines.append(url)
                        fixed_lines.append(url_end)
                        fixes += 1
                        break
                else:
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes

def add_reading_time(content: str) -> Tuple[str, int]:
    """Add Reading Time estimate if missing."""
    if 'Reading Time:' in content or 'Time Commitment:' in content:
        return content, 0

    # Estimate reading time (200 words per minute)
    word_count = len(content.split())
    minutes = max(1, word_count // 200)

    # Add before Last Updated
    if '**Last Updated:**' in content:
        reading_time = f'**Reading Time:** {minutes} min'
        content = content.replace('**Last Updated:**', f'{reading_time}\n**Last Updated:**')
        return content, 1

    return content, 0

def fix_file(file_path: Path) -> Tuple[int, List[str]]:
    """Fix a single documentation file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        fixes = 0
        details = []

        # Apply fixes
        content, updated = fix_last_updated(content, file_path)
        fixes += updated
        if updated:
            details.append("Added/Updated 'Last Updated'")

        content, updated = fix_code_blocks(content)
        fixes += updated
        if updated:
            details.append(f"Added syntax highlighting to {updated} code blocks")

        content, updated = fix_long_lines(content, file_path)
        fixes += updated
        if updated:
            details.append(f"Fixed {updated} long lines")

        content, updated = add_reading_time(content)
        fixes += updated
        if updated:
            details.append("Added reading time estimate")

        # Write back if changed
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            return fixes, details

        return 0, []

    except Exception as e:
        return 0, [f"Error: {e}"]

def main():
    """Main entry point."""
    docs_dir = Path("docs")

    if not docs_dir.exists():
        print(f"Error: {docs_dir} not found")
        sys.exit(1)

    total_fixes = 0
    fixed_files = 0

    print("🔧 Auto-fixing documentation linting issues...\n")

    # Fix all markdown files
    for md_file in docs_dir.rglob("*.md"):
        if "node_modules" in str(md_file):
            continue

        fixes, details = fix_file(md_file)

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
    print(f"\n✨ Auto-fix complete! Run check_doc_standards.py to verify.")

if __name__ == "__main__":
    main()
