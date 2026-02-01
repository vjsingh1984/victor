#!/usr/bin/env python3
"""Add reading time estimates and last updated dates to documentation."""

import re
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Average reading speed: 200 words per minute
WORDS_PER_MINUTE = 200


def estimate_reading_time(content: str) -> str:
    """Estimate reading time for content."""
    # Remove code blocks from word count
    content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content_no_code = re.sub(r'`[^`]+`', '', content_no_code)

    # Count words
    words = len(content_no_code.split())

    # Calculate minutes
    minutes = max(1, round(words / WORDS_PER_MINUTE))

    if minutes < 3:
        return f"{minutes} min"
    else:
        return f"{minutes} minutes"


def has_metadata(content: str) -> Tuple[bool, bool]:
    """Check if file has reading time and last updated metadata."""
    has_reading_time = (
        "**Reading Time:**" in content or
        "**Time Commitment:**" in content or
        "Time:" in content
    )
    has_last_updated = "**Last Updated:**" in content

    return has_reading_time, has_last_updated


def add_metadata_footer(content: str, file_path: Path) -> str:
    """Add metadata footer to content."""
    lines = content.split('\n')

    # Check if footer already exists
    if any("**Last Updated:**" in line for line in lines[-10:]):
        return content  # Already has footer

    # Calculate reading time
    reading_time = estimate_reading_time(content)

    # Get current date
    last_updated = datetime.now().strftime("%B %d, %Y")

    # Create footer
    footer = f"""

---

**Last Updated:** {last_updated}
**Reading Time:** {reading_time}
"""

    # Add footer before last blank line or at end
    content = content.rstrip() + footer

    return content


def process_file(file_path: Path) -> bool:
    """Process a single documentation file."""
    try:
        content = file_path.read_text(encoding='utf-8')

        # Skip if already has metadata
        has_reading_time, has_last_updated = has_metadata(content)
        if has_reading_time and has_last_updated:
            return False

        # Add metadata
        updated_content = add_metadata_footer(content, file_path)

        # Write back
        file_path.write_text(updated_content, encoding='utf-8')
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    docs_dir = Path("docs")

    # Skip these directories
    skip_dirs = {
        "docs/archive",
        "docs/diagrams",
        "docs/templates",
    }

    processed = 0
    skipped = 0

    for md_file in docs_dir.rglob("*.md"):
        # Skip if in skip directory
        if any(str(skip_dir) in str(md_file) for skip_dir in skip_dirs):
            skipped += 1
            continue

        # Skip templates
        if "templates/" in str(md_file):
            skipped += 1
            continue

        if process_file(md_file):
            processed += 1
            print(f"✓ Updated {md_file.relative_to(docs_dir)}")
        else:
            skipped += 1

    print()
    print(f"Processed: {processed} files")
    print(f"Skipped: {skipped} files (already have metadata)")


if __name__ == "__main__":
    main()
