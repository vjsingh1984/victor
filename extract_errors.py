#!/usr/bin/env python3
"""Extract errors 601-650 from mypy log."""

import re

errors = []
with open("/tmp/mypy_errors.log", "r") as f:
    for line in f:
        match = re.match(r"^(\d+)->(.+)", line)
        if match:
            error_num = int(match.group(1))
            if 601 <= error_num <= 650:
                errors.append((error_num, match.group(2)))

# Group by file
from collections import defaultdict

by_file = defaultdict(list)
for error_num, error_line in errors:
    # Extract file path
    file_match = re.match(r"^([^:]+):\d+:", error_line)
    if file_match:
        by_file[file_match.group(1)].append((error_num, error_line))

# Print summary
print(f"Total errors 601-650: {len(errors)}")
print(f"\nFiles affected: {len(by_file)}")
print("\n" + "=" * 80)

for file_path in sorted(by_file.keys()):
    error_list = by_file[file_path]
    print(f"\n{file_path}: {len(error_list)} errors")
    for error_num, error_line in error_list:
        print(f"  {error_num}: {error_line[:100]}...")
