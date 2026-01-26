#!/usr/bin/env python3
"""Analyze remaining mypy type-arg errors."""
import subprocess
import re

result = subprocess.run(
    ["mypy", "victor/", "--config-file", "pyproject.toml"],
    capture_output=True,
    text=True,
    cwd="/Users/vijaysingh/code/codingagent",
)

type_counts = {}
for line in result.stdout.split("\n"):
    if "type-arg" in line:
        match = re.search(r'Missing type parameters for generic type "([^"]+)"', line)
        if match:
            type_name = match.group(1)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

print("Remaining type-arg errors by type:")
for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {type_name}: {count}")
print(f"\nTotal: {sum(type_counts.values())}")
