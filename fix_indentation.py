#!/usr/bin/env python3
file_path = "/Users/vijaysingh/code/codingagent/victor/agent/coordinators/chat_coordinator.py"

with open(file_path, "r") as f:
    lines = f.readlines()

# Fix indentation for lines 2036-2063 (in 0-indexed: 2035-2062)
for i in range(2035, 2063):
    if i < len(lines):
        line = lines[i]
        # If line has 12 spaces at start (should be 16)
        if line.startswith("            ") and not line.startswith("                "):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                # Add 4 more spaces
                lines[i] = "    " + line

# Add return None after line 2063 (after the closing paren)
if 2063 < len(lines) and lines[2063].strip() == ")":
    # Insert return None at line 2064
    lines.insert(2064, "        return None\n")

with open(file_path, "w") as f:
    f.writelines(lines)

print("Fixed indentation and added return None")
