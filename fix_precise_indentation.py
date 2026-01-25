#!/usr/bin/env python3
"""Fix precise indentation in chat_coordinator.py."""

file_path = "/Users/vijaysingh/code/codingagent/victor/agent/coordinators/chat_coordinator.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Fix lines 2036-2062 (0-indexed: 2035-2061)
# These are the parameters inside handle_response() call
for i in range(2035, 2062):
    if i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Only fix non-empty, non-comment lines that start with exactly 12 spaces
        if stripped and not stripped.startswith("#"):
            if line.startswith("            ") and not line.startswith("                "):
                # This line has exactly 12 spaces, add 4 more
                lines[i] = "    " + line

# Add return None after line 2063 (the closing paren)
if 2063 < len(lines):
    if lines[2063].strip() == ")":
        # Check if next line already has return None
        if 2064 >= len(lines) or "return None" not in lines[2064]:
            lines.insert(2064, "        return None\n")

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Fixed indentation in chat_coordinator.py")
