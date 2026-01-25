#!/usr/bin/env python3
"""Fix specific indentation issue in chat_coordinator.py."""

file_path = "/Users/vijaysingh/code/codingagent/victor/agent/coordinators/chat_coordinator.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Fix lines 2036-2062 (0-indexed: 2035-2061)
# These lines need 4 more spaces of indentation
for i in range(2035, 2062):
    if i < len(lines):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            # Count current leading spaces
            current_spaces = len(lines[i]) - len(lines[i].lstrip(' '))
            # Should have 16 spaces total (12 for method body + 4 for if block)
            if current_spaces == 12:
                # Add 4 more spaces
                lines[i] = "    " + lines[i]

# Check if we need to add "return None" after line 2063
line_2063 = lines[2063].strip() if 2063 < len(lines) else ""
if line_2063 == ")" and (2064 >= len(lines) or "return None" not in lines[2064]):
    # Insert return None after the closing paren
    lines.insert(2064, "        return None\n")

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Fixed indentation in chat_coordinator.py")
