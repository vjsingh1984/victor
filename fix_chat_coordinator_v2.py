#!/usr/bin/env python3
"""Fix MyPy type errors in chat_coordinator.py for batch A4 (errors 151-200)."""

import sys


def apply_fixes():
    """Apply type fixes line by line."""
    file_path = "victor/agent/coordinators/chat_coordinator.py"

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Track modifications
    modified = False
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Fix 1: Line 2018 - Add None check for _recovery_integration
        if i == 2033 and "if orch._recovery_integration:" in line:
            # This is already good, fix indentation on next line
            if i + 1 < len(lines) and lines[i + 1].startswith("            return await"):
                # Fix indentation
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith(")"):
                    if lines[j].startswith("            ") and not lines[j].startswith("                "):
                        lines[j] = "                " + lines[j].strip() + "\n"
                        j += 1
                        modified = True
                    else:
                        j += 1

                # Add return None after the closing parenthesis
                # Find the line with the closing paren
                k = j
                while k < len(lines) and not lines[k].strip() == ")":
                    k += 1
                if k < len(lines):
                    # Insert return None after the closing paren line
                    lines.insert(k + 1, "        return None\n")
                    modified = True

        output_lines.append(line)
        i += 1

    if modified:
        with open(file_path, "w") as f:
            f.writelines(lines)
        print(f"Fixed {file_path}")
        return True
    else:
        print("No changes needed")
        return False


if __name__ == "__main__":
    apply_fixes()
