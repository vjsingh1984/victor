import shlex
import re


def split_command(command: str) -> list[str]:
    """
    Splits a shell-like command string into arguments, supporting
    triple quotes (\"\"\" and ''') which shlex does not handle natively.
    """
    extracted_blocks: list[str] = []

    def placeholder(index: int) -> str:
        return f"__VICTOR_CMD_BLOCK_{index}__"

    # Extract heredocs before triple quotes so Python docstrings inside
    # heredoc bodies are preserved as raw content.
    heredoc_pattern = re.compile(
        r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[ \t]*\n(.*?)\n[ \t]*\2(?=\s|$)",
        re.DOTALL,
    )

    def heredoc_replacer(match):
        extracted_blocks.append(match.group(3))
        return placeholder(len(extracted_blocks) - 1)

    sanitized_cmd = heredoc_pattern.sub(heredoc_replacer, command)

    # Regex to find triple quoted strings (both single and double).
    # The non-greedy .*? avoids swallowing text between separate triple quotes.
    pattern = re.compile(r"(\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\')", re.DOTALL)

    def replacer(match):
        # We store the inner content, stripping the actual triple quotes
        if match.group(2) is not None:
            # Matched """..."""
            content = match.group(2)
        else:
            # Matched '''...'''
            content = match.group(3)

        extracted_blocks.append(content)
        # Return a safe, unquoted placeholder string that shlex won't split
        return placeholder(len(extracted_blocks) - 1)

    # Replace all triple quotes with placeholders
    sanitized_cmd = pattern.sub(replacer, sanitized_cmd)

    # Split using shlex
    try:
        args = shlex.split(sanitized_cmd)
    except ValueError:
        # Fallback if shlex fails (e.g. mismatched single/double quotes)
        # Try to just split on spaces as a rough fallback
        args = sanitized_cmd.split()

    # Restore the extracted blocks
    restored_args = []
    for arg in args:
        if arg.startswith("__VICTOR_CMD_BLOCK_") and arg.endswith("__"):
            # Extract the index
            idx_str = arg[len("__VICTOR_CMD_BLOCK_") : -2]
            try:
                idx = int(idx_str)
                restored_args.append(extracted_blocks[idx])
            except ValueError:
                restored_args.append(arg)
        else:
            restored_args.append(arg)

    return restored_args
