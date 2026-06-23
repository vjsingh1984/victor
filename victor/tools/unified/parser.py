import shlex
import re


def split_command(command: str) -> list[str]:
    """
    Splits a shell-like command string into arguments, supporting
    triple quotes (\"\"\" and ''') which shlex does not handle natively.
    """
    # Regex to find triple quoted strings (both single and double)
    # The non-greedy .*? ensures we don't accidentally swallow text between two separate triple quotes
    pattern = re.compile(r"(\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\')", re.DOTALL)

    extracted_blocks = []

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
        return f"__MAGIC_TRIPLE_QUOTE_{len(extracted_blocks)-1}__"

    # Replace all triple quotes with placeholders
    sanitized_cmd = pattern.sub(replacer, command)

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
        if arg.startswith("__MAGIC_TRIPLE_QUOTE_") and arg.endswith("__"):
            # Extract the index
            idx_str = arg[21:-2]
            try:
                idx = int(idx_str)
                restored_args.append(extracted_blocks[idx])
            except ValueError:
                restored_args.append(arg)
        else:
            restored_args.append(arg)

    return restored_args
