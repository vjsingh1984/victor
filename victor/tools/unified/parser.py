import re
import shlex
from typing import Optional


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


# -----------------------------------------------------------------------------
# Shell-operator detection
# -----------------------------------------------------------------------------
# `fs`/`code`/etc. are NOT shells: they pass the command through `shlex` and
# `argparse`, so shell control/redirect operators are never interpreted — they
# just produce a cryptic argparse parse error. These helpers detect such
# operators on the *post-split token list* so the tool can return a crisp,
# actionable "use the `shell` tool" message instead.
#
# Scanning tokens (not the raw string) is what makes this accurate: an operator
# inside a quoted argument (e.g. `code python "a | b"`) survives `shlex` as a
# single token whose *content* contains the character, so it is only flagged
# when the whole token — or a leading redirect run — IS the operator.
# -----------------------------------------------------------------------------

# Control operators that are only meaningful as standalone (space-separated) tokens.
_CONTROL_OPERATORS = frozenset({"|", "||", "&&", ";", "&"})

# Redirect operators. These appear either as a bare token (`>`, `2>`) or glued
# to a target path (`>/dev/null`, `2>/dev/null`, `>>log.txt`). A legitimate
# filesystem/code argument never starts with one of these, so a prefix match is
# safe. Longer prefixes are listed first so the reported operator is specific.
_REDIRECT_PREFIXES = (">>", "2>>", "1>>", "&>>", "&>", "2>", "1>", ">", "<")


def detect_shell_operators(tokens: list[str]) -> str | None:
    """Return a shell operator if ``tokens`` contain one, else ``None``.

    Args:
        tokens: The post-:func:`split_command` token list (i.e. what argparse
            would see). Pass the list *after* stripping the tool-name prefix.

    Returns:
        The offending operator string (e.g. ``"||"``, ``"2>"``) for messaging,
        or ``None`` when the command is clean.

    Notes:
        - Operators embedded inside a quoted argument are NOT flagged, because
          ``shlex`` keeps them as a single token's content (e.g. the ``|`` in
          ``code python "a | b"`` is part of the token ``a | b``).
        - Deeply embedded, unspaced operators inside an unquoted code fragment
          (e.g. a bare ``a||b`` token) are intentionally NOT flagged to avoid
          false positives on legitimate code; the common space-separated and
          redirect forms — the ones that actually break argparse — are caught.
    """
    for tok in tokens:
        if tok in _CONTROL_OPERATORS:
            return tok
        for prefix in _REDIRECT_PREFIXES:
            if tok == prefix or tok.startswith(prefix):
                return tok
    return None


def shell_operator_rejection(tool_name: str, operator: str) -> str:
    """Build the actionable "use the ``shell`` tool" rejection message."""
    return (
        "### ⚠️ SHELL OPERATOR NOT SUPPORTED\n"
        f"`{tool_name}` is not a shell — `{operator}` is not interpreted here, "
        "so this command would silently mis-parse.\n"
        "  • For pipelines / redirects / `a || b` / `a && b`, use the `shell` "
        f"tool:  `shell --cmd '<your full command>'`\n"
        f"  • Or drop the operator and run each step as its own `{tool_name}` "
        "call."
    )


# ---------------------------------------------------------------------------
# String-result outcome classification (P2 telemetry truth; also used by the
# RL emitter). Unified tools return markdown strings whose leading marker is
# the only machine-readable outcome signal.
# ---------------------------------------------------------------------------

TOOL_ERROR_MARKER = "### ❌"
TOOL_WARNING_MARKER = "### ⚠️"


def classify_result_marker(result: object) -> Optional[str]:
    """Classify a unified-tool string result by its leading marker.

    Returns ``"tool_error"`` for strings starting with ``### ❌``, ``"warning"``
    for strings starting with ``### ⚠️`` (leading whitespace tolerated), else
    ``None`` (including all non-string results — dict outcomes carry their own
    ``success`` field and are classified elsewhere).
    """
    if not isinstance(result, str):
        return None
    stripped = result.lstrip()
    if stripped.startswith(TOOL_ERROR_MARKER):
        return "tool_error"
    if stripped.startswith(TOOL_WARNING_MARKER):
        return "warning"
    return None
