# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-based tool call extraction for models that output function calls as text.

Many open-weight models hosted on OpenAI-compatible providers (Cerebras, Groq,
Together, etc.) sometimes output tool calls as Python-like function calls in text
instead of proper structured JSON tool_calls. For example:

    read_file(path='victor/agent/orchestrator.py')
    shell(command="ls -la")
    edit(file_path="/path/to/file", old_string="foo", new_string="bar")

This module provides extraction logic to parse these text-based calls into
structured tool call format that Victor can execute.

Design principles (SOLID):
- Single Responsibility: Only handles text-to-tool-call extraction
- Open/Closed: New patterns can be added via EXTRACTION_PATTERNS
- Dependency Inversion: Returns standard ToolCall objects

Usage:
    from victor.agent.tool_calling.text_extractor import PythonCallExtractor

    extractor = PythonCallExtractor()
    result = extractor.extract_from_text(
        "I'll read the file: read_file(path='foo.py')",
        valid_tool_names={"read_file", "write_file", "shell"}
    )
    if result.tool_calls:
        for tc in result.tool_calls:
            print(f"Tool: {tc.name}, Args: {tc.arguments}")
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractedToolCall:
    """A tool call extracted from text.

    Attributes:
        name: Tool/function name
        arguments: Parsed arguments as dict
        raw_text: Original text that was parsed
        start_pos: Start position in source text
        end_pos: End position in source text
        confidence: Confidence score (0.0-1.0)
    """

    name: str
    arguments: Dict[str, Any]
    raw_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.8


@dataclass
class ExtractionResult:
    """Result of text-based tool call extraction.

    Attributes:
        tool_calls: List of extracted tool calls
        remaining_content: Content after removing tool calls
        parse_method: Method used for extraction
        confidence: Overall confidence score
        warnings: Any warnings during extraction
    """

    tool_calls: List[ExtractedToolCall] = field(default_factory=list)
    remaining_content: str = ""
    parse_method: str = "python_call"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether extraction found any tool calls."""
        return len(self.tool_calls) > 0


# Regex patterns for Python-like function calls
# Pattern matches: function_name(arg1='value', arg2="value", arg3=123, ...)
PYTHON_CALL_PATTERN = re.compile(
    r"""
    (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)  # Function name
    \s*\(\s*                           # Opening parenthesis
    (?P<args>[^)]*?)                   # Arguments (non-greedy)
    \s*\)                              # Closing parenthesis
    """,
    re.VERBOSE,
)

# Alternative pattern for multi-line calls with complex arguments
MULTILINE_CALL_PATTERN = re.compile(
    r"""
    (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)  # Function name
    \s*\(\s*                           # Opening parenthesis
    (?P<args>(?:[^()]*|\([^()]*\))*)   # Arguments (handles nested parens)
    \s*\)                              # Closing parenthesis
    """,
    re.VERBOSE | re.DOTALL,
)

# Known tool name patterns (common Victor tools)
KNOWN_TOOL_PATTERNS = frozenset(
    [
        "read",
        "read_file",
        "write",
        "write_file",
        "edit",
        "edit_file",
        "shell",
        "bash",
        "run",
        "grep",
        "find",
        "ls",
        "search",
        "replace",
        "extract",
        "analyze",
        "graph",
        "metrics",
        "scan",
        "review",
        "refs",
        "symbol",
        "overview",
        "deps",
        "workflow",
        "sandbox",
        "jira",
        "cicd",
        "docs_coverage",
        "organize_imports",
        "rename",
        "inline",
    ]
)


class PythonCallExtractor:
    """Extracts Python-like function calls from text content.

    This extractor handles cases where models output tool calls as text
    instead of structured JSON, common with open-weight models on various
    hosting providers.

    Supported formats:
    - read_file(path='foo.py')
    - shell(command="ls -la", timeout=30)
    - edit(file_path='/path', old_string="x", new_string="y")

    The extractor is designed to be:
    - Non-greedy: Only extracts calls that match known tool names
    - Safe: Uses AST parsing for argument extraction where possible
    - Configurable: Accepts custom valid tool name sets
    """

    def __init__(
        self,
        known_tools: Optional[Set[str]] = None,
        strict_mode: bool = False,
    ):
        """Initialize the extractor.

        Args:
            known_tools: Set of known/valid tool names. If None, uses defaults.
            strict_mode: If True, only extract calls for known tools.
        """
        self._known_tools = known_tools or KNOWN_TOOL_PATTERNS
        self._strict_mode = strict_mode

    def extract_from_text(
        self,
        content: str,
        valid_tool_names: Optional[Set[str]] = None,
    ) -> ExtractionResult:
        """Extract Python-like function calls from text content.

        Args:
            content: Text content to parse
            valid_tool_names: Optional set of valid tool names to filter by.
                If provided, only extracts calls matching these names.

        Returns:
            ExtractionResult with extracted tool calls and metadata
        """
        if not content or not content.strip():
            return ExtractionResult(remaining_content=content)

        # Combine known tools with valid names for filtering
        filter_names: Set[str] = set(valid_tool_names or self._known_tools)  # type: ignore[arg-type]
        if valid_tool_names:
            filter_names |= set(self._known_tools)  # type: ignore[arg-type]

        tool_calls: List[ExtractedToolCall] = []
        warnings: List[str] = []
        positions_to_remove: List[Tuple[int, int]] = []

        # Try simple pattern first
        for match in PYTHON_CALL_PATTERN.finditer(content):
            name = match.group("name")
            args_str = match.group("args")

            # Filter by known/valid tool names
            if self._strict_mode and name not in filter_names:
                continue

            # Skip if name doesn't look like a tool
            if not self._is_likely_tool_name(name, filter_names):
                continue

            # Parse arguments
            parsed_args, parse_warning = self._parse_arguments(args_str)
            if parse_warning:
                warnings.append(f"{name}: {parse_warning}")
                # Still try to use partial results
                if not parsed_args:
                    continue

            # Calculate confidence based on various factors
            confidence = self._calculate_confidence(name, parsed_args, filter_names)

            tool_calls.append(
                ExtractedToolCall(
                    name=name,
                    arguments=parsed_args,
                    raw_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                )
            )
            positions_to_remove.append((match.start(), match.end()))

        # Calculate remaining content (remove extracted calls)
        remaining = self._remove_positions(content, positions_to_remove)

        # Calculate overall confidence
        overall_confidence = 0.0
        if tool_calls:
            overall_confidence = sum(tc.confidence for tc in tool_calls) / len(tool_calls)

        return ExtractionResult(
            tool_calls=tool_calls,
            remaining_content=remaining.strip(),
            parse_method="python_call",
            confidence=overall_confidence,
            warnings=warnings,
        )

    def _is_likely_tool_name(self, name: str, valid_names: Set[str]) -> bool:
        """Check if a name is likely a tool name.

        Args:
            name: Function name to check
            valid_names: Set of valid tool names

        Returns:
            True if name looks like a tool name
        """
        # Direct match
        if name in valid_names:
            return True

        # Check against known patterns
        if name in self._known_tools:
            return True

        # Heuristics for tool-like names
        name_lower = name.lower()

        # Common tool name suffixes/prefixes
        tool_indicators = [
            "_file",
            "_dir",
            "_directory",
            "_code",
            "_text",
            "read_",
            "write_",
            "edit_",
            "get_",
            "set_",
            "list_",
            "find_",
            "search_",
            "run_",
            "execute_",
        ]

        for indicator in tool_indicators:
            if indicator in name_lower:
                return True

        # Avoid common Python builtins and methods
        builtins_to_skip = {
            "print",
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "bool",
            "type",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "open",
            "input",
            "format",
            "sorted",
            "reversed",
            "min",
            "max",
            "sum",
            "abs",
            "round",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
        }

        if name_lower in builtins_to_skip:
            return False

        return False

    def _parse_arguments(self, args_str: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse Python-style keyword arguments from a string.

        Handles formats like:
        - path='foo.py'
        - command="ls -la", timeout=30
        - file_path='/path/to/file', content="multi\\nline"

        Args:
            args_str: Argument string without parentheses

        Returns:
            Tuple of (parsed_dict, warning_message or None)
        """
        if not args_str or not args_str.strip():
            return {}, None

        args_str = args_str.strip()

        # Method 1: Try AST parsing (safest)
        try:
            # Wrap in a function call for AST parsing
            fake_call = f"_f({args_str})"
            tree = ast.parse(fake_call, mode="eval")

            if isinstance(tree.body, ast.Call):
                args_dict = {}
                for keyword in tree.body.keywords:
                    if keyword.arg:
                        value = self._ast_to_value(keyword.value)
                        args_dict[keyword.arg] = value

                # Also handle positional args (rare but possible)
                for i, arg in enumerate(tree.body.args):
                    value = self._ast_to_value(arg)
                    args_dict[f"arg_{i}"] = value

                return args_dict, None
        except (SyntaxError, ValueError):
            pass  # Fall through to regex parsing

        # Method 2: Regex-based parsing (fallback)
        return self._parse_arguments_regex(args_str)

    def _ast_to_value(self, node: ast.AST) -> Any:
        """Convert AST node to Python value.

        Args:
            node: AST node

        Returns:
            Python value (str, int, float, bool, None, list, dict)
        """
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compat
            return node.s
        elif isinstance(node, ast.Num):  # Python 3.7 compat
            return node.n
        elif isinstance(node, ast.NameConstant):  # Python 3.7 compat
            return node.value
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._ast_to_value(k) if k else None: self._ast_to_value(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Name):
            # Handle True, False, None as names
            if node.id == "True":
                return True
            elif node.id == "False":
                return False
            elif node.id == "None":
                return None
            return node.id  # Return as string
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers
            return -self._ast_to_value(node.operand)
        else:
            # Unknown node type - return string representation
            return ast.dump(node)

    def _parse_arguments_regex(self, args_str: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse arguments using regex (fallback method).

        Args:
            args_str: Argument string

        Returns:
            Tuple of (parsed_dict, warning_message or None)
        """
        args_dict: Dict[str, Any] = {}
        warning = None

        # Pattern for key=value pairs
        # Handles: key='value', key="value", key=123, key=True
        arg_pattern = re.compile(
            r"""
            (\w+)\s*=\s*        # Key name and equals
            (?:
                '([^']*)'       # Single-quoted string
                |"([^"]*)"      # Double-quoted string
                |(\d+\.?\d*)    # Number
                |(True|False|None)  # Boolean/None
                |(\w+)          # Unquoted identifier
            )
            """,
            re.VERBOSE,
        )

        for match in arg_pattern.finditer(args_str):
            key = match.group(1)
            # Check which capture group matched
            if match.group(2) is not None:
                value = match.group(2)
            elif match.group(3) is not None:
                value = match.group(3)
            elif match.group(4) is not None:
                num_str = match.group(4)
                value = float(num_str) if "." in num_str else int(num_str)
            elif match.group(5) is not None:
                bool_str = match.group(5)
                value = {"True": True, "False": False, "None": None}.get(bool_str)
            elif match.group(6) is not None:
                value = match.group(6)
            else:
                continue

            args_dict[key] = value

        if not args_dict and args_str.strip():
            warning = f"Could not parse arguments: {args_str[:50]}..."

        return args_dict, warning

    def _calculate_confidence(
        self,
        name: str,
        args: Dict[str, Any],
        valid_names: Set[str],
    ) -> float:
        """Calculate confidence score for an extracted tool call.

        Args:
            name: Tool name
            args: Parsed arguments
            valid_names: Set of valid tool names

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # Boost for exact match in valid names
        if name in valid_names:
            confidence += 0.3

        # Boost for known tool patterns
        if name in self._known_tools:
            confidence += 0.15

        # Boost if has arguments (less likely to be false positive)
        if args:
            confidence += 0.1

        # Boost for common argument names
        common_args = {"path", "file_path", "command", "content", "query", "pattern"}
        if any(arg in common_args for arg in args.keys()):
            confidence += 0.1

        return min(confidence, 1.0)

    def _remove_positions(
        self,
        content: str,
        positions: List[Tuple[int, int]],
    ) -> str:
        """Remove specified positions from content.

        Args:
            content: Original content
            positions: List of (start, end) positions to remove

        Returns:
            Content with positions removed
        """
        if not positions:
            return content

        # Sort positions in reverse order to avoid index shifting
        sorted_positions = sorted(positions, key=lambda x: x[0], reverse=True)

        result = content
        for start, end in sorted_positions:
            result = result[:start] + result[end:]

        return result


# Singleton instance for convenience
_default_extractor: Optional[PythonCallExtractor] = None


def get_extractor() -> PythonCallExtractor:
    """Get the default PythonCallExtractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = PythonCallExtractor()
    return _default_extractor


def extract_tool_calls_from_text(
    content: str,
    valid_tool_names: Optional[Set[str]] = None,
) -> ExtractionResult:
    """Convenience function to extract tool calls from text.

    Args:
        content: Text content to parse
        valid_tool_names: Optional set of valid tool names

    Returns:
        ExtractionResult with extracted tool calls
    """
    return get_extractor().extract_from_text(content, valid_tool_names)
