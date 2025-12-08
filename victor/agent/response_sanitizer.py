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

"""Response sanitization for Victor.

Handles cleaning and validation of model responses, including:
- Removing malformed XML/HTML tags
- Filtering training data leakage patterns
- Validating tool names
- Detecting garbage content from local models
- AST-based code extraction and cleanup (for LLM-generated code)
- Markdown code block extraction
- Python syntax validation
"""

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CodeSanitizationResult:
    """Result of code sanitization."""

    code: str  # The sanitized code
    is_valid: bool  # Whether the code is syntactically valid
    errors: List[str] = field(default_factory=list)  # List of errors/issues found
    fixes_applied: List[str] = field(default_factory=list)  # List of fixes applied
    function_names: List[str] = field(default_factory=list)  # Names of functions found


class ResponseSanitizer:
    """Sanitizes and validates model responses.

    Handles common issues from local models:
    - Repeated </function> or </parameter> tags
    - XML-like formatting artifacts
    - JSON-like tool call attempts in plain text
    - Instruction/example leakage from model training
    """

    # Patterns indicating training data leakage
    LEAKAGE_PATTERNS: List[str] = [
        r"Do not invent any new or additional parameters.*",
        r"The parameter value should be passed as a string.*",
        r"If you want to call multiple functions.*",
        r"Do NOT surround the function call.*",
        r"All parameters are required unless.*",
        r"The agent is not allowed to directly access.*",
        r"Begin by calling list_directory.*",
        r"execute_bash\(command=.*\)",
        r"No files read yet\. Avoid file-specific claims.*",
        r'list_directory\(path="[^"]*"\)',
        r'read_file\(path="[^"]*"\)',
    ]

    # Patterns indicating garbage/malformed output
    GARBAGE_PATTERNS: List[str] = [
        r"FUNCTION_CALL\s*\{",  # Raw function call syntax
        r"</function>\s*</function>",  # Repeated closing tags
        r"<parameter[^>]*>",  # Raw parameter tags
        r'^\s*\{\s*"name":\s*"[^"]+",\s*"arguments":',  # Raw JSON tool calls
        r"^\s*<IMPORTANT>",  # Instruction leakage
        r"^\s*<important>",  # Instruction leakage (lowercase)
        r"</important>",  # Instruction leakage closing tag
        r"^\s*Do NOT",  # Instruction leakage
        r"^\s*NEVER\s+",  # Instruction leakage
        r"^\s*ALWAYS include",  # Instruction leakage
        r"^\s*- Do NOT",  # List-style instruction leakage
        r"^\s*- Use lowercase",  # Formatting instruction leakage
        r"^\s*- The parameters",  # Parameter instruction leakage
        r"the XML tag",  # Format instruction leakage
        r"backticks.*markdown",  # Format instruction leakage
        r"JSON parsing",  # Technical instruction leakage
        r"file system structure",  # Security instruction leakage
        r"\[TOOL_REQUEST\]",  # LMStudio default format leakage
        r"\[END_TOOL_REQUEST\]",  # LMStudio default format leakage
    ]

    # Thinking token patterns to strip (DeepSeek, Qwen, and other reasoning models)
    # These tokens leak through in content when running locally via Ollama
    THINKING_TOKEN_PATTERNS: List[str] = [
        r"<｜end▁of▁thinking｜>",  # DeepSeek end-of-thinking marker
        r"<｜begin▁of▁thinking｜>",  # DeepSeek begin-of-thinking marker
        r"<\|end_of_thinking\|>",  # ASCII variant
        r"<\|begin_of_thinking\|>",  # ASCII variant
        r"<think>.*?</think>",  # Qwen3 thinking blocks
        r"</think>",  # Orphaned Qwen3 thinking close tag
        r"<think>",  # Orphaned Qwen3 thinking open tag
    ]

    # Patterns for invalid/hallucinated tool names
    INVALID_TOOL_PATTERNS: List[str] = [
        r"^example_",
        r"^func_",
        r"^function_",
        r"^tool_name",
        r"^my_",
        r"^test_tool",
        r"^sample_",
        r"/$",  # Ends with slash
        r"^<",  # Starts with XML tag
        r">$",  # Ends with XML tag
        r"\s",  # Contains whitespace
        r"^\d",  # Starts with number
    ]

    # ================================================================
    # Code Sanitization Patterns (for LLM-generated code)
    # ================================================================

    # Markdown code block extraction
    MARKDOWN_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    # LLM artifacts to remove from code
    CODE_ARTIFACTS = [
        re.compile(
            r"^Here(?:'s| is) (?:the |a |my )?(?:complete |corrected |fixed )?"
            r"(?:implementation|solution|code|function).*?:",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^(?:The )?(?:complete |corrected |fixed )?"
            r"(?:implementation|solution|code|function).*?:",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(r"^Note:.*$", re.MULTILINE),
        re.compile(r"^Explanation:.*$", re.MULTILINE),
        re.compile(r"^Output:.*$", re.MULTILINE),
        re.compile(r"^Example:.*$", re.MULTILINE),
        re.compile(r"^Test:.*$", re.MULTILINE),
        re.compile(r"^Usage:.*$", re.MULTILINE),
    ]

    # Patterns for detecting incomplete code
    INCOMPLETE_CODE_PATTERNS = [
        re.compile(r"\.\.\.$"),  # Trailing ellipsis
        re.compile(
            r"#\s*(?:TODO|FIXME|XXX|continue|rest of|implementation|more|etc).*$",
            re.IGNORECASE,
        ),
        re.compile(r"pass\s*#.*(?:implement|add|fill|complete).*$", re.IGNORECASE),
    ]

    def strip_markup(self, text: str) -> str:
        """Remove simple XML/HTML-like tags to salvage plain text.

        Args:
            text: Text potentially containing markup

        Returns:
            Plain text with markup removed
        """
        if not text:
            return text
        cleaned = re.sub(r"<[^>]+>", " ", text)
        return " ".join(cleaned.split())

    def sanitize(self, text: str) -> str:
        """Sanitize model response by removing malformed patterns.

        Handles common issues from local models:
        - Repeated </function> or </parameter> tags
        - XML-like formatting artifacts
        - JSON-like tool call attempts in plain text
        - Instruction/example leakage from model training

        Args:
            text: Raw response text from the model

        Returns:
            Cleaned text suitable for display
        """
        if not text:
            return text

        original_len = len(text)

        # Remove repeated closing tags (</function>, </parameter>, etc.)
        # These indicate the model is confused about tool calling format
        text = re.sub(r"(</\w+>\s*){3,}", "", text)

        # Remove orphaned XML-like tags
        text = re.sub(r"</?function[^>]*>", "", text)
        text = re.sub(r"</?parameter[^>]*>", "", text)
        text = re.sub(r"</?tool[^>]*>", "", text)
        text = re.sub(r"</?IMPORTANT[^>]*>", "", text)

        # Remove thinking tokens from reasoning models (DeepSeek, Qwen3)
        # Note: Don't use DOTALL for multi-character patterns to avoid stripping content
        for pattern in self.THINKING_TOKEN_PATTERNS:
            # Only use DOTALL for the <think>...</think> block pattern
            if ".*?" in pattern:
                # For think blocks, only match single-line content to be safe
                text = re.sub(pattern, "", text)
            else:
                text = re.sub(pattern, "", text)

        # Remove training data leakage patterns
        for pattern in self.LEAKAGE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove JSON-like tool call attempts embedded in text
        # Pattern: {"name": "tool_name", ...}
        text = re.sub(r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}', "", text)

        # Remove lines that are just tool call syntax
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that look like raw tool call attempts
            if stripped.startswith('{"name":') or stripped.startswith("</"):
                continue
            # Skip lines that are just parameter= syntax
            if re.match(r"^(parameter=|<parameter)", stripped):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Clean up excessive whitespace
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = text.strip()

        # Log if significant content was removed (indicates model confusion)
        if len(text) < original_len * 0.5 and original_len > 100:
            logger.warning(
                f"Sanitization removed {original_len - len(text)} chars "
                f"({100 - len(text) * 100 // original_len}% of response) - "
                "model may be confused about tool calling format"
            )

        return text

    def is_garbage_content(self, content: str) -> bool:
        """Detect if content is garbage/malformed output from local models.

        Args:
            content: Content chunk to check

        Returns:
            True if content appears to be garbage
        """
        if not content:
            return False

        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True

        return False

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if a tool name is valid and not a hallucination.

        Args:
            name: Tool name to validate

        Returns:
            True if the tool name appears valid
        """
        if not name or not isinstance(name, str):
            return False

        # Reject common hallucinated/example tool names
        for pattern in self.INVALID_TOOL_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                logger.debug(f"Rejecting invalid tool name: {name}")
                return False

        # Must be alphanumeric with underscores only
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            logger.debug(f"Rejecting malformed tool name: {name}")
            return False

        return True

    # ================================================================
    # Code Sanitization Methods (for LLM-generated code)
    # ================================================================

    def sanitize_code(self, code: str) -> CodeSanitizationResult:
        """Sanitize LLM-generated Python code.

        Performs AST-based cleanup:
        1. Extract code from markdown blocks
        2. Remove LLM artifacts (preamble text)
        3. Fix indentation issues
        4. Remove incomplete trailing code
        5. Validate Python syntax
        6. Extract function names

        Args:
            code: Raw code string from LLM

        Returns:
            CodeSanitizationResult with cleaned code and metadata
        """
        errors: List[str] = []
        fixes: List[str] = []

        # Step 1: Extract code from markdown blocks
        original_code = code
        code = self._extract_code_from_markdown(code)
        if code != original_code:
            fixes.append("extracted_from_markdown")

        # Step 2: Remove LLM artifacts (preamble text)
        code, artifact_fixes = self._remove_code_artifacts(code)
        fixes.extend(artifact_fixes)

        # Step 3: Fix common indentation issues
        code, indent_fixes = self._fix_code_indentation(code)
        fixes.extend(indent_fixes)

        # Step 4: Remove trailing incomplete code
        code, incomplete_fixes = self._remove_incomplete_code(code)
        fixes.extend(incomplete_fixes)

        # Step 5: Ensure valid Python (try to extract function if needed)
        code, extract_fixes = self._ensure_valid_python(code)
        fixes.extend(extract_fixes)

        # Step 6: Validate final result
        is_valid, validation_errors = self._validate_python_syntax(code)
        errors.extend(validation_errors)

        # Extract function names
        function_names = self._extract_function_names(code)

        return CodeSanitizationResult(
            code=code,
            is_valid=is_valid,
            errors=errors,
            fixes_applied=fixes,
            function_names=function_names,
        )

    def _extract_code_from_markdown(self, code: str) -> str:
        """Extract code from markdown code blocks."""
        matches = self.MARKDOWN_CODE_BLOCK.findall(code)
        if matches:
            # Use the longest code block
            return max(matches, key=len).strip()

        # Remove any remaining backticks
        code = code.replace("```python", "").replace("```", "")
        return code.strip()

    def _remove_code_artifacts(self, code: str) -> Tuple[str, List[str]]:
        """Remove common LLM preamble/artifacts from code."""
        fixes = []
        original = code

        for pattern in self.CODE_ARTIFACTS:
            if pattern.search(code):
                code = pattern.sub("", code)

        if code != original:
            fixes.append("removed_artifacts")

        # Remove leading non-code lines before first def/class/import
        lines = code.split("\n")
        code_start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "import ", "from ", "@", "#")):
                code_start_idx = i
                break
            elif stripped and not stripped.startswith(
                ("def ", "class ", "import ", "from ", "@", "#", '"""', "'''")
            ):
                continue

        if code_start_idx > 0:
            code = "\n".join(lines[code_start_idx:])
            fixes.append("removed_preamble")

        return code.strip(), fixes

    def _fix_code_indentation(self, code: str) -> Tuple[str, List[str]]:
        """Fix common indentation issues in code."""
        fixes = []
        lines = code.split("\n")

        # Detect and fix mixed tabs/spaces
        has_tabs = any("\t" in line for line in lines)
        has_spaces = any(line.startswith(" ") for line in lines)

        if has_tabs and has_spaces:
            code = code.replace("\t", "    ")
            fixes.append("converted_tabs_to_spaces")
            lines = code.split("\n")

        # Detect if code is over-indented
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
            if min_indent > 0:
                code = textwrap.dedent(code)
                fixes.append("fixed_over_indentation")

        return code.strip(), fixes

    def _remove_incomplete_code(self, code: str) -> Tuple[str, List[str]]:
        """Remove trailing incomplete code."""
        fixes = []
        lines = code.split("\n")

        while lines:
            last_line = lines[-1].strip()
            is_incomplete = False

            for pattern in self.INCOMPLETE_CODE_PATTERNS:
                if pattern.search(last_line):
                    is_incomplete = True
                    break

            if is_incomplete:
                lines.pop()
                fixes.append("removed_incomplete_line")
            else:
                break

        return "\n".join(lines), fixes

    def _ensure_valid_python(self, code: str) -> Tuple[str, List[str]]:
        """Ensure code is valid Python, trying to extract function if needed."""
        fixes = []

        # First, try to parse as-is
        try:
            ast.parse(code)
            return code, fixes
        except SyntaxError:
            pass

        # Try to extract just the function definition
        extracted = self._extract_function_def(code)
        if extracted and extracted != code:
            try:
                ast.parse(extracted)
                fixes.append("extracted_function_only")
                return extracted, fixes
            except SyntaxError:
                pass

        # Try adding common missing imports
        code_with_imports = self._add_common_imports(code)
        try:
            ast.parse(code_with_imports)
            fixes.append("added_missing_imports")
            return code_with_imports, fixes
        except SyntaxError:
            pass

        # Return original if we can't fix it
        return code, fixes

    def _extract_function_def(self, code: str) -> Optional[str]:
        """Extract just the function definition from code."""
        lines = code.split("\n")
        func_lines = []
        in_function = False
        base_indent = 0

        for line in lines:
            stripped = line.lstrip()

            if stripped.startswith("def "):
                in_function = True
                base_indent = len(line) - len(stripped)
                func_lines.append(line[base_indent:])
            elif in_function:
                if line.strip() == "":
                    func_lines.append("")
                elif len(line) - len(line.lstrip()) > base_indent:
                    func_lines.append(line[base_indent:])
                elif stripped.startswith(("def ", "class ", "@")):
                    break
                elif len(line) - len(line.lstrip()) <= base_indent and line.strip():
                    break
                else:
                    func_lines.append(line[base_indent:] if len(line) > base_indent else "")

        if func_lines:
            return "\n".join(func_lines)
        return None

    def _add_common_imports(self, code: str) -> str:
        """Add commonly needed imports if they're used but not imported."""
        imports_to_add = []

        module_checks = [
            ("re", r"\bre\."),
            ("math", r"\bmath\."),
            ("itertools", r"\bitertools\."),
            ("collections", r"\bcollections\."),
            ("functools", r"\bfunctools\."),
            ("typing", r"\b(?:List|Dict|Set|Tuple|Optional|Union|Any)\b"),
        ]

        for module, pattern in module_checks:
            if re.search(pattern, code) and f"import {module}" not in code:
                if module == "typing":
                    imports_to_add.append(
                        "from typing import List, Dict, Set, Tuple, Optional, Union, Any"
                    )
                else:
                    imports_to_add.append(f"import {module}")

        if imports_to_add:
            return "\n".join(imports_to_add) + "\n\n" + code
        return code

    def _validate_python_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax."""
        errors = []
        try:
            ast.parse(code)
            return True, errors
        except SyntaxError as e:
            errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
            return False, errors

    def _extract_function_names(self, code: str) -> List[str]:
        """Extract function names from code."""
        names = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    names.append(node.name)
        except SyntaxError:
            # Fallback to regex
            for match in re.finditer(r"def\s+(\w+)\s*\(", code):
                names.append(match.group(1))
        return names

    def is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python.

        Args:
            code: Code to check

        Returns:
            True if code is syntactically valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def extract_function(self, code: str, function_name: Optional[str] = None) -> Optional[str]:
        """Extract a function from code.

        Args:
            code: Code containing the function
            function_name: Name of function to extract (if None, extracts first)

        Returns:
            The function code or None if not found
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if function_name is None or node.name == function_name:
                        lines = code.split("\n")
                        start = node.lineno - 1
                        end = node.end_lineno if hasattr(node, "end_lineno") else len(lines)
                        return "\n".join(lines[start:end])
        except SyntaxError:
            pass
        return None


# Module-level singleton for convenience
_sanitizer = ResponseSanitizer()


def sanitize_response(text: str) -> str:
    """Sanitize model response (convenience function).

    Args:
        text: Raw response text

    Returns:
        Cleaned text
    """
    return _sanitizer.sanitize(text)


def is_garbage_content(content: str) -> bool:
    """Detect garbage content (convenience function).

    Args:
        content: Content to check

    Returns:
        True if garbage detected
    """
    return _sanitizer.is_garbage_content(content)


def is_valid_tool_name(name: str) -> bool:
    """Validate tool name (convenience function).

    Args:
        name: Tool name to validate

    Returns:
        True if valid
    """
    return _sanitizer.is_valid_tool_name(name)


def strip_markup(text: str) -> str:
    """Strip markup from text (convenience function).

    Args:
        text: Text with markup

    Returns:
        Plain text
    """
    return _sanitizer.strip_markup(text)


# ================================================================
# Code Sanitization Convenience Functions
# ================================================================


def sanitize_code(code: str) -> CodeSanitizationResult:
    """Sanitize LLM-generated Python code (convenience function).

    Args:
        code: Raw code from LLM

    Returns:
        CodeSanitizationResult with cleaned code and metadata
    """
    return _sanitizer.sanitize_code(code)


def extract_code(code: str) -> str:
    """Extract and sanitize code, returning just the code string.

    Args:
        code: Raw code from LLM

    Returns:
        Sanitized code string
    """
    result = _sanitizer.sanitize_code(code)
    return result.code


def is_valid_python(code: str) -> bool:
    """Check if code is valid Python (convenience function).

    Args:
        code: Code to check

    Returns:
        True if code is syntactically valid
    """
    return _sanitizer.is_valid_python(code)


def extract_function(code: str, function_name: Optional[str] = None) -> Optional[str]:
    """Extract a function from code (convenience function).

    Args:
        code: Code containing the function
        function_name: Name of function to extract (if None, extracts first)

    Returns:
        The function code or None if not found
    """
    return _sanitizer.extract_function(code, function_name)
