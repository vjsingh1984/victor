"""
Robust argument normalization for tool calls.
Handles malformed JSON from various LLM providers.

This module provides a multi-layer normalization pipeline that gracefully handles
common argument formatting issues, particularly Python-style syntax (single quotes)
being used instead of JSON (double quotes).
"""

import ast
import json
import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

# Import native extensions with unified availability check
try:
    from victor.processing.native import (
        repair_json as native_repair_json,
        is_native_available,
        coerce_string_type as native_coerce_string_type,
    )

    _NATIVE_AVAILABLE = is_native_available()
except ImportError:
    _NATIVE_AVAILABLE = False

    def native_coerce_string_type(value: str) -> tuple:
        """Fallback stub when native not available."""
        return ("string", value, None)


logger = logging.getLogger(__name__)


class ToolStats(BaseModel):
    """Statistics for tool argument normalization (Pydantic v2)."""

    model_config = {"validate_assignment": True}

    calls: int = Field(default=0, ge=0, description="Number of times tool was called")
    normalizations: int = Field(default=0, ge=0, description="Number of arguments normalized")

    # Dict-like interface for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class NormalizationStats(BaseModel):
    """Aggregate normalization statistics (Pydantic v2)."""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    total_calls: int = Field(default=0, ge=0, description="Total normalization calls")
    normalizations: Dict[str, int] = Field(
        default_factory=dict,
        description="Normalizations per strategy",
    )
    failures: int = Field(default=0, ge=0, description="Total normalization failures")
    by_tool: Dict[str, ToolStats] = Field(
        default_factory=dict,
        description="Per-tool statistics",
    )

    # Dict-like interface for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class NormalizationStrategy(Enum):
    """Strategies for normalizing malformed arguments."""

    DIRECT = "direct"  # Valid JSON, no changes needed
    PYTHON_AST = "python_ast"  # Python syntax → JSON via ast.literal_eval
    REGEX_QUOTES = "regex_quotes"  # Simple quote replacement
    MANUAL_REPAIR = "manual_repair"  # Tool-specific repairs
    FAILED = "failed"  # All strategies failed


class ArgumentNormalizer:
    """
    Multi-layer argument normalization with fallback strategies.

    Design Philosophy:
    - Fast path first (valid JSON) - zero overhead for 99%+ cases
    - Graceful degradation (multiple fallback strategies)
    - Complete transparency (log all normalizations)
    - Provider-aware (can customize per provider)
    - Alias normalization (model-specific param names → standard names)

    Usage:
        normalizer = ArgumentNormalizer(provider_name="ollama")
        normalized_args, strategy = normalizer.normalize_arguments(
            arguments={"operations": "[{'type': 'modify', ...}]"},
            tool_name="edit_files"
        )

        # With parameter aliases (e.g., gpt-oss uses line_start instead of offset)
        normalizer = ArgumentNormalizer(
            provider_name="ollama",
            config={"parameter_aliases": {
                "read": {"line_start": "offset", "line_end": "_line_end"}
            }}
        )
    """

    def __init__(self, provider_name: str = "unknown", config: Optional[Dict[str, Any]] = None):
        """
        Initialize argument normalizer.

        Args:
            provider_name: Name of the LLM provider (for logging/metrics)
            config: Optional configuration overrides
                - parameter_aliases: Dict[tool_name, Dict[model_param, standard_param]]
        """
        self.provider_name = provider_name
        self.config = config or {}
        # Built-in aliases for common model hallucinations (LLMs use wrong param names)
        _builtin_aliases: Dict[str, Dict[str, str]] = {
            "shell": {"command": "cmd"},
            "execute_bash": {"command": "cmd"},
            "read": {"file_path": "path", "filename": "path"},
            "read_file": {"file_path": "path", "filename": "path"},
            "write": {"file_path": "path", "filename": "path"},
            "ls": {"directory": "path", "dir": "path"},
            "list_directory": {"directory": "path", "dir": "path"},
        }
        user_aliases = self.config.get("parameter_aliases", {})
        # Merge: user overrides built-in
        self.parameter_aliases: Dict[str, Dict[str, str]] = {**_builtin_aliases, **user_aliases}
        self.stats: NormalizationStats = NormalizationStats(
            total_calls=0,
            normalizations={strategy.value: 0 for strategy in NormalizationStrategy},
            failures=0,
            by_tool={},
        )
        self._alias_stats = {"total": 0, "aliased": 0, "by_tool": {}}

    def normalize_parameter_aliases(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Normalize model-specific parameter names to standard tool parameter names.

        Some models (e.g., gpt-oss) use different parameter names than what the
        tools expect. This method maps those aliases to standard names.

        Example:
            Input:  {"line_start": 10, "line_end": 50} for tool "read"
            Output: {"offset": 10, "_line_end": 50} (with _line_end for special handling)

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool being called

        Returns:
            (normalized_arguments, was_aliased) - returns original if no aliases apply
        """
        tool_aliases = self.parameter_aliases.get(tool_name, {})
        if not tool_aliases:
            return arguments, False

        normalized = {}
        was_aliased = False

        for param, value in arguments.items():
            if param in tool_aliases:
                standard_param = tool_aliases[param]
                normalized[standard_param] = value
                was_aliased = True
                logger.debug(
                    f"[{self.provider_name}] {tool_name}: Aliased '{param}' -> '{standard_param}'"
                )
            else:
                normalized[param] = value

        if was_aliased:
            self._alias_stats["total"] += 1
            self._alias_stats["aliased"] += 1
            if tool_name not in self._alias_stats["by_tool"]:
                self._alias_stats["by_tool"][tool_name] = 0
            self._alias_stats["by_tool"][tool_name] += 1
            logger.info(
                f"[{self.provider_name}] Normalized parameter aliases for {tool_name}: "
                f"{list(set(arguments.keys()) - set(normalized.keys()))} -> "
                f"{list(set(normalized.keys()) - set(arguments.keys()))}"
            )

        return normalized, was_aliased

    def normalize_arguments(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Tuple[Dict[str, Any], NormalizationStrategy]:
        """
        Normalize tool call arguments through multi-layer pipeline.

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool being called

        Returns:
            (normalized_arguments, strategy_used)
        """
        self.stats.total_calls += 1

        # Track per-tool stats
        if tool_name not in self.stats.by_tool:
            self.stats.by_tool[tool_name] = ToolStats(calls=0, normalizations=0)
        self.stats.by_tool[tool_name].calls += 1

        # Layer 0: Normalize parameter aliases (model-specific -> standard names)
        # This runs first so subsequent layers work with standard param names
        arguments, was_aliased = self.normalize_parameter_aliases(arguments, tool_name)

        # Layer 0.1: Sanitize non-JSON-serializable objects (Ellipsis, Path, etc.)
        # This MUST happen before primitive type coercion to prevent serialization errors
        from victor.agent.argument_sanitizer import sanitize_arguments_for_serialization

        arguments = sanitize_arguments_for_serialization(arguments)
        logger.debug(
            f"[{self.provider_name}] {tool_name}: Sanitized arguments for JSON serialization"
        )

        # Layer 0.5: Coerce primitive types (str -> int/float/bool)
        # Some models (OpenRouter, Fireworks) output integers as strings like "0" or "30"
        arguments = self._coerce_primitive_types(arguments, tool_name)

        # AGGRESSIVE APPROACH: Check if any values look like JSON and try normalization FIRST
        # This ensures we catch malformed JSON even if basic validation passes
        has_json_like_strings = any(
            isinstance(v, str) and v.strip().startswith(("[", "{")) for v in arguments.values()
        )
        logger.debug(
            f"[{self.provider_name}] {tool_name}: has_json_like_strings={has_json_like_strings}"
        )

        if has_json_like_strings:
            # Try AST normalization preemptively for JSON-like strings
            try:
                logger.debug(
                    f"[{self.provider_name}] {tool_name}: Trying preemptive AST normalization"
                )
                normalized = self._normalize_via_ast(arguments)
                changed = normalized != arguments
                logger.debug(
                    f"[{self.provider_name}] {tool_name}: AST normalization changed={changed}"
                )
                # Verify normalization actually changed something or improved validity
                if changed:
                    is_valid = self._is_valid_json_dict(normalized)
                    logger.debug(
                        f"[{self.provider_name}] {tool_name}: Normalized version is_valid={is_valid}"
                    )
                    if is_valid:
                        self.stats.normalizations[NormalizationStrategy.PYTHON_AST.value] += 1
                        self.stats.by_tool[tool_name].normalizations += 1
                        logger.info(
                            f"[{self.provider_name}] Preemptively normalized {tool_name} arguments via AST"
                        )
                        return normalized, NormalizationStrategy.PYTHON_AST
            except Exception as e:
                logger.debug(
                    f"Preemptive AST normalization failed for {tool_name}: {e}",
                    exc_info=True,
                )

        # Layer 1: Check if already valid (fast path - most cases after preemptive normalization)
        logger.debug(f"[{self.provider_name}] {tool_name}: Layer 1 - Checking if already valid")
        try:
            is_valid = self._is_valid_json_dict(arguments)
            logger.debug(f"[{self.provider_name}] {tool_name}: Layer 1 - is_valid={is_valid}")
            if is_valid:
                self.stats.normalizations[NormalizationStrategy.DIRECT.value] += 1
                return arguments, NormalizationStrategy.DIRECT
        except Exception as e:
            logger.error(
                f"[{self.provider_name}] {tool_name}: Layer 1 validation threw exception: {e}",
                exc_info=True,
            )

        # Not valid - need normalization
        self.stats.by_tool[tool_name].normalizations += 1

        # Layer 2: Try Python AST conversion for string values (if not already tried)
        if not has_json_like_strings:
            try:
                normalized = self._normalize_via_ast(arguments)
                if self._is_valid_json_dict(normalized):
                    self.stats.normalizations[NormalizationStrategy.PYTHON_AST.value] += 1
                    logger.info(
                        f"[{self.provider_name}] Normalized {tool_name} arguments via AST conversion"
                    )
                    return normalized, NormalizationStrategy.PYTHON_AST
            except Exception as e:
                logger.debug(f"AST normalization failed for {tool_name}: {e}")

        # Layer 3: Regex-based quote replacement
        try:
            normalized = self._normalize_via_regex(arguments)
            if self._is_valid_json_dict(normalized):
                self.stats.normalizations[NormalizationStrategy.REGEX_QUOTES.value] += 1
                logger.info(f"[{self.provider_name}] Normalized {tool_name} arguments via regex")
                return normalized, NormalizationStrategy.REGEX_QUOTES
        except Exception as e:
            logger.debug(f"Regex normalization failed for {tool_name}: {e}")

        # Layer 4: Manual repair for known patterns
        try:
            normalized = self._normalize_via_manual_repair(arguments, tool_name)
            if self._is_valid_json_dict(normalized):
                self.stats.normalizations[NormalizationStrategy.MANUAL_REPAIR.value] += 1
                logger.info(
                    f"[{self.provider_name}] Normalized {tool_name} arguments via manual repair"
                )
                return normalized, NormalizationStrategy.MANUAL_REPAIR
        except Exception as e:
            logger.debug(f"Manual repair failed for {tool_name}: {e}")

        # All strategies failed
        self.stats.failures += 1
        logger.error(
            f"[{self.provider_name}] Failed to normalize {tool_name} arguments "
            f"after all strategies. Original: {arguments}"
        )
        return arguments, NormalizationStrategy.FAILED

    def _is_valid_json_dict(self, obj: Any) -> bool:
        """
        Check if object is valid for JSON serialization.

        Also validates that string values that look like JSON structures
        (starting with [ or {) are themselves valid JSON.

        NOTE: Some LLM providers (e.g., Ollama/Qwen) may output JSON strings
        with literal control characters (actual newlines instead of \\n).
        While technically invalid JSON, these can still be used by tools that
        handle raw content. We accept these as "valid enough" to avoid false
        failures while still catching truly malformed JSON.

        Args:
            obj: Object to validate

        Returns:
            True if object can be JSON-serialized (even if string values
                 contain control characters that make them technically invalid JSON)
        """
        try:
            # First check if the whole object can be JSON-serialized
            json.dumps(obj)
            logger.debug("_is_valid_json_dict: json.dumps() succeeded")

            # Additionally, check string values that look like JSON
            # NOTE: We intentionally do NOT validate with json.loads() here because
            # some providers output literal control characters (actual newlines),
            # which are invalid JSON but still work fine with tools that handle
            # raw content (like edit_files). Validating with json.loads() would
            # cause false failures.
            #
            # The edit_files tool will handle any necessary escaping internally.
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        stripped = value.strip()
                        if stripped.startswith(("[", "{")):
                            logger.debug(
                                "_is_valid_json_dict: Checking '%s' value that looks like JSON",
                                key,
                            )
                            # Just verify it starts with JSON-like syntax
                            # Don't use json.loads() as it rejects literal control chars
                            logger.debug(
                                "_is_valid_json_dict: '%s' value looks like JSON (syntax check only)",
                                key,
                            )

            logger.debug("_is_valid_json_dict: Returning True")
            return True
        except (TypeError, ValueError) as e:
            logger.debug("_is_valid_json_dict: Exception in validation: %s", e)
            return False

    def _normalize_via_ast(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Python syntax to JSON via AST.

        This is the primary normalization strategy for handling Python-style
        dict/list syntax (single quotes) in string arguments.

        Also handles type coercion: if a string looks like a simple list/dict
        (e.g., "[]", "{}"), convert it to the actual type.

        Example:
            Input:  {'operations': "[{'type': 'modify', 'path': 'file.sh'}]"}
            Output: {'operations': '[{"type": "modify", "path": "file.sh"}]'}

            Input:  {'patterns': '[]'}
            Output: {'patterns': []}

        Args:
            arguments: Raw arguments to normalize

        Returns:
            Normalized arguments with valid JSON strings or coerced types
        """
        normalized: Dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # First, check if the string LOOKS like JSON but may need normalization
                stripped = value.strip()
                if stripped.startswith(("[", "{")):
                    # Aggressively normalize: try Python AST first, then verify with json.loads
                    try:
                        # Use ast.literal_eval (SAFE - no code execution)
                        python_obj = ast.literal_eval(value)

                        # Type coercion: for simple empty structures, return the actual type
                        # This helps tools that expect List[str] instead of str
                        if isinstance(python_obj, (list, dict)):
                            # If it's a simple empty structure, return as-is (type coercion)
                            if not python_obj:  # Empty list [] or dict {}
                                normalized[key] = python_obj
                            else:
                                # Complex structure - keep as Python object, don't convert to string
                                # Verify it's JSON-serializable for provider compatibility
                                try:
                                    json.dumps(
                                        python_obj
                                    )  # Verify serializable (will raise if not)
                                    normalized[key] = python_obj  # Keep as dict/list
                                except (TypeError, ValueError) as e:
                                    # Not serializable - convert to string as fallback
                                    logger.warning(
                                        f"Parameter {key} contains non-serializable value: {e}"
                                    )
                                    normalized[key] = json.dumps(python_obj, default=str)
                        else:
                            # Primitive value - keep as-is
                            normalized[key] = value
                    except (ValueError, SyntaxError, json.JSONDecodeError):
                        # AST failed - keep original and let other layers handle it
                        normalized[key] = value
                else:
                    # Doesn't look like JSON - keep as-is
                    normalized[key] = value
            else:
                # Non-string value - keep as-is
                normalized[key] = value
        return normalized

    def _normalize_via_regex(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced regex-based normalization with fallback for malformed patterns.

        Handles common malformed tool call patterns:
        - YAML-style: key: value
        - Python type hints: key: tuple[str, ...]
        - Path objects: pdf_path: Path(...)

        Uses native Rust implementation when available for ~5-10x speedup.

        This is a fallback for cases where AST parsing fails but simple
        quote replacement might work.

        Example:
            Input:  {'key': "{'field': 'value'}"}
            Output: {'key': '{"field": "value"}'}

        Args:
            arguments: Raw arguments to normalize

        Returns:
            Normalized arguments with replaced quotes and cleaned patterns
        """
        import re

        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # Handle Python type hints in values
                # Example: "tuple[str, ...]" → "" (remove)
                # Example: "Path(...)" → string value
                value = self._clean_python_type_hints(value)

                # Handle Path(...) syntax
                if "Path(" in value:
                    # Extract path from Path(...) wrapper
                    # Example: Path('/path/to/file') → '/path/to/file'
                    path_match = re.search(r"Path\(['\"]([^'\"]+)['\"]\)", value)
                    if path_match:
                        value = path_match.group(1)

                # Handle tuple[str, ...] or list[str] syntax
                if "tuple[" in value or "list[" in value:
                    # Remove type annotation, use empty list as default
                    value = []

                # Now do the normal quote replacement
                if _NATIVE_AVAILABLE:
                    # Use native Rust JSON repair (handles more edge cases)
                    repaired = native_repair_json(value)
                else:
                    # Python fallback
                    # Replace escaped single quotes with double quotes
                    # Pattern: \' → "  and standalone ' → "
                    repaired = str(value).replace("\\'", '"')
                    # Also try replacing unescaped single quotes
                    # (careful not to replace quotes in strings)
                    if "'" in repaired:
                        repaired = repaired.replace("'", '"')
                normalized[key] = repaired
            else:
                normalized[key] = value
        return normalized

    def _clean_python_type_hints(self, value: str) -> str:
        """Remove Python type hints from string values.

        Args:
            value: String potentially containing type hints

        Returns:
            Cleaned string without type hints
        """
        import re

        # Remove standalone type annotations
        # Example: "tuple[str, ...]" → ""
        if re.match(r"^[a-z_]+\[[^\]]+\]$", value.strip()):
            return ""

        # Remove type annotations at end of value
        # Example: "'file.txt' : str" → "'file.txt'"
        value = re.sub(r"\s*:\s*[a-z_]+\[[^\]]*\]", "", value)

        return value

    def _normalize_via_manual_repair(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
        """
        Tool-specific manual repairs for known patterns.

        This allows adding custom repair logic for specific tools
        as edge cases are discovered.

        Args:
            arguments: Raw arguments to normalize
            tool_name: Name of the tool (for tool-specific repairs)

        Returns:
            Normalized arguments with tool-specific repairs applied
        """
        # Tool-specific repairs
        if tool_name == "edit_files":
            return self._repair_edit_files_args(arguments)

        # Add more tool-specific repairs here as needed
        # if tool_name == "other_tool":
        #     return self._repair_other_tool_args(arguments)

        return arguments

    def _repair_edit_files_args(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair common edit_files argument malformations.

        Known patterns:
        1. Python dict syntax in 'operations' field
        2. Missing quotes around field names
        3. Single quotes instead of double quotes

        Args:
            arguments: Raw edit_files arguments

        Returns:
            Repaired arguments
        """
        if "operations" in arguments:
            ops = arguments["operations"]
            if isinstance(ops, str):
                try:
                    # Try AST first (most robust)
                    python_obj = ast.literal_eval(ops)
                    arguments["operations"] = json.dumps(python_obj)
                except Exception:
                    # Fallback to regex
                    # This is a last resort for very malformed input
                    pass
        return arguments

    def _coerce_primitive_types(self, arguments: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Coerce string values to primitive types (int, float, bool) when appropriate.

        Some models (e.g., OpenRouter Llama, Fireworks) output integers as strings
        like "0", "30", or booleans as "true", "false". This method coerces them
        to proper Python types.

        Examples:
            {"line_start": "0", "line_end": "30"} -> {"line_start": 0, "line_end": 30}
            {"regex": "false"} -> {"regex": False}
            {"timeout": "30.5"} -> {"timeout": 30.5}

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool (for logging)

        Returns:
            Arguments with coerced primitive types
        """
        coerced = {}
        any_coerced = False

        for key, value in arguments.items():
            if isinstance(value, str):
                coerced_value = self._try_coerce_string(value)
                if coerced_value is not value:  # Identity check - was actually coerced
                    coerced[key] = coerced_value
                    any_coerced = True
                    logger.debug(
                        f"[{self.provider_name}] {tool_name}: Coerced '{key}' from str "
                        f"'{value}' to {type(coerced_value).__name__} {coerced_value}"
                    )
                else:
                    coerced[key] = value
            else:
                coerced[key] = value

        if any_coerced:
            logger.info(
                f"[{self.provider_name}] Coerced primitive types for {tool_name}: "
                f"{[k for k in arguments if coerced.get(k) is not arguments.get(k)]}"
            )

        return coerced

    def _try_coerce_string(self, value: str) -> Any:
        """
        Try to coerce a string value to a primitive type.

        Uses Rust accelerator when available for 3-5x speedup.

        Attempts conversions in order:
        1. Boolean ("true"/"false", case-insensitive)
        2. None ("null"/"none", case-insensitive)
        3. Integer (digits only, with optional negative sign)
        4. Float (digits with decimal point)

        Args:
            value: String value to potentially coerce

        Returns:
            Coerced value or original string if no coercion applies
        """
        # Use Rust accelerator when available (3-5x faster)
        if _NATIVE_AVAILABLE:
            type_name, coerced_str, _ = native_coerce_string_type(value)
            if type_name == "null":
                return None
            elif type_name == "bool":
                return coerced_str.lower() == "true"
            elif type_name == "int":
                # Don't coerce if it looks like a path
                if not value.strip().startswith("/"):
                    return int(coerced_str)
            elif type_name == "float":
                # Don't coerce if it looks like a path
                if not value.strip().startswith("/"):
                    return float(coerced_str)
            # For string/list/dict types, fall through to return original
            return value

        # Python fallback
        stripped = value.strip()
        lower = stripped.lower()

        # Boolean coercion
        if lower == "true":
            return True
        if lower == "false":
            return False

        # None/null coercion
        if lower in ("null", "none"):
            return None

        # Integer coercion (must be all digits, optionally with leading minus)
        # Don't coerce if it looks like a path or has special chars
        if stripped.lstrip("-").isdigit() and not stripped.startswith("/"):
            try:
                return int(stripped)
            except ValueError:
                pass

        # Float coercion (digits with one decimal point)
        if "." in stripped and not stripped.startswith("/"):
            parts = stripped.lstrip("-").split(".")
            if len(parts) == 2 and all(p.isdigit() for p in parts if p):
                try:
                    return float(stripped)
                except ValueError:
                    pass

        # No coercion - return original
        return value

    def get_stats(self) -> Dict[str, Any]:
        """
        Get normalization statistics for monitoring.

        Returns:
            Dictionary with normalization metrics
        """
        success_rate = (
            (self.stats.total_calls - self.stats.failures) / max(self.stats.total_calls, 1)
        ) * 100

        return {
            "provider": self.provider_name,
            "total_calls": self.stats.total_calls,
            "normalizations": self.stats.normalizations,
            "failures": self.stats.failures,
            "success_rate": round(success_rate, 2),
            "by_tool": self.stats.by_tool,
            "alias_stats": self._alias_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics (useful for testing)."""
        self.stats = NormalizationStats(
            total_calls=0,
            normalizations={strategy.value: 0 for strategy in NormalizationStrategy},
            failures=0,
            by_tool={},
        )
        self._alias_stats = {"total": 0, "aliased": 0, "by_tool": {}}

    def log_stats(self) -> None:
        """Log current statistics."""
        stats = self.get_stats()
        logger.info(f"Argument normalization stats: {stats}")
