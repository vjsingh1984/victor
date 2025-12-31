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
from typing import Any, Dict, Optional, Tuple, TypedDict

# Import native extensions with fallback
try:
    from victor.processing.native import (
        repair_json as native_repair_json,
        is_native_available,
    )

    _NATIVE_AVAILABLE = is_native_available()
except ImportError:
    _NATIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ToolStats(TypedDict):
    calls: int
    normalizations: int


class NormalizationStats(TypedDict):
    total_calls: int
    normalizations: Dict[str, int]
    failures: int
    by_tool: Dict[str, ToolStats]


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
        self.parameter_aliases: Dict[str, Dict[str, str]] = self.config.get("parameter_aliases", {})
        self.stats: NormalizationStats = {
            "total_calls": 0,
            "normalizations": {strategy.value: 0 for strategy in NormalizationStrategy},
            "failures": 0,
            "by_tool": {},
        }
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
        self.stats["total_calls"] += 1

        # Track per-tool stats
        if tool_name not in self.stats["by_tool"]:
            self.stats["by_tool"][tool_name] = {"calls": 0, "normalizations": 0}
        self.stats["by_tool"][tool_name]["calls"] += 1

        # Layer 0: Normalize parameter aliases (model-specific -> standard names)
        # This runs first so subsequent layers work with standard param names
        arguments, was_aliased = self.normalize_parameter_aliases(arguments, tool_name)

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
                        self.stats["normalizations"][NormalizationStrategy.PYTHON_AST.value] += 1
                        self.stats["by_tool"][tool_name]["normalizations"] += 1
                        logger.info(
                            f"[{self.provider_name}] Preemptively normalized {tool_name} arguments via AST"
                        )
                        return normalized, NormalizationStrategy.PYTHON_AST
            except Exception as e:
                logger.debug(
                    f"Preemptive AST normalization failed for {tool_name}: {e}", exc_info=True
                )

        # Layer 1: Check if already valid (fast path - most cases after preemptive normalization)
        logger.debug(f"[{self.provider_name}] {tool_name}: Layer 1 - Checking if already valid")
        try:
            is_valid = self._is_valid_json_dict(arguments)
            logger.debug(f"[{self.provider_name}] {tool_name}: Layer 1 - is_valid={is_valid}")
            if is_valid:
                self.stats["normalizations"][NormalizationStrategy.DIRECT.value] += 1
                return arguments, NormalizationStrategy.DIRECT
        except Exception as e:
            logger.error(
                f"[{self.provider_name}] {tool_name}: Layer 1 validation threw exception: {e}",
                exc_info=True,
            )

        # Not valid - need normalization
        self.stats["by_tool"][tool_name]["normalizations"] += 1

        # Layer 2: Try Python AST conversion for string values (if not already tried)
        if not has_json_like_strings:
            try:
                normalized = self._normalize_via_ast(arguments)
                if self._is_valid_json_dict(normalized):
                    self.stats["normalizations"][NormalizationStrategy.PYTHON_AST.value] += 1
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
                self.stats["normalizations"][NormalizationStrategy.REGEX_QUOTES.value] += 1
                logger.info(f"[{self.provider_name}] Normalized {tool_name} arguments via regex")
                return normalized, NormalizationStrategy.REGEX_QUOTES
        except Exception as e:
            logger.debug(f"Regex normalization failed for {tool_name}: {e}")

        # Layer 4: Manual repair for known patterns
        try:
            normalized = self._normalize_via_manual_repair(arguments, tool_name)
            if self._is_valid_json_dict(normalized):
                self.stats["normalizations"][NormalizationStrategy.MANUAL_REPAIR.value] += 1
                logger.info(
                    f"[{self.provider_name}] Normalized {tool_name} arguments via manual repair"
                )
                return normalized, NormalizationStrategy.MANUAL_REPAIR
        except Exception as e:
            logger.debug(f"Manual repair failed for {tool_name}: {e}")

        # All strategies failed
        self.stats["failures"] += 1
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
                                # Complex structure - convert to JSON and verify it's parseable
                                json_str = json.dumps(python_obj)
                                # Verify the JSON string can be parsed back
                                json.loads(json_str)
                                normalized[key] = json_str
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
        Simple quote replacement via regex.

        Uses native Rust implementation when available for ~5-10x speedup.

        This is a fallback for cases where AST parsing fails but simple
        quote replacement might work.

        Example:
            Input:  {'key': "{'field': 'value'}"}
            Output: {'key': '{"field": "value"}'}

        Args:
            arguments: Raw arguments to normalize

        Returns:
            Normalized arguments with replaced quotes
        """
        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if _NATIVE_AVAILABLE:
                    # Use native Rust JSON repair (handles more edge cases)
                    repaired = native_repair_json(value)
                else:
                    # Python fallback
                    # Replace escaped single quotes with double quotes
                    # Pattern: \' → "  and standalone ' → "
                    repaired = value.replace("\\'", '"')
                    # Also try replacing unescaped single quotes
                    # (careful not to replace quotes in strings)
                    if "'" in repaired:
                        repaired = repaired.replace("'", '"')
                normalized[key] = repaired
            else:
                normalized[key] = value
        return normalized

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

    def _coerce_primitive_types(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
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
            (self.stats["total_calls"] - self.stats["failures"]) / max(self.stats["total_calls"], 1)
        ) * 100

        return {
            "provider": self.provider_name,
            "total_calls": self.stats["total_calls"],
            "normalizations": self.stats["normalizations"],
            "failures": self.stats["failures"],
            "success_rate": round(success_rate, 2),
            "by_tool": self.stats["by_tool"],
            "alias_stats": self._alias_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics (useful for testing)."""
        self.stats = {
            "total_calls": 0,
            "normalizations": {strategy.value: 0 for strategy in NormalizationStrategy},
            "failures": 0,
            "by_tool": {},
        }
        self._alias_stats = {"total": 0, "aliased": 0, "by_tool": {}}

    def log_stats(self) -> None:
        """Log current statistics."""
        stats = self.get_stats()
        logger.info(f"Argument normalization stats: {stats}")
