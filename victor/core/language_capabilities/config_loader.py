"""
YAML configuration loader for language capabilities.

Loads language capability definitions from capabilities.yaml.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .types import (
    ASTAccessMethod,
    ASTCapability,
    LanguageTier,
    LSPCapability,
    TreeSitterCapability,
    UnifiedLanguageCapability,
)

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "config" / "languages" / "capabilities.yaml"
)


def _parse_ast_access_method(value: str) -> ASTAccessMethod:
    """Parse AST access method from string."""
    mapping = {
        "native": ASTAccessMethod.NATIVE,
        "python_lib": ASTAccessMethod.PYTHON_LIB,
        "ffi": ASTAccessMethod.FFI,
        "subprocess": ASTAccessMethod.SUBPROCESS,
        "lsp": ASTAccessMethod.LSP,
        "tree_sitter": ASTAccessMethod.TREE_SITTER,
    }
    return mapping.get(value, ASTAccessMethod.TREE_SITTER)


def _parse_language_tier(value: int) -> LanguageTier:
    """Parse language tier from int."""
    mapping = {
        1: LanguageTier.TIER_1,
        2: LanguageTier.TIER_2,
        3: LanguageTier.TIER_3,
        0: LanguageTier.UNSUPPORTED,
    }
    return mapping.get(value, LanguageTier.TIER_3)


def _parse_ast_capability(data: Optional[Dict[str, Any]]) -> Optional[ASTCapability]:
    """Parse AST capability from YAML data."""
    if not data:
        return None

    return ASTCapability(
        library=data.get("library", ""),
        access_method=_parse_ast_access_method(data.get("access_method", "native")),
        python_package=data.get("python_package"),
        requires_runtime=data.get("requires_runtime"),
        has_type_info=data.get("has_type_info", False),
        has_error_recovery=data.get("has_error_recovery", False),
        has_semantic_analysis=data.get("has_semantic_analysis", False),
        subprocess_command=data.get("subprocess_command"),
        output_format=data.get("output_format", "json"),
    )


def _parse_lsp_capability(data: Optional[Dict[str, Any]]) -> Optional[LSPCapability]:
    """Parse LSP capability from YAML data."""
    if not data:
        return None

    return LSPCapability(
        server_name=data.get("server_name", ""),
        language_id=data.get("language_id", ""),
        install_command=data.get("install_command"),
        has_diagnostics=data.get("has_diagnostics", True),
        has_completion=data.get("has_completion", True),
        has_hover=data.get("has_hover", True),
        has_type_info=data.get("has_type_info", False),
    )


def _parse_tree_sitter_capability(data: Optional[Dict[str, Any]]) -> Optional[TreeSitterCapability]:
    """Parse tree-sitter capability from YAML data."""
    if not data:
        return None

    return TreeSitterCapability(
        grammar_package=data.get("grammar_package", ""),
        language_function=data.get("language_function", "language"),
        has_highlights=data.get("has_highlights", True),
        has_injections=data.get("has_injections", False),
    )


def _parse_strategy(strategy: Optional[List[str]]) -> List[ASTAccessMethod]:
    """Parse indexing/validation strategy from YAML data."""
    if not strategy:
        return [ASTAccessMethod.NATIVE, ASTAccessMethod.TREE_SITTER, ASTAccessMethod.LSP]

    return [_parse_ast_access_method(s) for s in strategy]


def _parse_language_capability(name: str, data: Dict[str, Any]) -> UnifiedLanguageCapability:
    """Parse a single language capability from YAML data."""
    return UnifiedLanguageCapability(
        name=name,
        tier=_parse_language_tier(data.get("tier", 3)),
        extensions=data.get("extensions", []),
        filenames=data.get("filenames", []),
        native_ast=_parse_ast_capability(data.get("native_ast")),
        tree_sitter=_parse_tree_sitter_capability(data.get("tree_sitter")),
        lsp=_parse_lsp_capability(data.get("lsp")),
        indexing_enabled=data.get("indexing_enabled", True),
        validation_enabled=data.get("validation_enabled", True),
        indexing_strategy=_parse_strategy(data.get("indexing_strategy")),
        validation_strategy=_parse_strategy(data.get("validation_strategy")),
        fallback_on_unavailable=data.get("fallback_on_unavailable", "allow"),
        fallback_on_error=data.get("fallback_on_error", "warn"),
    )


def load_capabilities_from_yaml(
    config_path: Optional[Path] = None,
) -> List[UnifiedLanguageCapability]:
    """
    Load language capabilities from YAML configuration file.

    Args:
        config_path: Path to capabilities.yaml (uses default if None)

    Returns:
        List of UnifiedLanguageCapability objects

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return []

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "languages" not in data:
            logger.warning(f"No languages defined in {config_path}")
            return []

        capabilities = []
        languages = data.get("languages", {})

        for name, lang_data in languages.items():
            try:
                cap = _parse_language_capability(name, lang_data)
                capabilities.append(cap)
            except Exception as e:
                logger.warning(f"Failed to parse language '{name}': {e}")
                continue

        logger.debug(f"Loaded {len(capabilities)} language capabilities from {config_path}")
        return capabilities

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def load_feature_flags_from_yaml(
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load feature flags from YAML configuration file.

    Args:
        config_path: Path to capabilities.yaml (uses default if None)

    Returns:
        Dictionary of feature flags
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return data.get("feature_flags", {})

    except Exception as e:
        logger.warning(f"Failed to load feature flags: {e}")
        return {}
