
from typing import Dict
from tree_sitter import Language, Parser


# Language package mapping for tree-sitter 0.25+
# These use pre-compiled language packages instead of runtime compilation
LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
}

_language_cache: Dict[str, Language] = {}
_parser_cache: Dict[str, Parser] = {}


def get_language(language: str) -> Language:
    """
    Loads a tree-sitter Language object using pre-compiled language packages.

    This uses the tree-sitter 0.25+ API which requires pre-installed language packages
    (e.g., tree-sitter-python) instead of runtime compilation.
    """
    if language in _language_cache:
        return _language_cache[language]

    module_name = LANGUAGE_MODULES.get(language)
    if not module_name:
        raise ValueError(f"Unsupported language for tree-sitter: {language}")

    try:
        # Dynamically import the language module
        language_module = __import__(module_name)

        # Create Language object using the new API
        # In tree-sitter 0.25+, Language() takes a language object from the module
        lang = Language(language_module.language())

        _language_cache[language] = lang
        return lang

    except ImportError:
        raise ImportError(
            f"Language package '{module_name}' not installed. "
            f"Install it with: pip install {module_name.replace('_', '-')}"
        )


def get_parser(language: str) -> Parser:
    """
    Returns a tree-sitter Parser initialized with the specified language.

    In tree-sitter 0.25+, Parser() constructor takes the Language object directly.
    """
    if language in _parser_cache:
        return _parser_cache[language]

    lang = get_language(language)

    # New API: Parser takes Language object in constructor
    parser = Parser(lang)

    _parser_cache[language] = parser
    return parser

