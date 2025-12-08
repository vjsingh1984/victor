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


from typing import Dict
from tree_sitter import Language, Parser


# Language package mapping for tree-sitter 0.25+
# These use pre-compiled language packages instead of runtime compilation
# Install with: pip install tree-sitter-<language>
LANGUAGE_MODULES = {
    # Core languages (commonly used)
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    # Additional languages
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "c_sharp": "tree_sitter_c_sharp",
    "ruby": "tree_sitter_ruby",
    "php": "tree_sitter_php",
    "kotlin": "tree_sitter_kotlin",
    "swift": "tree_sitter_swift",
    "scala": "tree_sitter_scala",
    "bash": "tree_sitter_bash",
    "sql": "tree_sitter_sql",
    # Web languages
    "html": "tree_sitter_html",
    "css": "tree_sitter_css",
    "json": "tree_sitter_json",
    "yaml": "tree_sitter_yaml",
    "toml": "tree_sitter_toml",
    # Other
    "lua": "tree_sitter_lua",
    "elixir": "tree_sitter_elixir",
    "haskell": "tree_sitter_haskell",
    "r": "tree_sitter_r",
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
