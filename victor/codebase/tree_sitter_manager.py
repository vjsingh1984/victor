
import os
from pathlib import Path
from typing import Dict

from tree_sitter import Language, Parser
import git


LANGUAGES_REPO = {
    "python": "https://github.com/tree-sitter/tree-sitter-python.git",
    "javascript": "https://github.com/tree-sitter/tree-sitter-javascript.git",
    "typescript": "https://github.com/tree-sitter/tree-sitter-typescript.git",
    "java": "https://github.com/tree-sitter/tree-sitter-java.git",
    "go": "https://github.com/tree-sitter/tree-sitter-go.git",
    "rust": "https://github.com/tree-sitter/tree-sitter-rust.git",
}

_language_cache: Dict[str, Language] = {}


def get_language(language: str) -> Language:
    """
    Loads a tree-sitter Language object, compiling it from source if not already built.
    """
    if language in _language_cache:
        return _language_cache[language]

    grammar_dir = Path.home() / ".victor" / "grammars"
    grammar_dir.mkdir(parents=True, exist_ok=True)
    
    lib_path = grammar_dir / f"{language}.so"

    if not lib_path.exists():
        print(f"Compiling tree-sitter grammar for {language}...")
        
        repo_url = LANGUAGES_REPO.get(language)
        if not repo_url:
            raise ValueError(f"Unsupported language for tree-sitter: {language}")
            
        repo_path = grammar_dir / f"tree-sitter-{language}"
        
        if not repo_path.exists():
            git.Repo.clone_from(repo_url, repo_path)
            
        Language.build_library(
            str(lib_path),
            [str(repo_path)]
        )
        print(f"Finished compiling grammar for {language}.")

    lang = Language(str(lib_path), language)
    _language_cache[language] = lang
    return lang


def get_parser(language: str) -> Parser:
    """
    Returns a tree-sitter Parser initialized with the specified language.
    """
    lang = get_language(language)
    parser = Parser()
    parser.set_language(lang)
    return parser

