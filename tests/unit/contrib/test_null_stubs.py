"""Tests for contrib null stub implementations."""

from __future__ import annotations

import pytest

from victor.contrib.parsing.parser import NullTreeSitterParser
from victor.contrib.parsing.extractor import NullTreeSitterExtractor
from victor.contrib.codebase.indexer import NullCodebaseIndexFactory
from victor.contrib.codebase.symbol_store import NullSymbolStore
from victor.contrib.codebase.ignore_patterns import BasicIgnorePatterns
from victor.contrib.languages.registry import NullLanguageRegistry
from victor.contrib.prompts.task_hints import NullTaskTypeHinter


class TestNullTreeSitterParser:
    def test_get_parser_returns_none(self):
        parser = NullTreeSitterParser()
        assert parser.get_parser("python") is None

    def test_get_supported_languages_empty(self):
        parser = NullTreeSitterParser()
        assert parser.get_supported_languages() == []


class TestNullTreeSitterExtractor:
    def test_extract_symbols_empty(self):
        extractor = NullTreeSitterExtractor()
        assert extractor.extract_symbols("def foo(): pass", "python") == []


class TestNullCodebaseIndexFactory:
    def test_create_raises(self):
        factory = NullCodebaseIndexFactory()
        with pytest.raises(ImportError, match="victor-coding"):
            factory.create("/some/path")


class TestNullSymbolStore:
    def test_create_raises(self):
        store = NullSymbolStore()
        with pytest.raises(ImportError, match="victor-coding"):
            store.create("/some/path")


class TestBasicIgnorePatterns:
    def test_default_skip_dirs(self):
        patterns = BasicIgnorePatterns()
        dirs = patterns.get_default_skip_dirs()
        assert ".git" in dirs
        assert "node_modules" in dirs
        assert "__pycache__" in dirs

    def test_is_hidden_path(self):
        patterns = BasicIgnorePatterns()
        assert patterns.is_hidden_path(".git")
        assert patterns.is_hidden_path(".hidden")
        assert not patterns.is_hidden_path("visible/file.txt")

    def test_should_ignore_path(self):
        patterns = BasicIgnorePatterns()
        assert patterns.should_ignore_path(".git/config")
        assert patterns.should_ignore_path("project/node_modules/pkg")
        assert not patterns.should_ignore_path("src/main.py")

    def test_should_ignore_with_extra(self):
        patterns = BasicIgnorePatterns()
        assert patterns.should_ignore_path(
            "project/vendor/lib.py",
            extra_patterns=["vendor"],
        )


class TestNullLanguageRegistry:
    def test_discover_returns_zero(self):
        registry = NullLanguageRegistry()
        assert registry.discover_plugins() == 0

    def test_get_raises_key_error(self):
        registry = NullLanguageRegistry()
        with pytest.raises(KeyError, match="python"):
            registry.get("python")

    def test_supported_languages_empty(self):
        registry = NullLanguageRegistry()
        assert registry.get_supported_languages() == []


class TestNullTaskTypeHinter:
    def test_returns_empty_string(self):
        hinter = NullTaskTypeHinter()
        assert hinter.get_hint("coding") == ""
        assert hinter.get_hint("anything") == ""
