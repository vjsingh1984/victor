# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Receiver-type extraction tests for the Rust language plugin.

These tests pin down receiver-type inference for `obj.method()` calls so
the cross-file CALLS resolver in victor-ai can bind method calls to the
correct `impl T` block instead of fanning out to every `method` named
`callee_name` across the codebase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_rust")

from victor_coding.codebase.tree_sitter_manager import get_parser
from victor_coding.languages.base import CallEdge
from victor_coding.languages.plugins.rust import RustPlugin


def _detect(source: str) -> list[CallEdge]:
    plugin = RustPlugin()
    parser = get_parser("rust")
    tree = parser.parse(bytes(source, "utf-8"))
    result = plugin.detect_calls_edges(tree, source, Path("scratch.rs"))
    return result.calls


def _by_callee(calls: list[CallEdge], name: str) -> CallEdge:
    matches = [c for c in calls if c.callee_name == name]
    assert matches, f"no CallEdge with callee_name={name!r}; have {[c.callee_name for c in calls]}"
    assert len(matches) == 1, f"expected exactly one {name!r} call, got {len(matches)}"
    return matches[0]


def test_call_edge_default_receiver_type_is_none():
    """Backwards compatibility: receiver_type defaults to None when not set."""
    edge = CallEdge(caller_name="a", callee_name="b")
    assert edge.receiver_type is None


def test_typed_let_binds_receiver_type():
    """`let x: Foo = ...; x.method()` → receiver_type='Foo'."""
    source = """
fn caller() {
    let x: Foo = make();
    x.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.caller_name == "caller"
    assert edge.receiver_type == "Foo"


def test_constructor_let_binds_receiver_type():
    """`let x = Foo::new(); x.method()` → receiver_type='Foo'."""
    source = """
fn caller() {
    let x = Foo::new();
    x.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Foo"


def test_self_call_inside_impl_block_uses_impl_type():
    """`impl Foo { fn x(&self) { self.method() } }` → receiver_type='Foo'."""
    source = """
impl Foo {
    fn outer(&self) {
        self.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Foo"


def test_typed_parameter_binds_receiver_type():
    """`fn x(arg: Bar) { arg.method() }` → receiver_type='Bar'."""
    source = """
fn caller(arg: Bar) {
    arg.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"


def test_typed_reference_parameter_strips_reference():
    """`fn x(arg: &Bar)` and `fn x(arg: &mut Bar)` both yield receiver_type='Bar'."""
    source = """
fn caller(arg: &Bar) {
    arg.method();
}
fn caller_mut(arg: &mut Baz) {
    arg.other();
}
"""
    calls = _detect(source)
    assert _by_callee(calls, "method").receiver_type == "Bar"
    assert _by_callee(calls, "other").receiver_type == "Baz"


def test_plain_function_call_has_no_receiver_type():
    """`func()` → receiver_type=None (not a method call)."""
    source = """
fn caller() {
    func();
}
"""
    edge = _by_callee(_detect(source), "func")
    assert edge.receiver_type is None


def test_call_on_untracked_variable_has_no_receiver_type():
    """`unknown.method()` (no binding visible in scope) → receiver_type=None."""
    source = """
fn caller() {
    unknown.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type is None


def test_self_call_outside_impl_block_has_no_receiver_type():
    """A bare `fn` not inside `impl T` cannot bind `self` to a type."""
    source = """
fn outer(this: &Self) {
    this.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    # `this` is just a parameter named "this" with type Self; we don't model Self outside impl.
    # Receiver type should fall back to None (or the literal "Self") — assert it's not a real type.
    assert edge.receiver_type in (None, "Self")
