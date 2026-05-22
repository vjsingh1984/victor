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


# -----------------------------------------------------------------------------
# Struct-field receivers: `self.field.method()` resolves through the struct's
# field declarations to bind against the field's type, not Self. Common Rust
# pattern; intra-file only (cross-file struct field types are out of scope).
# -----------------------------------------------------------------------------


def test_self_field_method_resolves_through_struct_field_type():
    """`self.field.method()` inside `impl Foo` with `struct Foo { field: Bar }` -> Bar."""
    source = """
struct Foo {
    field: Bar,
}
impl Foo {
    fn outer(&self) {
        self.field.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"


def test_self_field_method_strips_reference_in_field_type():
    """`field: &Bar` and `field: &mut Bar` both yield 'Bar'."""
    source = """
struct Foo {
    field: &'static Bar,
}
impl Foo {
    fn outer(&self) {
        self.field.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"


def test_self_field_method_unwraps_generic_to_base_type():
    """`field: Vec<Bar>` resolves to 'Vec' (the base type; generic args dropped).

    Bar would be the right answer for `Vec`, `Option`, `Box`-style single-arg
    wrappers, but that requires whitelisting and we explicitly chose not to
    model standard-library smart-pointer transparency in this round.
    """
    source = """
struct Foo {
    field: Vec<Bar>,
}
impl Foo {
    fn outer(&self) {
        self.field.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Vec"


def test_chained_field_access_resolves_through_intermediate_struct():
    """`self.a.b.method()` walks through both struct field maps."""
    source = """
struct Outer {
    a: Inner,
}
struct Inner {
    b: Target,
}
impl Outer {
    fn run(&self) {
        self.a.b.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Target"


def test_unknown_struct_field_falls_back_to_none():
    """`self.nonexistent.method()` — struct doesn't declare the field, falls back to None."""
    source = """
struct Foo { real: Bar }
impl Foo {
    fn run(&self) {
        self.nonexistent.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type is None


def test_struct_field_inference_works_when_struct_appears_after_impl():
    """Rust allows forward references; the plugin must do a struct-collection
    pre-pass so a struct defined below its impl block still resolves."""
    source = """
impl Foo {
    fn run(&self) {
        self.field.method();
    }
}
struct Foo {
    field: Bar,
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"
