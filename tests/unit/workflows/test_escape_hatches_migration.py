# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

import pytest
from victor.workflows.escape_hatches import (
    CONDITIONS,
    TRANSFORMS,
    tests_passing,
    should_continue_fixing,
    complexity_check,
    ensure_global_escape_hatches_registered,
)


def test_tests_passing():
    # Scenario: No tests
    assert tests_passing({}) == "no_tests"

    # Scenario: Failing tests
    ctx = {"test_results": {"passed": 5, "failed": 1, "coverage": 0.9}}
    assert tests_passing(ctx) == "failing"

    # Scenario: Low coverage
    ctx = {"test_results": {"passed": 5, "failed": 0, "coverage": 0.5}, "min_coverage": 0.8}
    assert tests_passing(ctx) == "failing"

    # Scenario: Passing tests
    ctx = {"test_results": {"passed": 5, "failed": 0, "coverage": 0.9}}
    assert tests_passing(ctx) == "passing"

    # Scenario: Warnings are tolerated by default (non-strict mode)
    ctx = {"test_results": {"passed": 5, "failed": 0, "warnings": 3, "coverage": 0.9}}
    assert tests_passing(ctx) == "passing"

    # Scenario: Strict mode fails on ANY warning
    ctx = {
        "test_results": {"passed": 5, "failed": 0, "warnings": 1, "coverage": 0.9},
        "strict": True,
    }
    assert tests_passing(ctx) == "failing"

    # Scenario: Strict mode passes when there are no warnings
    ctx = {
        "test_results": {"passed": 5, "failed": 0, "warnings": 0, "coverage": 0.9},
        "strict": True,
    }
    assert tests_passing(ctx) == "passing"


def test_should_continue_fixing():
    # Scenario: Max iterations reached
    ctx = {"fix_iterations": 5, "max_iterations": 5}
    assert should_continue_fixing(ctx) == "submit_best_effort"

    # Scenario: High pass rate
    ctx = {"fix_iterations": 1, "test_results": {"passed": 95, "failed": 0}}
    assert should_continue_fixing(ctx) == "submit_best_effort"

    # Scenario: Making progress
    ctx = {"fix_iterations": 1, "progress_made": True, "test_results": {"passed": 5, "failed": 5}}
    assert should_continue_fixing(ctx) == "continue_fixing"

    # Scenario: No progress, escalate
    ctx = {"fix_iterations": 3, "progress_made": False, "test_results": {"passed": 5, "failed": 5}}
    assert should_continue_fixing(ctx) == "escalate"


def test_complexity_check():
    # Scenario: String analysis - complex
    ctx = {"task_analysis": "This is a very complex task involving major changes."}
    assert complexity_check(ctx) == "complex"

    # Scenario: String analysis - simple
    ctx = {"task_analysis": "Just a minor tweak, very straightforward."}
    assert complexity_check(ctx) == "simple"

    # Scenario: Dict analysis - medium
    ctx = {"task_analysis": {"complexity": "medium"}}
    assert complexity_check(ctx) == "medium"

    # Scenario: Team size - complex
    ctx = {"task_analysis": {"team_size": 4}}
    assert complexity_check(ctx) == "complex"


def test_generic_hatches_register_into_global_namespace():
    """The generic conditions/transforms must be registered into the global registry so
    every YAML workflow can use them (previously the module was orphaned — imported only
    by this test)."""
    from victor.framework.escape_hatch_registry import (
        EscapeHatchRegistry,
        get_escape_hatch_registry,
    )

    EscapeHatchRegistry.reset_instance()
    ensure_global_escape_hatches_registered()
    conditions, transforms = get_escape_hatch_registry().get_registry_for_vertical(
        "", include_global=True
    )
    for name in CONDITIONS:
        assert name in conditions, name
    for name in TRANSFORMS:
        assert name in transforms, name


def test_registration_is_idempotent():
    """Repeated registration must not raise (replace=True), so reconfiguring providers or
    resetting the registry is safe."""
    ensure_global_escape_hatches_registered()
    ensure_global_escape_hatches_registered()


def test_provider_hatches_take_precedence_over_global():
    """Provider-specific escape hatches override global generics on name conflict —
    the YAML config loader merges global beneath provider-specific."""
    from victor.framework.escape_hatch_registry import (
        EscapeHatchRegistry,
        get_escape_hatch_registry,
    )

    EscapeHatchRegistry.reset_instance()
    ensure_global_escape_hatches_registered()
    global_conditions, _ = get_escape_hatch_registry().get_registry_for_vertical(
        "", include_global=True
    )
    provider_conditions = {"complexity_check": lambda ctx: "provider_specific"}
    merged = {**global_conditions, **provider_conditions}
    assert merged["complexity_check"]({}) == "provider_specific"
    assert merged["tdd_cycle_status"]({"tests_written": False}) == "red"
