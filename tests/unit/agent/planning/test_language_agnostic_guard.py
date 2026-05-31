"""Guard tests preventing language-specific code from re-entering the planning framework.

Background
----------
The planning subsystem accumulated thirteen "fix rust review" band-aid commits before
the May 2026 refactor that stripped seven Rust/Cargo-specific compute nodes from
``victor/agent/planning/team_execution.py`` and replaced the rust workspace few-shot
example in ``victor/agent/planning/readable_schema.py`` with a language-agnostic
template.  These guard tests fail loudly if any of that hardcoding creeps back in,
forcing language-specific behaviour to live in verticals (via ``register_compute_node``
and ``register_manifest_handler``) rather than in the framework.

The only file allowed to contain language-specific tokens is
``victor/agent/planning/language_manifests.py``, which deliberately enumerates
manifest filenames for nine ecosystems behind a uniform ``LanguageManifestHandler``
Protocol.  ``repository_profile.py`` similarly contains a per-language guidance
dictionary that is treated as data, not as a routing branch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
PLANNING_DIR = ROOT / "victor" / "agent" / "planning"
SERVICES_DIR = ROOT / "victor" / "agent" / "services"

# Files where language-specific tokens are by design (data tables, manifest discovery,
# or per-language guidance strings used uniformly across ecosystems).
ALLOWED_FILES = {
    PLANNING_DIR / "language_manifests.py",
    PLANNING_DIR / "repository_profile.py",
}

# Tokens that indicate language-specific routing logic.  We scan as case-sensitive
# substrings because the bad pattern (``Path("rust") / "Cargo.toml"``) is exact.
LANGUAGE_SPECIFIC_TOKENS = (
    "Cargo.toml",
    'Path("rust")',
    "rust/crates",
    "_builtin_rust_",
    "_builtin_cargo_",
    "_builtin_cross_crate",
    "_builtin_parse_workspace_members",
    "_cargo_dependency_map",
    "_rust_hotspot_scan",
    "_rust_crate_review",
    "_rust_prioritized_report",
)


def _iter_framework_files() -> list[Path]:
    files: list[Path] = []
    for directory in (PLANNING_DIR, SERVICES_DIR):
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if path in ALLOWED_FILES:
                continue
            files.append(path)
    return files


@pytest.mark.parametrize("token", LANGUAGE_SPECIFIC_TOKENS)
def test_no_language_specific_token_in_framework(token: str) -> None:
    """Framework planning code must not hardcode language-specific identifiers."""
    offenders: list[tuple[Path, int, str]] = []
    for path in _iter_framework_files():
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            # Skip docstring examples that legitimately mention rust in prose
            # (illustrative inputs to generic inference functions).  We allow
            # the token in lines that look like comments containing
            # "example" or "e.g." so docstrings can stay informative.
            stripped = line.strip()
            is_doc_example = (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
                or stripped.startswith('"')
                or stripped.startswith("'")
            ) and re.search(r"example|e\.g\.|illustrat", line, re.IGNORECASE)
            if token in line and not is_doc_example:
                offenders.append((path, line_no, line.rstrip()))
    assert not offenders, (
        f"Language-specific token {token!r} re-appeared in framework planning code. "
        "If you really need this behaviour, add it to a vertical "
        "(register_compute_node / register_manifest_handler) instead of the framework.\n"
        + "\n".join(f"  {p.relative_to(ROOT)}:{n}: {ln}" for p, n, ln in offenders[:10])
    )


def test_framework_registers_only_generic_compute_nodes() -> None:
    """The framework must ship at most two compute nodes — both language-agnostic.

    Verticals add language-specific nodes via ``register_compute_node`` at plugin init.
    The framework itself must not pre-register anything domain-specific.
    """
    # Import fresh so module-load registrations run.
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    registered = set(PlanningTeamExecutionAdapter._COMPUTE_NODES.keys())
    expected = {"_checklist_artifact", "_aggregate_target_findings"}
    extra = registered - expected
    assert not extra, (
        f"Framework registered unexpected compute nodes at import: {sorted(extra)}. "
        "Language-specific or domain-specific compute nodes belong in verticals, "
        "not in the framework's import-time registration block."
    )


def test_compute_node_routing_requires_explicit_node_name() -> None:
    """Plan steps must not be routed to compute nodes by description substring.

    The historical band-aid was a regex that auto-routed any step whose description
    contained "Cargo.toml" or "workspace member" to a built-in parser.  That routing
    coupled LLM prose to framework dispatch, which broke for paraphrased prompts
    and non-English descriptions.  Routing now requires either explicit
    ``step.context["node"]`` or the very narrow checklist-artifact heuristic.
    """
    from victor.agent.planning.base import PlanStep, StepType
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    # A step whose description LOOKS like the old rust auto-route trigger but
    # declares neither execution=compute nor a registered node.  Must NOT route
    # to any compute node — has to fall through to the agent path.
    step = PlanStep(
        id="2",
        description="Inventory Rust workspace members from rust/Cargo.toml",
        step_type=StepType.RESEARCH,
        context={"produces": "workspace_members"},
    )
    assert PlanningTeamExecutionAdapter._compute_node_for_step(step) is None

    step_with_cargo_dep_desc = PlanStep(
        id="3",
        description="Map cargo.toml dependencies across crates",
        step_type=StepType.RESEARCH,
        context={"produces": "dependency_findings"},
    )
    assert PlanningTeamExecutionAdapter._compute_node_for_step(step_with_cargo_dep_desc) is None


def test_prompt_template_uses_generic_vocabulary() -> None:
    """The complex few-shot example must not present a rust workspace as the canonical shape."""
    from victor.agent.planning.readable_schema import ReadableTaskPlan

    prompt = ReadableTaskPlan.get_llm_prompt()
    # The old template used "Rust best practices review" as the canonical
    # complex example name.  The replacement template is multi-target and
    # language-agnostic.
    assert "Rust best practices review" not in prompt
    # The new generic example uses these tokens — sanity-check that the replacement
    # actually landed and wasn't reverted.
    assert "review_targets" in prompt
    assert "per_target_findings" in prompt
    assert "cross_target_findings" in prompt
    assert "Codebase review (workspace-by-workspace)" in prompt
