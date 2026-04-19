"""Tests for SDK-owned YAML tool dependency helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from victor_sdk.verticals.tool_dependencies import (
    ToolDependency,
    ToolDependencyLoadError,
    ToolDependencyLoader,
    YAMLToolDependencyProvider,
    create_tool_dependency_provider,
    get_cached_provider,
    invalidate_provider_cache,
    load_tool_dependency_yaml,
)


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    """Create a representative tool dependency config."""

    path = tmp_path / "tool_dependencies.yaml"
    path.write_text(
        """
version: "1.0"
vertical: sdk_test
transitions:
  read_file:
    - tool: write_file
      weight: 0.4
clusters:
  file_ops:
    - read_file
    - write_file
sequences:
  edit:
    - read_file
    - write_file
dependencies:
  - tool: write_file
    depends_on:
      - read_file
    enables:
      - run_tests
    weight: 0.8
required_tools:
  - read_file
optional_tools:
  - run_tests
default_sequence:
  - read_file
  - write_file
""".strip(),
        encoding="utf-8",
    )
    return path


def test_load_tool_dependency_yaml_canonicalizes_aliases(yaml_path: Path) -> None:
    """Canonicalization should normalize aliases to SDK-owned names."""

    config = load_tool_dependency_yaml(yaml_path, canonicalize=True, use_cache=False)

    assert config.required_tools == {"read"}
    assert config.optional_tools == {"test"}
    assert config.transitions == {"read": [("write", 0.4)]}
    assert config.sequences["edit"] == ["read", "write"]
    assert config.dependencies[0].tool_name == "write"
    assert config.dependencies[0].depends_on == {"read"}
    assert config.dependencies[0].enables == {"test"}


def test_load_tool_dependency_yaml_can_preserve_raw_names(yaml_path: Path) -> None:
    """Callers can opt out of canonicalization for vertical-specific semantics."""

    config = load_tool_dependency_yaml(yaml_path, canonicalize=False)

    assert config.required_tools == {"read_file"}
    assert config.optional_tools == {"run_tests"}
    assert config.transitions == {"read_file": [("write_file", 0.4)]}


def test_loader_raises_for_missing_vertical(tmp_path: Path) -> None:
    """A missing vertical field should fail fast with a clear error."""

    broken = tmp_path / "broken.yaml"
    broken.write_text("version: '1.0'\ntransitions: {}\n", encoding="utf-8")

    with pytest.raises(ToolDependencyLoadError, match="vertical"):
        ToolDependencyLoader().load(broken, use_cache=False)


def test_yaml_provider_merges_without_mutating_cached_config(yaml_path: Path) -> None:
    """Provider-level merge operations should not corrupt shared cached configs."""

    base_config = load_tool_dependency_yaml(yaml_path, canonicalize=True, use_cache=True)

    provider = YAMLToolDependencyProvider(
        yaml_path,
        additional_dependencies=[
            ToolDependency(tool_name="graph", depends_on={"read"}, enables={"write"}, weight=0.5)
        ],
        additional_sequences={"review": ["read", "graph", "write"]},
    )

    assert provider.vertical == "sdk_test"
    assert provider.yaml_path == yaml_path
    assert any(dependency.tool_name == "graph" for dependency in provider.get_dependencies())
    assert ["read", "graph", "write"] in provider.get_tool_sequences()

    cached_again = load_tool_dependency_yaml(yaml_path, canonicalize=True, use_cache=True)
    assert cached_again is base_config
    assert all(dependency.tool_name != "graph" for dependency in cached_again.dependencies)
    assert "review" not in cached_again.sequences


def test_provider_cache_reuses_and_invalidates_instances(yaml_path: Path) -> None:
    """Cached providers should be reused until explicitly invalidated."""

    invalidate_provider_cache()
    provider_one = get_cached_provider(str(yaml_path))
    provider_two = get_cached_provider(str(yaml_path))

    assert provider_one is provider_two

    invalidate_provider_cache(str(yaml_path))
    provider_three = get_cached_provider(str(yaml_path))
    assert provider_three is not provider_one


def test_create_tool_dependency_provider_returns_base_provider(yaml_path: Path) -> None:
    """Factory helper should return a usable provider instance."""

    provider = create_tool_dependency_provider(yaml_path)

    assert provider.get_required_tools() == {"read"}
    assert provider.get_recommended_sequence("missing") == ["read", "write"]
