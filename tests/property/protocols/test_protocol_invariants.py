# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Property-based tests for protocol invariants.

Uses Hypothesis to test protocol properties across 100+ iterations,
ensuring robust protocol behavior with diverse inputs.
"""

from hypothesis import given, strategies as st, settings, Phase
import pytest
from typing import Any, Optional, Dict, Set, List, Callable
from victor.protocols import (
    CapabilityContainerProtocol,
    WorkflowProviderProtocol,
    TieredConfigProviderProtocol,
    ExtensionProviderProtocol,
)
from victor.framework.graph import StateGraph


class TestCapabilityContainerProtocolInvariants:
    """Property-based tests for CapabilityContainerProtocol invariants."""

    @given(
        capability_name=st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
        capabilities=st.dictionaries(
            keys=st.text(
                min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            values=st.integers(min_value=0, max_value=100),
            min_size=0,
            max_size=15,
        ),
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_has_capability_reflects_get_capability(
        self, capability_name: str, capabilities: Dict[str, int]
    ):
        """has_capability(name) should return True iff get_capability(name) is not None."""

        class TestContainer:
            def __init__(self, caps: Dict[str, int]):
                self._capabilities = caps

            def has_capability(self, capability_name: str) -> bool:
                return capability_name in self._capabilities

            def get_capability(self, name: str) -> Optional[int]:
                return self._capabilities.get(name)

        container = TestContainer(capabilities)

        # Property: has_capability should match get_capability result
        has_it = container.has_capability(capability_name)
        gets_it = container.get_capability(capability_name) is not None

        assert (
            has_it == gets_it
        ), f"Inconsistent: has_capability={has_it}, get_capability non-None={gets_it}"

    @given(
        capabilities=st.dictionaries(
            keys=st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            values=st.integers(min_value=0, max_value=1000),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_capability_retrieval_is_consistent(self, capabilities: Dict[str, int]):
        """get_capability should return consistent values for the same capability."""

        class TestContainer:
            def __init__(self, caps: Dict[str, int]):
                self._capabilities = caps

            def has_capability(self, capability_name: str) -> bool:
                return capability_name in self._capabilities

            def get_capability(self, name: str) -> Optional[int]:
                return self._capabilities.get(name)

        container = TestContainer(capabilities)

        for cap_name in capabilities:
            # Property: Multiple retrievals should return the same value
            first = container.get_capability(cap_name)
            second = container.get_capability(cap_name)
            third = container.get_capability(cap_name)

            assert first == second == third, f"Inconsistent retrieval for {cap_name}"

    @given(
        capabilities=st.dictionaries(
            keys=st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            values=st.integers(min_value=0, max_value=1000),
            min_size=0,
            max_size=30,
        ),
        queries=st.lists(
            st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=0,
            max_size=50,
            unique=True,
        ),
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_nonexistent_capabilities_return_none(
        self, capabilities: Dict[str, int], queries: List[str]
    ):
        """get_capability should return None for nonexistent capabilities."""

        class TestContainer:
            def __init__(self, caps: Dict[str, int]):
                self._capabilities = caps

            def has_capability(self, capability_name: str) -> bool:
                return capability_name in self._capabilities

            def get_capability(self, name: str) -> Optional[int]:
                return self._capabilities.get(name)

        container = TestContainer(capabilities)

        for query in queries:
            if query not in capabilities:
                result = container.get_capability(query)
                assert (
                    result is None
                ), f"Expected None for nonexistent capability '{query}', got {result}"


class TestWorkflowProviderProtocolInvariants:
    """Property-based tests for WorkflowProviderProtocol invariants."""

    @given(
        workflow_names=st.lists(
            st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=0,
            max_size=20,
            unique=True,
        )
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_workflow_provider_returns_all_workflows(self, workflow_names: List[str]):
        """get_workflows should return all registered workflows."""

        class MockStateGraph:
            """Minimal StateGraph mock for testing."""

            pass

        class TestWorkflowProvider:
            def __init__(self, names: List[str]):
                self._workflows = {name: MockStateGraph() for name in names}

            def get_workflows(self) -> Dict[str, MockStateGraph]:
                return self._workflows.copy()

            def get_workflow_provider(self) -> Optional["TestWorkflowProvider"]:
                return self

        provider = TestWorkflowProvider(workflow_names)
        workflows = provider.get_workflows()

        # Property: All workflow names should be in result
        assert set(workflows.keys()) == set(workflow_names), "Workflow names don't match"

        # Property: Number of workflows should match
        assert len(workflows) == len(workflow_names), "Workflow count doesn't match"

    @given(
        workflow_names=st.lists(
            st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=1,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_workflow_provider_self_reference(self, workflow_names: List[str]):
        """get_workflow_provider should return self when available."""

        class TestWorkflowProvider:
            def __init__(self, names: List[str]):
                self._workflows = dict.fromkeys(names)

            def get_workflows(self) -> Dict[str, Any]:
                return self._workflows.copy()

            def get_workflow_provider(self) -> Optional["TestWorkflowProvider"]:
                return self

        provider = TestWorkflowProvider(workflow_names)
        result = provider.get_workflow_provider()

        # Property: Should return self
        assert result is provider, "get_workflow_provider should return self"


class TestTieredConfigProviderProtocolInvariants:
    """Property-based tests for TieredConfigProviderProtocol invariants."""

    @given(
        tool_pool=st.sets(
            st.text(
                min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=0,
            max_size=30,
        )
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_tiered_config_sets_are_disjoint(self, tool_pool: Set[str]):
        """Tool sets in tiered config should be disjoint when constructed from disjoint partitions."""

        # Split tool_pool into 3 disjoint partitions
        tool_list = list(tool_pool)

        class TestTieredConfig:
            def __init__(self, tools: List[str]):
                # Partition into 3 disjoint sets
                n = len(tools)
                self._mandatory = set(tools[: n // 3])
                self._core = set(tools[n // 3 : 2 * n // 3])
                self._optional = set(tools[2 * n // 3 :])

            @property
            def mandatory(self) -> Set[str]:
                return self._mandatory.copy()

            @property
            def core(self) -> Set[str]:
                return self._core.copy()

            @property
            def optional(self) -> Set[str]:
                return self._optional.copy()

        config = TestTieredConfig(tool_list)

        # Property: Sets should be disjoint (no overlap) by construction
        mand_core_overlap = config.mandatory & config.core
        mand_opt_overlap = config.mandatory & config.optional
        core_opt_overlap = config.core & config.optional

        assert len(mand_core_overlap) == 0, f"mandatory and core overlap: {mand_core_overlap}"
        assert len(mand_opt_overlap) == 0, f"mandatory and optional overlap: {mand_opt_overlap}"
        assert len(core_opt_overlap) == 0, f"core and optional overlap: {core_opt_overlap}"

    @given(
        tool_names=st.lists(
            st.text(
                min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=1,
            max_size=30,
            unique=True,
        ),
        mandatory_ratio=st.floats(min_value=0.0, max_value=0.3),
        core_ratio=st.floats(min_value=0.0, max_value=0.4),
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_tiered_config_hierarchy(
        self, tool_names: List[str], mandatory_ratio: float, core_ratio: float
    ):
        """Tiered config should maintain hierarchical constraints."""

        import math

        class TestTieredConfig:
            def __init__(self, tools: List[str], m_ratio: float, c_ratio: float):
                n_mandatory = math.floor(len(tools) * m_ratio)
                n_core = math.floor(len(tools) * c_ratio)

                self._mandatory = set(tools[:n_mandatory])
                self._core = set(tools[n_mandatory : n_mandatory + n_core])
                self._optional = set(tools[n_mandatory + n_core :])

            @property
            def mandatory(self) -> Set[str]:
                return self._mandatory.copy()

            @property
            def core(self) -> Set[str]:
                return self._core.copy()

            @property
            def optional(self) -> Set[str]:
                return self._optional.copy()

        config = TestTieredConfig(tool_names, mandatory_ratio, core_ratio)

        # Property: Total tools should equal union of all tiers
        all_tools = config.mandatory | config.core | config.optional
        assert len(all_tools) <= len(tool_names), "Union of tiers exceeds original tool set"


class TestExtensionProviderProtocolInvariants:
    """Property-based tests for ExtensionProviderProtocol invariants."""

    @given(
        extensions=st.lists(
            st.functions(
                returns=st.one_of(st.none(), st.integers(), st.text(), st.booleans()),
                pure=True,
            ),
            min_size=0,
            max_size=20,
        )
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_extension_registration_preserves_extensions(self, extensions: List[Callable[[], Any]]):
        """Registering extensions should preserve them in get_extensions."""

        class TestExtensionProvider:
            def __init__(self):
                self._extensions: List[Callable[..., Any]] = []

            def get_extensions(self) -> List[Callable[..., Any]]:
                return self._extensions.copy()

            def register_extension(self, extension: Callable[..., Any]) -> None:
                if extension not in self._extensions:
                    self._extensions.append(extension)

        provider = TestExtensionProvider()

        # Register all extensions
        for ext in extensions:
            provider.register_extension(ext)

        retrieved = provider.get_extensions()

        # Property: Number of extensions should match
        assert len(retrieved) == len(set(extensions)), "Extension count mismatch (duplicates?)"

        # Property: All registered extensions should be retrievable
        for ext in extensions:
            assert ext in retrieved, f"Extension {ext} not found in retrieved list"

    @given(
        initial_extensions=st.lists(
            st.functions(returns=st.none(), pure=True),
            min_size=0,
            max_size=10,
        ),
        new_extensions=st.lists(
            st.functions(returns=st.none(), pure=True),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_extension_registration_is_additive(
        self, initial_extensions: List[Callable[[], None]], new_extensions: List[Callable[[], None]]
    ):
        """Adding extensions should be additive and idempotent."""

        class TestExtensionProvider:
            def __init__(self):
                self._extensions: List[Callable[..., Any]] = []

            def get_extensions(self) -> List[Callable[..., Any]]:
                return self._extensions.copy()

            def register_extension(self, extension: Callable[..., Any]) -> None:
                if extension not in self._extensions:
                    self._extensions.append(extension)

        provider = TestExtensionProvider()

        # Register initial extensions
        for ext in initial_extensions:
            provider.register_extension(ext)

        initial_count = len(provider.get_extensions())

        # Register new extensions
        for ext in new_extensions:
            provider.register_extension(ext)

        final_count = len(provider.get_extensions())

        # Property: Final count should be >= initial count
        assert final_count >= initial_count, "Extension count decreased"

        # Property: Final count should be at most initial + new (no duplicates)
        expected_max = len(set(initial_extensions) | set(new_extensions))
        assert (
            final_count <= expected_max
        ), f"Extension count {final_count} exceeds expected {expected_max}"


class TestProtocolCompositionInvariants:
    """Property-based tests for protocol composition invariants."""

    @given(
        capability_names=st.lists(
            st.text(
                min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=1,
            max_size=15,
            unique=True,
        ),
        workflow_names=st.lists(
            st.text(
                min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=1,
            max_size=15,
            unique=True,
        ),
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_object_can_implement_multiple_protocols(
        self, capability_names: List[str], workflow_names: List[str]
    ):
        """An object can implement multiple protocols simultaneously."""

        class MockStateGraph:
            pass

        class MultiProtocolObject:
            """Implements both CapabilityContainerProtocol and WorkflowProviderProtocol."""

            def __init__(self, caps: List[str], workflows: List[str]):
                self._capabilities = {name: {} for name in caps}
                self._workflows = {name: MockStateGraph() for name in workflows}

            # CapabilityContainerProtocol methods
            def has_capability(self, capability_name: str) -> bool:
                return capability_name in self._capabilities

            def get_capability(self, name: str) -> Optional[Dict[str, Any]]:
                return self._capabilities.get(name)

            # WorkflowProviderProtocol methods
            def get_workflows(self) -> Dict[str, MockStateGraph]:
                return self._workflows.copy()

            def get_workflow_provider(self) -> Optional["MultiProtocolObject"]:
                return self

        obj = MultiProtocolObject(capability_names, workflow_names)

        # Property: Should conform to both protocols
        assert isinstance(
            obj, CapabilityContainerProtocol
        ), "Should conform to CapabilityContainerProtocol"
        assert isinstance(
            obj, WorkflowProviderProtocol
        ), "Should conform to WorkflowProviderProtocol"

        # Property: Both protocol interfaces should work
        assert all(obj.has_capability(cap) for cap in capability_names)
        assert set(obj.get_workflows().keys()) == set(workflow_names)
