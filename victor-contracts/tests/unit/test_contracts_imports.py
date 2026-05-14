"""Tests for the canonical victor_contracts package."""

from __future__ import annotations


def test_contracts_top_level_exports_contract_symbols():
    import victor_contracts

    assert hasattr(victor_contracts, "VerticalBase")
    assert hasattr(victor_contracts, "VictorPlugin")
    assert victor_contracts.__version__


def test_contracts_nested_imports_share_top_level_identity():
    from victor_contracts import VerticalBase as TopLevelVerticalBase
    from victor_contracts.verticals.protocols.base import VerticalBase as ContractsVerticalBase

    assert ContractsVerticalBase is TopLevelVerticalBase
