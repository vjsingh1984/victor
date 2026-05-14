"""Tests for the semantic victor_contracts alias package."""

from __future__ import annotations


def test_contracts_top_level_alias_exports_sdk_symbols():
    import victor_contracts
    import victor_sdk

    assert victor_contracts.VerticalBase is victor_sdk.VerticalBase
    assert victor_contracts.VictorPlugin is victor_sdk.VictorPlugin
    assert victor_contracts.__version__ == victor_sdk.__version__


def test_contracts_nested_import_aliases_sdk_module_identity():
    from victor_contracts.verticals.protocols.base import VerticalBase as ContractsVerticalBase
    from victor_sdk.verticals.protocols.base import VerticalBase as SdkVerticalBase

    assert ContractsVerticalBase is SdkVerticalBase
