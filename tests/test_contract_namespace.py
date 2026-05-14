from pathlib import Path

import tomllib

from victor_contracts import VictorPlugin
from victor_contracts.verticals.protocols.base import VerticalBase

from victor_dataanalysis.assistant import DataAnalysisAssistant
from victor_dataanalysis.plugin import DataAnalysisPlugin, plugin

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _entry_points() -> dict:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["entry-points"]


def test_pyproject_registers_contract_extension_entry_points() -> None:
    entry_points = _entry_points()

    assert "victor.sdk.protocols" not in entry_points
    assert entry_points["victor.extension.protocols"] == {
        "dataanalysis-tools": "victor_dataanalysis.protocols:DataAnalysisToolProvider",
        "dataanalysis-safety": "victor_dataanalysis.protocols:DataAnalysisSafetyProvider",
        "dataanalysis-prompts": "victor_dataanalysis.protocols:DataAnalysisPromptProvider",
        "dataanalysis-workflows": "victor_dataanalysis.protocols:DataAnalysisWorkflowProvider",
    }


def test_public_contract_modules_avoid_legacy_import_namespace() -> None:
    contract_modules = [
        "victor_dataanalysis/assistant.py",
        "victor_dataanalysis/plugin.py",
        "victor_dataanalysis/protocols.py",
        "victor_dataanalysis/prompts.py",
        "victor_dataanalysis/safety.py",
    ]
    banned_imports = (
        "from victor" "_sdk import",
        "from victor" "_sdk.capabilities import",
        "from victor" "_sdk.core.types import",
        "from victor" "_sdk.verticals.protocols import",
        "from victor" "_sdk.verticals.protocols.base import",
        "from victor.framework.extensions import",
    )

    for module in contract_modules:
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        for banned in banned_imports:
            assert banned not in source, f"{module} still imports {banned}"


def test_plugin_and_assistant_use_contract_protocols() -> None:
    assert isinstance(plugin, VictorPlugin)
    assert isinstance(plugin, DataAnalysisPlugin)
    assert issubclass(DataAnalysisAssistant, VerticalBase)
