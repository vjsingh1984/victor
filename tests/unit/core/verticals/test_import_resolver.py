# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for vertical import path resolution helpers."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import patch

from victor.core.verticals.import_resolver import (
    import_module_with_fallback,
    module_import_candidates,
    vertical_module_candidates,
)


def test_vertical_module_candidates_prefers_external_package() -> None:
    """External package path should be first, with legacy/contrib fallbacks."""
    candidates = vertical_module_candidates("research", "escape_hatches")

    assert candidates[0] == "victor_research.escape_hatches"
    assert "victor.research.escape_hatches" in candidates
    assert "victor.verticals.contrib.research.escape_hatches" in candidates


def test_vertical_module_candidates_handles_dataanalysis_alias() -> None:
    """Historical data-analysis spellings should map to victor_dataanalysis."""
    candidates = vertical_module_candidates("data-analysis", "capabilities")
    assert candidates[0] == "victor_dataanalysis.capabilities"


def test_module_import_candidates_for_legacy_path() -> None:
    """Legacy victor.<vertical> paths should expand to external-first candidates."""
    candidates = module_import_candidates("victor.research.escape_hatches")
    assert candidates[:3] == [
        "victor_research.escape_hatches",
        "victor.research.escape_hatches",
        "victor.verticals.contrib.research.escape_hatches",
    ]


def test_module_import_candidates_for_external_path() -> None:
    """External paths should retain themselves first with backward fallbacks."""
    candidates = module_import_candidates("victor_research.escape_hatches")
    assert candidates[:3] == [
        "victor_research.escape_hatches",
        "victor.research.escape_hatches",
        "victor.verticals.contrib.research.escape_hatches",
    ]


def test_import_module_with_fallback_returns_first_importable_candidate() -> None:
    """Importer should skip missing candidates and return the first importable one."""
    legacy_module = ModuleType("victor.research.escape_hatches")

    def _fake_import(module_name: str):
        if module_name == "victor_research.escape_hatches":
            raise ImportError("external package missing")
        if module_name == "victor.research.escape_hatches":
            return legacy_module
        raise ImportError("missing")

    with patch("importlib.import_module", side_effect=_fake_import):
        module, resolved = import_module_with_fallback("victor.research.escape_hatches")

    assert module is legacy_module
    assert resolved == "victor.research.escape_hatches"


def test_import_module_with_fallback_returns_none_when_all_candidates_fail() -> None:
    """Importer should return (None, None) when no candidate is importable."""
    with patch("importlib.import_module", side_effect=ImportError("missing")):
        module, resolved = import_module_with_fallback("victor.research.escape_hatches")

    assert module is None
    assert resolved is None
