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

"""Polyglot index-build timeout heuristic.

Regression for the 60s timeout that always fired on large non-Python repos:
the heuristic previously counted only ``.py`` files, so a Rust codebase scored
the bare 60s base timeout regardless of size.
"""

from pathlib import Path

from victor.tools.code_search_tool import _calculate_index_build_timeout


def test_rust_only_repo_scales_above_base_timeout(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(120):
        (src / f"mod_{i}.rs").write_text("pub fn f() {}\n" * 40)

    timeout = _calculate_index_build_timeout(tmp_path)

    # 120 source files * 5s = 600s file term alone; must exceed the 60s base.
    assert timeout > 60.0


def test_empty_repo_uses_base_timeout(tmp_path: Path) -> None:
    assert _calculate_index_build_timeout(tmp_path) == 60.0


def test_victor_cache_is_not_counted(tmp_path: Path) -> None:
    """Files under .victor (caches) must not inflate the estimate."""
    cache = tmp_path / ".victor" / "swe_bench_cache"
    cache.mkdir(parents=True)
    for i in range(200):
        (cache / f"junk_{i}.py").write_text("x = 1\n")

    assert _calculate_index_build_timeout(tmp_path) == 60.0
