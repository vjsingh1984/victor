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

"""CLI smoke tests for the offline calibration runner (EVR-2, ADR-011).

The harness/corpus/adapters have unit coverage, but the runner script itself is the
user-facing entry point — an argparse regression, broken import, or report-path bug
there would ship invisibly without these tests. Scripted judges only: fast, no LLM.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "benchmarks" / "judge_calibration" / "run_offline_calibration.py"


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


def test_runner_executes_offline_and_writes_reports(tmp_path: Path) -> None:
    result = _run(["--variants", "1", "--out", str(tmp_path)])
    assert result.returncode == 0, result.stderr[-2000:]
    # All three scripted judges reported with a gate verdict.
    for judge in ("credulous", "evidence", "rubric-heuristic"):
        assert f"[{judge}]" in result.stdout
        report = json.loads((tmp_path / f"{judge}.json").read_text())
        assert report["n"] == 6
        assert "trusted" in report["gate"]
    assert "TRUSTED" in result.stdout  # at least one verdict line rendered


def test_runner_rejects_unknown_judge_profile(tmp_path: Path) -> None:
    result = _run(
        ["--variants", "1", "--out", str(tmp_path), "--judge-profile", "no-such-profile-xyz"]
    )
    assert result.returncode != 0
    assert "not found" in result.stderr
    assert "Available:" in result.stderr


def test_runner_help_documents_the_profile_flag() -> None:
    result = _run(["--help"], timeout=60)
    assert result.returncode == 0
    for flag in ("--judge-profile", "--llm-judge-provider", "--llm-judge-base-url", "--variants"):
        assert flag in result.stdout
