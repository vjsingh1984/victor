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

"""Tests for the offline judge-calibration harness and verifiable corpus (EVR-2, ADR-011)."""

import json
from pathlib import Path

import pytest

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    Transcript,
    VerifiableTask,
    alternating_scripted_executor,
    binary_categorize,
    make_scripted_executor,
)

# --- Corpus integrity ------------------------------------------------------------------------------


def test_corpus_task_ids_are_unique_and_families_covered() -> None:
    tasks = default_corpus(variants=3)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)) == 18
    assert {t.family for t in tasks} == {"file-create", "code-fix", "refactor", "docs", "qa"}


def test_corpus_rejects_zero_variants() -> None:
    with pytest.raises(ValueError):
        default_corpus(variants=0)


def test_every_task_verifies_zero_when_unsolved(tmp_path: Path) -> None:
    executor = make_scripted_executor(lambda _t: False)
    for i, task in enumerate(default_corpus(variants=2)):
        workspace = tmp_path / f"unsolved_{i}"
        workspace.mkdir()
        task.setup(workspace)
        transcript = executor(task, workspace)
        assert task.verify(workspace, transcript) == 0.0, task.task_id


def test_every_task_verifies_one_when_solved(tmp_path: Path) -> None:
    executor = make_scripted_executor(lambda _t: True)
    for i, task in enumerate(default_corpus(variants=2)):
        workspace = tmp_path / f"solved_{i}"
        workspace.mkdir()
        task.setup(workspace)
        transcript = executor(task, workspace)
        assert task.verify(workspace, transcript) == 1.0, task.task_id


# --- Harness end-to-end ----------------------------------------------------------------------------


def _evidence_judge(_prompt: str, transcript: Transcript, _workspace: Path) -> float:
    return 1.0 if transcript.tool_steps() else 0.0


def _constant_judge(_prompt: str, _transcript: Transcript, _workspace: Path) -> float:
    return 1.0


def test_perfectly_agreeing_judge_is_trusted(tmp_path: Path) -> None:
    harness = JudgeCalibrationHarness(default_corpus(variants=4))
    report = harness.run(
        alternating_scripted_executor(period=5),
        _evidence_judge,
        workspace_root=tmp_path,
    )
    assert report.gate_decision.trusted
    assert report.overall.krippendorff_alpha == pytest.approx(1.0)
    assert len(report.samples) == 24
    # Both outcomes must actually occur, or agreement is vacuous.
    golds = {s.gold for s in report.samples}
    assert golds == {0.0, 1.0}


def test_constant_judge_fails_gate_on_mixed_gold(tmp_path: Path) -> None:
    harness = JudgeCalibrationHarness(default_corpus(variants=4))
    report = harness.run(
        alternating_scripted_executor(period=5),
        _constant_judge,
        workspace_root=tmp_path,
    )
    assert not report.gate_decision.trusted


def test_report_shape_and_json_roundtrip(tmp_path: Path) -> None:
    harness = JudgeCalibrationHarness(default_corpus(variants=2))
    report = harness.run(
        alternating_scripted_executor(period=5),
        _evidence_judge,
        workspace_root=tmp_path / "ws",
    )
    out = tmp_path / "reports" / "report.json"
    report.save(out)
    data = json.loads(out.read_text())
    assert data["n"] == 12
    assert set(data["per_family"]) == {"file-create", "code-fix", "refactor", "docs", "qa"}
    assert data["gate"]["trusted"] is True
    assert "completion verdicts only" in data["gate"]["scope"]
    assert all({"task_id", "family", "gold", "judged"} <= set(s) for s in data["samples"])


def test_judge_is_blinded_to_gold(tmp_path: Path) -> None:
    """The judge callable only ever receives (prompt, transcript, workspace)."""
    seen: list[tuple] = []

    def spy_judge(prompt: str, transcript: Transcript, workspace: Path) -> float:
        seen.append((prompt, transcript, workspace))
        return 0.0

    harness = JudgeCalibrationHarness(default_corpus(variants=1))
    harness.run(make_scripted_executor(lambda _t: True), spy_judge, workspace_root=tmp_path)
    assert len(seen) == 6
    for prompt, transcript, workspace in seen:
        assert isinstance(prompt, str)
        assert isinstance(transcript, Transcript)
        assert isinstance(workspace, Path) and workspace.is_dir()


def test_scripted_executor_requires_reference_solution(tmp_path: Path) -> None:
    task = VerifiableTask(
        task_id="no-solve",
        family="qa",
        prompt="unanswerable",
        setup=lambda ws: None,
        verify=lambda ws, t: 0.0,
        solve=None,
    )
    executor = make_scripted_executor(lambda _t: True)
    with pytest.raises(ValueError, match="no reference solution"):
        executor(task, tmp_path)


def test_binary_categorize_threshold() -> None:
    assert binary_categorize(0.5) is True
    assert binary_categorize(0.49) is False
