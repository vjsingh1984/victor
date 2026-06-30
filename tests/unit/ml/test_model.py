# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012: model artifact + numpy inference round-trip + gating + delta blend."""

import numpy as np
import pytest

pytest.importorskip("sklearn")  # trainer is dev-only ([ml] extra)

from victor.ml.features import FEATURE_SPEC_VERSION  # noqa: E402
from victor.ml.model import EdgeClassifierModel  # noqa: E402
from victor.ml.trainer import train_model  # noqa: E402

# Synthetic, linearly-separable-ish data per decision type.
_SAMPLES = {
    "task_type_classification": [
        ("create a new module for user auth", "create"),
        ("write a fresh handler", "create"),
        ("scaffold the api layer", "create"),
        ("debug the failing login test", "debug"),
        ("fix the auth bug in login", "debug"),
        ("investigate the crash on startup", "debug"),
        ("explain how the cache works", "explain"),
        ("describe the indexing pipeline", "explain"),
        ("summarize the routing logic", "explain"),
    ]
    * 4,  # replicate so each class has enough samples
}


def _trained() -> EdgeClassifierModel:
    return train_model(_SAMPLES, model_version="test-1", threshold=0.5)


def test_train_export_load_roundtrip(tmp_path):
    model = _trained()
    path = str(tmp_path / "edge.npz")
    model.save(path)
    loaded = EdgeClassifierModel.load(path)

    assert loaded.feature_spec_version == FEATURE_SPEC_VERSION
    assert loaded.model_version == "test-1"
    assert set(loaded.heads) == {"task_type_classification"}
    head = loaded.heads["task_type_classification"]
    assert set(head.labels) == {"create", "debug", "explain"}

    # The loaded model predicts a sane label for a training-like input.
    label, conf = loaded.predict("task_type_classification", "fix the broken login bug")
    assert label == "debug"
    assert conf >= head.threshold


def test_unknown_decision_type_defers():
    model = _trained()
    label, conf = model.predict("not_a_real_type", "anything")
    assert label is None
    assert conf == 0.0


def test_confidence_gating_returns_none_below_threshold(tmp_path):
    model = _trained()
    model.heads["task_type_classification"].threshold = 0.99  # force below-gate
    label, conf = model.predict("task_type_classification", "fix the login bug")
    assert label is None  # defers to heuristic
    assert 0.0 <= conf < 0.99


def test_feature_spec_mismatch_raises(tmp_path):
    model = _trained()
    path = str(tmp_path / "edge.npz")
    model.save(path)
    # Corrupt the spec version in the saved artifact.
    data = dict(np.load(path, allow_pickle=False))
    import numpy as _np

    data["feature_spec_version"] = _np.array("999")
    _np.savez(path, **data)
    with pytest.raises(ValueError, match="feature_spec_version"):
        EdgeClassifierModel.load(path)


def test_delta_blend_changes_prediction(tmp_path):
    """A per-project delta overlay can flip a prediction toward a label."""
    model = _trained()
    model.alpha = 1.0
    head = model.heads["task_type_classification"]
    text = "fix the login bug"
    base_label, _ = model.predict("task_type_classification", text)

    from victor.ml.features import extract_features

    feats = extract_features(text)
    target = "explain" if base_label != "explain" else "create"
    target_idx = head.labels.index(target)
    # Build a delta that strongly boosts `target` on every active feature.
    delta = {h: _label_onehot(head.labels, target_idx) * 50.0 for h in feats}
    model.heads["task_type_classification"].threshold = 0.0  # ignore gating here
    new_label, _ = model.predict("task_type_classification", text, delta=delta)
    assert new_label == target


def _label_onehot(labels, idx):
    v = np.zeros(len(labels))
    v[idx] = 1.0
    return v
