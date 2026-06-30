# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Offline trainer for the edge classifier (FEP-0012 Phase 3).

**DEV-ONLY** — depends on ``scikit-learn``/``scipy`` (the ``[ml]`` extra). The
runtime path (``victor.ml.features`` + ``victor.ml.model``) needs only numpy.

Trains one multiclass logistic head per DecisionType on ``(text, label)``
samples and exports sparse weights via :meth:`EdgeClassifierModel.save`.

v1 supervision uses whatever labels the caller supplies (e.g. LLM-sourced from
``decisions.jsonl``). The production path supervises on
``decision_outcome.attributed_reward`` (reward, not imitation) once that junction
is populated — swapping the label source is the only change here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from victor.ml.features import HASH_SPACE, extract_features
from victor.ml.model import DecisionHead, EdgeClassifierModel

# sklearn / scipy are dev-only; imported here so a missing [ml] extra fails
# loudly only when training, never at runtime inference.
from scipy.sparse import csr_matrix  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def train_head(
    decision_type: str,
    samples: List[Tuple[str, str]],
    *,
    threshold: float = 0.6,
    C: float = 1.0,
) -> DecisionHead:
    """Train a multiclass logistic head on ``(text, label)`` samples.

    Args:
        decision_type: The DecisionType key for this head.
        samples: ``[(text, label), ...]``.
        threshold: Confidence gate τ stored on the head.
        C: LogisticRegression regularization (inverse strength).

    Returns:
        A :class:`DecisionHead` with sparse ``{hash: coef_per_label}`` weights.

    Raises:
        ValueError: If fewer than 2 distinct labels are present.
    """
    label_set = sorted({label for _, label in samples})
    if len(label_set) < 2:
        raise ValueError(f"need >=2 distinct labels to train {decision_type}, got {label_set}")
    label_to_idx = {label: i for i, label in enumerate(label_set)}

    texts = [t for t, _ in samples]
    y = np.array([label_to_idx[label] for _, label in samples])

    # Featurize -> sparse matrix [n_samples, HASH_SPACE].
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i, text in enumerate(texts):
        for h, val in extract_features(text).items():
            rows.append(i)
            cols.append(h)
            data.append(val)
    X = csr_matrix((data, (rows, cols)), shape=(len(samples), HASH_SPACE), dtype=float)

    clf = LogisticRegression(C=C, max_iter=300, solver="lbfgs")
    clf.fit(X, y)

    # coef_: [n_labels, HASH_SPACE]. Keep only non-zero columns (sparse shipping).
    coef = np.asarray(clf.coef_)
    nonzero_cols = np.flatnonzero(coef.any(axis=0))
    weights = {int(col): coef[:, col].astype(float) for col in nonzero_cols}
    bias = np.asarray(clf.intercept_).astype(float)

    return DecisionHead(
        decision_type=decision_type,
        labels=label_set,
        weights=weights,
        bias=bias,
        threshold=threshold,
    )


def train_model(
    per_type_samples: Dict[str, List[Tuple[str, str]]],
    *,
    model_version: str = "0",
    threshold: float = 0.6,
) -> EdgeClassifierModel:
    """Train a head per DecisionType and return a shippable model.

    Args:
        per_type_samples: ``{decision_type: [(text, label), ...]}``.
        model_version: Version stamp stored on the artifact.
        threshold: Confidence gate τ applied to every head.

    Returns:
        An :class:`EdgeClassifierModel` ready to ``save()``.
    """
    heads: Dict[str, DecisionHead] = {}
    for decision_type, samples in per_type_samples.items():
        if not samples:
            continue
        heads[decision_type] = train_head(decision_type, samples, threshold=threshold)
    return EdgeClassifierModel(heads=heads, model_version=model_version)
