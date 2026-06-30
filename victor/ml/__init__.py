# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Edge-classifier ML core (FEP-0012).

Versioned feature extraction + a shipped linear-head artifact + pure-numpy
inference. The *training* side (``trainer.py``) uses sklearn and is a DEV-only
dependency; the runtime (features + model) needs only numpy.

See FEP-0012 and ``victor/ml/features.py`` for the versioning contract.
"""

from victor.ml.features import (
    FEATURE_SPEC_VERSION,
    HASH_SPACE,
    extract_features,
)
from victor.ml.model import DecisionHead, EdgeClassifierModel

__all__ = [
    "FEATURE_SPEC_VERSION",
    "HASH_SPACE",
    "extract_features",
    "DecisionHead",
    "EdgeClassifierModel",
]
