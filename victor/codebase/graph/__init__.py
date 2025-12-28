# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Codebase graph storage module.

.. deprecated:: 0.3.0
    This module has moved to ``victor.graph``.
    Please update your imports. This shim will be removed in version 0.5.0.

For new code, prefer importing directly from victor.graph:
    from victor.graph import GraphNode, GraphEdge, create_graph_store
"""

import warnings

warnings.warn(
    "Importing from 'victor.codebase.graph' is deprecated. "
    "Please use 'victor.graph' instead. "
    "This compatibility shim will be removed in version 0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from victor.graph (the canonical location)
from victor.graph import *  # noqa: F401, F403
from victor.graph import __all__  # noqa: F401
