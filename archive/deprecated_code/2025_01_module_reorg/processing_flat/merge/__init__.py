# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""This module has moved to victor.processing.merge.

This module is maintained for backward compatibility only.
Please update your imports to use the new location:

    # OLD:
    from victor.merge import MergeManager

    # NEW (preferred):
    from victor.processing.merge import MergeManager
"""

# Re-export everything from the new location for backward compatibility
from victor.processing.merge import (
    # Manager
    MergeManager,
    # Protocols
    ConflictResolverProtocol,
    MergeAnalyzerProtocol,
    # Data classes
    ConflictType,
    ResolutionStrategy,
    ConflictComplexity,
    ConflictHunk,
    FileConflict,
    Resolution,
    FileResolution,
    MergeContext,
    ConflictAnalysis,
    # Resolvers
    TrivialResolver,
    UnionResolver,
    ImportResolver,
    DefaultMergeAnalyzer,
    # Registry functions
    get_resolvers,
    get_analyzer,
)

__all__ = [
    # Manager
    "MergeManager",
    # Protocols
    "ConflictResolverProtocol",
    "MergeAnalyzerProtocol",
    # Data classes
    "ConflictType",
    "ResolutionStrategy",
    "ConflictComplexity",
    "ConflictHunk",
    "FileConflict",
    "Resolution",
    "FileResolution",
    "MergeContext",
    "ConflictAnalysis",
    # Resolvers
    "TrivialResolver",
    "UnionResolver",
    "ImportResolver",
    "DefaultMergeAnalyzer",
    # Registry functions
    "get_resolvers",
    "get_analyzer",
]
