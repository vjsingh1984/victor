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

"""Code indexing modules for building and maintaining code graphs.

This package provides:
- CCG Builder: Builds Control Flow, Control Dependence, and Data Dependence Graphs
- Symbol extraction: Language-aware symbol extraction using Tree-sitter
- Graph indexing: Incremental indexing with staleness detection
"""

from victor.core.indexing.ccg_builder import CodeContextGraphBuilder

__all__ = [
    "CodeContextGraphBuilder",
]
