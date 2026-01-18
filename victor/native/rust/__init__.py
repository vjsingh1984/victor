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

"""Rust implementation wrappers.

These wrappers provide protocol-compliant interfaces around the Rust
native functions, adding observability hooks and maintaining compatibility
with the Python fallback implementations.
"""

try:
    from victor.native.rust.arg_normalizer import RustArgumentNormalizer
    from victor.native.rust.ast_indexer import RustAstIndexer
    from victor.native.rust.chunker import RustTextChunker
    from victor.native.rust.similarity import RustSimilarityComputer
    from victor.native.rust.tool_selector import (
        ToolSelectorAccelerator,
        get_tool_selector_accelerator,
        reset_tool_selector_accelerator,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

    # Create stub classes for when Rust is not available
    class RustArgumentNormalizer:  # type: ignore
        pass

    class RustAstIndexer:  # type: ignore
        pass

    class RustTextChunker:  # type: ignore
        pass

    class RustSimilarityComputer:  # type: ignore
        pass

    class ToolSelectorAccelerator:  # type: ignore
        pass

    def get_tool_selector_accelerator(*args, **kwargs):  # type: ignore
        return None

    def reset_tool_selector_accelerator(*args, **kwargs):  # type: ignore
        pass


__all__ = [
    "RustArgumentNormalizer",
    "RustAstIndexer",
    "RustSimilarityComputer",
    "RustTextChunker",
    "ToolSelectorAccelerator",
    "get_tool_selector_accelerator",
    "reset_tool_selector_accelerator",
    "RUST_AVAILABLE",
]
