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

"""Native accelerator modules for high-performance operations.

This package provides Python wrappers around Rust implementations
for performance-critical operations with automatic fallback to Python.
"""

from victor.native.accelerators.ast_processor import (
    AstProcessorAccelerator,
    get_ast_processor,
)
from victor.native.accelerators.embedding_ops import (
    EmbeddingOpsAccelerator,
    get_embedding_accelerator,
)
from victor.native.accelerators.regex_engine import (
    RegexEngineAccelerator,
    get_regex_engine,
)
from victor.native.accelerators.signature import (
    SignatureAccelerator,
    get_signature_accelerator,
)
from victor.native.accelerators.file_ops import (
    FileOpsAccelerator,
    get_file_ops_accelerator,
)
from victor.native.accelerators.graph_algorithms import (
    GraphAlgorithmsAccelerator,
    get_graph_algorithms_accelerator,
)
from victor.native.accelerators.batch_processor import (
    BatchProcessorAccelerator,
    get_batch_processor_accelerator,
)
from victor.native.accelerators.serialization import (
    SerializationAccelerator,
    get_serialization_accelerator,
)

__all__ = [
    "AstProcessorAccelerator",
    "get_ast_processor",
    "EmbeddingOpsAccelerator",
    "get_embedding_accelerator",
    "RegexEngineAccelerator",
    "get_regex_engine",
    "SignatureAccelerator",
    "get_signature_accelerator",
    "FileOpsAccelerator",
    "get_file_ops_accelerator",
    # Tier 3 Accelerators
    "GraphAlgorithmsAccelerator",
    "get_graph_algorithms_accelerator",
    "BatchProcessorAccelerator",
    "get_batch_processor_accelerator",
    "SerializationAccelerator",
    "get_serialization_accelerator",
]
