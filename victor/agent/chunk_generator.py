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

"""Deprecated: Re-exports ChunkGenerator from victor.agent.services.chunk_runtime.

This module is deprecated. Import from victor.agent.services.chunk_runtime instead.
"""

import warnings

from victor.agent.services.chunk_runtime import ChunkGenerator

warnings.warn(
    "victor.agent.chunk_generator is deprecated. "
    "Import ChunkGenerator from victor.agent.services.chunk_runtime instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ChunkGenerator"]
