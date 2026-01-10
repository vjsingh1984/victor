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

"""Emergent collaboration patterns framework.

This module provides protocols and utilities for pattern discovery,
recommendation, and evolution from workflow execution traces.
"""

from victor.framework.patterns.protocols import (
    PatternMinerProtocol,
    PatternValidatorProtocol,
    PatternRecommenderProtocol,
)

__all__ = [
    "PatternMinerProtocol",
    "PatternValidatorProtocol",
    "PatternRecommenderProtocol",
]
