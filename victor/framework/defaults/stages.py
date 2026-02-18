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

"""Default stage definitions for verticals.

Re-exports ``get_default_stages`` from the canonical location in
``victor.framework.stage_manager`` so external verticals can import via::

    from victor.framework.defaults import get_default_stages
"""

from __future__ import annotations

from victor.framework.stage_manager import get_default_stages

__all__ = [
    "get_default_stages",
]
