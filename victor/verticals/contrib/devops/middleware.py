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

"""Runtime middleware defaults for the DevOps vertical."""

from __future__ import annotations

from typing import Any, List

from victor.framework.middleware import MiddlewareComposer


def get_middleware() -> List[Any]:
    """Build the default DevOps middleware stack."""

    return (
        MiddlewareComposer()
        .git_safety(
            block_dangerous=True,
            warn_on_risky=True,
            protected_branches={"production", "staging"},
        )
        .secret_masking(
            replacement="[REDACTED]",
            mask_in_arguments=True,
        )
        .logging(
            include_arguments=True,
            include_results=True,
        )
        .build()
    )


__all__ = ["get_middleware"]
