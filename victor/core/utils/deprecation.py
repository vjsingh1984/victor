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

"""Shared deprecation warning utilities.

Provides consistent deprecation messaging across the codebase with
standardized format, version tracking, and migration guidance.
"""

from __future__ import annotations

import warnings
from typing import Optional


def emit_deprecation(
    message: str,
    removal_version: str = "v1.0",
    alternative: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Emit a deprecation warning with consistent formatting.

    Args:
        message: Description of what is deprecated.
        removal_version: Version when the deprecated feature will be removed.
        alternative: Suggested replacement, if any.
        stacklevel: How many stack frames to skip (default 2 = caller's caller).
    """
    parts = [message]
    if alternative:
        parts.append(f"Use {alternative} instead.")
    parts.append(f"Will be removed in {removal_version}.")

    warnings.warn(
        " ".join(parts),
        DeprecationWarning,
        stacklevel=stacklevel,
    )
