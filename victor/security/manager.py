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

"""Security manager for orchestrating security scanning operations.

.. deprecated:: 0.6.0
    This module is deprecated. Please migrate to ``victor.security_analysis.tools.manager``.
    This module will be removed in v1.0.0.

Migration Guide:
    Old (deprecated):
        from victor.security.manager import SecurityManager

    New (recommended):
        from victor.security_analysis.tools import SecurityManager
        # or
        from victor.security_analysis.tools.manager import SecurityManager

Provides a high-level API for security scanning and vulnerability management.
"""

import warnings

warnings.warn(
    "victor.security.manager is deprecated and will be removed in v1.0.0. "
    "Use victor.security_analysis.tools.manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from victor.security_analysis.tools.manager import (
    SecurityManager,
    get_security_manager,
    reset_security_manager,
)

__all__ = [
    "SecurityManager",
    "get_security_manager",
    "reset_security_manager",
]
