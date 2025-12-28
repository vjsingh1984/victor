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

"""Code review and analysis.

.. deprecated:: 0.3.0
    This module has moved to ``victor_coding.review``.
    Please update your imports.
"""

import warnings

warnings.warn(
    "Importing from 'victor.review' is deprecated. "
    "Please use 'victor_coding.review' instead. "
    "This compatibility shim will be removed in version 0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from victor_coding for backward compatibility
from victor_coding.review import *  # noqa: F401, F403
from victor_coding.review import __all__  # noqa: F401
