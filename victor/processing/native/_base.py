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

"""Shared state for native extension modules.

This module holds the native extension availability flag and reference,
shared by all submodules in victor.processing.native.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import the native extension
_NATIVE_AVAILABLE = False
_native = None

try:
    import victor_native as _native

    _NATIVE_AVAILABLE = True
    logger.info(f"Native extensions loaded (version {_native.__version__})")
except ImportError:
    logger.debug("Native extensions not available, using pure Python fallback")


def is_native_available() -> bool:
    """Check if native Rust extensions are available."""
    return _NATIVE_AVAILABLE


def get_native_version() -> Optional[str]:
    """Get the version of the native extension, if available."""
    if _NATIVE_AVAILABLE:
        return _native.__version__
    return None
