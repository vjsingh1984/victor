# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Languages manager module shim."""
import warnings
warnings.warn("Importing from 'victor.languages.manager' is deprecated. Please use 'victor_coding.languages.manager' instead.", DeprecationWarning, stacklevel=2)
from victor_coding.languages.manager import *  # noqa: F401, F403
