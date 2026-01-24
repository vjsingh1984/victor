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

"""Security utilities for protecting against deserialization attacks.

This module provides safe pickle serialization/deserialization with HMAC signing
to prevent cache poisoning attacks via malicious pickle data.

CWE-502: Deserialization of Untrusted Data
"""

import hashlib
import hmac
import os
import pickle
import secrets
from typing import Any

# Environment variable for cache signing key
CACHE_SIGNING_KEY_ENV = "VICTOR_CACHE_SIGNING_KEY"


_signing_key_cache: bytes | None = None


def _get_signing_key() -> bytes:
    """Get the signing key for cache data.

    Returns:
        Bytes signing key from environment or generated once per process.

    Note:
        In production, set VICTOR_CACHE_SIGNING_KEY to a persistent secret
        so cache data survives process restarts. Otherwise, a random key
        is generated per process (cache will be invalidated on restart).
    """
    global _signing_key_cache

    key = os.getenv(CACHE_SIGNING_KEY_ENV)
    if key:
        return key.encode("utf-8")

    # Generate a random key for this process lifetime
    # This means cache data will be invalidated on process restart
    # which is acceptable for development/testing
    if _signing_key_cache is None:
        _signing_key_cache = secrets.token_bytes(32)  # 256-bit key

    return _signing_key_cache


def safe_pickle_dumps(obj: Any) -> bytes:
    """Safely serialize object with HMAC signature.

    Args:
        obj: Python object to serialize

    Returns:
        Signed pickle data (signature + payload)

    Example:
        >>> data = {"key": "value"}
        >>> signed = safe_pickle_dumps(data)
        >>> len(signed)  # 32 bytes signature + pickle payload
    """
    # Serialize object
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate HMAC signature
    key = _get_signing_key()
    signature = hmac.new(key, payload, hashlib.sha256).digest()

    # Return signature + payload
    return signature + payload


def safe_pickle_loads(data: bytes) -> Any:
    """Safely deserialize object with HMAC verification.

    Args:
        data: Signed pickle data (signature + payload)

    Returns:
        Deserialized Python object

    Raises:
        ValueError: If signature verification fails
        TypeError: If data is not bytes
        pickle.UnpicklingError: If deserialization fails

    Example:
        >>> signed = safe_pickle_dumps({"key": "value"})
        >>> obj = safe_pickle_loads(signed)
        >>> obj
        {'key': 'value'}

    Security:
        - Verifies HMAC signature before deserialization
        - Uses constant-time comparison to prevent timing attacks
        - Fails closed on signature mismatch
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")

    if len(data) < 32:
        raise ValueError("Data too short to contain valid signature")

    # Split signature and payload
    signature = data[:32]  # SHA256 = 32 bytes
    payload = data[32:]

    # Verify signature
    key = _get_signing_key()
    expected = hmac.new(key, payload, hashlib.sha256).digest()

    if not hmac.compare_digest(signature, expected):
        raise ValueError(
            "Cache signature verification failed. "
            "This may indicate cache poisoning or corrupted data. "
            f"Set {CACHE_SIGNING_KEY_ENV} environment variable if using persistent cache."
        )

    # Deserialize after verification
    return pickle.loads(payload)


def is_signed_pickle_data(data: bytes) -> bool:
    """Check if data appears to be signed pickle data.

    Args:
        data: Data to check

    Returns:
        True if data has valid signature format

    Note:
        This only checks format, not validity of signature.
        Use safe_pickle_loads to actually verify and deserialize.
    """
    return isinstance(data, bytes) and len(data) >= 32


__all__ = [
    "safe_pickle_dumps",
    "safe_pickle_loads",
    "is_signed_pickle_data",
    "CACHE_SIGNING_KEY_ENV",
]
