"""Lightweight AST helpers — backward compatibility shim.

Canonical implementation promoted to victor-sdk.
This module re-exports for backward compatibility.

Preferred import:
    from victor_sdk.utils.ast_helpers import extract_symbols, build_signature
"""

from victor_sdk.utils.ast_helpers import *  # noqa: F401, F403
from victor_sdk.utils.ast_helpers import __all__  # noqa: F401
