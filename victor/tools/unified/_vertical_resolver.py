# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Vertical-tool delegation resolver for unified command tools.

Unified command-shell dispatchers (``git``, ``code``) own the canonical tool
names and *delegate* to sister-repo implementations when the optional vertical
package is importable, falling back to a core implementation otherwise. This
keeps the framework-owned dispatchers as the sole canonical LLM-facing surface
while reusing the richer sister-repo logic (e.g. AI commit messages, semantic
search) where available.

Resolution is direct module import today (the realistic path for the
installed-editable sister repos). A future ``victor.tool_callables`` entry-point
group would make this discovery-based; until then the dotted module path is the
contract the dispatchers depend on.

This module must never import a sister-repo package at module top level —
imports happen lazily inside :func:`resolve_vertical_callable` so that an absent
optional package degrades gracefully rather than breaking framework import.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_vertical_callable(
    module_path: str,
    attribute: str,
) -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    """Resolve a callable from an optional vertical package.

    Args:
        module_path: Dotted module path within the vertical package
            (e.g. ``"victor_devops.tools.git_tool"``).
        attribute: Attribute name to fetch from the module
            (e.g. ``"git"``).

    Returns:
        ``(callable, source)`` where ``source`` is the resolved module path on
        success, else ``(None, None)``. Never raises: a missing or
        unloadable optional package returns ``(None, None)`` so callers can
        fall back to a core implementation.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.debug(
            "Vertical module '%s' not available; will use fallback: %s",
            module_path,
            exc,
        )
        return None, None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Vertical module '%s' failed to import: %s", module_path, exc)
        return None, None

    candidate = getattr(module, attribute, None)
    if not callable(candidate):
        logger.debug("Vertical module '%s' has no callable '%s'", module_path, attribute)
        return None, None
    return candidate, module_path


__all__ = ["resolve_vertical_callable"]
