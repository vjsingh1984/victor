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

"""Unified LSP tool for code intelligence operations.

Consolidates all LSP operations into a single tool for better token efficiency.
Supports: status, start, stop, completions, hover, definition, references,
diagnostics, open, close.
"""

from typing import Any, Optional

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

# Completion kind mapping
KIND_NAMES = {
    1: "Text",
    2: "Method",
    3: "Function",
    4: "Constructor",
    5: "Field",
    6: "Variable",
    7: "Class",
    8: "Interface",
    9: "Module",
    10: "Property",
    11: "Unit",
    12: "Value",
    13: "Enum",
    14: "Keyword",
    15: "Snippet",
    16: "Color",
    17: "File",
    18: "Reference",
    19: "Folder",
    20: "EnumMember",
    21: "Constant",
    22: "Struct",
    23: "Event",
    24: "Operator",
    25: "TypeParameter",
}


async def _do_status() -> dict[str, Any]:
    """Get LSP status."""
    from victor.coding.lsp.manager import get_lsp_manager

    manager = get_lsp_manager()
    status = manager.get_status()
    available = manager.get_available_servers()

    return {
        "success": True,
        "servers": {
            lang: {
                "name": s.server_name,
                "running": s.running,
                "initialized": s.initialized,
                "open_documents": s.open_documents,
                "capabilities": s.capabilities,
            }
            for lang, s in status.items()
        },
        "available": [
            {
                "language": s["language"],
                "name": s["name"],
                "installed": s["installed"],
                "running": s["running"],
            }
            for s in available
        ],
    }


async def _do_start(language: str) -> dict[str, Any]:
    """Start a language server."""
    from victor.coding.lsp.manager import get_lsp_manager
    from victor.coding.lsp.config import LANGUAGE_SERVERS

    if not language:
        return {"success": False, "error": "Missing required parameter: language"}

    manager = get_lsp_manager()

    if language not in LANGUAGE_SERVERS:
        available = list(LANGUAGE_SERVERS.keys())
        return {
            "success": False,
            "error": f"Unknown language: {language}",
            "available_languages": available,
        }

    config = LANGUAGE_SERVERS[language]
    success = await manager.start_server(language)

    if success:
        return {
            "success": True,
            "message": f"Started {config.name} for {language}",
            "server_name": config.name,
        }
    else:
        return {
            "success": False,
            "error": f"Failed to start {config.name}. Install with: {config.install_command}",
            "install_command": config.install_command,
        }


async def _do_stop(language: str) -> dict[str, Any]:
    """Stop a language server."""
    from victor.coding.lsp.manager import get_lsp_manager

    if not language:
        return {"success": False, "error": "Missing required parameter: language"}

    manager = get_lsp_manager()
    await manager.stop_server(language)

    return {
        "success": True,
        "message": f"Stopped language server for {language}",
    }


async def _do_completions(
    file_path: str,
    line: Optional[int],
    character: Optional[int],
    max_items: int = 20,
) -> dict[str, Any]:
    """Get code completions."""
    from victor.coding.lsp.manager import get_lsp_manager

    if not file_path:
        return {"success": False, "error": "Missing required parameter: file_path"}
    if line is None:
        return {"success": False, "error": "Missing required parameter: line"}
    if character is None:
        return {"success": False, "error": "Missing required parameter: character"}

    manager = get_lsp_manager()
    await manager.open_document(file_path)
    completions = await manager.get_completions(file_path, line, character)

    return {
        "success": len(completions) > 0,
        "count": len(completions),
        "completions": [
            {
                "label": c.label,
                "kind": KIND_NAMES.get(c.kind, "Unknown"),
                "detail": c.detail,
                "insert_text": c.insert_text or c.label,
            }
            for c in completions[:max_items]
        ],
    }


async def _do_hover(
    file_path: str, line: Optional[int], character: Optional[int]
) -> dict[str, Any]:
    """Get hover information."""
    from victor.coding.lsp.manager import get_lsp_manager

    if not file_path:
        return {"success": False, "error": "Missing required parameter: file_path"}
    if line is None:
        return {"success": False, "error": "Missing required parameter: line"}
    if character is None:
        return {"success": False, "error": "Missing required parameter: character"}

    manager = get_lsp_manager()
    await manager.open_document(file_path)
    hover = await manager.get_hover(file_path, line, character)

    if hover:
        return {"success": True, "contents": hover.contents}
    else:
        return {"success": False, "message": "No hover information available"}


async def _do_definition(
    file_path: str, line: Optional[int], character: Optional[int]
) -> dict[str, Any]:
    """Go to definition."""
    from victor.coding.lsp.manager import get_lsp_manager

    if not file_path:
        return {"success": False, "error": "Missing required parameter: file_path"}
    if line is None:
        return {"success": False, "error": "Missing required parameter: line"}
    if character is None:
        return {"success": False, "error": "Missing required parameter: character"}

    manager = get_lsp_manager()
    await manager.open_document(file_path)
    locations = await manager.get_definition(file_path, line, character)

    return {
        "success": len(locations) > 0,
        "count": len(locations),
        "locations": locations,
    }


async def _do_references(
    file_path: str,
    line: Optional[int],
    character: Optional[int],
    max_results: int = 50,
) -> dict[str, Any]:
    """Find references."""
    from victor.coding.lsp.manager import get_lsp_manager

    if not file_path:
        return {"success": False, "error": "Missing required parameter: file_path"}
    if line is None:
        return {"success": False, "error": "Missing required parameter: line"}
    if character is None:
        return {"success": False, "error": "Missing required parameter: character"}

    manager = get_lsp_manager()
    await manager.open_document(file_path)
    locations = await manager.get_references(file_path, line, character)

    return {
        "success": len(locations) > 0,
        "count": len(locations),
        "locations": locations[:max_results],
    }


async def _do_diagnostics(file_path: str) -> dict[str, Any]:
    """Get diagnostics."""
    import asyncio
    from victor.coding.lsp.manager import get_lsp_manager

    if not file_path:
        return {"success": False, "error": "Missing required parameter: file_path"}

    manager = get_lsp_manager()
    await manager.open_document(file_path)
    await asyncio.sleep(0.5)
    diagnostics = manager.get_diagnostics(file_path)

    errors = sum(1 for d in diagnostics if d["severity"] == "error")
    warnings = sum(1 for d in diagnostics if d["severity"] == "warning")

    return {
        "success": True,
        "errors": errors,
        "warnings": warnings,
        "info": len(diagnostics) - errors - warnings,
        "diagnostics": diagnostics,
    }


async def _do_open(file_path: str) -> dict[str, Any]:
    """Open a file in LSP."""
    from victor.coding.lsp.manager import get_lsp_manager

    if not file_path:
        return {"success": False, "error": "Missing required parameter: file_path"}

    manager = get_lsp_manager()
    success = await manager.open_document(file_path)

    if success:
        return {"success": True, "message": f"Opened {file_path} in language server"}
    else:
        return {
            "success": False,
            "error": f"Could not open {file_path}. No language server available.",
        }


async def _do_close(file_path: str) -> dict[str, Any]:
    """Close a file in LSP."""
    from victor.coding.lsp.manager import get_lsp_manager

    if not file_path:
        return {"success": False, "error": "Missing required parameter: file_path"}

    manager = get_lsp_manager()
    manager.close_document(file_path)

    return {"success": True, "message": f"Closed {file_path}"}


@tool(
    category="lsp",
    priority=Priority.MEDIUM,  # Task-specific code intelligence
    access_mode=AccessMode.MIXED,  # Manages LSP processes, reads files
    danger_level=DangerLevel.SAFE,  # No file modifications
    keywords=["lsp", "language server", "hover", "definition", "references", "diagnostics"],
)
async def lsp(
    action: str,
    language: Optional[str] = None,
    file_path: Optional[str] = None,
    line: Optional[int] = None,
    character: Optional[int] = None,
    max_items: int = 20,
    max_results: int = 50,
) -> dict[str, Any]:
    """Language Server Protocol operations for code intelligence.

    Actions: status, start, stop, completions, hover, definition, references, diagnostics.
    Position-based actions require: file_path, line, character.
    """
    action_lower = action.lower().strip()

    if action_lower == "status":
        return await _do_status()

    elif action_lower == "start":
        return await _do_start(language or "")

    elif action_lower == "stop":
        return await _do_stop(language or "")

    elif action_lower == "completions":
        return await _do_completions(file_path or "", line, character, max_items)

    elif action_lower == "hover":
        return await _do_hover(file_path or "", line, character)

    elif action_lower == "definition":
        return await _do_definition(file_path or "", line, character)

    elif action_lower == "references":
        return await _do_references(file_path or "", line, character, max_results)

    elif action_lower == "diagnostics":
        return await _do_diagnostics(file_path or "")

    elif action_lower == "open":
        return await _do_open(file_path or "")

    elif action_lower == "close":
        return await _do_close(file_path or "")

    else:
        return {
            "success": False,
            "error": f"Unknown action: {action}. Valid actions: status, start, stop, "
            "completions, hover, definition, references, diagnostics, open, close",
        }
