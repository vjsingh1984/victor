"""Multi-file editing with diff preview and rollback.

Provides safe, atomic operations for modifying multiple files with:
- Diff preview before applying
- Rollback capability
- Transaction-like editing
- Backup management
"""

from victor.editing.editor import FileEditor, EditOperation, EditTransaction

__all__ = ["FileEditor", "EditOperation", "EditTransaction"]
