# Multi-File Editing with Victor

Victor provides **transaction-based multi-file editing** with diff preview, automatic backups, and complete rollback capability. This ensures safe code modifications across multiple files.

## Features

### ✅ Transaction-Based Safety
- All operations are atomic (all-or-nothing)
- Automatic rollback on errors
- No partial modifications left behind

### ✅ Rich Diff Preview
- Unified diff with syntax highlighting
- Shows exactly what will change
- Preview before committing

### ✅ Automatic Backups
- Files backed up before modification
- Stored in `~/.victor/backups`
- Used for rollback recovery

### ✅ Multiple Operation Types
- **CREATE**: Add new files
- **MODIFY**: Change existing files
- **DELETE**: Remove files
- **RENAME**: Move/rename files

### ✅ Dry-Run Mode
- Test changes without applying
- Perfect for validation
- Preview with no side effects

### ✅ Complete Rollback
- Undo all changes in transaction
- Restores from backups
- Safe error recovery

---

## Usage

### Command Line Demo

```bash
# Run the comprehensive demo
python examples/multi_file_editing_demo.py
```

This demo shows all features:
1. Creating new files
2. Modifying existing files with diff preview
3. Dry-run mode
4. Rollback capability
5. Multiple operations in one transaction
6. Rename operations
7. Delete operations

### Python API

```python
from victor.editing import FileEditor

# Create editor
editor = FileEditor(backup_dir="~/.victor/backups")

# Start transaction
editor.start_transaction("Update authentication module")

# Queue operations
editor.add_create("auth.py", "def authenticate(): pass")
editor.add_modify("config.py", new_content)
editor.add_delete("old_file.py")
editor.add_rename("utils.py", "helpers.py")

# Preview changes
editor.preview_diff()

# Commit (or use dry_run=True to test)
success = editor.commit()

# Or rollback if needed
# editor.rollback()
```

### Agent Tool Integration

Victor's agent can use the file editor through the `file_editor` tool:

```python
# Available in agent conversations
# The LLM can call these operations:

# Start transaction
file_editor(operation="start_transaction", description="Update auth")

# Add operations
file_editor(operation="add_create", path="file.py", content="...")
file_editor(operation="add_modify", path="file.py", new_content="...")
file_editor(operation="add_delete", path="file.py")
file_editor(operation="add_rename", path="old.py", new_path="new.py")

# Preview
file_editor(operation="preview")

# Commit or rollback
file_editor(operation="commit")
file_editor(operation="rollback")

# Check status
file_editor(operation="status")
```

---

## Architecture

### Core Classes

#### `FileEditor`
Main class for managing multi-file edits.

**Key Methods:**
- `start_transaction(description)` - Begin new transaction
- `add_create(path, content)` - Queue file creation
- `add_modify(path, new_content)` - Queue file modification
- `add_delete(path)` - Queue file deletion
- `add_rename(old_path, new_path)` - Queue file rename
- `preview_diff(context_lines=3)` - Show diff preview
- `commit(dry_run=False)` - Apply all changes
- `rollback()` - Undo all changes
- `abort()` - Cancel without applying
- `get_transaction_summary()` - Get transaction info

#### `EditTransaction`
Represents a transaction of multiple file edits.

**Fields:**
- `id` - Unique transaction ID (timestamp-based)
- `operations` - List of queued operations
- `committed` - Whether transaction was committed
- `rolled_back` - Whether transaction was rolled back
- `timestamp` - Transaction creation time
- `description` - User-provided description

#### `EditOperation`
Represents a single file operation.

**Fields:**
- `type` - Operation type (CREATE/MODIFY/DELETE/RENAME)
- `path` - File path
- `old_content` - Original content (for modify/delete)
- `new_content` - New content (for create/modify)
- `new_path` - New path (for rename)
- `backup_path` - Path to backup file
- `applied` - Whether operation was applied

#### `OperationType`
Enum of operation types.

**Values:**
- `CREATE` - Create new file
- `MODIFY` - Modify existing file
- `DELETE` - Delete file
- `RENAME` - Rename/move file

---

## Example Workflows

### 1. Simple File Update

```python
from victor.editing import FileEditor

editor = FileEditor()

# Start transaction
editor.start_transaction("Fix typo in README")

# Modify file
with open("README.md") as f:
    content = f.read()

new_content = content.replace("Vicotr", "Victor")
editor.add_modify("README.md", new_content)

# Preview and commit
editor.preview_diff()
editor.commit()
```

### 2. Refactoring with Multiple Files

```python
editor = FileEditor()
editor.start_transaction("Extract utilities to separate module")

# Create new utils file
editor.add_create("utils.py", """
def helper_function():
    pass
""")

# Update main file to import from utils
editor.add_modify("main.py", """
from utils import helper_function

def main():
    helper_function()
""")

# Remove old helper file
editor.add_delete("old_helpers.py")

# Preview all changes
editor.preview_diff()

# Commit atomically
if editor.commit():
    print("Refactoring complete!")
```

### 3. Safe Batch Updates with Dry-Run

```python
editor = FileEditor()
editor.start_transaction("Update copyright headers")

# Queue updates for multiple files
for file in ["auth.py", "config.py", "utils.py"]:
    with open(file) as f:
        content = f.read()

    new_content = f"# Copyright 2025 MyCompany\n{content}"
    editor.add_modify(file, new_content)

# Test with dry-run first
editor.preview_diff()
editor.commit(dry_run=True)  # No changes applied

# If everything looks good, commit for real
editor = FileEditor()
editor.start_transaction("Update copyright headers")
# ... add operations again ...
editor.commit()  # Actually apply changes
```

### 4. Error Recovery with Rollback

```python
editor = FileEditor()
editor.start_transaction("Risky refactoring")

try:
    # Queue potentially dangerous operations
    editor.add_modify("critical.py", new_content)
    editor.add_delete("important.py")

    # Try to commit
    editor.commit()

    # Run tests
    import subprocess
    result = subprocess.run(["pytest"], capture_output=True)

    if result.returncode != 0:
        # Tests failed! Need to rollback
        editor.rollback()
        print("Tests failed, changes rolled back")

except Exception as e:
    # Automatic rollback on error
    editor.rollback()
    print(f"Error occurred, rolled back: {e}")
```

---

## Advanced Features

### Custom Backup Directory

```python
editor = FileEditor(backup_dir="/custom/backup/path")
```

### Disable Auto-Backup

```python
editor = FileEditor(auto_backup=False)
```

### Custom Console Output

```python
from rich.console import Console

console = Console(highlight=False)
editor = FileEditor(console=console)
```

### Transaction History

```python
# FileEditor maintains transaction history
for transaction in editor.transaction_history:
    print(f"Transaction {transaction.id}:")
    print(f"  Description: {transaction.description}")
    print(f"  Operations: {len(transaction.operations)}")
    print(f"  Committed: {transaction.committed}")
    print(f"  Rolled back: {transaction.rolled_back}")
```

---

## Safety Guarantees

### Atomic Operations
- All operations in a transaction succeed together or fail together
- No partial modifications left behind
- Database-like ACID properties

### Automatic Backups
- Files backed up before any modification or deletion
- Backups used for rollback recovery
- Stored with timestamp in filename

### Error Handling
- Exceptions trigger automatic rollback
- All changes undone in reverse order
- Original state restored from backups

### Validation
- File existence checked before operations
- Prevents creating files that already exist
- Prevents modifying files that don't exist
- Prevents renaming to paths that already exist

---

## Integration with Victor Agent

The multi-file editor is registered as a tool in Victor's agent orchestrator:

**File:** `victor/agent/orchestrator.py`

```python
from victor.tools.file_editor_tool import FileEditorTool

def _register_default_tools(self):
    # ... other tools ...
    self.tools.register(FileEditorTool())
```

This allows the LLM to safely modify multiple files during conversations:

**Example conversation:**
```
User: "Extract the authentication logic into a separate module"

Victor: I'll help refactor the authentication logic. Let me:
1. Create a new auth.py module
2. Move the authentication functions there
3. Update the imports in main.py

[Victor uses file_editor tool to perform transaction-based refactoring]

Victor: Done! Created auth.py and updated main.py.
        All changes committed atomically.
```

---

## Testing

Run the test suite:

```bash
# Run file editor tool tests
python tests/test_file_editor_tool.py

# Run comprehensive demo
python examples/multi_file_editing_demo.py
```

Both tests verify:
- Transaction management
- All operation types (CREATE/MODIFY/DELETE/RENAME)
- Diff preview generation
- Commit and rollback
- Dry-run mode
- Error handling

---

## Comparison with Other Approaches

### vs. Direct File I/O
❌ **Direct I/O**: No preview, no rollback, error-prone
✅ **FileEditor**: Preview changes, safe rollback, atomic

### vs. Manual Backups
❌ **Manual**: Tedious, easy to forget, manual restore
✅ **FileEditor**: Automatic backups, automatic restore

### vs. Git Commits
❌ **Git**: Requires git repo, affects commit history
✅ **FileEditor**: Works anywhere, no git pollution

### vs. IDE Refactoring
❌ **IDE**: GUI-only, not scriptable, no API
✅ **FileEditor**: Programmatic, CLI, LLM-friendly

---

## Best Practices

### 1. Use Descriptive Transaction Names
```python
# Good
editor.start_transaction("Extract UserAuth class to separate module")

# Bad
editor.start_transaction("refactor")
```

### 2. Always Preview Before Committing
```python
editor.preview_diff()
if input("Commit? (y/n): ") == "y":
    editor.commit()
else:
    editor.abort()
```

### 3. Use Dry-Run for Testing
```python
# Test first
editor.commit(dry_run=True)

# Then commit for real
editor = FileEditor()
editor.start_transaction(description)
# ... add operations ...
editor.commit()
```

### 4. Handle Errors Gracefully
```python
try:
    editor.commit()
except Exception as e:
    print(f"Error: {e}")
    editor.rollback()
```

### 5. Keep Transactions Focused
```python
# Good - single purpose
editor.start_transaction("Add logging to auth module")

# Bad - multiple unrelated changes
editor.start_transaction("Update everything")
```

---

## Future Enhancements

Potential future improvements:

1. **Conflict Detection**: Detect concurrent modifications
2. **Merge Support**: Smart merging of changes
3. **History Replay**: Replay transaction history
4. **Undo Stack**: Multiple levels of undo
5. **Change Tracking**: Track who made what changes
6. **Semantic Diff**: Context-aware diff generation
7. **Partial Rollback**: Rollback specific operations
8. **Distributed Transactions**: Coordinate across machines

---

## Summary

Victor's multi-file editing system provides:

✅ **Safety**: Transaction-based with automatic rollback
✅ **Visibility**: Rich diff preview before applying
✅ **Flexibility**: Create, modify, delete, rename operations
✅ **Reliability**: Automatic backups and error recovery
✅ **Usability**: Python API and agent tool integration
✅ **Quality**: Comprehensive tests and examples

Perfect for safe code modifications, refactoring, and automated file management!

🏆 **"Code to Victory with Any AI"** ⚡
