# Session Management

Save, restore, and manage your conversation sessions with SQLite-based persistence.

## Overview

Victor stores conversation sessions in the project database (`.victor/project.db`), providing:
- **Fast queries** with indexed tables
- **Single source of truth** (no JSON file duplication)
- **Interactive session browser** with visual selection
- **Combined operations** (resume + switch model in one command)
- **Project-level storage** (sessions stored per project)

## Database Schema

Sessions are stored in two tables:

```sql
-- Sessions table: metadata and conversation state
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,              -- Session ID (timestamp-based)
    name TEXT,                         -- Session title
    provider TEXT,                     -- Provider name
    model TEXT,                        -- Model name
    profile TEXT,                      -- Profile used
    data TEXT,                         -- JSON: full session data
    created_at TEXT,                   -- ISO timestamp
    updated_at TEXT                    -- ISO timestamp
);

-- Messages table: individual messages
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,                   -- Foreign key to sessions.id
    role TEXT,                         -- 'user', 'assistant', 'system'
    content TEXT,                      -- Message content
    tool_calls TEXT,                   -- JSON: tool call data (optional)
    created_at TEXT,                   -- ISO timestamp
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

## Commands

### `/save` - Save Session

Save your current conversation to SQLite.

```bash
/save                                    # Auto-generate title
/save "Refactoring Authentication"       # Custom title
```

**Example Output**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Session Saved                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Session saved to SQLite database!     │
│                                        │
│ Session ID: 20250107_153045           │
│ Database: /project/.victor/project.db │
│                                        │
│ Use '/resume 20250107_153045' to      │
│ restore this session                   │
└────────────────────────────────────────┘
```

### `/sessions` - List Sessions

List saved sessions with metadata.

```bash
/sessions              # List last 10 sessions
/sessions 20           # List last 20 sessions
```

**Example Output**:
```
╔═══════════╤═══════════════════════════════╤═══════════════════╤══════════╤════════════╤═════════╗
║ ID        │ Title                         │ Model             │ Provider │ Messages   │ Created ║
╠═══════════╪═══════════════════════════════╪═══════════════════╪══════════╪════════════╪═════════╣
║ 20250107… │ Refactoring Authentication…   │ claude-sonnet-4-2 │ anthropic │ 42         │ 14:30   ║
║ 20250107… │ API Testing                   │ gpt-4             │ openai   │ 18         │ 12:15   ║
║ 20250106… │ Code Review                   │ qwen2.5-coder:7b  │ ollama   │ 67         │ 09:45   ║
╚───────────┴───────────────────────────────┴───────────────────┴──────────┴────────────┴─────────╝

Use '/resume <session_id>' to restore a session
Or '/switch <model> --resume <session_id>' to resume and switch
```

### `/resume` - Restore Session

Restore a previously saved session.

```bash
/resume                                 # Interactive selection
/resume 20250107_153045                 # Direct session ID
```

**Interactive Mode**:
Shows a numbered list of recent sessions. Enter a number to select.

```
╔════╤═══════════╤═══════════════════════════════╤═══════════════════╤════════════╤══════════╗
║ #  │ ID        │ Title                         │ Model             │ Messages   │ Date     ║
╠════╪═══════════╪═══════════════════════════════╪═══════════════════╪════════════╪══════════╣
║ 1  │ 20250107… │ Refactoring Authentication    │ claude-sonnet-4-2 │ 42         │ 14:30    ║
║ 2  │ 20250107… │ API Testing                   │ gpt-4             │ 18         │ 12:15    ║
║ 3  │ 20250106… │ Code Review                   │ qwen2.5-coder:7b  │ 67         │ 09:45    ║
╚════╧═══════════╧═══════════════════════════════╧═══════════════════╧════════════╧══════════╝

Enter session number to resume (1-3)
Or use: /resume <session_id>
```

**Restored Session**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       Session Resumed                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Session restored from SQLite!         │
│                                        │
│ ID: 20250107_153045                   │
│ Title: Refactoring Authentication     │
│ Model: claude-sonnet-4-20250514       │
│ Provider: anthropic                   │
│ Messages: 42                          │
│ Created: 2025-01-07T14:30:45          │
└────────────────────────────────────────┘
```

### `/switch` - Switch Model/Provider

Switch models or providers, optionally resuming a session first.

```bash
/switch                                 # Show current model
/switch claude-opus-4-20250514          # Switch model only
/switch anthropic:claude-opus-4         # Switch provider and model
/switch --resume                        # Resume last session, then show current
/switch --resume 20250107_153045        # Resume specific session, show current
/switch claude-opus-4 --resume          # Resume last, then switch to this model
/switch ollama:qwen2.5-coder:7b --resume 20250107_153045  # Combined
```

**Examples**:

1. **Switch model only**:
   ```bash
   /switch claude-opus-4-20250514
   ```
   ```
   ✓ Switched to claude-opus-4-20250514
     Native tools: True, Thinking: True
   ```

2. **Switch provider and model**:
   ```bash
   /switch ollama:qwen2.5-coder:7b
   ```
   ```
   ✓ Switched to ollama:qwen2.5-coder:7b
     Native tools: False, Thinking: False
   ```

3. **Resume then switch**:
   ```bash
   /switch claude-opus-4 --resume 20250107_153045
   ```
   ```
   ✓ Resumed: Refactoring Authentication (42 messages)

   ✓ Switched to claude-opus-4-20250514
     Native tools: True, Thinking: True
   ```

### `/compact` - Reduce Context Size

Compress conversation history to reduce token usage.

```bash
/compact                         # Simple truncation (keeps last 6 messages)
/compact --smart                # AI-powered summarization
/compact --smart --keep 10      # Keep last 10 messages + summary
```

**Smart Compaction Example**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    Smart Compaction Complete         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Compacted 42 messages to 8.           │
│                                        │
│ Summary:                               │
│ The user requested help refactoring   │
│ the authentication module. We:        │
│ 1. Analyzed the current auth flow     │
│ 2. Identified security issues         │
│ 3. Implemented JWT token validation   │
│ 4. Added unit tests                   │
│ 5. Updated documentation              │
└────────────────────────────────────────┘
```

## Advanced Usage

### Conversation State Machine

When you save a session, Victor preserves the conversation state machine, including:
- Current stage (plan, explore, build, etc.)
- Mode transitions
- Task progress
- Tool selection statistics

This means when you resume, the conversation state is fully restored.

### Cross-Model Workflows

Leverage different model strengths in a single session:

```bash
# 1. Start with Claude for complex reasoning
victor chat --provider anthropic "Design a REST API"
# ... work on design ...

# 2. Save the session
/save API Design

# 3. Resume with GPT-4 for implementation
/switch gpt-4 --resume

# 4. Finish with local model for privacy
/switch ollama:qwen2.5-coder:7b
```

### Session Organization

Use descriptive titles to organize sessions:

```bash
/save "Authentication Refactoring - Phase 1"
/save "Bug Fix: Race condition in payment processing"
/save "Feature: User profile management"
```

Then use `/sessions` to browse and find what you need.

## Database Management

### Location

Sessions are stored in the project database:
```
<project-root>/.victor/project.db
```

### Direct SQL Access

You can query sessions directly using SQLite:

```bash
# List all sessions
sqlite3 .victor/project.db "SELECT id, name, model, created_at FROM sessions ORDER BY created_at DESC;"

# Count messages per session
sqlite3 .victor/project.db "
  SELECT s.name, COUNT(m.id) as message_count
  FROM sessions s
  LEFT JOIN messages m ON s.id = m.session_id
  GROUP BY s.id
  ORDER BY message_count DESC;
"

# Search sessions by title
sqlite3 .victor/project.db "
  SELECT id, name, model, created_at
  FROM sessions
  WHERE name LIKE '%authentication%'
  ORDER BY created_at DESC;
"
```

### Backup and Restore

```bash
# Backup sessions
cp .victor/project.db .victor/project.db.backup

# Restore sessions
cp .victor/project.db.backup .victor/project.db
```

## Migration from JSON (Optional)

If you have existing JSON sessions in `~/.victor/sessions/`, you can still access them. The system is backward compatible.

To migrate JSON sessions to SQLite (manual process):

1. List JSON sessions:
   ```bash
   ls -la ~/.victor/sessions/
   ```

2. For each JSON session, manually:
   - Load the JSON file
   - Extract conversation, model, provider data
   - Use `/save` to create a new SQLite session
   - Copy conversation content from JSON

A migration utility may be added in the future (see Phase 6 in CODE_PRUNING_SUMMARY.md).

## Troubleshooting

### Session not found

**Error**: `Session not found: 20250107_153045`

**Solutions**:
1. Use `/sessions` to list available sessions
2. Check the session ID is correct
3. Verify you're in the correct project directory

### Database locked

**Error**: `database is locked`

**Solutions**:
1. Close other Victor instances
2. Check for background processes: `ps aux | grep victor`
3. Ensure no other tools are accessing `.victor/project.db`

### Empty sessions list

**Error**: `/sessions` shows "No saved sessions found"

**Solutions**:
1. Save a session first: `/save My Session`
2. Check database exists: `ls .victor/project.db`
3. Verify database tables: `sqlite3 .victor/project.db ".tables"`

### Conversation state not restored

**Issue**: Resumed session but conversation state lost

**Cause**: Session saved before conversation state machine was implemented

**Solution**: Save new sessions will include conversation state

## Best Practices

1. **Save frequently**: After completing significant work
   ```bash
   /save "Completed authentication refactoring"
   ```

2. **Use descriptive titles**: Include context and status
   ```bash
   /save "API Design - Draft 1 - In Progress"
   ```

3. **Leverage model switching**: Use different models for different phases
   ```bash
   # Design with Claude
   /save "Design phase"
   # Implement with GPT-4
   /switch gpt-4 --resume
   /save "Implementation phase"
   # Test with local model
   /switch ollama:qwen2.5-coder:7b --resume
   ```

4. **Compact long sessions**: Reduce context before resuming
   ```bash
   /resume 20250107_153045
   /compact --smart
   /save "Refactored - Compact version"
   ```

5. **Organize by task**: Use consistent naming
   ```bash
   /save "Feature: User authentication - Design"
   /save "Feature: User authentication - Implementation"
   /save "Feature: User authentication - Testing"
   ```

## API Reference

### Python API

You can also use session persistence programmatically:

```python
from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

# Get persistence instance
persistence = get_sqlite_session_persistence()

# Save session
session_id = persistence.save_session(
    conversation=agent.conversation,
    model=agent.model,
    provider=agent.provider_name,
    title="My Session",
    conversation_state=agent.conversation_state,
)

# Load session
session_data = persistence.load_session(session_id)

# List sessions
sessions = persistence.list_sessions(limit=10)

# Search sessions
sessions = persistence.search_sessions("authentication", limit=5)

# Delete session
persistence.delete_session(session_id)
```

See also:
- [CLI Reference](cli-reference.md) - All slash commands
- [Provider Switching](../reference/providers/index.md) - Switching providers
- [Architecture Deep Dive](../development/architecture/deep-dive.md) - Database design
