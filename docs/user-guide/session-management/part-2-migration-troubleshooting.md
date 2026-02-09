# Session Management Guide - Part 2

**Part 2 of 2:** Database Management, Migration from JSON, Troubleshooting, Best Practices, and API Reference

---

## Navigation

- [Part 1: Core Features](part-1-core-features.md)
- **[Part 2: Migration & Troubleshooting](#)** (Current)
- [**Complete Guide](../session-management.md)**

---
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

A migration utility may be added in a future release.

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
- [Architecture Overview](../architecture/overview.md) - Database design

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
