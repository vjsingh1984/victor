# Coordinator Usage Guide - Part 2

**Part 2 of 2:** Provider, Session, and Additional Coordinators

---

## Navigation

- [Part 1: Core Coordinators](part-1-config-prompt-context-chat.md)
- **[Part 2: Provider, Session, Additional](#)** (Current)
- [**Complete Guide**](../coordinators.md)

---

## ProviderCoordinator

**Purpose**: LLM provider management and switching

**File Location**: `/victor/agent/coordinators/provider_coordinator.py`

**Dependencies**: ConfigCoordinator, ProviderRegistry

**Key Methods**:

```python
class ProviderCoordinator:
    """Provider management coordinator."""

    def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None
    ) -> None:
        """
        Switch to a different provider.

        Args:
            provider_name: Name of provider to switch to
            model: Optional model to use

        Example:
            >>> provider_coord.switch_provider("anthropic")
            >>> provider_coord.switch_provider("openai", "gpt-4")
        """
```

### SessionCoordinator

**Purpose**: Session management and state

**File Location**: `/victor/agent/coordinators/session_coordinator.py`

**Dependencies**: ContextCoordinator

**Key Methods**:

```python
class SessionCoordinator:
    """Session management coordinator."""

    def create_session(
        self,
        session_id: Optional[str] = None
    ) -> Session:
        """
        Create a new session.

        Args:
            session_id: Optional session ID

        Returns:
            Session: Created session

        Example:
            >>> session = session_coord.create_session()
            >>> print(session.id)
            'session_123'
        """
```

### Additional Coordinators

[Content continues with additional coordinators...]

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 1 min
**Last Updated:** January 31, 2026
