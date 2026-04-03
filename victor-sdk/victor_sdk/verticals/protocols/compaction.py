"""Session compaction protocol definitions.

These protocols enable external verticals to customize how conversation
history is compacted when approaching context window limits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class CompactionProvider(Protocol):
    """Protocol for providing compaction configuration.

    Verticals implement this to customize compaction behavior
    for their specific conversation patterns.
    """

    def get_compaction_config(self) -> Dict[str, Any]:
        """Return compaction configuration for this vertical.

        Returns:
            Dict with optional keys:
            - 'preserve_recent_messages': int (default 4)
            - 'max_estimated_tokens': int (default 10000)
            - 'auto_compact': bool (default False)
            - 'summary_template': str - custom summary format
        """
        ...

    def get_compaction_priorities(self) -> List[str]:
        """Return message types to prioritize during compaction.

        Messages matching these types are kept longer before
        being compacted.

        Returns:
            List of message type identifiers to preserve.
            E.g.: ["tool_result", "code_block", "error"]
        """
        ...
