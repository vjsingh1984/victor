"""Usage logger — consolidated into EnhancedUsageLogger.

UsageLogger is now an alias for EnhancedUsageLogger. All callers
get the enhanced version with PII scrubbing, encryption, and
rotation support. The enhanced features are opt-in via constructor
params (disabled by default), so this is a drop-in replacement.
"""

from victor.observability.analytics.enhanced_logger import (
    EnhancedUsageLogger as UsageLogger,
)

__all__ = ["UsageLogger"]
