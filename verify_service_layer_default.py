#!/usr/bin/env python
"""Quick verification that USE_SERVICE_LAYER_FOR_AGENT is enabled by default."""

from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag, reset_feature_flag_manager

# Reset to ensure clean state
reset_feature_flag_manager()

# Check default state
manager = get_feature_flag_manager()
is_enabled = manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)
is_opt_in = FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT.is_opt_in_by_default()

print("=" * 70)
print("Service Layer Default Behavior Verification")
print("=" * 70)
print()
print(f"USE_SERVICE_LAYER_FOR_AGENT enabled by default: {is_enabled}")
print(f"USE_SERVICE_LAYER_FOR_AGENT is opt-in: {is_opt_in}")
print()

if is_enabled and not is_opt_in:
    print("✅ SUCCESS: Service layer is enabled by default (opt-out)")
    print()
    print("This means:")
    print("  - Agent.run() will use ChatService by default")
    print("  - Agent.stream() will use ChatService by default")
    print("  - No configuration needed for correct architecture")
    print()
    print("To disable (for testing legacy behavior):")
    print("  export VICTOR_USE_SERVICE_LAYER_FOR_AGENT=false")
else:
    print("❌ FAILURE: Service layer is not the default")
    print()
    print("Expected:")
    print("  - is_enabled should be True")
    print("  - is_opt_in should be False")

print("=" * 70)
