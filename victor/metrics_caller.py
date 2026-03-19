"""Utility script to demonstrate calling metrics on ArgumentNormalizer.

This script instantiates an ArgumentNormalizer, performs a few sample
normalizations, and then prints the statistics collected via
`get_stats()`.  It is intended for quick manual inspection or as a
starting point for unit tests.
"""

from victor.agent.argument_normalizer import ArgumentNormalizer

# Create normalizer instance
normalizer = ArgumentNormalizer(provider_name="mock")

# Sample arguments – some malformed, some correct
samples = [
    {"operations": "[{'type': 'modify', 'path': 'file.txt'}]"},  # Python style quotes
    {"json": "{'key': 'value'}"},                               # Invalid JSON
    {"valid": {"foo": "bar"}},                                 # Already valid
]

for i, args in enumerate(samples, 1):
    normalized, strategy = normalizer.normalize_arguments(args, tool_name="test_tool")
    print(f"Sample {i}: strategy={strategy.name}, result={normalized}")

print("\nNormalization stats:")
print(normalizer.get_stats())

# Reset stats for cleanliness
normalizer.reset_stats()
print("\nAfter reset: ")
print(normalizer.get_stats())
