#!/usr/bin/env python
"""Check coverage for base.py."""
import subprocess
import sys
import os

# Clean coverage files
for f in ['.coverage', '.coverage.*', 'coverage.json', 'coverage.xml']:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

# Run pytest with coverage
result = subprocess.run([
    sys.executable, '-m', 'pytest',
    'tests/unit/providers/test_base_provider_protocols.py',
    'tests/unit/providers/test_provider_error_handling.py',
    '--cov=victor.providers.base',
    '--cov-report=term',
    '-v'
], capture_output=True, text=True)

# Parse output for base.py coverage
lines = result.stdout.split('\n')
for i, line in enumerate(lines):
    if 'victor/providers/base.py' in line and '%' in line:
        # Print this line and context
        print('\n'.join(lines[max(0, i-2):min(len(lines), i+3)]))
        print('\n' + '='*80)
        # Extract coverage percentage
        parts = line.split()
        for j, part in enumerate(parts):
            if '%' in part:
                print(f"Coverage for victor/providers/base.py: {part}")
                break
        break

# Print test summary
if 'passed' in result.stdout:
    summary_lines = [l for l in lines if 'passed' in l]
    if summary_lines:
        print('\nTest Summary:', summary_lines[-1])

sys.exit(result.returncode)
