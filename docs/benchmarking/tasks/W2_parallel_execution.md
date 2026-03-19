# Task W2: Parallel Execution

## Objective
Execute multiple independent analysis tasks in parallel and aggregate results.

## Requirements
Create an agentic workflow that:
1. Takes a list of 5 file paths as input
2. Analyzes each file independently in parallel:
   - Counts lines of code
   - Identifies programming language
   - Detects TODO/FIXME comments
3. Aggregates results into a summary report
4. Completes in less than 30 seconds total

## Input Prompt
```
Analyze these 5 files in parallel and create a summary report showing:
- Total lines of code across all files
- Programming language breakdown
- All TODO/FIXME comments found

Files:
- src/main.py
- src/utils.py
- src/config.py
- tests/test_main.py
- README.md
```

## Success Criteria
1. All files analyzed
2. Results correctly aggregated
3. Parallel execution confirmed (completion time < sequential time)
4. Summary report includes all required information
5. No partial failures (all files succeed or graceful handling)

## Scoring Rubric
- **5 points**: Perfect parallel execution, correct aggregation, < sequential time
- **4 points**: Parallel execution, correct aggregation, timing borderline
- **3 points**: Sequential execution or correct results but no parallelism
- **2 points**: Partial results, some files failed
- **1 point**: Attempted but failed to complete
- **0 points**: No output or completely unrelated

## Test Environment
- **Allowed tools**: File system, parallel execution
- **Timeout**: 30 seconds
- **LLM temperature**: 0.3
- **Max tokens**: 1000

## Validation Steps
1. Check that all files were processed
2. Verify summary report accuracy
3. Measure execution time vs sequential baseline
4. Confirm no data loss or corruption

## Notes
- Sequential baseline: ~5 seconds per file = 25 seconds total
- Parallel target: < 10 seconds total (with 5 files)
- Framework must support concurrent task execution
