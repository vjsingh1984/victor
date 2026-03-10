# Task C1: Single-File Generation

## Objective
Generate a Python class from natural language requirements.

## Requirements
Generate a Python class named `DataProcessor` with the following:
- Constructor taking `data_path: str` and `batch_size: int = 100`
- Method `process()` that loads CSV, applies transformations, returns DataFrame
- Method `save(output_path: str)` that saves processed data to CSV
- Type hints and docstrings for all methods
- Error handling for missing files (FileNotFoundError with helpful message)

## Input Prompt
```
Create a Python class called DataProcessor that loads a CSV file from data_path,
processes it in batches of size batch_size, handles missing values by dropping rows,
and saves the result using the save() method. Include type hints and docstrings.
```

## Success Criteria
1. Code compiles without syntax errors
2. Class has all required methods with correct signatures
3. Type hints present on all methods
4. Docstrings present for class and methods
5. Error handling for FileNotFoundError
6. Follows PEP 8 style (line length, naming)

## Scoring Rubric
- **5 points**: All criteria met, production-ready code, good error messages
- **4 points**: All criteria met, minor style issues
- **3 points**: Most criteria met, minor missing features
- **2 points**: Basic functionality only, missing type hints or docs
- **1 point**: Attempted but non-functional or major issues
- **0 points**: No output or completely unrelated

## Test Environment
- **Allowed tools**: File system (read/write)
- **Timeout**: 60 seconds
- **LLM temperature**: 0.7
- **Max tokens**: 500

## Validation Steps
1. Generate code
2. Parse with Python AST to check syntax
3. Inspect class to verify methods exist
4. Check type hints with `mypy --strict` (optional)
5. Run pylint/flake8 for style check
