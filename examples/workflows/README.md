# Ready Workflows

These scripts run common tasks using `victor chat`. Each accepts a target file
path as the first argument.

## Usage

```bash
bash examples/workflows/refactor.sh src/app.py
bash examples/workflows/review.sh src/app.py
bash examples/workflows/tests.sh src/app.py
```

## Optional: choose a profile

```bash
VICTOR_PROFILE=local bash examples/workflows/refactor.sh src/app.py
```
