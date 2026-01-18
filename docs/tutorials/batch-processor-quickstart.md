# Batch Processor Quick Start Guide

## Installation

The batch processor is included with Victor AI v0.5.1+. No additional installation required if you have Victor installed.

```bash
# Verify installation
python -c "from victor.native.rust.batch_processor import RUST_AVAILABLE; print(f'Rust available: {RUST_AVAILABLE}')"
```

## Basic Usage

### 1. Simple Batch Processing

```python
from victor.native.rust.batch_processor import BatchProcessor, BatchTask

# Create processor
processor = BatchProcessor(
    max_concurrent=10,      # Max 10 parallel tasks
    timeout_ms=30000,       # 30 second timeout
    retry_policy="exponential",  # Retry with exponential backoff
    aggregation_strategy="unordered"  # Return results as they complete
)

# Define task executor
def my_executor(task_dict):
    # Your task logic here
    return f"Processed {task_dict['task_id']}"

# Create tasks
tasks = [
    BatchTask(task_id=f"task{i}", task_data=my_executor, priority=1.0)
    for i in range(10)
]

# Process batch
summary = processor.process_batch(tasks, my_executor)

# Check results
print(f"Completed: {summary.successful_count}/{len(summary.results)}")
print(f"Throughput: {summary.throughput_per_second:.2f} tasks/sec")
print(f"Duration: {summary.total_duration_ms:.2f}ms")
```

### 2. Tasks with Dependencies

```python
# Create tasks with dependencies
tasks = [
    BatchTask(
        task_id="download",
        task_data=download_file,
        priority=1.0,
        dependencies=[]  # No dependencies
    ),
    BatchTask(
        task_id="process",
        task_data=process_file,
        priority=0.8,
        dependencies=["download"]  # Requires download to complete first
    ),
    BatchTask(
        task_id="upload",
        task_data=upload_result,
        priority=0.6,
        dependencies=["process"]  # Requires process to complete first
    ),
]

# Process with automatic dependency resolution
summary = processor.process_batch(tasks, my_executor)
```

### 3. Streaming Results

```python
def result_callback(result):
    """Called as each task completes."""
    if result.success:
        print(f"✓ {result.task_id}: {result.result}")
    else:
        print(f"✗ {result.task_id}: {result.error}")

# Process with streaming
summary = processor.process_batch_streaming(
    tasks,
    executor=my_executor,
    callback=result_callback
)
```

## Common Patterns

### Parallel File Processing

```python
import os

def file_processor(task_dict):
    filepath = task_dict['task_id']
    with open(filepath, 'r') as f:
        content = f.read()
    # Process file content
    return len(content)

# Get all Python files
files = [f for f in os.listdir('.') if f.endswith('.py')]

# Create tasks
tasks = [
    BatchTask(task_id=f, task_data=file_processor, priority=1.0)
    for f in files
]

# Process in parallel
summary = processor.process_batch(tasks, file_processor)
print(f"Processed {summary.successful_count} files")
```

### Parallel API Calls

```python
import requests

def api_caller(task_dict):
    endpoint = task_dict['task_id']
    response = requests.get(endpoint, timeout=5)
    return response.json()

endpoints = [
    "https://api.example.com/data1",
    "https://api.example.com/data2",
    "https://api.example.com/data3",
]

tasks = [
    BatchTask(task_id=url, task_data=api_caller, priority=1.0, timeout_ms=5000)
    for url in endpoints
]

summary = processor.process_batch(tasks, api_caller)
```

### Multi-Agent Coordination

```python
# Define agent tasks
researcher = BatchTask(
    task_id="research",
    task_data=research_agent.run,
    priority=1.0,
    timeout_ms=30000
)

writer = BatchTask(
    task_id="write",
    task_data=writer_agent.run,
    priority=0.9,
    dependencies=["research"],  # Waits for research
    timeout_ms=20000
)

reviewer = BatchTask(
    task_id="review",
    task_data=reviewer_agent.run,
    priority=0.8,
    dependencies=["write"],  # Waits for writing
    timeout_ms=15000
)

# Execute in dependency order
summary = processor.process_batch(
    [researcher, writer, reviewer],
    agent_executor
)
```

## Configuration Options

### Retry Policies

```python
# No retry - fail immediately
processor = BatchProcessor(retry_policy="none")

# Fixed delay - 1000ms between retries
processor = BatchProcessor(retry_policy="fixed")

# Linear backoff - 1000ms, 1500ms, 2000ms, ...
processor = BatchProcessor(retry_policy="linear")

# Exponential backoff - 1000ms, 2000ms, 4000ms, ... (default)
processor = BatchProcessor(retry_policy="exponential")
```

### Aggregation Strategies

```python
# Maintain original order
processor = BatchProcessor(aggregation_strategy="ordered")

# Return as they complete (fastest, default)
processor = BatchProcessor(aggregation_strategy="unordered")

# Stream results via callback
processor = BatchProcessor(aggregation_strategy="streaming")

# Higher priority first
processor = BatchProcessor(aggregation_strategy="priority")
```

### Load Balancing

```python
# Set load balancing strategy
processor.set_load_balancer("round_robin")     # Distribute evenly
processor.set_load_balancer("least_loaded")    # Worker with fewest tasks
processor.set_load_balancer("weighted")        # Use task priority
processor.set_load_balancer("random")          # Random assignment
```

## Error Handling

```python
try:
    summary = processor.process_batch(tasks, my_executor)

    # Check for failures
    if summary.failed_count > 0:
        print(f"Failed tasks: {summary.failed_count}")

        # Inspect errors
        for result in summary.results:
            if not result.success:
                print(f"  {result.task_id}: {result.error}")

except ValueError as e:
    if "Circular dependency" in str(e):
        print("Error: Tasks have circular dependencies!")
    raise
```

## Performance Tips

### 1. Choose Right Concurrency

```python
import os

# CPU-bound tasks: use number of cores
processor = BatchProcessor(max_concurrent=os.cpu_count())

# I/O-bound tasks: use higher concurrency
processor = BatchProcessor(max_concurrent=50)
```

### 2. Set Appropriate Timeouts

```python
# Quick tasks
quick_task = BatchTask(
    task_id="quick",
    task_data=quick_fn,
    timeout_ms=1000  # 1 second
)

# Slow tasks
slow_task = BatchTask(
    task_id="slow",
    task_data=slow_fn,
    timeout_ms=60000  # 60 seconds
)
```

### 3. Use Priority for Important Tasks

```python
critical_task = BatchTask(
    task_id="critical",
    task_data=critical_fn,
    priority=1.0  # Highest priority
)

background_task = BatchTask(
    task_id="background",
    task_data=background_fn,
    priority=0.1  # Lowest priority
)
```

### 4. Batch Large Workloads

```python
from victor.native.rust.batch_processor import create_task_batches_py

# Split 1000 tasks into batches of 100
batches = create_task_batches_py(all_tasks, batch_size=100)

all_summaries = []
for batch in batches:
    summary = processor.process_batch(batch, my_executor)
    all_summaries.append(summary)

# Merge results
from victor.native.rust.batch_processor import merge_batch_summaries_py
final_summary = merge_batch_summaries_py(all_summaries)
```

## Utility Functions

### Calculate Optimal Batch Size

```python
from victor.native.rust.batch_processor import calculate_optimal_batch_size_py

# 1000 tasks, 10 workers -> 100 tasks per batch
batch_size = calculate_optimal_batch_size_py(1000, 10)
print(f"Optimal batch size: {batch_size}")
```

### Estimate Duration

```python
from victor.native.rust.batch_processor import estimate_batch_duration_py

# 100 tasks, 100ms avg duration, 10 concurrent
duration = estimate_batch_duration_py(100, 100.0, 10)
print(f"Estimated duration: {duration:.2f}ms")
```

### Validate Dependencies

```python
# Check if dependencies are valid
if processor.validate_dependencies(tasks):
    print("Dependencies are valid!")
else:
    print("Invalid dependencies (circular reference?)")
```

### Resolve Execution Order

```python
# Get execution layers
layers = processor.resolve_execution_order(tasks)
for i, layer in enumerate(layers):
    print(f"Layer {i}: {layer}")
# Output:
# Layer 0: ['task1', 'task2']
# Layer 1: ['task3', 'task4']
# Layer 2: ['task5']
```

## Advanced Usage

### Custom Task Data

```python
# Pass custom data with tasks
def custom_executor(task_dict):
    task_id = task_dict['task_id']
    custom_data = task_dict['task_data']
    priority = task_dict['priority']
    timeout = task_dict['timeout_ms']

    # Your custom logic
    return process_custom_data(custom_data)

task = BatchTask(
    task_id="custom",
    task_data={"key": "value", "items": [1, 2, 3]},
    priority=1.0
)
```

### Progress Monitoring

```python
# Get current progress
progress = processor.get_progress()
print(f"Progress: {progress.progress_percentage:.1f}%")
print(f"Completed: {progress.completed_tasks}/{progress.total_tasks}")
print(f"ETA: {progress.estimated_remaining_ms:.0f}ms remaining")
```

### Load Balancing

```python
# Assign tasks to workers
assignments = processor.assign_tasks(tasks, workers=4)

for i, worker_tasks in enumerate(assignments):
    print(f"Worker {i}: {len(worker_tasks)} tasks")
```

## Testing Your Setup

```python
# Quick test
from victor.native.rust.batch_processor import BatchProcessor, BatchTask

def test_executor(task_dict):
    return f"Result: {task_dict['task_id']}"

processor = BatchProcessor(max_concurrent=5)
tasks = [
    BatchTask(task_id=f"test{i}", task_data=test_executor)
    for i in range(10)
]

summary = processor.process_batch(tasks, test_executor)
assert summary.successful_count == 10
print("✓ Batch processor working!")
```

## Troubleshooting

### Import Error

```python
# If you get ImportError, Rust extension may not be compiled
from victor.native.rust.batch_processor import RUST_AVAILABLE

if not RUST_AVAILABLE:
    print("Rust implementation not available, using Python fallback")
    # The module will still work, just slower
```

### Circular Dependencies

```python
# Error: "Circular dependency detected in tasks"
# Check your task dependencies
tasks = [
    BatchTask(task_id="a", dependencies=["b"]),
    BatchTask(task_id="b", dependencies=["a"]),  # Circular!
]
# Fix: Remove circular dependency
```

### Timeout Issues

```python
# If tasks timeout frequently, increase timeout
task = BatchTask(
    task_id="slow_task",
    task_data=slow_fn,
    timeout_ms=120000  # 2 minutes
)
```

### Memory Issues

```python
# For large batches, process in smaller chunks
batches = create_task_batches_py(all_tasks, batch_size=100)
for batch in batches:
    summary = processor.process_batch(batch, executor)
    # Results are automatically cleaned up between batches
```

## Next Steps

- Read [full documentation](./batch_processor.md) for detailed API reference
- Check [implementation details](../BATCH_PROCESSOR_IMPLEMENTATION.md) for technical info
- Run [unit tests](../../tests/unit/test_batch_processor.py) for examples
- Explore [use cases](./batch_processor.md#use-cases) for real-world examples

## Support

- **Documentation**: `/Users/vijaysingh/code/codingagent/docs/batch_processor.md`
- **Implementation**: `/Users/vijaysingh/code/codingagent/BATCH_PROCESSOR_IMPLEMENTATION.md`
- **Tests**: `/Users/vijaysingh/code/codingagent/tests/unit/test_batch_processor.py`
- **Source**: `/Users/vijaysingh/code/codingagent/rust/src/batch_processor.rs`
