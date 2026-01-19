# Victor AI - Jupyter Notebooks

This directory contains interactive Jupyter notebooks for learning Victor AI.

## Available Notebooks

### 1. Introductory Tutorial
**File:** `introductory_tutorial.ipynb`

**Topics:**
- Installation and setup
- Basic chat functionality
- Code analysis
- Code generation
- Refactoring
- Multi-turn conversations

**Prerequisites:** Basic Python knowledge

**Run:**
```bash
jupyter notebook introductory_tutorial.ipynb
```

### 2. Advanced Features
**File:** `advanced_features.ipynb`

**Topics:**
- Workflow creation and execution
- Custom tool development
- Multi-agent coordination
- Event handling
- Performance optimization

**Prerequisites:** Complete introductory tutorial

**Run:**
```bash
jupyter notebook advanced_features.ipynb
```

### 3. Custom Tools Development
**File:** `custom_tools.ipynb`

**Topics:**
- Tool architecture
- Creating custom tools
- Parameter validation
- Error handling
- Tool testing
- Tool registration

**Prerequisites:** Understanding of Victor basics

**Run:**
```bash
jupyter notebook custom_tools.ipynb
```

### 4. API Usage
**File:** `api_usage.ipynb`

**Topics:**
- Direct API integration
- Async/await patterns
- Streaming responses
- Error handling
- Provider switching
- Configuration management

**Prerequisites:** Basic async Python knowledge

**Run:**
```bash
jupyter notebook api_usage.ipynb
```

## Setup

### Install Dependencies

```bash
# Install Victor AI with Jupyter support
pip install victor-ai[jupyter]

# Or install all dependencies
pip install -e ".[dev]"

# Install Jupyter if not already installed
pip install jupyter notebook

# Install additional notebook dependencies
pip install ipywidgets matplotlib seaborn pandas
```

### Start Jupyter

```bash
# Start Jupyter notebook server
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Usage Tips

### 1. Run Cells Sequentially

Notebooks are designed to be run from top to bottom. Execute cells in order:

```
Cell 1 -> Cell 2 -> Cell 3 -> ...
```

### 2. Clear Outputs Before Re-running

```bash
# Clear all outputs
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

### 3. Restart Kernel if Needed

If you encounter issues, restart the kernel:
- **Jupyter:** Kernel -> Restart Kernel
- **Keyboard:** Esc, 0, 0 (press twice)

### 4. Save Your Work

- **Manual:** File -> Save and Checkpoint
- **Auto-save:** Jupyter auto-saves every few minutes
- **Keyboard:** Ctrl+S (or Cmd+S on Mac)

## Notebook Features

### Interactive Code Execution

Run code cells and see results immediately:

```python
# Run this in a notebook cell
from victor import Agent

agent = await Agent.create()
response = await agent.run("Hello!")
print(response.content)
```

### Rich Output Display

Notebooks support rich output including:
- Text
- HTML
- Markdown
- Images
- DataFrames
- Plots

### Visualizations

```python
import matplotlib.pyplot as plt

# Create plots that display inline
plt.plot([1, 2, 3, 4])
plt.show()
```

## Common Issues

### Issue: Module Not Found

**Solution:**
```bash
# Install Victor AI
pip install victor-ai

# Restart Jupyter kernel after installation
```

### Issue: Async/Await Errors

**Solution:**
```python
# Use asyncio.run() in notebooks
import asyncio

async def main():
    agent = await Agent.create()
    return agent

agent = asyncio.run(main())
```

### Issue: Kernel Hanging

**Solution:**
- Restart kernel: Kernel -> Restart Kernel
- Check if Ollama is running (if using local provider)
- Try a different provider

## Best Practices

### 1. Document Your Code

```python
# Add comments and markdown cells to explain
# what each section does
```

### 2. Test Incrementally

```python
# Test each cell before moving to the next
# This makes debugging easier
```

### 3. Use Checkpoints

```python
# Save important intermediate results
# in case you need to restart
```

### 4. Keep Cells Focused

```python
# Each cell should do one thing
# This makes notebooks easier to understand
```

## Converting Notebooks

### To Python Script

```bash
jupyter nbconvert notebook.ipynb --to python
```

### To Markdown

```bash
jupyter nbconvert notebook.ipynb --to markdown
```

### To HTML

```bash
jupyter nbconvert notebook.ipynb --to html
```

## Sharing Notebooks

### GitHub

1. Commit `.ipynb` files to GitHub
2. GitHub renders notebooks automatically
3. Share link with others

### NBViewer

```bash
# Upload to NBViewer for sharing
# https://nbviewer.jupyter.org/
```

### Google Colab

1. Upload notebook to Google Drive
2. Open with Colab
3. Share link

## Learning Path

### Beginner
1. Start with **Introductory Tutorial**
2. Practice with your own code
3. Explore different providers

### Intermediate
1. Complete **Advanced Features**
2. Build custom tools
3. Create workflows

### Advanced
1. Master **API Usage**
2. Contribute to Victor
3. Create your own notebooks

## Resources

- **Victor Documentation:** `docs/`
- **Jupyter Documentation:** https://jupyter-notebook.readthedocs.io/
- **Python Asyncio:** https://docs.python.org/3/library/asyncio.html
- **Victor Examples:** `examples/`

## Contributing

Have ideas for new notebooks?

1. Follow the existing structure
2. Include clear explanations
3. Test all code cells
4. Add documentation
5. Submit a pull request

Happy learning! 📓
