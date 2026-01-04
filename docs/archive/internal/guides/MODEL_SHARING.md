# GGUF Model Sharing Guide: Ollama & LMStudio

Complete guide for sharing GGUF models between Ollama and LMStudio to save disk space.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Understanding Model Storage](#understanding-model-storage)
3. [Solution Options](#solution-options)
4. [Recommended: Using Gollama](#recommended-using-gollama)
5. [Alternative: Manual Symlinks](#alternative-manual-symlinks)
6. [Testing & Verification](#testing-verification)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Problem Overview

When using both Ollama and LMStudio, models are stored in separate directories, leading to:

- **Wasted Disk Space**: Each model stored twice (10-40GB each)
- **Redundant Downloads**: Same model pulled multiple times
- **Difficult Management**: Updating models requires changes in multiple places

**Solution**: Use symbolic links to share GGUF model files between applications.

### Disk Space Savings Example

| Scenario | Disk Usage |
|----------|-----------|
| 5 models × 2 apps (separate) | ~100-200GB |
| 5 models shared via symlinks | ~50-100GB |
| **Space Saved** | **~50-100GB** |

---

## Understanding Model Storage

### Ollama Storage

**Location**:
```
~/.ollama/models/
```

**Structure**:
```
~/.ollama/models/
├── manifests/
│   └── registry.ollama.ai/
│       └── library/
│           ├── qwen2.5-coder/
│           └── llama3/
└── blobs/
    ├── sha256-abc123...
    ├── sha256-def456...
    └── ...
```

**Format**: Ollama stores models as blobs with SHA256 hashes, but the underlying files are often GGUF format.

### LMStudio Storage

**Location**:
```
~/.cache/lm-studio/models/
```

**Structure**:
```
~/.cache/lm-studio/models/
├── TheBloke/
│   └── Llama-2-7B-GGUF/
│       └── llama-2-7b.Q4_K_M.gguf
├── lmstudio-community/
│   └── Qwen2.5-Coder-7B-Instruct-GGUF/
│       └── qwen2.5-coder-7b-instruct-q4_k_m.gguf
└── ...
```

**Format**: LMStudio uses GGUF files directly with HuggingFace-style directory structure.

---

## Solution Options

### Option 1: Gollama (Recommended) ⭐

**Pros**:
- Automated model linking
- Interactive TUI interface
- Bidirectional sync (Ollama ↔ LMStudio)
- Active development and maintenance
- Safe operations with dry-run mode

**Cons**:
- Requires Go installation
- macOS/Linux only

### Option 2: Manual Symlinks

**Pros**:
- No additional tools required
- Works on all platforms
- Full control over linking

**Cons**:
- Manual process
- Error-prone
- Requires understanding of both systems

### Option 3: Llamalink (Archived)

**Status**: Project archived, replaced by Gollama

**Note**: Use Gollama instead for better features and support.

---

## Recommended: Using Gollama

### Step 1: Install Gollama

#### Option A: Using Go (Recommended)

```bash
# Install Gollama
go install github.com/sammcj/gollama@HEAD

# Add to PATH if needed
echo 'export PATH="$HOME/go/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
gollama --version
```

#### Option B: Using Homebrew

```bash
# Check if available via brew
brew install gollama

# Or build from source
git clone https://github.com/sammcj/gollama.git
cd gollama
go build
sudo mv gollama /usr/local/bin/
```

### Step 2: Configure Gollama

```bash
# Create config directory
mkdir -p ~/.config/gollama

# Run Gollama once to create default config
gollama -l

# Edit config if needed
cat ~/.config/gollama/config.json
```

Example config:
```json
{
  "ollama_api_url": "http://localhost:11434",
  "lm_studio_dir": "/Users/yourusername/.cache/lm-studio/models",
  "ollama_models_dir": "/Users/yourusername/.ollama/models",
  "sort_order": "size",
  "editor": "vim",
  "log_level": "info"
}
```

### Step 3: Interactive Linking (Recommended)

Launch Gollama TUI for interactive management:

```bash
# Start Gollama
gollama
```

**Keyboard Controls**:
- `Space` - Select models to link
- `l` - Link selected models to LMStudio
- `L` - Link ALL models to LMStudio
- `C` - Create Ollama models from LMStudio
- `q` - Quit

**Steps**:
1. Navigate with arrow keys
2. Press `Space` to select models you want to link
3. Press `l` to link selected models to LMStudio
4. Confirm the operation
5. Models will now be accessible in both applications

### Step 4: Command-Line Linking

For automation or scripting:

```bash
# Link all Ollama models to LMStudio
gollama --link-lmstudio

# Dry run (see what will happen without doing it)
gollama --link-lmstudio -n

# Link specific model
gollama -s "qwen2.5-coder:7b" --link-lmstudio

# Create Ollama models from LMStudio
gollama --create-from-lmstudio

# List all models
gollama -l
```

### Step 5: Verify Links

```bash
# Check Ollama models
ollama list

# Check symlinks
ls -la ~/.cache/lm-studio/models/

# You should see symlinks pointing to ~/.ollama/models/
```

---

## Alternative: Manual Symlinks

If you can't use Gollama, here's how to create symlinks manually.

### Understanding the Process

1. Identify GGUF model files in Ollama
2. Find corresponding location in LMStudio structure
3. Create symbolic links

### Step 1: Locate Ollama Model Files

```bash
# List Ollama models
ollama list

# Show model file location
ollama show qwen2.5-coder:7b --modelfile

# Find actual GGUF files
find ~/.ollama/models/blobs -type f -name "sha256-*" -size +1G
```

### Step 2: Create LMStudio Directory Structure

```bash
# Create directory for model
mkdir -p ~/.cache/lm-studio/models/ollama-imports/qwen2.5-coder-7b

# Note: Use descriptive names that match the model
```

### Step 3: Create Symbolic Links

```bash
# Example: Link qwen2.5-coder:7b
# First, find the blob hash from ollama show
BLOB_HASH="sha256-abc123def456..."  # Get from ollama show output

# Create symlink
ln -s ~/.ollama/models/blobs/${BLOB_HASH} \
      ~/.cache/lm-studio/models/ollama-imports/qwen2.5-coder-7b/model.gguf

# Verify link
ls -lh ~/.cache/lm-studio/models/ollama-imports/qwen2.5-coder-7b/
```

### Step 4: Verify in LMStudio

1. Open LMStudio
2. Go to "My Models" tab
3. Look for "ollama-imports" folder
4. Models should appear with correct size
5. Try loading a model to test

---

## Testing & Verification

### Test 1: Verify Symbolic Links

```bash
# Check if symlinks exist
ls -la ~/.cache/lm-studio/models/ | grep "\->"

# Count symlinks
find ~/.cache/lm-studio/models/ -type l | wc -l

# Check symlink targets
find ~/.cache/lm-studio/models/ -type l -ls
```

### Test 2: Verify Disk Space Savings

```bash
# Check actual disk usage (follows symlinks)
du -sh ~/.ollama/models/
du -sh ~/.cache/lm-studio/models/

# Check apparent size (includes symlinks)
du -sh --apparent-size ~/.cache/lm-studio/models/

# The apparent size should be much larger than actual disk usage
```

### Test 3: Test in LMStudio

1. Open LMStudio
2. Go to "Local Server" tab
3. Select a linked model from dropdown
4. Click "Load Model"
5. Should load successfully without errors

### Test 4: Test in Ollama

```bash
# List models
ollama list

# Run a model
ollama run qwen2.5-coder:7b "Write a hello world function"

# Should work normally
```

### Test 5: Test with Victor

```bash
# Run Victor test script
python test_all_backends.py --backend ollama
python test_all_backends.py --backend lmstudio

# Both should pass
```

---

## Troubleshooting

### Issue: Gollama Command Not Found

**Solution**:
```bash
# Check if Go bin is in PATH
echo $PATH | grep "go/bin"

# If not, add it
echo 'export PATH="$HOME/go/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Or use full path
~/go/bin/gollama
```

### Issue: Symlink Broken

**Symptoms**: LMStudio shows model but can't load it

**Solution**:
```bash
# Find broken symlinks
find ~/.cache/lm-studio/models/ -type l ! -exec test -e {} \; -print

# Remove broken symlinks
find ~/.cache/lm-studio/models/ -type l ! -exec test -e {} \; -delete

# Recreate links using Gollama
gollama --link-lmstudio
```

### Issue: Model File Not Found

**Symptoms**: Ollama model exists but can't find GGUF file

**Solution**:
```bash
# Get model details
ollama show qwen2.5-coder:7b --modelfile

# Look for "FROM" line which shows blob location
# Example: FROM /Users/you/.ollama/models/blobs/sha256-abc123...

# Verify blob exists
ls -lh ~/.ollama/models/blobs/sha256-*

# If missing, pull model again
ollama pull qwen2.5-coder:7b
```

### Issue: Permission Denied

**Symptoms**: Can't create symlinks

**Solution**:
```bash
# Check directory permissions
ls -ld ~/.cache/lm-studio/models/

# Fix permissions
chmod 755 ~/.cache/lm-studio/models/

# Create directory if missing
mkdir -p ~/.cache/lm-studio/models/

# Try linking again
```

### Issue: LMStudio Doesn't Show Linked Models

**Symptoms**: Symlinks exist but LMStudio doesn't list them

**Solution**:
1. Restart LMStudio application
2. Click "Rescan" or "Refresh" in My Models tab
3. Check if models are in subdirectories (LMStudio expects author/model structure)
4. Recreate with proper structure:
```bash
mkdir -p ~/.cache/lm-studio/models/ollama-community/qwen2.5-coder-7b
ln -s [source] ~/.cache/lm-studio/models/ollama-community/qwen2.5-coder-7b/model.gguf
```

### Issue: Models Taking Up Same Space

**Symptoms**: Disk usage not reduced after linking

**Solution**:
```bash
# Check if you created copies instead of symlinks
file ~/.cache/lm-studio/models/ollama-imports/*/model.gguf

# Should show: "symbolic link to ..."
# If it shows "data" or file type, you created a copy

# Remove copies
rm ~/.cache/lm-studio/models/ollama-imports/*/model.gguf

# Create proper symlinks
gollama --link-lmstudio
```

---

## Best Practices

### 1. Use Gollama for Management

✅ **DO**: Use Gollama's TUI for interactive management
✅ **DO**: Run `gollama -l` regularly to see model status
✅ **DO**: Use dry-run mode (`-n`) before major operations

❌ **DON'T**: Mix manual and automated linking
❌ **DON'T**: Delete source models after linking

### 2. Maintain Model Consistency

✅ **DO**: Keep Ollama as the primary model store
✅ **DO**: Pull new models via Ollama first
✅ **DO**: Link to LMStudio after Ollama has the model

❌ **DON'T**: Download same model in both apps
❌ **DON'T**: Delete Ollama models if LMStudio has symlinks

### 3. Backup Before Major Changes

```bash
# Backup Ollama models list
ollama list > ~/ollama-models-backup.txt

# Backup LMStudio model list
ls -R ~/.cache/lm-studio/models/ > ~/lmstudio-models-backup.txt

# Backup Gollama config
cp ~/.config/gollama/config.json ~/.config/gollama/config.json.backup
```

### 4. Monitor Disk Space

```bash
# Check disk usage regularly
df -h

# Check model directories
du -sh ~/.ollama/models/
du -sh ~/.cache/lm-studio/models/

# List largest models
du -sh ~/.ollama/models/blobs/* | sort -h | tail -10
```

### 5. Clean Up Periodically

```bash
# Remove unused Ollama models
ollama rm model-name

# Clean up broken symlinks
find ~/.cache/lm-studio/models/ -type l ! -exec test -e {} \; -delete

# Prune old models (via Gollama)
gollama
# Press 'D' to delete selected models
```

---

## Advanced: Automated Setup Script

Create a script to automate the entire process:

```bash
cat > ~/setup-model-sharing.sh << 'EOF'
#!/bin/bash
# Automated Model Sharing Setup for Ollama & LMStudio

set -e

echo "================================"
echo "Model Sharing Setup Script"
echo "================================"
echo ""

# Step 1: Check if Ollama is running
echo "1. Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "   ✅ Ollama is running"
else
    echo "   ❌ Ollama is not running"
    echo "   Start with: ollama serve"
    exit 1
fi

# Step 2: Check if Go is installed
echo "2. Checking Go installation..."
if command -v go &> /dev/null; then
    echo "   ✅ Go is installed ($(go version))"
else
    echo "   ❌ Go is not installed"
    echo "   Install with: brew install go"
    exit 1
fi

# Step 3: Install Gollama
echo "3. Installing Gollama..."
go install github.com/sammcj/gollama@HEAD

# Add to PATH
if ! echo $PATH | grep -q "$HOME/go/bin"; then
    echo 'export PATH="$HOME/go/bin:$PATH"' >> ~/.zshrc
    export PATH="$HOME/go/bin:$PATH"
fi

echo "   ✅ Gollama installed"

# Step 4: Create config directory
echo "4. Setting up configuration..."
mkdir -p ~/.config/gollama
echo "   ✅ Config directory created"

# Step 5: Link models
echo "5. Linking Ollama models to LMStudio..."
gollama --link-lmstudio -n  # Dry run first

read -p "   Proceed with linking? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gollama --link-lmstudio
    echo "   ✅ Models linked successfully"
else
    echo "   ⚠️  Linking cancelled"
fi

# Step 6: Verify
echo ""
echo "6. Verification..."
SYMLINK_COUNT=$(find ~/.cache/lm-studio/models/ -type l 2>/dev/null | wc -l)
echo "   Found $SYMLINK_COUNT symlinks"

# Step 7: Calculate savings
OLLAMA_SIZE=$(du -sh ~/.ollama/models/ 2>/dev/null | awk '{print $1}')
LMSTUDIO_APPARENT=$(du -sh --apparent-size ~/.cache/lm-studio/models/ 2>/dev/null | awk '{print $1}')
LMSTUDIO_ACTUAL=$(du -sh ~/.cache/lm-studio/models/ 2>/dev/null | awk '{print $1}')

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo "Ollama models: $OLLAMA_SIZE"
echo "LMStudio apparent: $LMSTUDIO_APPARENT"
echo "LMStudio actual: $LMSTUDIO_ACTUAL"
echo ""
echo "Next steps:"
echo "1. Open LMStudio and verify models appear"
echo "2. Run: gollama (to launch TUI)"
echo "3. Test with: python test_all_backends.py"
EOF

chmod +x ~/setup-model-sharing.sh
~/setup-model-sharing.sh
```

---

## Summary

### Quick Reference

```bash
# Install Gollama
go install github.com/sammcj/gollama@HEAD

# Link all models
gollama --link-lmstudio

# Launch TUI
gollama

# Verify
ollama list
ls -la ~/.cache/lm-studio/models/
```

### Expected Results

After proper setup:
- ✅ Models accessible in both Ollama and LMStudio
- ✅ ~50% disk space savings
- ✅ Single source of truth (Ollama)
- ✅ Easy model management via Gollama TUI
- ✅ No duplicate downloads needed

---

## Sources

- [LM Studio Import Models Docs](https://lmstudio.ai/docs/app/advanced/import-model)
- [Gollama GitHub Repository](https://github.com/sammcj/gollama)
- [Llamalink Project](https://github.com/sammcj/llamalink)
- [Running LLMs Locally Guide](https://notes.suhaib.in/docs/tech/how-to/how-to-run-llms-locally-with-ollama-lm-studio-and-gguf-models/)
- [Ollama Issue #6589: Sharing with LM Studio](https://github.com/ollama/ollama/issues/6589)
- [GGUF and Ollama Primer](https://polarsparc.github.io/GenAI/GGUF-Ollama.html)

---

**Happy Model Sharing!** 💾✨
