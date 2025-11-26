# Air-Gapped Deployment Guide for Victor

This guide provides step-by-step instructions for setting up and running Victor in a fully air-gapped environment where no internet connection is available.

## Overview

Running in an air-gapped environment requires pre-installing all necessary components on the target machine. The core components are:

1.  **Python Environment:** With Victor and its dependencies installed.
2.  **Docker:** To run the secure code execution sandbox.
3.  **Docker Images:** The specific Docker image used for code execution.
4.  **Local LLM Provider:** Such as Ollama.
5.  **LLM Models:** The actual model files to be run by Ollama.

## Step 1: Prepare the Components on a Machine with Internet

First, you need to gather all the necessary files on a machine that has internet access.

### 1.1. Download Victor Project

Clone the Victor repository or download the source code as a ZIP file.

```bash
git clone https://github.com/vijaysingh/victor.git
```

### 1.2. Download Python Dependencies

Create a `requirements.txt` file and download the wheels for all of Victor's dependencies.

```bash
# Make sure you are in the victor project directory
pip download -r requirements.txt -d /path/to/your/offline_packages
```
*Note: Make sure you download packages compatible with your air-gapped machine's OS and architecture.*

### 1.3. Download Docker

Download the appropriate Docker Desktop or Docker Engine installer for your target machine's operating system from the official Docker website.

### 1.4. Save the Code Execution Docker Image

Pull the Docker image used by the `CodeExecutorTool` and save it as a tarball.

```bash
# Pull the image
docker pull python:3.11-slim

# Save the image to a file
docker save python:3.11-slim -o /path/to/your/python_3_11_slim.tar
```

### 1.5. Download Ollama

Download the Ollama installer for your target OS from the [Ollama website](https://ollama.com/).

### 1.6. Download LLM Models via Ollama

On the machine with internet, use Ollama to pull the models you want to use on your air-gapped machine. For example, to get Llama 3:

```bash
ollama pull llama3
```

Then, find the model file (blobs) in the Ollama directory (e.g., `~/.ollama/models` on macOS/Linux) and copy them.

## Step 2: Set Up the Air-Gapped Machine

Transfer all the files you downloaded in Step 1 to the air-gapped machine (e.g., via a USB drive).

### 2.1. Install Python and Dependencies

1.  Install Python on the machine if it's not already present.
2.  Create a virtual environment.
3.  Install Victor and its dependencies from the wheels you downloaded:

```bash
# In the victor project directory
pip install --no-index --find-links=/path/to/your/offline_packages -e .
```

### 2.2. Install and Configure Docker

1.  Install Docker using the installer you downloaded.
2.  Load the code executor Docker image:

```bash
docker load -i /path/to/your/python_3_11_slim.tar
```

### 2.3. Install and Configure Ollama

1.  Install Ollama using its installer.
2.  Copy the model files you saved into the Ollama models directory on the air-gapped machine.

## Step 3: Configure Victor for Air-Gapped Mode

The final step is to configure Victor to run in `airgapped_mode`.

1.  Create a `profiles.yaml` file in the `~/.victor/` directory on the air-gapped machine.
2.  Configure a profile that uses Ollama and enables `airgapped_mode`.

**Example `profiles.yaml`:**

```yaml
profiles:
  local_secure:
    provider: ollama
    model: llama3 # The model you made available locally
    temperature: 0.5
    airgapped_mode: true

# You can also set it globally
settings:
  airgapped_mode: true
```

## Step 4: Run Victor

You are now ready to run Victor in a fully offline, air-gapped environment.

```bash
# Run with your specific air-gapped profile
victor --profile local_secure

# If you set airgapped_mode globally, you can just run:
victor
```

Victor will now run without making any external network requests, using your local Ollama instance for LLM inference and the local Docker image for secure code execution.
