# Victor Security and Privacy

The security of your code and data is a top priority for Victor. This document outlines the security model, data handling practices, and the measures taken to ensure a safe and private user experience, especially when using the "air-gapped" mode.

> Reality check: `security_scan` currently performs regex-based secret/config checks and a small dependency hint list only. There is no CVE/IaC/package-audit integration yet; treat results as lightweight hygiene, not a full security review.

## Code Execution Sandboxing

**Problem:** Allowing a Large Language Model (LLM) to generate and execute code on your local machine is inherently risky. A malicious or flawed code suggestion could potentially access sensitive files, modify your system, or make unwanted network requests.

**Solution:** Victor addresses this risk by executing all LLM-generated code within a **secure, isolated Docker container**.

### How it Works:

1.  **Isolation:** When the `CodeExecutor` tool is invoked, it does not run the code on your machine directly. Instead, it starts a new, clean Docker container.
2.  **Controlled Environment:** The code is executed inside this container, which has its own isolated filesystem, process space, and network stack. It has **no access** to your personal files, environment variables, or local network by default.
3.  **Minimalist Image:** The default Docker image (`python:3.11-slim`) is a minimal Python environment. While it includes common libraries for data analysis (like `pandas` and `numpy`), it does not contain unnecessary tools or permissions.
4.  **Ephemeral State:** Each container is ephemeral. It is created for a single code execution, and it is **destroyed** immediately after the code finishes running, ensuring that no state persists between executions.
5.  **Timeout:** All code executions have a strict timeout (defaulting to 60 seconds) to prevent runaway processes or infinite loops from consuming system resources.

This sandboxing model ensures that even if the LLM generates harmful code, its impact is confined to the temporary container and cannot affect your host system.

## Data Handling and Privacy

We are committed to handling your data with the utmost respect for your privacy.

### API Keys

-   Your LLM provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) are stored and managed by you.
-   You can provide them via environment variables or a local `profiles.yaml` file in your `~/.victor` directory.
-   Victor loads these keys into memory for the duration of a session to authenticate with the respective provider APIs. They are **never** logged or stored elsewhere.

### Code and Prompts

-   When you are using a **cloud-based LLM provider** (e.g., Anthropic, OpenAI, Google), your prompts and the relevant context (including code snippets) are sent to that provider's API for processing. Please refer to the privacy policy of your chosen provider for details on how they handle your data.
-   When you are using a **local LLM provider** (e.g., Ollama), all data remains on your local machine and is only sent to your local LLM instance.

## Air-Gapped Mode

For maximum privacy and for use in environments with no internet access, Victor provides a dedicated **air-gapped mode**.

### How to Enable It:

You can enable this mode by setting `airgapped_mode: true` in your `~/.victor/profiles.yaml` file under a specific profile or globally.

### Guarantees:

When `airgapped_mode` is set to `true`:

1.  **No Outbound Network Calls:** Victor programmatically disables all tools that require internet access. Currently, this includes:
    *   `WebSearchTool`
2.  **Local-Only Providers:** The application will fail to connect if your configured profile points to a cloud-based LLM provider. You must use a local provider like Ollama.
3.  **Local-Only Embeddings:** Codebase indexing and semantic search will rely exclusively on locally-run embedding models.

This mode is designed to give you full confidence that none of your data can leave your machine.
