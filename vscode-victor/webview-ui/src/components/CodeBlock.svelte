<script lang="ts">
  import { onMount } from 'svelte';
  import hljs from 'highlight.js';
  import { sendToExtension } from '../stores';

  export let code: string;
  export let language: string = 'text';
  export let showCopy: boolean = true;
  export let showApply: boolean = true;

  let codeElement: HTMLElement;
  let copied = false;
  let applied = false;

  // Normalize language name for highlight.js
  function normalizeLanguage(lang: string): string {
    const langMap: Record<string, string> = {
      js: 'javascript',
      ts: 'typescript',
      py: 'python',
      sh: 'bash',
      shell: 'bash',
      zsh: 'bash',
      yml: 'yaml',
      md: 'markdown',
    };
    return langMap[lang?.toLowerCase()] || lang?.toLowerCase() || 'plaintext';
  }

  // Highlight code on mount and when code changes
  $: if (codeElement && code) {
    highlightCode();
  }

  function highlightCode() {
    if (codeElement) {
      const normalizedLang = normalizeLanguage(language);
      try {
        if (hljs.getLanguage(normalizedLang)) {
          codeElement.innerHTML = hljs.highlight(code, { language: normalizedLang }).value;
        } else {
          codeElement.innerHTML = hljs.highlightAuto(code).value;
        }
      } catch {
        // Fallback to plain text
        codeElement.textContent = code;
      }
    }
  }

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(code);
      copied = true;
      setTimeout(() => { copied = false; }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }

  function applyCode() {
    sendToExtension({ type: 'applyCode', code });
    applied = true;
    setTimeout(() => { applied = false; }, 2000);
  }
</script>

<div class="code-block">
  {#if language && language !== 'text' && language !== 'plaintext'}
    <span class="language-label">{language}</span>
  {/if}

  <div class="code-actions">
    {#if showApply}
      <button
        class="action-btn apply-btn"
        class:success={applied}
        on:click={applyCode}
        title="Apply to editor"
      >
        {applied ? 'Applied!' : 'Apply'}
      </button>
    {/if}

    {#if showCopy}
      <button
        class="action-btn copy-btn"
        class:success={copied}
        on:click={copyCode}
        title="Copy to clipboard"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
    {/if}
  </div>

  <pre><code bind:this={codeElement} class="hljs language-{normalizeLanguage(language)}">{code}</code></pre>
</div>

<style>
  .code-block {
    position: relative;
    margin: var(--spacing-sm) 0;
    background: var(--surface);
    border-radius: var(--radius-md);
    border: 1px solid var(--panel-highlight);
    overflow: hidden;
  }

  .language-label {
    position: absolute;
    top: 0;
    left: 0;
    background: var(--primary-dim);
    color: var(--text-bright);
    padding: 2px 8px;
    font-size: var(--font-size-xs);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-radius: 0 0 var(--radius-sm) 0;
    z-index: 1;
  }

  .code-actions {
    position: absolute;
    top: var(--spacing-xs);
    right: var(--spacing-xs);
    display: flex;
    gap: var(--spacing-xs);
    opacity: 0;
    transition: opacity var(--transition-fast);
    z-index: 1;
  }

  .code-block:hover .code-actions {
    opacity: 1;
  }

  .action-btn {
    background: var(--panel);
    color: var(--text-muted);
    border: 1px solid var(--panel-highlight);
    padding: 4px 8px;
    font-size: var(--font-size-xs);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .action-btn:hover {
    background: var(--panel-elevated);
    color: var(--text);
    border-color: var(--primary-dim);
  }

  .action-btn.success {
    background: var(--success-dim);
    color: var(--success-bright);
    border-color: var(--success);
  }

  pre {
    margin: 0;
    padding: var(--spacing-md);
    padding-top: calc(var(--spacing-md) + 20px);
    overflow-x: auto;
    font-family: var(--vscode-editor-font-family, 'Fira Code', 'JetBrains Mono', monospace);
    font-size: var(--font-size-sm);
    line-height: 1.5;
  }

  code {
    background: none;
    padding: 0;
  }

  /* Syntax highlighting - matches TUI theme */
  :global(.hljs) {
    background: transparent;
    color: var(--text);
  }

  :global(.hljs-keyword) {
    color: var(--primary);
    font-weight: 500;
  }

  :global(.hljs-built_in) {
    color: var(--accent-cyan);
  }

  :global(.hljs-type) {
    color: var(--accent-cyan);
  }

  :global(.hljs-literal) {
    color: var(--primary-bright);
  }

  :global(.hljs-number) {
    color: var(--accent-purple);
  }

  :global(.hljs-string) {
    color: var(--success-bright);
  }

  :global(.hljs-symbol) {
    color: var(--accent-cyan);
  }

  :global(.hljs-regexp) {
    color: var(--accent-pink);
  }

  :global(.hljs-function) {
    color: var(--warning-bright);
  }

  :global(.hljs-title) {
    color: var(--warning-bright);
  }

  :global(.hljs-params) {
    color: var(--text);
  }

  :global(.hljs-comment) {
    color: var(--text-subtle);
    font-style: italic;
  }

  :global(.hljs-doctag) {
    color: var(--success);
  }

  :global(.hljs-meta) {
    color: var(--accent-purple);
  }

  :global(.hljs-attr) {
    color: var(--primary-bright);
  }

  :global(.hljs-variable) {
    color: var(--text);
  }

  :global(.hljs-template-variable) {
    color: var(--accent-pink);
  }

  :global(.hljs-tag) {
    color: var(--error-bright);
  }

  :global(.hljs-name) {
    color: var(--primary);
  }

  :global(.hljs-selector-class) {
    color: var(--warning);
  }

  :global(.hljs-selector-id) {
    color: var(--warning-bright);
  }

  :global(.hljs-selector-tag) {
    color: var(--primary);
  }

  :global(.hljs-property) {
    color: var(--primary-bright);
  }

  :global(.hljs-addition) {
    color: var(--success);
    background: var(--success-glow);
  }

  :global(.hljs-deletion) {
    color: var(--error);
    background: var(--error-glow);
  }
</style>
