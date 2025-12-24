<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { sendToExtension } from '../stores';

  export let disabled: boolean = false;
  export let placeholder: string = 'Ask Victor anything...';

  const dispatch = createEventDispatcher<{ send: string }>();

  let message = '';
  let textareaElement: HTMLTextAreaElement;

  // Auto-resize textarea based on content
  function autoResize() {
    if (textareaElement) {
      textareaElement.style.height = 'auto';
      textareaElement.style.height = Math.min(textareaElement.scrollHeight, 150) + 'px';
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    } else if (event.key === 'Escape') {
      message = '';
      autoResize();
      textareaElement?.blur();
    }
  }

  function sendMessage() {
    const trimmed = message.trim();
    if (!trimmed || disabled) return;

    dispatch('send', trimmed);
    sendToExtension({ type: 'sendMessage', message: trimmed });
    message = '';
    autoResize();
  }

  function useShortcut(prefix: string) {
    message = prefix + ' ';
    textareaElement?.focus();
  }

  // Expose methods for parent components
  export function focus() {
    textareaElement?.focus();
  }

  export function clear() {
    message = '';
    autoResize();
  }
</script>

<div class="input-area">
  <div class="shortcuts">
    <button class="shortcut" on:click={() => useShortcut('Explain the selected code')}>
      💡 Explain
    </button>
    <button class="shortcut" on:click={() => useShortcut('Refactor this code')}>
      🔧 Refactor
    </button>
    <button class="shortcut" on:click={() => useShortcut('Write tests for')}>
      🧪 Test
    </button>
    <button class="shortcut" on:click={() => useShortcut('Fix the bugs in')}>
      🐛 Fix
    </button>
    <button class="shortcut" on:click={() => useShortcut('Add documentation to')}>
      📖 Document
    </button>
  </div>

  <div class="input-container">
    <textarea
      bind:this={textareaElement}
      bind:value={message}
      {placeholder}
      {disabled}
      rows="1"
      on:input={autoResize}
      on:keydown={handleKeydown}
    ></textarea>

    <button
      class="send-btn"
      disabled={disabled || !message.trim()}
      on:click={sendMessage}
      title="Send message (Enter)"
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2Z" />
      </svg>
    </button>
  </div>

  <div class="help-text">
    <kbd>Enter</kbd> to send
    <span class="separator">|</span>
    <kbd>Shift+Enter</kbd> for new line
    <span class="separator">|</span>
    <kbd>Esc</kbd> to clear
  </div>
</div>

<style>
  .input-area {
    padding: var(--spacing-md);
    border-top: 1px solid var(--panel-highlight);
    background: var(--panel);
  }

  .shortcuts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
    margin-bottom: var(--spacing-sm);
  }

  .shortcut {
    background: var(--surface);
    border: 1px solid var(--panel-highlight);
    color: var(--text-muted);
    padding: 4px 10px;
    border-radius: var(--radius-md);
    font-size: var(--font-size-xs);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .shortcut:hover {
    background: var(--surface-elevated);
    border-color: var(--primary-dim);
    color: var(--text);
  }

  .input-container {
    display: flex;
    gap: var(--spacing-sm);
    align-items: flex-end;
  }

  textarea {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--panel-highlight);
    color: var(--text);
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-md);
    font-family: inherit;
    font-size: var(--font-size-base);
    resize: none;
    min-height: 40px;
    max-height: 150px;
    line-height: 1.5;
    transition: border-color var(--transition-fast);
  }

  textarea:focus {
    outline: none;
    border-color: var(--primary);
  }

  textarea:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  textarea::placeholder {
    color: var(--text-subtle);
  }

  .send-btn {
    background: var(--primary);
    color: var(--text-bright);
    border: none;
    width: 40px;
    height: 40px;
    border-radius: var(--radius-md);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition-fast);
  }

  .send-btn:hover:not(:disabled) {
    background: var(--primary-bright);
    transform: translateY(-1px);
  }

  .send-btn:active:not(:disabled) {
    transform: translateY(0);
  }

  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .help-text {
    font-size: var(--font-size-xs);
    color: var(--text-subtle);
    text-align: center;
    margin-top: var(--spacing-sm);
  }

  .separator {
    margin: 0 var(--spacing-xs);
    color: var(--panel-highlight);
  }

  kbd {
    background: var(--surface);
    border: 1px solid var(--panel-highlight);
    border-radius: 3px;
    padding: 1px 5px;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 10px;
    box-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);
  }
</style>
