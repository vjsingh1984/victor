<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{ useShortcut: string }>();

  const shortcuts = [
    { icon: '💡', label: 'Explain', prompt: 'Explain the selected code' },
    { icon: '🔧', label: 'Refactor', prompt: 'Refactor this code' },
    { icon: '🧪', label: 'Test', prompt: 'Write tests for' },
    { icon: '🐛', label: 'Fix', prompt: 'Fix the bugs in' },
    { icon: '📖', label: 'Document', prompt: 'Add documentation to' },
    { icon: '⚡', label: 'Optimize', prompt: 'Optimize this code' },
  ];
</script>

<div class="welcome">
  <div class="logo">
    <span class="logo-icon">🤖</span>
    <h1>Victor AI Assistant</h1>
  </div>

  <p class="tagline">Ask me anything about your code!</p>

  <div class="features">
    <div class="feature">
      <span class="feature-icon">🔍</span>
      <span class="feature-text">Semantic code search</span>
    </div>
    <div class="feature">
      <span class="feature-icon">🛠️</span>
      <span class="feature-text">45+ enterprise tools</span>
    </div>
    <div class="feature">
      <span class="feature-icon">🌐</span>
      <span class="feature-text">Multi-provider support</span>
    </div>
  </div>

  <div class="shortcuts">
    <p class="shortcuts-title">Quick actions</p>
    <div class="shortcuts-grid">
      {#each shortcuts as shortcut}
        <button
          class="shortcut-btn"
          on:click={() => dispatch('useShortcut', shortcut.prompt)}
        >
          <span class="shortcut-icon">{shortcut.icon}</span>
          <span class="shortcut-label">{shortcut.label}</span>
        </button>
      {/each}
    </div>
  </div>

  <div class="keyboard-hints">
    <kbd>Enter</kbd> to send
    <span class="separator">•</span>
    <kbd>Shift+Enter</kbd> new line
    <span class="separator">•</span>
    <kbd>Esc</kbd> clear
  </div>
</div>

<style>
  .welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-xl);
    text-align: center;
    animation: fadeIn 0.5s ease;
    flex: 1;
  }

  .logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-lg);
  }

  .logo-icon {
    font-size: 48px;
    animation: pulse 2s infinite;
  }

  .logo h1 {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--text-bright);
    margin: 0;
  }

  .tagline {
    color: var(--text-muted);
    font-size: var(--font-size-base);
    margin-bottom: var(--spacing-lg);
  }

  .features {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-xl);
  }

  .feature {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: var(--spacing-xs) var(--spacing-sm);
    background: var(--surface);
    border-radius: var(--radius-md);
    font-size: var(--font-size-sm);
    color: var(--text-muted);
  }

  .feature-icon {
    font-size: 14px;
  }

  .shortcuts-title {
    font-size: var(--font-size-sm);
    color: var(--text-subtle);
    margin-bottom: var(--spacing-sm);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .shortcuts-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-lg);
  }

  .shortcut-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--spacing-xs);
    padding: var(--spacing-md);
    background: var(--surface);
    border: 1px solid var(--panel-highlight);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all var(--transition-fast);
    color: var(--text);
  }

  .shortcut-btn:hover {
    background: var(--surface-elevated);
    border-color: var(--primary);
    transform: translateY(-2px);
  }

  .shortcut-icon {
    font-size: 20px;
  }

  .shortcut-label {
    font-size: var(--font-size-xs);
    font-weight: 500;
  }

  .keyboard-hints {
    font-size: var(--font-size-xs);
    color: var(--text-subtle);
  }

  .separator {
    margin: 0 var(--spacing-xs);
    opacity: 0.5;
  }

  kbd {
    background: var(--panel);
    border: 1px solid var(--panel-highlight);
    border-radius: 3px;
    padding: 2px 6px;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 10px;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }
</style>
