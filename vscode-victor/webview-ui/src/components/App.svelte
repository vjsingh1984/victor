<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    chatStore,
    hasMessages,
    sendToExtension,
    handleExtensionMessage,
    type ExtensionMessage,
  } from '../stores';
  import ChatMessage from './ChatMessage.svelte';
  import ThinkingIndicator from './ThinkingIndicator.svelte';
  import InputArea from './InputArea.svelte';
  import Welcome from './Welcome.svelte';

  let chatContainer: HTMLDivElement;
  let inputArea: InputArea;

  // Scroll to bottom of chat
  function scrollToBottom() {
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  // Handle messages from VS Code extension
  function handleMessage(event: MessageEvent<ExtensionMessage>) {
    handleExtensionMessage(event.data);
    // Scroll to bottom after state updates
    setTimeout(scrollToBottom, 0);
  }

  // Handle shortcut from welcome screen
  function handleUseShortcut(event: CustomEvent<string>) {
    inputArea?.focus();
    // Note: The shortcut text is handled by InputArea
  }

  // Notify extension that webview is ready
  onMount(() => {
    window.addEventListener('message', handleMessage);
    sendToExtension({ type: 'webviewReady' });
  });

  onDestroy(() => {
    window.removeEventListener('message', handleMessage);
  });

  // Auto-scroll when new messages arrive
  $: if ($chatStore.messages.length > 0) {
    setTimeout(scrollToBottom, 0);
  }

  // Auto-scroll during streaming
  $: if ($chatStore.currentStreamContent) {
    setTimeout(scrollToBottom, 0);
  }
</script>

<div class="app">
  <div class="chat-container" bind:this={chatContainer}>
    {#if $hasMessages}
      {#each $chatStore.messages as message (message.id)}
        <ChatMessage {message} />
      {/each}

      {#if $chatStore.currentStreamContent}
        <ChatMessage
          message={{
            id: 'streaming',
            role: 'assistant',
            content: $chatStore.currentStreamContent,
            timestamp: Date.now(),
          }}
          isStreaming={true}
        />
      {/if}

      {#if $chatStore.isThinking && !$chatStore.currentStreamContent}
        <ThinkingIndicator />
      {/if}

      {#if $chatStore.error}
        <div class="error-message">
          <span class="error-icon">⚠️</span>
          <span class="error-text">{$chatStore.error}</span>
        </div>
      {/if}
    {:else}
      <Welcome on:useShortcut={handleUseShortcut} />
    {/if}
  </div>

  <InputArea
    bind:this={inputArea}
    disabled={$chatStore.isThinking}
  />
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--background);
  }

  .chat-container {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: var(--spacing-md);
    display: flex;
    flex-direction: column;
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-md);
    background: var(--error-bg);
    border: 1px solid var(--error-border);
    border-radius: var(--radius-md);
    color: var(--error-bright);
    margin: var(--spacing-sm) 0;
    animation: fadeIn var(--transition-normal);
  }

  .error-icon {
    font-size: 16px;
  }

  .error-text {
    font-size: var(--font-size-sm);
  }
</style>
