<script lang="ts">
  export let message: string = 'Victor is thinking...';
</script>

<div class="thinking">
  <div class="thinking-dots">
    <span></span>
    <span></span>
    <span></span>
  </div>
  <span class="thinking-text">{message}</span>
</div>

<style>
  .thinking {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-md);
    color: var(--text-muted);
    background: var(--thinking-bg);
    border: 1px solid var(--thinking-border);
    border-radius: var(--radius-md);
    margin: var(--spacing-sm) 0;
  }

  .thinking-dots {
    display: flex;
    gap: 4px;
  }

  .thinking-dots span {
    width: 8px;
    height: 8px;
    background: var(--accent-purple);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out;
  }

  .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
  .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
  }

  .thinking-text {
    font-size: var(--font-size-sm);
    font-style: italic;
  }
</style>
