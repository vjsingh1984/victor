<script lang="ts">
  import type { ChatMessage } from '../stores';
  import CodeBlock from './CodeBlock.svelte';
  import ToolCall from './ToolCall.svelte';

  export let message: ChatMessage;
  export let isStreaming: boolean = false;

  // Parse content into segments (text and code blocks)
  interface ContentSegment {
    type: 'text' | 'code';
    content: string;
    language?: string;
  }

  function parseContent(content: string): ContentSegment[] {
    const segments: ContentSegment[] = [];
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;

    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Add text before this code block
      if (match.index > lastIndex) {
        const textContent = content.slice(lastIndex, match.index);
        if (textContent.trim()) {
          segments.push({ type: 'text', content: textContent });
        }
      }

      // Add code block
      segments.push({
        type: 'code',
        language: match[1] || 'text',
        content: match[2],
      });

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < content.length) {
      const textContent = content.slice(lastIndex);
      if (textContent.trim()) {
        segments.push({ type: 'text', content: textContent });
      }
    }

    return segments.length > 0 ? segments : [{ type: 'text', content }];
  }

  // Format text with markdown-like styling
  function formatText(text: string): string {
    let html = escapeHtml(text);

    // Headers
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold and italic
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

    // Strikethrough
    html = html.replace(/~~([^~]+)~~/g, '<del>$1</del>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Blockquotes
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');

    // Lists
    html = html.replace(/^[\*\-] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/gs, '<ul>$&</ul>');
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    // Clean up
    html = html.replace(/<\/(h[1-4]|ul|ol|blockquote)><br>/g, '</$1>');
    html = html.replace(/<br><(h[1-4]|ul|ol|blockquote)/g, '<$1');

    return html;
  }

  function escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Format timestamp
  function formatTime(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  $: segments = parseContent(message.content);
  $: roleClass = message.role;
  $: timeString = formatTime(message.timestamp);
</script>

<div class="message {roleClass}" class:streaming={isStreaming}>
  <div class="message-header">
    <span class="role-badge">
      {#if message.role === 'user'}
        👤 You
      {:else if message.role === 'assistant'}
        🤖 Victor
      {:else}
        ⚙️ System
      {/if}
    </span>
    <span class="timestamp">{timeString}</span>
  </div>

  <div class="message-content">
    {#each segments as segment}
      {#if segment.type === 'code'}
        <CodeBlock
          code={segment.content}
          language={segment.language || 'text'}
        />
      {:else}
        <div class="text-content">
          {@html formatText(segment.content)}
        </div>
      {/if}
    {/each}

    {#if isStreaming}
      <span class="cursor-blink">▋</span>
    {/if}
  </div>

  {#if message.toolCalls && message.toolCalls.length > 0}
    <div class="tool-calls">
      {#each message.toolCalls as toolCall}
        <ToolCall {toolCall} />
      {/each}
    </div>
  {/if}
</div>

<style>
  .message {
    padding: var(--spacing-md);
    border-radius: var(--radius-lg);
    margin-bottom: var(--spacing-md);
    animation: fadeIn var(--transition-normal);
  }

  .message.user {
    background: var(--user-message-bg);
    border: 1px solid var(--user-message-border);
    margin-left: var(--spacing-xl);
  }

  .message.assistant {
    background: var(--assistant-message-bg);
    border: 1px solid var(--assistant-message-border);
    margin-right: var(--spacing-xl);
  }

  .message.system {
    background: var(--panel);
    border: 1px solid var(--text-subtle);
    text-align: center;
    font-size: var(--font-size-sm);
    color: var(--text-muted);
  }

  .message.streaming {
    border-color: var(--primary);
    box-shadow: 0 0 10px var(--primary-glow);
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-sm);
    font-size: var(--font-size-sm);
  }

  .role-badge {
    font-weight: 600;
    color: var(--text);
  }

  .user .role-badge {
    color: var(--success-bright);
  }

  .assistant .role-badge {
    color: var(--primary-bright);
  }

  .timestamp {
    color: var(--text-subtle);
    font-size: var(--font-size-xs);
  }

  .message-content {
    color: var(--text);
    line-height: 1.6;
  }

  .text-content {
    word-wrap: break-word;
  }

  .text-content :global(h1),
  .text-content :global(h2),
  .text-content :global(h3),
  .text-content :global(h4) {
    margin: var(--spacing-md) 0 var(--spacing-sm) 0;
    color: var(--text-bright);
  }

  .text-content :global(h1) { font-size: 1.4em; border-bottom: 1px solid var(--panel-highlight); padding-bottom: var(--spacing-xs); }
  .text-content :global(h2) { font-size: 1.2em; }
  .text-content :global(h3) { font-size: 1.1em; }
  .text-content :global(h4) { font-size: 1em; }

  .text-content :global(strong) {
    font-weight: 600;
    color: var(--text-bright);
  }

  .text-content :global(em) {
    font-style: italic;
    color: var(--text-muted);
  }

  .text-content :global(a) {
    color: var(--primary);
    text-decoration: none;
  }

  .text-content :global(a:hover) {
    text-decoration: underline;
  }

  .text-content :global(code.inline-code) {
    background: var(--surface);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 0.9em;
    color: var(--accent-pink);
  }

  .text-content :global(blockquote) {
    border-left: 3px solid var(--primary);
    margin: var(--spacing-sm) 0;
    padding: var(--spacing-xs) var(--spacing-md);
    color: var(--text-muted);
    background: var(--panel-alt);
  }

  .text-content :global(ul),
  .text-content :global(ol) {
    margin: var(--spacing-sm) 0;
    padding-left: var(--spacing-xl);
  }

  .text-content :global(li) {
    margin: var(--spacing-xs) 0;
  }

  .text-content :global(hr) {
    border: none;
    border-top: 1px solid var(--panel-highlight);
    margin: var(--spacing-md) 0;
  }

  .cursor-blink {
    animation: blink 1s step-end infinite;
    color: var(--primary);
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }

  .tool-calls {
    margin-top: var(--spacing-md);
    padding-top: var(--spacing-md);
    border-top: 1px solid var(--panel-highlight);
  }
</style>
