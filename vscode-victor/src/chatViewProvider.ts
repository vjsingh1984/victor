/**
 * Chat View Provider
 *
 * Provides the webview-based chat panel for interacting with Victor.
 */

import * as vscode from 'vscode';
import { VictorClient, ChatMessage, ToolCall } from './victorClient';

export class ChatViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
    public static readonly viewType = 'victor.chatView';
    private _view?: vscode.WebviewView;
    private _messages: ChatMessage[] = [];
    private _disposables: vscode.Disposable[] = [];

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _client: VictorClient
    ) {}

    public dispose(): void {
        // Clean up all registered event listeners
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview - store disposable for cleanup
        const messageListener = webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.sendMessage(data.message);
                    break;
                case 'clearHistory':
                    this._messages = [];
                    this._updateMessages();
                    break;
                case 'applyCode':
                    await this._applyCode(data.code, data.file);
                    break;
            }
        });
        this._disposables.push(messageListener);

        // Clean up when webview is disposed
        const viewDisposedListener = webviewView.onDidDispose(() => {
            this._view = undefined;
        });
        this._disposables.push(viewDisposedListener);

        // Initialize with welcome message
        this._postMessage({
            type: 'init',
            messages: this._messages,
        });
    }

    public async sendMessage(content: string): Promise<void> {
        if (!this._view) {
            return;
        }

        // Add user message
        const userMessage: ChatMessage = { role: 'user', content };
        this._messages.push(userMessage);
        this._updateMessages();

        // Show thinking indicator
        this._postMessage({ type: 'thinking', thinking: true });

        try {
            // Stream response
            let assistantContent = '';
            const toolCalls: ToolCall[] = [];

            await this._client.streamChat(
                this._messages,
                (chunk: string) => {
                    assistantContent += chunk;
                    this._postMessage({
                        type: 'stream',
                        content: assistantContent,
                    });
                },
                (toolCall: ToolCall) => {
                    toolCalls.push(toolCall);
                    this._postMessage({
                        type: 'toolCall',
                        toolCall,
                    });
                }
            );

            // Add assistant message
            const assistantMessage: ChatMessage = {
                role: 'assistant',
                content: assistantContent,
                toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
            };
            this._messages.push(assistantMessage);
            this._updateMessages();

        } catch (error) {
            this._postMessage({
                type: 'error',
                message: `Error: ${error}`,
            });
        } finally {
            this._postMessage({ type: 'thinking', thinking: false });
        }
    }

    private async _applyCode(code: string, file?: string): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor to apply code');
            return;
        }

        const edit = new vscode.WorkspaceEdit();
        const selection = editor.selection;

        if (selection.isEmpty) {
            // Insert at cursor
            edit.insert(editor.document.uri, selection.start, code);
        } else {
            // Replace selection
            edit.replace(editor.document.uri, selection, code);
        }

        await vscode.workspace.applyEdit(edit);
        vscode.window.showInformationMessage('Code applied');
    }

    private _updateMessages(): void {
        this._postMessage({
            type: 'messages',
            messages: this._messages,
        });
    }

    private _postMessage(message: unknown): void {
        if (this._view) {
            this._view.webview.postMessage(message);
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const nonce = this._getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <title>Victor Chat</title>
    <style>
        :root {
            --bg-primary: var(--vscode-editor-background);
            --bg-secondary: var(--vscode-sideBar-background);
            --text-primary: var(--vscode-editor-foreground);
            --text-secondary: var(--vscode-descriptionForeground);
            --border-color: var(--vscode-panel-border);
            --user-bg: var(--vscode-button-background);
            --assistant-bg: var(--vscode-editor-inactiveSelectionBackground);
            --accent: var(--vscode-button-background);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--text-primary);
            background: var(--bg-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .message {
            max-width: 90%;
            padding: 10px 14px;
            border-radius: 12px;
            line-height: 1.5;
        }

        .message.user {
            align-self: flex-end;
            background: var(--user-bg);
            color: var(--vscode-button-foreground);
        }

        .message.assistant {
            align-self: flex-start;
            background: var(--assistant-bg);
        }

        .message pre {
            background: var(--vscode-textCodeBlock-background);
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 8px 0;
            font-family: var(--vscode-editor-font-family);
            font-size: 12px;
            position: relative;
        }

        .message pre .copy-btn {
            position: absolute;
            top: 4px;
            right: 4px;
            background: var(--accent);
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .message pre:hover .copy-btn {
            opacity: 1;
        }

        .message code {
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: var(--vscode-editor-font-family);
        }

        .tool-call {
            background: var(--vscode-inputValidation-infoBackground);
            border: 1px solid var(--vscode-inputValidation-infoBorder);
            padding: 8px;
            border-radius: 6px;
            margin: 8px 0;
            font-size: 12px;
        }

        .tool-call-name {
            font-weight: bold;
            color: var(--accent);
        }

        .thinking {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
            padding: 10px;
        }

        .thinking-dots {
            display: flex;
            gap: 4px;
        }

        .thinking-dots span {
            width: 8px;
            height: 8px;
            background: var(--accent);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .input-container {
            padding: 12px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }

        .input-wrapper {
            display: flex;
            gap: 8px;
        }

        #messageInput {
            flex: 1;
            background: var(--vscode-input-background);
            border: 1px solid var(--vscode-input-border);
            color: var(--vscode-input-foreground);
            padding: 10px 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: inherit;
            resize: none;
            min-height: 40px;
            max-height: 150px;
        }

        #messageInput:focus {
            outline: none;
            border-color: var(--accent);
        }

        #sendBtn {
            background: var(--accent);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
        }

        #sendBtn:hover {
            opacity: 0.9;
        }

        #sendBtn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .welcome {
            text-align: center;
            color: var(--text-secondary);
            padding: 40px 20px;
        }

        .welcome h2 {
            margin-bottom: 12px;
            color: var(--text-primary);
        }

        .shortcuts {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            margin-top: 16px;
        }

        .shortcut {
            background: var(--bg-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            border: 1px solid var(--border-color);
        }

        .shortcut:hover {
            border-color: var(--accent);
        }
    </style>
</head>
<body>
    <div class="chat-container" id="chatContainer">
        <div class="welcome" id="welcome">
            <h2>Victor AI Assistant</h2>
            <p>Ask me anything about your code!</p>
            <div class="shortcuts">
                <span class="shortcut" onclick="useShortcut('Explain the selected code')">Explain</span>
                <span class="shortcut" onclick="useShortcut('Refactor this code')">Refactor</span>
                <span class="shortcut" onclick="useShortcut('Write tests for')">Test</span>
                <span class="shortcut" onclick="useShortcut('Fix the bugs in')">Fix</span>
            </div>
        </div>
    </div>

    <div class="thinking" id="thinking" style="display: none;">
        <div class="thinking-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <span>Victor is thinking...</span>
    </div>

    <div class="input-container">
        <div class="input-wrapper">
            <textarea
                id="messageInput"
                placeholder="Ask Victor anything..."
                rows="1"
            ></textarea>
            <button id="sendBtn">Send</button>
        </div>
    </div>

    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const thinking = document.getElementById('thinking');
        const welcome = document.getElementById('welcome');

        let isStreaming = false;
        let streamingElement = null;

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });

        // Send on Enter (Shift+Enter for newline)
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendBtn.addEventListener('click', sendMessage);

        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isStreaming) return;

            vscode.postMessage({ type: 'sendMessage', message });
            messageInput.value = '';
            messageInput.style.height = 'auto';
        }

        function useShortcut(prefix) {
            messageInput.value = prefix + ' ';
            messageInput.focus();
        }

        // Handle messages from extension
        window.addEventListener('message', (event) => {
            const data = event.data;

            switch (data.type) {
                case 'init':
                case 'messages':
                    renderMessages(data.messages);
                    break;

                case 'stream':
                    if (!streamingElement) {
                        streamingElement = createMessageElement('assistant', '');
                        chatContainer.appendChild(streamingElement);
                    }
                    streamingElement.innerHTML = formatContent(data.content);
                    scrollToBottom();
                    break;

                case 'thinking':
                    isStreaming = data.thinking;
                    thinking.style.display = data.thinking ? 'flex' : 'none';
                    sendBtn.disabled = data.thinking;
                    if (!data.thinking) {
                        streamingElement = null;
                    }
                    break;

                case 'toolCall':
                    appendToolCall(data.toolCall);
                    break;

                case 'error':
                    appendError(data.message);
                    break;
            }
        });

        function renderMessages(messages) {
            // Hide welcome if messages exist
            welcome.style.display = messages.length > 0 ? 'none' : 'block';

            // Clear existing messages (except welcome)
            const existingMessages = chatContainer.querySelectorAll('.message');
            existingMessages.forEach(m => m.remove());

            messages.forEach(msg => {
                const el = createMessageElement(msg.role, msg.content);
                chatContainer.appendChild(el);

                if (msg.toolCalls) {
                    msg.toolCalls.forEach(tc => {
                        const tcEl = createToolCallElement(tc);
                        chatContainer.appendChild(tcEl);
                    });
                }
            });

            scrollToBottom();
        }

        function createMessageElement(role, content) {
            const el = document.createElement('div');
            el.className = 'message ' + role;
            el.innerHTML = formatContent(content);
            return el;
        }

        function createToolCallElement(toolCall) {
            const el = document.createElement('div');
            el.className = 'tool-call';
            el.innerHTML = '<span class="tool-call-name">' + toolCall.name + '</span>';
            return el;
        }

        function appendToolCall(toolCall) {
            const el = createToolCallElement(toolCall);
            chatContainer.appendChild(el);
            scrollToBottom();
        }

        function appendError(message) {
            const el = document.createElement('div');
            el.className = 'message assistant';
            el.style.color = 'var(--vscode-errorForeground)';
            el.textContent = message;
            chatContainer.appendChild(el);
            scrollToBottom();
        }

        function formatContent(content) {
            // Basic markdown-like formatting
            let html = escapeHtml(content);

            // Code blocks
            html = html.replace(/\`\`\`(\\w*)\\n([\\s\\S]*?)\`\`\`/g, (match, lang, code) => {
                return '<pre><code class="language-' + lang + '">' + code + '</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></pre>';
            });

            // Inline code
            html = html.replace(/\`([^\`]+)\`/g, '<code>$1</code>');

            // Bold
            html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');

            // Line breaks
            html = html.replace(/\\n/g, '<br>');

            return html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function copyCode(btn) {
            const code = btn.previousSibling.textContent;
            navigator.clipboard.writeText(code);
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy', 1500);
        }

        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>`;
    }

    private _getNonce(): string {
        let text = '';
        const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        for (let i = 0; i < 32; i++) {
            text += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        return text;
    }
}
