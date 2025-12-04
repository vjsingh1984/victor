/**
 * Settings View Provider
 *
 * Provides a webview panel for configuring Victor AI settings:
 * - Provider and model selection
 * - API key management
 * - Tool preferences
 * - Appearance settings
 */

import * as vscode from 'vscode';

interface VictorSettings {
    provider: string;
    model: string;
    mode: string;
    serverPort: number;
    autoStart: boolean;
    showInlineCompletions: boolean;
    semanticSearchEnabled: boolean;
    semanticSearchMaxResults: number;
}

/**
 * Settings View Provider
 */
export class SettingsViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
    public static readonly viewType = 'victor.settingsView';

    private _view?: vscode.WebviewView;
    private _disposables: vscode.Disposable[] = [];

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public dispose(): void {
        // Clean up all registered event listeners
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlContent(webviewView.webview);

        // Handle messages from webview - store disposable for cleanup
        const messageListener = webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'saveSettings':
                    await this._saveSettings(data.settings);
                    break;
                case 'loadSettings':
                    this._sendSettings();
                    break;
                case 'testConnection':
                    await this._testConnection();
                    break;
            }
        });
        this._disposables.push(messageListener);

        // Clean up when webview is disposed
        const viewDisposedListener = webviewView.onDidDispose(() => {
            this._view = undefined;
        });
        this._disposables.push(viewDisposedListener);

        // Send initial settings
        this._sendSettings();
    }

    private _getSettings(): VictorSettings {
        const config = vscode.workspace.getConfiguration('victor');
        return {
            provider: config.get('provider', 'anthropic'),
            model: config.get('model', 'claude-sonnet-4-20250514'),
            mode: config.get('mode', 'build'),
            serverPort: config.get('serverPort', 8765),
            autoStart: config.get('autoStart', true),
            showInlineCompletions: config.get('showInlineCompletions', true),
            semanticSearchEnabled: config.get('semanticSearch.enabled', true),
            semanticSearchMaxResults: config.get('semanticSearch.maxResults', 10),
        };
    }

    private async _saveSettings(settings: VictorSettings): Promise<void> {
        const config = vscode.workspace.getConfiguration('victor');

        await config.update('provider', settings.provider, true);
        await config.update('model', settings.model, true);
        await config.update('mode', settings.mode, true);
        await config.update('serverPort', settings.serverPort, true);
        await config.update('autoStart', settings.autoStart, true);
        await config.update('showInlineCompletions', settings.showInlineCompletions, true);
        await config.update('semanticSearch.enabled', settings.semanticSearchEnabled, true);
        await config.update('semanticSearch.maxResults', settings.semanticSearchMaxResults, true);

        vscode.window.showInformationMessage('Victor settings saved');
    }

    private _sendSettings(): void {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'settings',
                settings: this._getSettings(),
            });
        }
    }

    private async _testConnection(): Promise<void> {
        const config = vscode.workspace.getConfiguration('victor');
        const port = config.get('serverPort', 8765);

        try {
            const response = await fetch(`http://localhost:${port}/health`);
            if (response.ok) {
                this._view?.webview.postMessage({
                    type: 'connectionStatus',
                    status: 'connected',
                    message: 'Server is running',
                });
            } else {
                throw new Error('Server not healthy');
            }
        } catch {
            this._view?.webview.postMessage({
                type: 'connectionStatus',
                status: 'disconnected',
                message: 'Server not running',
            });
        }
    }

    private _getHtmlContent(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Victor Settings</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 10px;
            margin: 0;
        }

        .section {
            margin-bottom: 20px;
        }

        .section-title {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid var(--vscode-widget-border);
        }

        .setting {
            margin-bottom: 12px;
        }

        label {
            display: block;
            margin-bottom: 4px;
            color: var(--vscode-descriptionForeground);
        }

        select, input[type="text"], input[type="number"] {
            width: 100%;
            padding: 6px 8px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            border-radius: 2px;
            box-sizing: border-box;
        }

        select:focus, input:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }

        .checkbox-setting {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .checkbox-setting input {
            width: auto;
        }

        .checkbox-setting label {
            margin-bottom: 0;
            color: var(--vscode-foreground);
        }

        button {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            border-radius: 2px;
            cursor: pointer;
            margin-right: 8px;
        }

        button:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        button.secondary {
            background-color: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }

        .status {
            padding: 8px;
            border-radius: 4px;
            margin-top: 10px;
        }

        .status.connected {
            background-color: var(--vscode-testing-iconPassed);
            color: white;
        }

        .status.disconnected {
            background-color: var(--vscode-testing-iconFailed);
            color: white;
        }

        .button-row {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="section">
        <div class="section-title">Provider Settings</div>

        <div class="setting">
            <label for="provider">AI Provider</label>
            <select id="provider">
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="openai">OpenAI (GPT-4)</option>
                <option value="google">Google (Gemini)</option>
                <option value="ollama">Ollama (Local)</option>
                <option value="lmstudio">LM Studio (Local)</option>
                <option value="vllm">vLLM (Local)</option>
                <option value="xai">xAI (Grok)</option>
            </select>
        </div>

        <div class="setting">
            <label for="model">Model</label>
            <select id="model">
                <optgroup label="Anthropic" data-provider="anthropic">
                    <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
                    <option value="claude-opus-4-5-20251101">Claude Opus 4.5</option>
                </optgroup>
                <optgroup label="OpenAI" data-provider="openai">
                    <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                </optgroup>
                <optgroup label="Google" data-provider="google">
                    <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                    <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                </optgroup>
                <optgroup label="Ollama" data-provider="ollama">
                    <option value="qwen2.5-coder:14b">Qwen 2.5 Coder 14B</option>
                    <option value="llama3.1:8b">Llama 3.1 8B</option>
                    <option value="codellama:7b">Code Llama 7B</option>
                </optgroup>
            </select>
        </div>

        <div class="setting">
            <label for="mode">Agent Mode</label>
            <select id="mode">
                <option value="build">Build - Full implementation mode</option>
                <option value="plan">Plan - Read-only analysis</option>
                <option value="explore">Explore - Codebase exploration</option>
            </select>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Server Settings</div>

        <div class="setting">
            <label for="serverPort">Server Port</label>
            <input type="number" id="serverPort" min="1024" max="65535" value="8765">
        </div>

        <div class="setting checkbox-setting">
            <input type="checkbox" id="autoStart" checked>
            <label for="autoStart">Auto-start server on extension activation</label>
        </div>

        <button class="secondary" onclick="testConnection()">Test Connection</button>
        <div id="connectionStatus"></div>
    </div>

    <div class="section">
        <div class="section-title">Features</div>

        <div class="setting checkbox-setting">
            <input type="checkbox" id="showInlineCompletions" checked>
            <label for="showInlineCompletions">Show inline code completions</label>
        </div>

        <div class="setting checkbox-setting">
            <input type="checkbox" id="semanticSearchEnabled" checked>
            <label for="semanticSearchEnabled">Enable semantic code search</label>
        </div>

        <div class="setting">
            <label for="semanticSearchMaxResults">Semantic search max results</label>
            <input type="number" id="semanticSearchMaxResults" min="1" max="50" value="10">
        </div>
    </div>

    <div class="button-row">
        <button onclick="saveSettings()">Save Settings</button>
        <button class="secondary" onclick="loadSettings()">Reset</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;

            switch (message.type) {
                case 'settings':
                    applySettings(message.settings);
                    break;
                case 'connectionStatus':
                    showConnectionStatus(message.status, message.message);
                    break;
            }
        });

        function applySettings(settings) {
            document.getElementById('provider').value = settings.provider;
            document.getElementById('model').value = settings.model;
            document.getElementById('mode').value = settings.mode;
            document.getElementById('serverPort').value = settings.serverPort;
            document.getElementById('autoStart').checked = settings.autoStart;
            document.getElementById('showInlineCompletions').checked = settings.showInlineCompletions;
            document.getElementById('semanticSearchEnabled').checked = settings.semanticSearchEnabled;
            document.getElementById('semanticSearchMaxResults').value = settings.semanticSearchMaxResults;

            updateModelOptions();
        }

        function getSettings() {
            return {
                provider: document.getElementById('provider').value,
                model: document.getElementById('model').value,
                mode: document.getElementById('mode').value,
                serverPort: parseInt(document.getElementById('serverPort').value),
                autoStart: document.getElementById('autoStart').checked,
                showInlineCompletions: document.getElementById('showInlineCompletions').checked,
                semanticSearchEnabled: document.getElementById('semanticSearchEnabled').checked,
                semanticSearchMaxResults: parseInt(document.getElementById('semanticSearchMaxResults').value),
            };
        }

        function saveSettings() {
            vscode.postMessage({
                type: 'saveSettings',
                settings: getSettings(),
            });
        }

        function loadSettings() {
            vscode.postMessage({ type: 'loadSettings' });
        }

        function testConnection() {
            vscode.postMessage({ type: 'testConnection' });
        }

        function showConnectionStatus(status, message) {
            const statusDiv = document.getElementById('connectionStatus');
            statusDiv.className = 'status ' + status;
            statusDiv.textContent = message;
        }

        function updateModelOptions() {
            const provider = document.getElementById('provider').value;
            const modelSelect = document.getElementById('model');

            // Show/hide optgroups based on provider
            const optgroups = modelSelect.querySelectorAll('optgroup');
            optgroups.forEach(group => {
                const options = group.querySelectorAll('option');
                options.forEach(opt => {
                    opt.style.display = group.dataset.provider === provider ? '' : 'none';
                });
            });

            // Select first visible option if current is hidden
            const currentOption = modelSelect.querySelector('option:checked');
            if (currentOption && currentOption.style.display === 'none') {
                const firstVisible = modelSelect.querySelector('option[style=""]');
                if (firstVisible) {
                    modelSelect.value = firstVisible.value;
                }
            }
        }

        // Update models when provider changes
        document.getElementById('provider').addEventListener('change', updateModelOptions);

        // Request initial settings
        loadSettings();
    </script>
</body>
</html>`;
    }
}

/**
 * Register settings commands
 */
export function registerSettingsCommands(context: vscode.ExtensionContext): void {
    // Open settings
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.openSettings', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'victor');
        })
    );

    // Open settings JSON
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.openSettingsJson', () => {
            vscode.commands.executeCommand('workbench.action.openSettingsJson');
        })
    );
}
