/**
 * Victor AI VS Code Extension
 *
 * Provides AI-powered coding assistance through the Victor backend.
 * Features automatic server management for seamless experience.
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';
import { ChatViewProvider } from './chatViewProvider';
import { SemanticSearchProvider } from './semanticSearch';
import { InlineCompletionProvider } from './inlineCompletion';
import { ServerManager, ServerStatus, createServerManager } from './serverManager';
import { HistoryViewProvider, registerHistoryCommands } from './historyView';
import { ToolsViewProvider, registerToolsCommands } from './toolsView';
import { DiffViewProvider, registerDiffCommands } from './diffView';
import { createFileWatcher, FileWatcher, ViewRefreshManager } from './fileWatcher';
import { ContextProvider, registerContextCommands } from './contextProvider';
import { TerminalProvider, TerminalHistoryProvider, registerTerminalCommands } from './terminalProvider';
import { registerCodeActionCommands } from './codeActions';
import { DiagnosticsViewProvider, registerDiagnosticsCommands } from './diagnosticsView';
import { SettingsViewProvider, registerSettingsCommands } from './settingsView';
import { getStore, selectors, AgentMode, ModelInfo } from './state';

// Provider instances (managed centrally, accessed via module)
interface ExtensionProviders {
    victorClient: VictorClient;
    chatViewProvider: ChatViewProvider;
    semanticSearchProvider: SemanticSearchProvider;
    inlineCompletionProvider: InlineCompletionProvider;
    serverManager: ServerManager;
    historyViewProvider: HistoryViewProvider;
    toolsViewProvider: ToolsViewProvider;
    diffViewProvider: DiffViewProvider;
    fileWatcher: FileWatcher;
    refreshManager: ViewRefreshManager;
    contextProvider: ContextProvider;
    terminalProvider: TerminalProvider;
    terminalHistoryProvider: TerminalHistoryProvider;
    diagnosticsViewProvider: DiagnosticsViewProvider;
    settingsViewProvider: SettingsViewProvider;
    statusBarItem: vscode.StatusBarItem;
    serverStatusBarItem: vscode.StatusBarItem;
}

let providers: ExtensionProviders | null = null;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Victor AI extension activating...');

    // Initialize centralized state store
    const store = getStore();
    await store.initialize(context);

    const config = vscode.workspace.getConfiguration('victor');

    // Initialize server manager
    const serverManager = createServerManager();

    // Create server status bar item
    const serverStatusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        99
    );
    serverStatusBarItem.command = 'victor.serverStatus';
    updateServerStatusBar(serverStatusBarItem, ServerStatus.Stopped);
    serverStatusBarItem.show();
    context.subscriptions.push(serverStatusBarItem);

    // Listen for server status changes and update state store
    serverManager.onStatusChange((status) => {
        store.setServerStatus(status);
        updateServerStatusBar(serverStatusBarItem, status);
    });

    // Initialize the Victor client with server URL from state
    const victorClient = new VictorClient(store.select(selectors.serverUrl));

    // Create mode status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.command = 'victor.switchMode';
    updateStatusBar(statusBarItem, store.select(selectors.mode));
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Subscribe to mode changes from state store
    context.subscriptions.push(
        store.subscribeToSelector(selectors.mode, (mode) => {
            updateStatusBar(statusBarItem, mode);
        })
    );

    // Initialize providers
    const chatViewProvider = new ChatViewProvider(context.extensionUri, victorClient);
    const semanticSearchProvider = new SemanticSearchProvider(victorClient);
    const inlineCompletionProvider = new InlineCompletionProvider(victorClient);
    const historyViewProvider = new HistoryViewProvider(victorClient);
    const toolsViewProvider = new ToolsViewProvider();

    // Register chat view
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'victor.chatView',
            chatViewProvider
        )
    );

    // Register history view
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.historyView',
            historyViewProvider
        )
    );
    registerHistoryCommands(context, victorClient, historyViewProvider);

    // Register tools view
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.toolsView',
            toolsViewProvider
        )
    );
    registerToolsCommands(context, toolsViewProvider);

    // Initialize diff view provider
    const diffViewProvider = new DiffViewProvider();
    registerDiffCommands(context, diffViewProvider);
    context.subscriptions.push({ dispose: () => diffViewProvider.dispose() });

    // Initialize file watcher for auto-refresh
    const watcherComponents = createFileWatcher(context);
    const fileWatcher = watcherComponents.fileWatcher;
    const refreshManager = watcherComponents.refreshManager;

    // Register views for auto-refresh on file changes
    refreshManager.registerView('victor.historyView', () => historyViewProvider.refresh());
    refreshManager.registerView('victor.toolsView', () => toolsViewProvider.refresh());

    // Initialize context provider for @-mentions
    const contextProvider = new ContextProvider();
    registerContextCommands(context, contextProvider);

    // Initialize terminal provider
    const terminalProvider = new TerminalProvider();
    const terminalHistoryProvider = new TerminalHistoryProvider(terminalProvider);
    registerTerminalCommands(context, terminalProvider);
    context.subscriptions.push(terminalProvider);
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.terminalHistoryView',
            terminalHistoryProvider
        )
    );

    // Initialize diagnostics view
    const diagnosticsViewProvider = new DiagnosticsViewProvider();
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.diagnosticsView',
            diagnosticsViewProvider
        )
    );
    // Helper function to send messages to chat
    const sendToChat = async (message: string) => {
        await chatViewProvider.sendMessage(message);
        await vscode.commands.executeCommand('victor.chatView.focus');
    };
    registerDiagnosticsCommands(context, diagnosticsViewProvider, sendToChat);
    context.subscriptions.push(diagnosticsViewProvider);

    // Register code action commands
    registerCodeActionCommands(context, victorClient, sendToChat);

    // Initialize settings view
    const settingsViewProvider = new SettingsViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            SettingsViewProvider.viewType,
            settingsViewProvider
        )
    );
    registerSettingsCommands(context);

    // Store all providers in centralized object
    providers = {
        victorClient,
        chatViewProvider,
        semanticSearchProvider,
        inlineCompletionProvider,
        serverManager,
        historyViewProvider,
        toolsViewProvider,
        diffViewProvider,
        fileWatcher,
        refreshManager,
        contextProvider,
        terminalProvider,
        terminalHistoryProvider,
        diagnosticsViewProvider,
        settingsViewProvider,
        statusBarItem,
        serverStatusBarItem,
    };

    // Register inline completions if enabled
    if (config.get('showInlineCompletions', true)) {
        context.subscriptions.push(
            vscode.languages.registerInlineCompletionItemProvider(
                { pattern: '**' },
                inlineCompletionProvider
            )
        );
    }

    // Register commands with providers
    registerCommands(context, providers);

    // Auto-start server if configured
    if (config.get('autoStart', true)) {
        await autoStartServer(serverManager);
    }

    // Register dispose handler
    context.subscriptions.push({
        dispose: () => {
            serverManager.dispose();
        }
    });

    console.log('Victor AI extension activated');
}

/**
 * Auto-start the server with user feedback
 */
async function autoStartServer(serverManager: ServerManager): Promise<void> {
    // First check if server is already running (e.g., user started manually)
    if (await serverManager.checkHealth()) {
        vscode.window.showInformationMessage('Victor: Connected to existing server');
        return;
    }

    // Show progress notification while starting
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Victor: Starting server...',
            cancellable: false
        },
        async (progress) => {
            progress.report({ increment: 0 });

            const started = await serverManager.start();

            if (started) {
                vscode.window.showInformationMessage('Victor: Server started');
            } else {
                const action = await vscode.window.showErrorMessage(
                    'Victor: Failed to start server. Make sure Victor is installed.',
                    'Show Logs',
                    'Manual Start',
                    'Install Help'
                );

                if (action === 'Show Logs') {
                    serverManager.showOutput();
                } else if (action === 'Manual Start') {
                    vscode.commands.executeCommand('victor.startServer');
                } else if (action === 'Install Help') {
                    vscode.env.openExternal(
                        vscode.Uri.parse('https://github.com/vjsingh1984/victor#quick-start')
                    );
                }
            }
        }
    );
}

function registerCommands(context: vscode.ExtensionContext, p: ExtensionProviders) {
    const store = getStore();
    const { victorClient, chatViewProvider, semanticSearchProvider, serverManager } = p;

    // Chat command
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.chat', async () => {
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Explain selected code
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.explain', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            await chatViewProvider.sendMessage(
                `Explain this code:\n\`\`\`\n${text}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Refactor selected code
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refactor', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const suggestion = await vscode.window.showInputBox({
                prompt: 'What refactoring would you like?',
                placeHolder: 'e.g., Extract to function, simplify logic...'
            });

            if (suggestion) {
                await chatViewProvider.sendMessage(
                    `Refactor this code (${suggestion}):\n\`\`\`\n${text}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        })
    );

    // Fix issues in selection
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.fix', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            // Get diagnostics for the selection
            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri)
                .filter(d => selection.contains(d.range))
                .map(d => `- ${d.message}`)
                .join('\n');

            const prompt = diagnostics
                ? `Fix these issues:\n${diagnostics}\n\nIn this code:\n\`\`\`\n${text}\n\`\`\``
                : `Fix any issues in this code:\n\`\`\`\n${text}\n\`\`\``;

            await chatViewProvider.sendMessage(prompt);
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Generate tests
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.test', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const language = editor.document.languageId;
            await chatViewProvider.sendMessage(
                `Generate comprehensive unit tests for this ${language} code:\n\`\`\`${language}\n${text}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Add documentation
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.document', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const language = editor.document.languageId;
            await chatViewProvider.sendMessage(
                `Add documentation/docstrings to this ${language} code:\n\`\`\`${language}\n${text}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Semantic search
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.search', async () => {
            const query = await vscode.window.showInputBox({
                prompt: 'Semantic code search',
                placeHolder: 'e.g., function that handles user authentication'
            });

            if (query) {
                await semanticSearchProvider.search(query);
            }
        })
    );

    // Switch model - now uses state store
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.switchModel', async () => {
            const models: Array<{ label: string; value: string; provider: string }> = [
                { label: 'Claude Sonnet 4', value: 'claude-sonnet-4-20250514', provider: 'anthropic' },
                { label: 'Claude Opus 4.5', value: 'claude-opus-4-5-20251101', provider: 'anthropic' },
                { label: 'GPT-4 Turbo', value: 'gpt-4-turbo', provider: 'openai' },
                { label: 'GPT-4o', value: 'gpt-4o', provider: 'openai' },
                { label: 'Gemini 2.0 Flash', value: 'gemini-2.0-flash', provider: 'google' },
                { label: 'Qwen 2.5 Coder (Local)', value: 'qwen2.5-coder:14b', provider: 'ollama' },
                { label: 'Llama 3.1 (Local)', value: 'llama3.1:8b', provider: 'ollama' },
            ];

            const selected = await vscode.window.showQuickPick(models, {
                placeHolder: 'Select AI model'
            });

            if (selected) {
                try {
                    await victorClient.switchModel(selected.provider, selected.value);

                    // Update centralized state store (also updates VS Code config)
                    await store.setModel({
                        provider: selected.provider as ModelInfo['provider'],
                        modelId: selected.value,
                        displayName: selected.label,
                    });

                    vscode.window.showInformationMessage(`Switched to ${selected.label}`);
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to switch model: ${error}`);
                }
            }
        })
    );

    // Switch mode - now uses state store
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.switchMode', async () => {
            const modes = [
                { label: '$(tools) Build', description: 'Implementation mode - all tools available', value: 'build' as AgentMode },
                { label: '$(search) Plan', description: 'Planning mode - read-only analysis', value: 'plan' as AgentMode },
                { label: '$(eye) Explore', description: 'Exploration mode - understand codebase', value: 'explore' as AgentMode },
            ];

            const selected = await vscode.window.showQuickPick(modes, {
                placeHolder: 'Select agent mode'
            });

            if (selected) {
                try {
                    await victorClient.switchMode(selected.value);

                    // Update centralized state store (also updates VS Code config)
                    await store.setMode(selected.value);

                    vscode.window.showInformationMessage(`Switched to ${selected.value} mode`);
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to switch mode: ${error}`);
                }
            }
        })
    );

    // Server status command
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.serverStatus', async () => {
            const status = serverManager.getStatus();
            const actions: string[] = [];

            switch (status) {
                case ServerStatus.Running:
                    actions.push('Restart Server', 'Stop Server', 'Show Logs');
                    break;
                case ServerStatus.Stopped:
                case ServerStatus.Error:
                    actions.push('Start Server', 'Show Logs');
                    break;
                case ServerStatus.Starting:
                    actions.push('Show Logs');
                    break;
            }

            const action = await vscode.window.showQuickPick(actions, {
                placeHolder: `Victor Server: ${status}`
            });

            if (action === 'Start Server') {
                await serverManager.start();
            } else if (action === 'Stop Server') {
                await serverManager.stop();
            } else if (action === 'Restart Server') {
                await serverManager.restart();
            } else if (action === 'Show Logs') {
                serverManager.showOutput();
            }
        })
    );

    // Undo
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.undo', async () => {
            try {
                const result = await victorClient.undo();
                if (result.success) {
                    vscode.window.showInformationMessage(result.message);
                } else {
                    vscode.window.showWarningMessage(result.message);
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Undo failed: ${error}`);
            }
        })
    );

    // Redo
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.redo', async () => {
            try {
                const result = await victorClient.redo();
                if (result.success) {
                    vscode.window.showInformationMessage(result.message);
                } else {
                    vscode.window.showWarningMessage(result.message);
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Redo failed: ${error}`);
            }
        })
    );

    // Start server (manual)
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.startServer', async () => {
            const started = await serverManager.start();
            if (started) {
                vscode.window.showInformationMessage('Victor server started');
            }
        })
    );

    // Stop server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.stopServer', async () => {
            await serverManager.stop();
            vscode.window.showInformationMessage('Victor server stopped');
        })
    );

    // Restart server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.restartServer', async () => {
            await serverManager.restart();
        })
    );

    // Show server logs
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showServerLogs', async () => {
            serverManager.showOutput();
        })
    );
}

function updateStatusBar(statusBarItem: vscode.StatusBarItem, mode: string): void {
    const icons: Record<string, string> = {
        'build': '$(tools)',
        'plan': '$(search)',
        'explore': '$(eye)'
    };
    statusBarItem.text = `${icons[mode] || '$(hubot)'} Victor: ${mode}`;
    statusBarItem.tooltip = `Victor AI - ${mode} mode\nClick to switch modes`;
}

function updateServerStatusBar(serverStatusBarItem: vscode.StatusBarItem, status: ServerStatus): void {
    const statusConfig: Record<ServerStatus, { icon: string; color?: string; tooltip: string }> = {
        [ServerStatus.Running]: {
            icon: '$(circle-filled)',
            color: 'statusBarItem.prominentBackground',
            tooltip: 'Victor Server: Running\nClick for options'
        },
        [ServerStatus.Starting]: {
            icon: '$(sync~spin)',
            tooltip: 'Victor Server: Starting...'
        },
        [ServerStatus.Stopped]: {
            icon: '$(circle-outline)',
            tooltip: 'Victor Server: Stopped\nClick to start'
        },
        [ServerStatus.Error]: {
            icon: '$(error)',
            color: 'statusBarItem.errorBackground',
            tooltip: 'Victor Server: Error\nClick for options'
        }
    };

    const config = statusConfig[status];
    serverStatusBarItem.text = `${config.icon} Server`;
    serverStatusBarItem.tooltip = config.tooltip;

    if (config.color) {
        serverStatusBarItem.backgroundColor = new vscode.ThemeColor(config.color);
    } else {
        serverStatusBarItem.backgroundColor = undefined;
    }
}

export function deactivate() {
    console.log('Victor AI extension deactivating...');

    // Dispose state store
    const store = getStore();
    store.dispose();

    // Dispose all providers
    if (providers) {
        if (providers.serverManager) {
            providers.serverManager.dispose();
        }
        if (providers.historyViewProvider) {
            providers.historyViewProvider.dispose();
        }
        if (providers.terminalProvider) {
            providers.terminalProvider.dispose();
        }
        if (providers.diagnosticsViewProvider) {
            providers.diagnosticsViewProvider.dispose();
        }
        if (providers.diffViewProvider) {
            providers.diffViewProvider.dispose();
        }
        providers = null;
    }

    console.log('Victor AI extension deactivated');
}

/**
 * Get the extension providers (for use by other modules)
 */
export function getProviders(): ExtensionProviders | null {
    return providers;
}
