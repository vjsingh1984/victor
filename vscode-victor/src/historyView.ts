/**
 * History View Provider
 *
 * Provides a tree view of change history for undo/redo operations.
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

interface HistoryEntry {
    id: string;
    timestamp: string;
    toolName: string;
    description: string;
    fileCount: number;
}

export class HistoryViewProvider implements vscode.TreeDataProvider<HistoryItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<HistoryItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private _history: HistoryEntry[] = [];
    private _refreshInterval: NodeJS.Timeout | null = null;

    constructor(private readonly _client: VictorClient) {
        // Auto-refresh every 10 seconds
        this._refreshInterval = setInterval(() => this.refresh(), 10000);
    }

    async refresh(): Promise<void> {
        try {
            this._history = await this._client.getHistory(20);
            this._onDidChangeTreeData.fire(undefined);
        } catch {
            // Ignore refresh errors
        }
    }

    getTreeItem(element: HistoryItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: HistoryItem): HistoryItem[] {
        if (element) {
            return [];
        }

        if (this._history.length === 0) {
            return [new HistoryItem(
                'No changes yet',
                '',
                'No AI-made changes to show',
                0,
                vscode.TreeItemCollapsibleState.None
            )];
        }

        return this._history.map((entry, index) => new HistoryItem(
            entry.toolName || 'Unknown',
            entry.id,
            entry.description || `${entry.fileCount} file(s) modified`,
            entry.fileCount,
            vscode.TreeItemCollapsibleState.None,
            index === 0 // Most recent is undoable
        ));
    }

    dispose(): void {
        if (this._refreshInterval) {
            clearInterval(this._refreshInterval);
        }
    }
}

class HistoryItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly id: string,
        public readonly description: string,
        public readonly fileCount: number,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly isUndoable: boolean = false
    ) {
        super(label, collapsibleState);

        this.tooltip = `${this.label}\n${this.description}`;
        this.iconPath = new vscode.ThemeIcon(this._getIcon());

        if (this.isUndoable && this.id) {
            this.contextValue = 'undoable';
        }
    }

    private _getIcon(): string {
        const toolIcons: Record<string, string> = {
            'write_file': 'new-file',
            'edit_files': 'edit',
            'edit_file': 'edit',
            'bash': 'terminal',
            'git_commit': 'git-commit',
            'refactor': 'symbol-method',
            'delete_file': 'trash',
        };
        return toolIcons[this.label.toLowerCase()] || 'history';
    }
}

/**
 * Register history view commands
 */
export function registerHistoryCommands(
    context: vscode.ExtensionContext,
    client: VictorClient,
    provider: HistoryViewProvider
): void {
    // Refresh history
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshHistory', () => {
            provider.refresh();
        })
    );

    // Undo from history
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.undoFromHistory', async (item: HistoryItem) => {
            if (item.isUndoable) {
                try {
                    const result = await client.undo();
                    if (result.success) {
                        vscode.window.showInformationMessage(result.message);
                        provider.refresh();
                    } else {
                        vscode.window.showWarningMessage(result.message);
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Undo failed: ${error}`);
                }
            }
        })
    );
}
